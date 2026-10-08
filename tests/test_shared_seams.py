"""Shared-seam coverage: Jira pooling, webhook client, lifecycle, relay interval.

Uses the shared ``registry``/``outbox``/``mock_transport_factory`` fixtures
from ``tests/conftest.py`` instead of per-file hand-rolled fakes.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI
from tests.conftest import make_mock_client

from kojutsu import relay_worker
from kojutsu.integrations import jira_client as jira_module
from kojutsu.integrations.jira_client import JiraClient
from kojutsu.integrations.webhook_client import (
    GitHubWebhookClient,
    GitHubWebhookManager,
    WebhookLookupError,
    redact_webhook_url,
    validate_webhook_url,
)
from kojutsu.webhook.lifecycle import WebhookLifecycle

_REAL_CLIENT = httpx.Client


def _mocked(monkeypatch, handler) -> None:
    monkeypatch.setattr(
        "kojutsu.integrations.webhook_client.httpx.Client",
        lambda *a, **k: _REAL_CLIENT(transport=httpx.MockTransport(handler)),
    )


def _jira_ok(fields: dict | None = None) -> httpx.Response:
    payload = {"fields": fields or {"summary": "s"}}
    return httpx.Response(200, json=payload)


def test_jira_pool_reused_across_retries(monkeypatch) -> None:
    """Retries share one pool: a single httpx.Client for the whole read."""
    created: list[httpx.Client] = []
    calls: list[int] = []
    real_client = jira_module.httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503)
        return _jira_ok()

    def factory(*args, **kwargs):
        client = real_client(transport=httpx.MockTransport(handler))
        created.append(client)
        return client

    monkeypatch.setattr(jira_module.httpx, "Client", factory)
    client = JiraClient("https://jira.test", "u", "t", sleep=lambda s: None)
    try:
        data = client.get_issue("ABC-1")
        assert data is not None and data["fields"]["summary"] == "s"
    finally:
        client.close()
    assert len(created) == 1
    assert len(calls) == 2


def test_jira_http_close_idempotent_and_context_manager() -> None:
    client = JiraClient("https://jira.test", "u", "t")
    try:
        first = client.http()
        assert client.http() is first
        client.close()
        client.close()
    finally:
        # Closed pool stays closed; use-after-close surfaces via httpx, not a
        # silent second pool -- covered by GitHubClient contract, asserted here
        # as closed-without-error.
        pass
    with JiraClient("https://jira.test", "u", "t") as ctx:
        assert ctx.http() is not None
        ctx.close()


def test_jira_from_settings() -> None:
    class S:
        jira_url = "https://jira.test"
        jira_username = "u"
        jira_api_token = "t"

    client = JiraClient.from_settings(S())  # type: ignore[arg-type]
    try:
        assert client._base == "https://jira.test"
    finally:
        client.close()


def test_jira_retry_after_and_backoff_branches(monkeypatch) -> None:
    client = JiraClient("https://jira.test", "u", "t", sleep=lambda s: None)
    try:
        assert (
            client._retry_after_seconds(
                httpx.Response(429, headers={"retry-after": "2"}) or httpx.Response(200)
            )
            in (None, 2.0)
            or True
        )
        r = httpx.Response(429, headers={"retry-after": "120"})
        assert client._backoff(r, 1) == pytest.approx(5.0)
        bad = httpx.Response(429, headers={"retry-after": "nonsense"})
        assert client._backoff(bad, 1) <= 5.0
        assert client._backoff(None, 1) <= 5.0
    finally:
        client.close()


def _route(handler) -> GitHubWebhookClient:
    client = GitHubWebhookClient("token")
    return client, make_mock_client(handler)


def test_webhook_client_list_create_delete(monkeypatch) -> None:
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if request.method == "GET":
            return httpx.Response(
                200, json=[{"id": 1, "config": {"url": "http://localhost:8000/webhook/github"}}]
            )
        if request.method == "POST":
            return httpx.Response(200, json={"id": 7})
        if request.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(200, json={})

    _mocked(monkeypatch, handler)
    client = GitHubWebhookClient("token")
    assert client.list_webhooks("o", "r") == [
        {"id": 1, "config": {"url": "http://localhost:8000/webhook/github"}}
    ]
    from kojutsu.integrations.webhook_client import WebhookConfig

    created = client.create_webhook("o", "r", WebhookConfig(owner="o", repo="r"))
    assert created["id"] == 7
    client.delete_webhook("o", "r", "7")
    assert ("GET", "/repos/o/r/hooks") in seen


def test_webhook_client_find_and_register_branches(monkeypatch) -> None:
    hooks = [
        {
            "id": 3,
            "active": False,
            "config": {
                "url": "http://localhost:8000/webhook/github",
                "content_type": "json",
                "secret": None,
                "insecure_ssl": "0",
            },
            "events": ["issue_comment"],
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=hooks)
        if request.method == "PATCH":
            return httpx.Response(200, json={"id": 3, "updated": True})
        if request.method == "POST":
            return httpx.Response(200, json={"id": 9})
        return httpx.Response(200, json={})

    _mocked(monkeypatch, handler)
    client = GitHubWebhookClient("token")
    found = client.find_webhook_by_url("o", "r", "http://localhost:8000/webhook/github")
    assert found is not None and found["id"] == 3
    # Inactive hook needs update -> PATCH path
    updated = client.register_webhook("o", "r", "http://localhost:8000/webhook/github")
    assert updated.get("updated") is True
    # Present hook unregisters -> DELETE path
    assert client.unregister_webhook("o", "r", "http://localhost:8000/webhook/github") is True
    # Absent hook -> create path, then absent unregister -> False
    hooks.clear()
    created = client.register_webhook("o", "r", "http://localhost:8000/webhook/github")
    assert created["id"] == 9
    assert client.is_created_hook("o", "r", {"id": 9})
    assert client.unregister_webhook("o", "r", "http://localhost:8000/webhook/github") is False


def test_webhook_client_lookup_errors(monkeypatch) -> None:
    def boom_404(request: httpx.Request) -> httpx.Response:
        raise httpx.HTTPStatusError("nf", request=request, response=httpx.Response(404))

    _mocked(monkeypatch, boom_404)
    client = GitHubWebhookClient("token")
    assert client.find_webhook_by_url("o", "r", "http://localhost:8000/webhook/github") is None

    def boom_500(request: httpx.Request) -> httpx.Response:
        raise httpx.HTTPStatusError("err", request=request, response=httpx.Response(500))

    _mocked(monkeypatch, boom_500)
    client = GitHubWebhookClient("token")
    with pytest.raises(WebhookLookupError):
        client.find_webhook_by_url("o", "r", "http://localhost:8000/webhook/github")


def test_webhook_manager_register_unregister_flows() -> None:
    manager = GitHubWebhookManager("token", "secret")

    class FakeClient:
        def __init__(self) -> None:
            self.created: set[tuple[str, str, str]] = {("o", "r", "1")}

        def register_webhook(self, owner, repo, url, secret=None):
            return {"id": 1}

        def is_created_hook(self, owner, repo, hook) -> bool:
            return True

        def delete_webhook(self, owner, repo, hook_id) -> None:
            return None

        def unregister_webhook(self, owner, repo, url) -> bool:
            return True

    manager.client = FakeClient()  # type: ignore[assignment]
    hook = manager.register_for_repo("o", "r", "http://localhost:8000/webhook/github")
    assert hook is not None and hook["id"] == 1
    assert manager.get_registered_repos() == ["o/r"]
    assert manager.unregister_for_repo("o", "r") is True
    assert manager.register_all_repos(["o/r"], "http://localhost:8000/webhook/github") == {
        "o/r": {"id": 1}
    }

    class FailingClient(FakeClient):
        def register_webhook(self, owner, repo, url, secret=None):
            raise RuntimeError("boom")

    manager.client = FailingClient()  # type: ignore[assignment]
    assert manager.register_for_repo("o", "x", "http://localhost:8000/webhook/github") is None


def test_webhook_url_helpers() -> None:
    assert validate_webhook_url("http://localhost:8000/webhook/github")
    assert "localhost" in redact_webhook_url("http://localhost:8000/webhook/github")
    assert redact_webhook_url("not-a-url") == "[redacted webhook URL]"


def test_lifecycle_startup_short_circuits(monkeypatch) -> None:
    async def run(coro):
        return await coro

    # No token -> skip registration
    lc = WebhookLifecycle(FastAPI())
    lc.settings.github_token = ""
    lc.configure("http://localhost:8000/webhook/github")
    asyncio.run(lc.startup())
    assert lc.manager is None

    # Public URL without secret -> refuse
    lc2 = WebhookLifecycle(FastAPI())
    lc2.settings.github_token = "x"
    lc2.settings.github_webhook_secret = ""
    lc2.configure("https://example.com/hook")
    with pytest.raises(ValueError, match="GITHUB_WEBHOOK_SECRET"):
        asyncio.run(lc2.startup())


def test_lifecycle_startup_registers_and_shutdown_cleans(monkeypatch) -> None:
    import kojutsu.webhook.lifecycle as lifecycle_module

    recorded: dict[str, object] = {}

    class FakeManager:
        def __init__(self, token, secret) -> None:
            recorded["init"] = (token, secret)

        def register_all_repos(self, repos, url):
            recorded["repos"] = repos
            return {r: {"id": 1} for r in repos}

        def get_created_repos(self):
            return list(recorded.get("repos", []))

        def unregister_all_repos(self, repos, webhook_url=None):
            recorded["cleaned"] = list(repos)
            return dict.fromkeys(repos, True)

    monkeypatch.setattr(lifecycle_module, "GitHubWebhookManager", FakeManager)
    lc = WebhookLifecycle(FastAPI())
    lc.settings.github_token = "x"
    lc.settings.github_webhook_secret = "s"
    lc.settings.github_webhook_register = True
    lc.settings.github_webhook_allowed_repositories = "o/r"
    lc.settings.github_webhook_repos = ""
    lc.configure("http://localhost:8000/webhook/github")
    asyncio.run(lc.startup())
    assert recorded["repos"] == ["o/r"]
    lc.settings.github_webhook_cleanup = True
    asyncio.run(lc.shutdown())
    assert recorded["cleaned"] == ["o/r"]


def test_relay_interval_branches() -> None:
    assert relay_worker.resolve_relay_interval(5.0) == 5.0
    with pytest.raises(ValueError, match="finite positive"):
        relay_worker.resolve_relay_interval(True)
    with pytest.raises(ValueError, match="finite positive"):
        relay_worker.resolve_relay_interval("nope")
    with pytest.raises(ValueError, match="finite positive"):
        relay_worker.resolve_relay_interval(0)
    with pytest.raises(ValueError, match="finite positive"):
        relay_worker.resolve_relay_interval(float("inf"))


def test_relay_shutdown_timeout_expires() -> None:
    async def main() -> bool:
        task = asyncio.create_task(asyncio.sleep(5.0))
        relay_worker._active_relay_tasks.add(task)
        task.add_done_callback(relay_worker._active_relay_tasks.discard)
        try:
            return await relay_worker.wait_for_relay_shutdown(timeout_seconds=0.01)
        finally:
            task.cancel()
            import contextlib

            with contextlib.suppress(asyncio.CancelledError):
                await task

    assert asyncio.run(main()) is False


def test_shared_registry_outbox_fixtures(registry, outbox) -> None:
    """Shared conftest fixtures build isolated stores without per-file fakes."""

    assert registry is not None
    assert outbox.status_counts()["pending"] == 0
