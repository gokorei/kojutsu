"""Tests for webhook endpoint: signature verification and event handling."""

import asyncio
import hashlib
import hmac
import itertools
import sqlite3
from contextlib import closing
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import kojutsu.webhook.server as server_module
from kojutsu import relay_worker
from kojutsu import runtime as runtime_module
from kojutsu.core.outbox import OutboxOwnershipError
from kojutsu.integrations.webhook_client import (
    GitHubWebhookClient,
    GitHubWebhookManager,
    redact_webhook_url,
    validate_webhook_url,
)
from kojutsu.webhook import app
from kojutsu.webhook.lifecycle import WebhookLifecycle, create_webhook_app
from kojutsu.webhook.server import _sqlite_writable, _verify_signature

client = TestClient(app)
WEBHOOK_SECRET = "test-secret"


def _signed_headers(body: bytes, event: str) -> dict[str, str]:
    digest = hmac.new(WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return {
        "Content-Type": "application/json",
        "X-GitHub-Event": event,
        "X-Hub-Signature-256": f"sha256={digest}",
    }


def test_verify_signature_valid() -> None:
    """Valid HMAC signature passes."""
    body = b'{"action":"created"}'
    secret = "my-secret"
    sig = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert _verify_signature(body, sig, secret) is True


def test_verify_signature_invalid() -> None:
    """Wrong signature fails."""
    body = b'{"action":"created"}'
    assert _verify_signature(body, "sha256=wrong", "my-secret") is False


def test_webhook_ignores_non_issue_comment(monkeypatch) -> None:
    """Non issue_comment/pull_request event returns ignored."""
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", WEBHOOK_SECRET)
    body = b'{"action":"opened"}'
    r = client.post(
        "/webhook/github",
        content=body,
        headers=_signed_headers(body, "push"),
    )
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ignored"
    assert data.get("event") == "push"


def test_webhook_ignores_issue_comment_not_created(monkeypatch) -> None:
    """issue_comment with action other than created is ignored."""
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", WEBHOOK_SECRET)
    body = (
        b'{"action":"edited","comment":{"id":1,"body":"x",'
        b'"user":{"login":"u"},"created_at":"2023-01-01T00:00:00Z"},'
        b'"issue":{"number":1},"repository":{"full_name":"org/repo"}}'
    )
    r = client.post(
        "/webhook/github",
        content=body,
        headers=_signed_headers(body, "issue_comment"),
    )
    assert r.status_code == 200
    assert r.json().get("status") == "ignored"


def test_webhook_rejects_bad_signature_when_secret_set() -> None:
    """When webhook secret is set, invalid signature returns 401."""
    with patch("kojutsu.webhook.server.get_settings") as mock_settings:
        mock_settings.return_value.github_webhook_secret = "secret"
        mock_settings.return_value.github_token = ""
        r = client.post(
            "/webhook/github",
            content=b'{"action":"created"}',
            headers={
                "X-GitHub-Event": "issue_comment",
                "X-Hub-Signature-256": "sha256=invalid",
                "Content-Type": "application/json",
            },
        )
        assert r.status_code == 401


def test_create_webhook_app_includes_all_routes() -> None:
    app = create_webhook_app()
    app_client = TestClient(app)

    assert app_client.get("/webhook/health").status_code == 200
    assert app_client.get("/webhook/status").status_code == 503


def test_webhook_ready_builds_and_verifies_owned_runtime(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", WEBHOOK_SECRET)
    monkeypatch.setenv("GITHUB_WEBHOOK_ALLOWED_REPOSITORIES", "org/repo")
    monkeypatch.setenv("TANSEKI_URL", "https://tanseki.test")

    missing_token = client.get("/webhook/ready")
    monkeypatch.setenv("GITHUB_TOKEN", "x")

    class Runtime:
        def status(self) -> dict[str, object]:
            return {"tanseki_reachable": True, "outbox_path": "/owned/outbox.db"}

    get_runtime_calls: list[int] = []
    monkeypatch.setattr(
        server_module, "get_runtime", lambda: get_runtime_calls.append(1) or Runtime()
    )
    ready = client.get("/webhook/ready")
    monkeypatch.setattr(server_module, "_sqlite_writable", lambda path: False)
    unwritable = client.get("/webhook/ready")

    assert missing_token.status_code == 503
    assert missing_token.json() == {"detail": "GITHUB_TOKEN is not configured"}
    assert ready.status_code == 200
    assert ready.json()["sqlite_writable"] is True
    assert ready.json()["runtime_ready"] is True
    assert ready.json()["outbox_owned"] is True
    assert get_runtime_calls == [1]
    assert unwritable.status_code == 503
    assert unwritable.json() == {"detail": "Local SQLite storage is not writable"}


def test_webhook_ready_fails_closed_when_another_process_owns_outbox(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", WEBHOOK_SECRET)
    monkeypatch.setenv("GITHUB_WEBHOOK_ALLOWED_REPOSITORIES", "org/repo")
    monkeypatch.setenv("TANSEKI_URL", "https://tanseki.test")

    def unavailable() -> object:
        raise OutboxOwnershipError("outbox owned by worker 2")

    monkeypatch.setattr(server_module, "get_runtime", unavailable)

    response = client.get("/webhook/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "Outbox is owned by another process"}


@pytest.mark.parametrize("allowlist", ["", "*"])
def test_readiness_rejects_empty_or_wildcard_repository_allowlist(
    monkeypatch, allowlist: str
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", WEBHOOK_SECRET)
    monkeypatch.setenv("GITHUB_WEBHOOK_ALLOWED_REPOSITORIES", allowlist)
    monkeypatch.setenv("TANSEKI_URL", "https://tanseki.test")

    response = client.get("/webhook/ready")

    assert response.status_code == 503
    assert "No exact repositories" in response.json()["detail"]


def test_status_requires_authentication_and_uses_configured_url(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", WEBHOOK_SECRET)
    monkeypatch.setenv("GITHUB_WEBHOOK_ALLOWED_REPOSITORIES", "org/repo")
    configured_url = "https://hooks.example.test/custom/github"
    protected_app = create_webhook_app(webhook_url=configured_url)
    protected_client = TestClient(protected_app)

    unauthorized = protected_client.get("/webhook/status")
    authorized = protected_client.get(
        "/webhook/status",
        headers={"Authorization": f"Bearer {WEBHOOK_SECRET}"},
    )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    assert authorized.json()["webhook_url"] == configured_url
    assert "registry_path" not in authorized.json()
    assert "tanseki_url" not in authorized.json()


def test_status_rejects_empty_repository_allowlist(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", WEBHOOK_SECRET)
    monkeypatch.setenv("GITHUB_WEBHOOK_ALLOWED_REPOSITORIES", "*")
    protected_client = TestClient(create_webhook_app())

    response = protected_client.get(
        "/webhook/status",
        headers={"Authorization": f"Bearer {WEBHOOK_SECRET}"},
    )

    assert response.status_code == 503
    assert "No exact repositories" in response.json()["detail"]


@pytest.mark.asyncio
async def test_public_lifecycle_refuses_to_start_without_webhook_secret(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    lifecycle = WebhookLifecycle(app)
    lifecycle.configure("https://hooks.example.test/webhook/github")

    with pytest.raises(ValueError, match="GITHUB_WEBHOOK_SECRET"):
        await lifecycle.startup()


@pytest.mark.asyncio
async def test_lifecycle_shutdown_without_github_token() -> None:
    lifecycle = WebhookLifecycle(app)
    lifecycle.configure("http://localhost:8000/webhook/github")

    await lifecycle.startup()
    await lifecycle.shutdown()


@pytest.mark.asyncio
async def test_initial_relay_startup_propagates_outbox_ownership_error(monkeypatch) -> None:
    protected_app = create_webhook_app()

    async def unavailable() -> int:
        raise OutboxOwnershipError("outbox owned by worker 2")

    monkeypatch.setattr(relay_worker, "relay_once", unavailable)
    startup = protected_app.router.on_startup[-1]

    with pytest.raises(OutboxOwnershipError, match="worker 2"):
        await startup()

    assert not hasattr(protected_app.state, "relay_task")


@pytest.mark.asyncio
async def test_webhook_processing_guard_bounds_concurrency_and_drains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_module, "WEBHOOK_MAX_CONCURRENT_REQUESTS", 1)
    guard = server_module._WebhookProcessingGuard()
    assert guard.try_begin() is True
    assert guard.try_begin() is False
    guard.finish()
    assert await asyncio.to_thread(guard.wait_for_drain, 0) is True

    assert guard.try_begin() is True
    drain = asyncio.create_task(asyncio.to_thread(guard.wait_for_drain, 0.01))
    assert await drain is False
    guard.finish()
    assert await asyncio.to_thread(guard.wait_for_drain, 0.1) is True


def test_webhook_processing_guard_rate_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_module, "WEBHOOK_RATE_LIMIT", 1)
    guard = server_module._WebhookProcessingGuard()

    assert guard.try_begin() is True
    guard.finish()
    assert guard.try_begin() is False


@pytest.mark.asyncio
async def test_lifecycle_defers_runtime_reset_until_inflight_webhooks_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected_app = create_webhook_app()
    reset_calls: list[bool] = []

    async def relay_drained() -> bool:
        return True

    async def webhook_drained() -> bool:
        return False

    monkeypatch.setattr("kojutsu.relay_worker.wait_for_relay_shutdown", relay_drained)
    monkeypatch.setattr(server_module, "wait_for_webhook_processing", webhook_drained)
    monkeypatch.setattr(runtime_module, "reset_runtime", lambda: reset_calls.append(True))
    shutdown = protected_app.router.on_shutdown[-1]

    await shutdown()

    assert reset_calls == []


def test_webhook_registration_updates_mismatched_hook(monkeypatch) -> None:
    github_client = GitHubWebhookClient("token")
    updates = []
    monkeypatch.setattr(
        github_client,
        "find_webhook_by_url",
        lambda owner, repo, url: {
            "id": 7,
            "active": False,
            "config": {"url": url, "content_type": "json"},
            "events": ["issues"],
        },
    )
    monkeypatch.setattr(
        github_client,
        "update_webhook",
        lambda owner, repo, webhook_id, config: updates.append(config) or {"id": 7},
    )

    result = github_client.register_webhook(
        "org",
        "repo",
        "https://example.test/webhook/github",
        "secret",
    )

    assert result == {"id": 7}
    assert len(updates) == 1
    # Registration must subscribe to the review events, not just comments and PRs:
    # a hook without them never receives a verdict or an inline comment, and the
    # missing capture would look like a bug in the capture path rather than a
    # missing subscription.
    assert updates[0].events == [
        "issue_comment",
        "pull_request",
        "pull_request_review",
        "pull_request_review_comment",
    ]
    assert updates[0].secret == "secret"


def test_manager_only_auto_cleans_up_hooks_it_created(monkeypatch) -> None:
    manager = GitHubWebhookManager("token", "secret")
    target_url = "https://example.test/webhook/github"
    existing = {
        "id": 7,
        "active": True,
        "config": {
            "url": target_url,
            "content_type": "json",
            "secret": "secret",
            "insecure_ssl": "0",
        },
        # Already subscribed to everything, so no update should be issued. The
        # full list is listed because a partial match is exactly the case that
        # silently drops review events.
        "events": [
            "issue_comment",
            "pull_request",
            "pull_request_review",
            "pull_request_review_comment",
        ],
    }
    deleted: list[int] = []

    def list_webhooks(_owner: str, _repo: str) -> list[dict[str, object]]:
        return [existing] if _repo == "existing" else []

    def create_webhook(_owner: str, _repo: str, _config: object) -> dict[str, object]:
        return {"id": 8}

    def delete_webhook(_owner: str, _repo: str, webhook_id: str) -> None:
        deleted.append(int(webhook_id))

    monkeypatch.setattr(manager.client, "list_webhooks", list_webhooks)
    monkeypatch.setattr(manager.client, "create_webhook", create_webhook)
    monkeypatch.setattr(manager.client, "delete_webhook", delete_webhook)

    results = manager.register_all_repos(["org/existing", "org/new"], target_url)

    assert results["org/existing"] is existing
    assert manager.get_registered_repos() == ["org/existing", "org/new"]
    assert manager.get_created_repos() == ["org/new"]
    assert manager.unregister_for_repo("org", "existing", target_url) is True
    assert deleted == [7]


def test_webhook_url_rejects_credentials_query_and_fragment() -> None:
    with pytest.raises(ValueError):
        validate_webhook_url("https://user:password@example.test/hook")
    with pytest.raises(ValueError):
        validate_webhook_url("https://example.test/hook?token=secret")
    with pytest.raises(ValueError):
        validate_webhook_url("https://example.test/hook#secret")
    assert (
        redact_webhook_url("https://user:password@example.test/hook?token=secret#fragment")
        == "https://example.test/hook"
    )


def test_readiness_storage_check_does_not_mutate_existing_database(tmp_path) -> None:
    path = tmp_path / "registry.db"
    with closing(sqlite3.connect(path)) as db:
        db.execute("CREATE TABLE questions (id TEXT PRIMARY KEY)")
        db.execute("INSERT INTO questions (id) VALUES ('q1')")
        db.commit()

    assert _sqlite_writable(str(path)) is True

    with closing(sqlite3.connect(path)) as db:
        assert db.execute("SELECT id FROM questions").fetchall() == [("q1",)]
        assert db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall() == [("questions",)]


def test_visibility_wait_yields_the_event_loop(monkeypatch) -> None:
    """Three fruitless polls must wait without pinning the loop thread.

    The waits are ``asyncio.sleep`` on the request path: a blocking sleep here
    would stall every concurrent delivery for the full retry window, which is
    the starvation this test pins -- the ticker's worst gap must stay far below
    the retry delay.
    """
    monkeypatch.setattr(server_module, "COMMENT_VISIBILITY_DELAY_SECONDS", 0.1)

    class _NoComments:
        def __init__(self, token: str) -> None:
            self.token = token

        def __enter__(self):
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def list_issue_comments(self, owner: str, repo: str, issue_number: int):
            from kojutsu.integrations.github import PagedResult

            return PagedResult(items=[], truncated=False)

    monkeypatch.setattr(server_module, "GitHubClient", _NoComments)

    async def main() -> tuple[object, list[float]]:
        loop = asyncio.get_running_loop()
        stamps: list[float] = [loop.time()]

        async def ticker() -> None:
            for _ in range(60):
                await asyncio.sleep(0.005)
                stamps.append(loop.time())

        tick_task = asyncio.create_task(ticker())
        parent = await server_module._await_parent_comment("tok", "o", "r", 1, 2, "q1")
        await tick_task
        return parent, stamps

    parent, stamps = asyncio.run(main())
    assert parent is None
    gaps = [b - a for a, b in itertools.pairwise(stamps)]
    assert max(gaps) < 0.08, f"event loop stalled during visibility wait: {max(gaps):.3f}s"
