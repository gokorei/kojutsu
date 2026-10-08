"""Tests for normalized Jira failures."""

import httpx
import pytest

from kojutsu.integrations.jira_client import (
    MAX_JIRA_TIMEOUT_SECONDS,
    JiraAuthenticationError,
    JiraClient,
    JiraConfigurationError,
    JiraPayloadError,
    JiraUnavailableError,
)


def _mock_jira(monkeypatch, response: httpx.Response) -> None:
    import kojutsu.integrations.jira_client as module

    real_client = module.httpx.Client
    monkeypatch.setattr(
        module.httpx,
        "Client",
        lambda *args, **kwargs: real_client(
            transport=httpx.MockTransport(lambda request: response)
        ),
    )


def test_remote_http_is_rejected_and_loopback_http_is_accepted() -> None:
    with pytest.raises(JiraConfigurationError, match="HTTPS"):
        JiraClient("http://jira.example.com", "user", "token")

    client = JiraClient("http://127.0.0.1:8080", "user", "token")
    assert client is not None


def test_jira_authentication_failure_is_surfaced(monkeypatch) -> None:
    _mock_jira(monkeypatch, httpx.Response(401))
    with pytest.raises(JiraAuthenticationError, match="authentication"):
        JiraClient("https://jira.test", "user", "token").get_issue_fields("ABC-1")


def test_jira_unavailable_failure_is_surfaced(monkeypatch) -> None:
    _mock_jira(monkeypatch, httpx.Response(503))
    with pytest.raises(JiraUnavailableError):
        JiraClient("https://jira.test", "user", "token").get_issue_fields("ABC-1")


def test_jira_invalid_shape_is_normalized(monkeypatch) -> None:
    _mock_jira(monkeypatch, httpx.Response(200, json=[]))
    with pytest.raises(JiraPayloadError):
        JiraClient("https://jira.test", "user", "token").get_issue_fields("ABC-1")


def test_jira_timeout_is_bounded(monkeypatch) -> None:
    import kojutsu.integrations.jira_client as module

    captured = {}
    real_client = module.httpx.Client

    def client_factory(*args, **kwargs):
        captured.update(kwargs)
        return real_client(transport=httpx.MockTransport(lambda request: httpx.Response(404)))

    monkeypatch.setattr(module.httpx, "Client", client_factory)

    JiraClient("https://jira.test", "user", "token", timeout_seconds=60).get_issue("ABC-1")

    assert captured["timeout"] == MAX_JIRA_TIMEOUT_SECONDS


def _mock_jira_sequence(monkeypatch, effects: list) -> list:
    """Serve a script of responses (or exceptions) in order, recording calls."""
    import kojutsu.integrations.jira_client as module

    calls: list[httpx.Request] = []
    remaining = list(effects)
    real_client = module.httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        effect = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        if isinstance(effect, BaseException):
            raise effect
        return effect

    monkeypatch.setattr(
        module.httpx,
        "Client",
        lambda *args, **kwargs: real_client(transport=httpx.MockTransport(handler)),
    )
    return calls


def _issue_payload() -> httpx.Response:
    return httpx.Response(200, json={"fields": {"summary": "s", "description": ""}})


def test_transient_503_is_retried_then_succeeds(monkeypatch) -> None:
    slept: list[float] = []
    calls = _mock_jira_sequence(monkeypatch, [httpx.Response(503), _issue_payload()])

    client = JiraClient("https://jira.test", "user", "token", sleep=slept.append)
    data = client.get_issue("ABC-1")

    assert data is not None and data["fields"]["summary"] == "s"
    assert len(calls) == 2
    assert len(slept) == 1 and slept[0] > 0


def test_rate_limit_is_retried_honouring_retry_after(monkeypatch) -> None:
    slept: list[float] = []
    calls = _mock_jira_sequence(
        monkeypatch,
        [httpx.Response(429, headers={"Retry-After": "0"}), _issue_payload()],
    )

    client = JiraClient("https://jira.test", "user", "token", sleep=slept.append)
    assert client.get_issue("ABC-1") is not None

    assert len(calls) == 2
    assert slept == [0.0]


def test_transport_blip_is_retried(monkeypatch) -> None:
    slept: list[float] = []
    calls = _mock_jira_sequence(
        monkeypatch,
        [httpx.ConnectError("connection reset"), _issue_payload()],
    )

    client = JiraClient("https://jira.test", "user", "token", sleep=slept.append)
    assert client.get_issue("ABC-1") is not None

    assert len(calls) == 2
    assert len(slept) == 1


def test_unrelenting_outage_raises_after_bounded_attempts(monkeypatch) -> None:
    calls = _mock_jira_sequence(monkeypatch, [httpx.Response(503)])

    client = JiraClient("https://jira.test", "user", "token", sleep=lambda _s: None)
    with pytest.raises(JiraUnavailableError):
        client.get_issue("ABC-1")

    assert len(calls) == 3, "one attempt plus two retries, then give up"


def test_timeout_clamp_is_announced_not_silent(caplog) -> None:
    with caplog.at_level("WARNING", logger="kojutsu.integrations.jira_client"):
        JiraClient("https://jira.test", "user", "token", timeout_seconds=60)

    assert any("clamped" in record.message for record in caplog.records)


def test_non_string_summary_is_coerced_not_passed_through(monkeypatch) -> None:
    """Every caller reads the summary as a string; a shaped object landing
    there would surface far from the seam that let it through."""
    _mock_jira(
        monkeypatch,
        httpx.Response(200, json={"fields": {"summary": {"text": "shaped"}}}),
    )
    fields = JiraClient("https://jira.test", "user", "token").get_issue_fields("ABC-1")

    assert isinstance(fields["summary"], str)
    assert fields["summary"] == "{'text': 'shaped'}"
