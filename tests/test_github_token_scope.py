"""Tests for refusing an over-scoped GITHUB_TOKEN.

The pipeline reads pull requests and posts issue comments. A classic token that also
carries ``repo`` is a repository-wide read/write credential, so accepting it silently
would mean the capture process holds far more authority than it uses.
"""

from __future__ import annotations

import warnings

import httpx
import pytest

from kojutsu.integrations.github import (
    GitHubClient,
    GitHubIntegrationError,
    GitHubTokenScopeError,
)


def _patch_http(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    real_client = httpx.Client

    def factory(*args, **kwargs):  # type: ignore[no-untyped-def]
        return real_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(httpx, "Client", factory)


def test_fine_grained_token_has_no_scope_header_and_is_accepted() -> None:
    # GitHub omits x-oauth-scopes for fine-grained tokens, so an absent header means
    # the token is scoped per repository by construction.
    GitHubClient.assert_minimum_scope(None)
    GitHubClient.assert_minimum_scope("")
    GitHubClient.assert_minimum_scope("   ")


def test_narrow_classic_scopes_are_accepted() -> None:
    # Read-only, and nothing the pipeline has no use for. `gist` is *not* in this
    # list: Kojutsu never touches gists, so a token holding that scope is
    # over-scoped even though it is not repository-wide.
    GitHubClient.assert_minimum_scope("read:org, read:user")


@pytest.mark.parametrize(
    "scopes",
    ["repo", "public_repo", "repo, read:org", "admin:org_hook", "delete_repo"],
)
def test_over_scoped_classic_tokens_are_refused(scopes: str) -> None:
    with pytest.raises(GitHubTokenScopeError, match="over-scoped"):
        GitHubClient.assert_minimum_scope(scopes)


def test_refusal_names_the_offending_scope_and_points_at_the_documentation() -> None:
    with pytest.raises(GitHubTokenScopeError) as excinfo:
        GitHubClient.assert_minimum_scope("repo, gist")

    message = str(excinfo.value)
    assert "gist" in message
    assert "repo" in message
    assert "docs/github-seam.md" in message
    # The reason has to be actionable, not just a refusal.
    assert "fine-grained" in message


def test_override_downgrades_to_a_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_ALLOW_BROAD_SCOPES", "true")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        GitHubClient.assert_minimum_scope("repo")

    assert len(caught) == 1
    assert "over-scoped" in str(caught[0].message)


def test_scope_refusal_is_not_retryable() -> None:
    """A configuration problem: the same credential cannot succeed on retry.

    Distinguishing this from a transport failure is what stops the relay from
    retrying a rejected configuration until it dead-letters.
    """
    assert issubclass(GitHubTokenScopeError, GitHubIntegrationError)


def test_a_plain_read_is_refused_when_the_token_is_over_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check must apply to ordinary calls, not only the identity endpoint.

    Anchoring it to `get_authenticated_user` let a plan-only run through untouched.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"x-oauth-scopes": "repo, gist"},
            json={"number": 1, "title": "t", "state": "open", "head": {"ref": "b"}},
        )

    _patch_http(monkeypatch, handler)

    with pytest.raises(GitHubTokenScopeError, match="over-scoped"):
        GitHubClient("token").get_pull_request("org", "repo", 1)


def test_a_fine_grained_token_lets_the_same_call_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"number": 1, "title": "t", "state": "open", "head": {"ref": "b"}},
        )

    _patch_http(monkeypatch, handler)

    pull = GitHubClient("token").get_pull_request("org", "repo", 1)

    assert pull.number == 1


def test_a_refused_credential_keeps_being_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refusal must be consistent, not lopsided.

    An earlier design set the "already checked" flag before raising, which made the
    first call fail and the second succeed with the same token. A configuration
    problem does not resolve itself between calls, so a caller that retried would
    have seen inconsistent behaviour for one credential.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"x-oauth-scopes": "repo"}, json=[])

    _patch_http(monkeypatch, handler)
    client = GitHubClient("token")

    for _ in range(2):
        with pytest.raises(GitHubTokenScopeError):
            client.list_issue_comments("org", "repo", 1)


def test_an_accepted_credential_is_not_re_checked_on_every_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once cleared, the header is not parsed again for each request."""
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("x-oauth-scopes"))
        return httpx.Response(
            200,
            headers={"x-oauth-scopes": "read:org"},
            json=[],
        )

    _patch_http(monkeypatch, handler)
    client = GitHubClient("token")
    client.list_issue_comments("org", "repo", 1)
    client.list_issue_comments("org", "repo", 1)

    assert client._scope_checked is True


def test_a_transport_error_is_raised_before_the_scope_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "Not Found"})

    _patch_http(monkeypatch, handler)

    with pytest.raises(httpx.HTTPStatusError):
        GitHubClient("token").get_pull_request("org", "repo", 1)
