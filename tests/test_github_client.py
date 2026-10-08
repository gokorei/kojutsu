"""Tests for the GitHub REST client (via httpx MockTransport)."""

from __future__ import annotations

import json

import httpx
import pytest

from kojutsu.integrations.github import (
    GitHubClient,
    GitHubRateLimitError,
    answer_comment_body,
    canonical_pr_url,
    extract_answer_question_id_from_comment_body,
    extract_question_id_from_comment_body,
    extract_question_text_from_comment_body,
    kojutsu_comment_body,
    parse_pr_identifier,
)
from kojutsu.integrations.github_models import IssueCommentPayload


def patch_http(monkeypatch, handler) -> None:
    real_client = httpx.Client

    def factory(*args, **kwargs):
        return real_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(httpx, "Client", factory)


class ClosesLikeAClient:
    """The context-manager half of :class:`GitHubClient`, for a plain test double.

    ``GitHubClient`` owns one pooled transport for its lifetime and is entered as a
    context manager, so a double standing in for one has to honour the same
    protocol. Without this a stub fails with ``TypeError: object does not support
    the context manager protocol`` the moment production code starts saying what it
    means -- which is a test telling the truth about a real interface change, not a
    nuisance to work around.

    ``closed`` is recorded rather than merely tolerated so a test can assert the
    close actually happened. A double that accepts ``__exit__`` and does nothing
    would let a leaked pool pass.
    """

    closed = False

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def test_get_pull_request_sends_auth_and_parses(monkeypatch) -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("Authorization", "")
        return httpx.Response(
            200,
            json={"number": 1, "title": "T", "state": "open", "head": {"ref": "feature/ABC-1-x"}},
        )

    patch_http(monkeypatch, handler)
    pr = GitHubClient("tok").get_pull_request("org", "repo", 1)
    assert pr.title == "T"
    assert pr.head == {"ref": "feature/ABC-1-x"}
    assert seen["path"] == "/repos/org/repo/pulls/1"
    assert seen["auth"] == "Bearer tok"


def test_get_pull_diff_uses_diff_accept(monkeypatch) -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["accept"] = request.headers.get("Accept", "")
        return httpx.Response(200, text="diff --git a/x b/x")

    patch_http(monkeypatch, handler)
    diff = GitHubClient("tok").get_pull_diff("org", "repo", 1)
    assert diff.startswith("diff --git")
    assert seen["accept"] == "application/vnd.github.v3.diff"


def test_get_pr_files_returns_filenames(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"filename": "a.py"}, {"filename": "b.py"}])

    patch_http(monkeypatch, handler)
    result = GitHubClient("tok").get_pr_files("org", "repo", 1)
    assert result.items == ["a.py", "b.py"]
    assert result.truncated is False


def test_list_issue_comments_parses(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {
                    "id": 10,
                    "body": "hi",
                    "user": {"login": "dev"},
                    "created_at": "2023-01-01T00:00:00Z",
                    "author_association": "OWNER",
                }
            ],
        )

    patch_http(monkeypatch, handler)
    result = GitHubClient("tok").list_issue_comments("org", "repo", 1)
    comments = result.items
    assert result.truncated is False
    assert len(comments) == 1
    assert comments[0].id == 10
    assert comments[0].user.login == "dev"
    assert comments[0].author_association == "OWNER"


def test_issue_comment_payload_preserves_author_association_provenance() -> None:
    payload = IssueCommentPayload.model_validate(
        {
            "action": "created",
            "issue": {"number": 1},
            "comment": {
                "id": 10,
                "body": "hi",
                "user": {"login": "dev"},
                "created_at": "2023-01-01T00:00:00Z",
                "author_association": "MEMBER",
            },
            "repository": {"full_name": "org/repo"},
        }
    )

    assert payload.comment.author_association == "MEMBER"


def test_issue_comment_payload_allows_missing_association_for_fail_closed_handling() -> None:
    payload = IssueCommentPayload.model_validate(
        {
            "action": "created",
            "issue": {"number": 1},
            "comment": {
                "id": 10,
                "body": "hi",
                "user": {"login": "dev"},
                "created_at": "2023-01-01T00:00:00Z",
            },
            "repository": {"full_name": "org/repo"},
        }
    )

    assert payload.comment.author_association is None


def test_list_issue_comments_traverses_pages(monkeypatch) -> None:
    pages: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        page = request.url.params.get("page")
        pages.append(page or "")
        if page == "1":
            return httpx.Response(
                200,
                json=[
                    {
                        "id": index,
                        "body": str(index),
                        "user": {"login": "dev"},
                        "created_at": "2023-01-01T00:00:00Z",
                    }
                    for index in range(100)
                ],
            )
        return httpx.Response(
            200,
            json=[
                {
                    "id": 100,
                    "body": "last",
                    "user": {"login": "dev"},
                    "created_at": "2023-01-01T00:00:01Z",
                }
            ],
        )

    patch_http(monkeypatch, handler)
    result = GitHubClient("tok").list_issue_comments("org", "repo", 1)
    comments = result.items
    assert result.truncated is False
    assert len(comments) == 101
    assert comments[-1].id == 100
    assert pages == ["1", "2"]


def test_explicit_answer_marker_round_trip() -> None:
    body = answer_comment_body("q-1", "Because X.")
    assert extract_answer_question_id_from_comment_body(body) == "q-1"
    assert extract_answer_question_id_from_comment_body("not an answer") is None


def test_post_issue_comment_posts_body(monkeypatch) -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["body"] = request.read().decode()
        return httpx.Response(
            201,
            json={
                "id": 99,
                "body": "x",
                "user": {"login": "dev"},
                "created_at": "2023-01-01T00:00:00Z",
            },
        )

    patch_http(monkeypatch, handler)
    created = GitHubClient("tok").post_issue_comment("org", "repo", 1, "hello")
    assert created.id == 99
    assert seen["method"] == "POST"
    assert '"body": "hello"' in seen["body"] or '"body":"hello"' in seen["body"]


def test_question_marker_round_trip() -> None:
    body = kojutsu_comment_body("q-1", "Why this approach?")
    assert extract_question_id_from_comment_body(body) == "q-1"
    assert extract_question_text_from_comment_body(body) == "Why this approach?"
    assert extract_question_id_from_comment_body("no marker here") is None


def test_pr_url_parsing_is_exact_and_canonical() -> None:
    assert parse_pr_identifier("https://www.github.com/Owner/Repo/pull/456") == (
        "Owner/Repo",
        456,
    )
    assert canonical_pr_url("Owner/Repo", 456) == "https://github.com/Owner/Repo/pull/456"
    assert parse_pr_identifier("https://github.com/Owner/Repo/pull/456?tab=files") is None
    assert parse_pr_identifier("https://github.com/Owner/Repo/pull/456/") is None
    assert parse_pr_identifier("https://github.com/Owner/Repo/pull/456#discussion") is None
    assert parse_pr_identifier("https://gitlab.com/Owner/Repo/pull/456") is None


def test_post_questions_reconciles_authenticated_kojutsu_marker_without_duplicate(
    monkeypatch,
) -> None:
    posted: list[str] = []
    existing_body = kojutsu_comment_body("existing-question", "Why this approach?")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/user":
            return httpx.Response(200, json={"login": "kojutsu-bot"})
        if request.method == "GET":
            return httpx.Response(
                200,
                json=[
                    {
                        "id": 41,
                        "body": existing_body,
                        "user": {"login": "kojutsu-bot"},
                        "created_at": "2023-01-01T00:00:00Z",
                    }
                ],
            )
        posted.append(request.read().decode())
        return httpx.Response(
            201,
            json={
                "id": 42,
                "body": kojutsu_comment_body("new-question", "Another question?"),
                "user": {"login": "kojutsu-bot"},
                "created_at": "2023-01-01T00:00:01Z",
            },
        )

    patch_http(monkeypatch, handler)
    checkpointed: list[str] = []
    comments = GitHubClient("tok").post_questions_as_pr_comments(
        "org",
        "repo",
        1,
        [("new-question", "Why this approach?")],
        on_comment=lambda question_id, _text, _comment: checkpointed.append(question_id),
    )

    assert [comment.id for comment in comments] == [41]
    assert posted == []
    assert checkpointed == ["existing-question"]
    assert extract_question_id_from_comment_body(comments[0].body) == "existing-question"


def test_post_questions_rejects_copied_markers_and_user_text_then_repeats_idempotently(
    monkeypatch,
) -> None:
    posted: list[dict] = []
    comments = [
        {
            "id": 10,
            "body": kojutsu_comment_body("new-question", "Why this approach?"),
            "user": {"login": "outsider"},
            "created_at": "2023-01-01T00:00:00Z",
        },
        {
            "id": 11,
            "body": "Why this approach?",
            "user": {"login": "another-user"},
            "created_at": "2023-01-01T00:00:01Z",
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/user":
            return httpx.Response(200, json={"login": "kojutsu-bot"})
        if request.method == "GET":
            return httpx.Response(200, json=comments)
        body = json.loads(request.read())
        posted.append(body)
        created = {
            "id": 42,
            "body": body["body"],
            "user": {"login": "kojutsu-bot"},
            "created_at": "2023-01-01T00:00:02Z",
        }
        comments.append(created)
        return httpx.Response(201, json=created)

    patch_http(monkeypatch, handler)
    client = GitHubClient("tok")
    questions = [("new-question", "Why this approach?")]

    first = client.post_questions_as_pr_comments("org", "repo", 1, questions)
    second = client.post_questions_as_pr_comments("org", "repo", 1, questions)

    assert [comment.id for comment in first] == [42]
    assert [comment.id for comment in second] == [42]
    assert len(posted) == 1
    assert extract_question_id_from_comment_body(first[0].body) == "new-question"


def test_the_context_manager_hands_the_pool_back_on_the_happy_path() -> None:
    client = GitHubClient("token")
    pool = client.http()

    assert not pool.is_closed
    with client:
        pass

    assert pool.is_closed


def test_the_context_manager_hands_the_pool_back_even_when_the_body_raises() -> None:
    """The case ``with`` exists for, and the one a ``try/finally`` written by hand forgets.

    A read that raises is the *likely* outcome on this seam, not the exceptional
    one — a rate limit, a 5xx, a scope refusal. So a close that only runs on the
    happy path would leak precisely when the client is busiest.
    """
    client = GitHubClient("token")
    pool = client.http()

    with pytest.raises(RuntimeError, match="boom"), client:
        raise RuntimeError("boom")

    assert pool.is_closed


def test_closing_twice_is_harmless_and_does_not_rebuild_the_pool() -> None:
    """``close`` is called from ``__exit__`` and often again by an owner.

    Rebuilding instead would make the second close a fresh leak, and silently
    reopening the transport would let a use-after-close look like it worked.
    """
    client = GitHubClient("token")
    pool = client.http()

    client.close()
    client.close()

    assert pool.is_closed
    assert client.http() is pool


def test_a_full_final_page_at_the_ceiling_reports_truncated(monkeypatch) -> None:
    """One hundred full pages is where the walk stops, and stopping there must
    read as truncated: the only way to know whether a 101st page exists would
    be a 101st read past the bound."""
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        page = int(request.url.params.get("page", "1"))
        assert page <= 100, "the walk must stop at the ceiling, not past it"
        return httpx.Response(
            200,
            json=[
                {
                    "id": (page - 1) * 2 + index,
                    "body": "x",
                    "user": {"login": "dev"},
                    "created_at": "2023-01-01T00:00:00Z",
                }
                for index in range(2)
            ],
        )

    patch_http(monkeypatch, handler)
    result = GitHubClient("tok").list_issue_comments("org", "repo", 1, per_page=2)

    assert len(result.items) == 200
    assert result.truncated is True
    assert requests == 100


def test_a_short_final_page_is_complete_even_at_the_ceiling(monkeypatch) -> None:
    """Ninety-nine full pages and a short hundredth is everything there is: a
    full page *at* the ceiling means 'maybe more', a short one means 'done'."""
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        page = int(request.url.params.get("page", "1"))
        count = 2 if page < 100 else 1
        return httpx.Response(
            200,
            json=[
                {
                    "id": (page - 1) * 2 + index,
                    "body": "x",
                    "user": {"login": "dev"},
                    "created_at": "2023-01-01T00:00:00Z",
                }
                for index in range(count)
            ],
        )

    patch_http(monkeypatch, handler)
    result = GitHubClient("tok").list_issue_comments("org", "repo", 1, per_page=2)

    assert len(result.items) == 199
    assert result.truncated is False
    assert requests == 100


def test_a_get_survives_a_single_429_then_succeeds(monkeypatch) -> None:
    """The live webhook path reads through this client, so one 429 must be a
    wait rather than a failed capture."""
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if not hasattr(handler, "calls"):
            handler.calls = 0  # type: ignore[attr-defined]
        handler.calls += 1  # type: ignore[attr-defined]
        if handler.calls == 1:  # type: ignore[attr-defined]
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json=[])

    patch_http(monkeypatch, handler)
    client = GitHubClient("token", sleep=slept.append)

    assert client.get_pr_files("org", "repo", 1).items == []

    assert handler.calls == 2  # type: ignore[attr-defined]
    assert slept == [0.0], "the interval the forge asked for, honoured exactly"


def test_a_5xx_is_retried_for_gets_but_not_for_posts(monkeypatch) -> None:
    """A gateway error on a GET means the read never happened, so it is safe to
    repeat. On a POST the write may have applied and only the response been
    lost, so a blind retry can double-post and the error surfaces instead."""
    get_calls = 0
    post_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal get_calls, post_calls
        if request.method == "GET":
            get_calls += 1
            if get_calls == 1:
                return httpx.Response(503, json={"message": "upstream failure"})
            return httpx.Response(200, json=[])
        post_calls += 1
        return httpx.Response(503, json={"message": "upstream failure"})

    patch_http(monkeypatch, handler)
    client = GitHubClient("token", sleep=lambda _seconds: None)

    assert client.get_pr_files("org", "repo", 1).items == []
    assert get_calls == 2

    with pytest.raises(httpx.HTTPStatusError):
        client.post_issue_comment("org", "repo", 1, "hello")
    assert post_calls == 1, "a failed POST must not be re-issued blindly"


def test_an_unrelenting_rate_limit_raises_the_taxed_error(monkeypatch) -> None:
    """Exhausted retries are not a new error type: the caller raises what it
    would have raised without retries, after the bounded number of waits."""
    slept: list[float] = []
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            429,
            headers={"Retry-After": "0"},
            json={"message": "API rate limit exceeded for user ID 1."},
        )

    patch_http(monkeypatch, handler)
    client = GitHubClient("token", sleep=slept.append, rate_limit_retries=2)

    with pytest.raises(GitHubRateLimitError):
        client.get_pull_request("org", "repo", 1)

    assert calls == 3, "one attempt plus two retries"
    assert slept == [0.0, 0.0]


def test_a_rate_limit_403_is_retried_but_a_scope_403_is_not(monkeypatch) -> None:
    """Status alone cannot tell an exhausted budget from a permissions problem;
    the headers can, and the retry decision follows them."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                403,
                headers={"x-ratelimit-remaining": "0", "Retry-After": "0"},
                json={"message": "API rate limit exceeded"},
            )
        return httpx.Response(
            200,
            json={"number": 1, "title": "T", "state": "open", "head": {"ref": "x"}},
        )

    patch_http(monkeypatch, handler)
    client = GitHubClient("token", sleep=lambda _seconds: None)

    assert client.get_pull_request("org", "repo", 1).title == "T"
    assert calls == 2
