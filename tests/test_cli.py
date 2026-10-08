"""Tests for CLI: ask/collect/search invoke the right logic (contract tests)."""

import json
import socket
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from kojutsu.cli import app
from kojutsu.cli_ask import derived_question_id
from kojutsu.integrations.github import PagedResult
from kojutsu.integrations.webhook_client import WebhookURLError, validate_webhook_url
from test_github_client import ClosesLikeAClient

runner = CliRunner()


def test_ask_requires_pr_spec() -> None:
    """Without PR argument or --pr/--repo (and without token), ask exits with error."""
    result = runner.invoke(app, ["ask"])
    assert result.exit_code != 0
    assert "GITHUB_TOKEN" in result.output or "PR" in result.output or "pr" in result.output.lower()


def test_ask_invalid_pr_spec() -> None:
    """Invalid PR spec exits with error."""
    result = runner.invoke(app, ["ask", "not-a-valid-pr"])
    assert result.exit_code != 0


def test_ask_defaults_to_plan_without_github_mutation(monkeypatch) -> None:
    from kojutsu import cli_ask as cli_module
    from kojutsu.models import Question, QuestionCategory

    monkeypatch.setenv("GITHUB_TOKEN", "token")

    class ForbiddenMutation:
        """Refuses construction, so it needs only the protocol half.

        It cannot use :class:`ClosesLikeAClient` because inheriting ``close`` would
        be inheriting the one method whose purpose is to be called on an object that
        was never built. The context-manager half still has to be present: the plan
        path enters the client it is forbidden from creating, so if that path ever
        did construct one, the constructor raising *is* the assertion.
        """

        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("plan must not construct a GitHub client")

        def __enter__(self):
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def fake_generate(**kwargs):
        assert kwargs["pr_spec"] == "https://github.com/org/repo/pull/7"
        return (
            [
                Question(
                    id="q-1", text="Why this approach?", category=QuestionCategory.DESIGN_DECISION
                )
            ],
            {
                "pr_url": "https://github.com/org/repo/pull/7",
                "pr_number": 7,
                "repo": "org/repo",
                "owner": "org",
                "repo_name": "repo",
                "branch_name": "feature/ABC-1-x",
                "jira_ticket_key": "ABC-1",
                "files_changed": [],
            },
        )

    monkeypatch.setattr(cli_module, "generate_questions_for_pr", fake_generate)
    monkeypatch.setattr(cli_module, "GitHubClient", ForbiddenMutation)
    monkeypatch.setattr(cli_module, "build_registry", ForbiddenMutation)

    result = runner.invoke(app, ["ask", "org/repo#7"])

    assert result.exit_code == 0
    assert "Plan only" in result.output
    assert "Why this approach?" in result.output


def test_ask_plan_artifact_is_exact_and_concurrent_applies_are_idempotent(
    monkeypatch, tmp_path: Path
) -> None:
    from kojutsu import cli_ask as cli_module
    from kojutsu.models import Question, QuestionCategory

    generated = [
        Question(id="q-exact", text="Why this exact approach?", category=QuestionCategory.EDGE_CASE)
    ]
    context = {
        "pr_url": "https://github.com/org/repo/pull/7",
        "pr_number": 7,
        "repo": "org/repo",
        "owner": "org",
        "repo_name": "repo",
        "branch_name": "feature/ABC-1-x",
        "jira_ticket_key": "ABC-1",
        "files_changed": ["src/x.py"],
    }
    plan_path = tmp_path / "ask-plan.json"
    posted: list[tuple[str, str]] = []
    comments: list[SimpleNamespace] = []
    expected_id = derived_question_id(context["pr_url"], generated[0])

    class FakeRegistry:
        def __enter__(self):
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def record_session(self, session) -> None:
            return None

        def record_question(self, **kwargs) -> None:
            assert kwargs["question_id"] == expected_id
            assert kwargs["question_text"] == "Why this exact approach?"

    class FakeGitHub(ClosesLikeAClient):
        def __init__(self, token: str) -> None:
            assert token == "token"

        def list_issue_comments(self, owner: str, repo: str, pr_number: int) -> PagedResult:
            return PagedResult(items=list(comments), truncated=False)

        def post_questions_as_pr_comments(
            self, owner, repo, pr_number, questions, *, existing_comments, on_comment
        ):
            assert questions == [(expected_id, "Why this exact approach?")]
            if not comments:
                posted.extend(questions)
                comment = SimpleNamespace(
                    id=91,
                    body=f"<!-- kojutsu:question:{expected_id} -->\n\nWhy this exact approach?",
                    user=SimpleNamespace(login="kojutsu-bot"),
                )
                comments.append(comment)
            for question_id, text in questions:
                on_comment(question_id, text, comments[0])
            return [comments[0]]

    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(cli_module, "build_registry", lambda _settings: FakeRegistry())
    monkeypatch.setattr(cli_module, "GitHubClient", FakeGitHub)
    monkeypatch.setattr(
        cli_module,
        "generate_questions_for_pr",
        lambda **_kwargs: (generated, context),
    )

    planned = runner.invoke(app, ["ask", "org/repo#7", "--plan-file", str(plan_path)])
    assert planned.exit_code == 0, planned.output
    assert plan_path.stat().st_mode & 0o777 == 0o600
    artifact = json.loads(plan_path.read_text(encoding="utf-8"))
    # The id is the derived one, not the id the generator handed out. The
    # artifact records what `ask` will apply, and applying is what has to be
    # repeatable, so an unrepeatable id must not reach it.
    assert artifact["questions"] == [
        {
            "id": expected_id,
            "text": "Why this exact approach?",
            "category": "edge_case",
        }
    ]

    monkeypatch.setattr(
        cli_module,
        "generate_questions_for_pr",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("apply regenerated questions")),
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                cli_module.ask,
                None,
                None,
                None,
                True,
                True,
                plan_path,
            )
            for _ in range(2)
        ]
        results = [future.result() for future in futures]
    repeated = runner.invoke(app, ["ask", "--apply", "--plan-file", str(plan_path)])

    assert results == [None, None]
    assert repeated.exit_code == 0, repeated.output
    assert posted == [(expected_id, "Why this exact approach?")]
    assert "Applied 1 question(s)" in repeated.output


def test_ask_does_not_repost_questions_a_crashed_run_already_posted(monkeypatch) -> None:
    """A crash between the POST and the record must not become a duplicate comment.

    Dedupe is keyed on the marker, and the marker is read back off the pull request
    rather than out of the local registry. That ordering is the entire fix: the crash
    being guarded is precisely the one that leaves the registry without a row, so a
    key derived from the registry would be the thing that lost the information.

    Only the three network edges are faked. ``post_questions_as_pr_comments`` is the
    real one — a fake that re-implemented the dedupe, as the harness above does, is
    asserting its own arithmetic instead of the code's.
    """
    from kojutsu import cli_ask as cli_module
    from kojutsu.integrations.github import GitHubClient
    from kojutsu.integrations.github_models import GitHubComment, GitHubUser
    from kojutsu.models import Question, QuestionCategory

    context = {
        "pr_url": "https://github.com/org/repo/pull/7",
        "pr_number": 7,
        "repo": "org/repo",
        "owner": "org",
        "repo_name": "repo",
        "branch_name": "feature/ABC-1-x",
        "jira_ticket_key": "ABC-1",
        "files_changed": [],
    }
    # The ids the generator hands out are deliberately not the identity, and are
    # re-derived from the text before anything is posted.
    generated = [
        Question(id="raw-1", text="Why this approach?", category=QuestionCategory.DESIGN_DECISION),
        Question(id="raw-2", text="What breaks on rollback?", category=QuestionCategory.EDGE_CASE),
    ]
    posted: list[str] = []
    comments: list[GitHubComment] = []
    recorded: list[tuple[str, str]] = []
    lose_writes = True

    class FakeRegistry:
        def __enter__(self):
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def record_session(self, session) -> None:
            return None

        def record_question(self, *, question_id: str, question_text: str, **kwargs) -> None:
            # The failure this ticket is about: the comment is on the pull request,
            # and the write that would have said so never lands.
            if lose_writes:
                return
            recorded.append((question_id, question_text))

    class FakeGitHub(GitHubClient):
        def __init__(self, token: str) -> None:
            super().__init__(token)
            self._authenticated_login = "kojutsu-bot"

        def list_issue_comments(self, owner, repo, issue_number, *, per_page=100):
            return PagedResult(items=list(comments), truncated=False)

        def post_issue_comment(self, owner, repo, issue_number, body: str) -> GitHubComment:
            posted.append(body)
            comment = GitHubComment(
                id=1000 + len(comments),
                body=body,
                user=GitHubUser(login="kojutsu-bot"),
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
            comments.append(comment)
            return comment

    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(cli_module, "build_registry", lambda _settings: FakeRegistry())
    monkeypatch.setattr(cli_module, "GitHubClient", FakeGitHub)
    monkeypatch.setattr(
        cli_module,
        "generate_questions_for_pr",
        lambda **_kwargs: (generated, context),
    )

    crashed = runner.invoke(app, ["ask", "org/repo#7", "--apply"])
    assert crashed.exit_code == 0, crashed.output
    assert len(posted) == 2
    assert recorded == []

    lose_writes = False
    retried = runner.invoke(app, ["ask", "org/repo#7", "--apply"])
    assert retried.exit_code == 0, retried.output

    # One set of question comments on the pull request, not two — and the retry
    # still reports the row it recovered, because the comment it did not post is
    # still a real question that somebody has to answer.
    assert len(comments) == 2
    assert len(posted) == 2
    assert "0 posted, 2 already present" in retried.output
    assert recorded == [
        (derived_question_id(context["pr_url"], generated[0]), "Why this approach?"),
        (derived_question_id(context["pr_url"], generated[1]), "What breaks on rollback?"),
    ]


def test_ask_keys_dedupe_on_the_marker_so_a_reworded_comment_is_still_the_same_question(
    monkeypatch,
) -> None:
    """The marker settles identity; the prose beneath it is the answerable surface.

    Pinning this separately matters because the two look interchangeable. If dedupe
    keyed on question text alone, rewording a posted question — a reviewer tidying
    Kojutsu's grammar, say — would read as a new question and post the original
    wording beside it. That is the duplicate this command exists to prevent, arriving
    through the most innocent edit.
    """
    from kojutsu import cli_ask as cli_module
    from kojutsu.integrations.github import GitHubClient, kojutsu_comment_body
    from kojutsu.integrations.github_models import GitHubComment, GitHubUser
    from kojutsu.models import Question, QuestionCategory

    context = {
        "pr_url": "https://github.com/org/repo/pull/7",
        "pr_number": 7,
        "repo": "org/repo",
        "owner": "org",
        "repo_name": "repo",
        "branch_name": "feature/ABC-1-x",
        "jira_ticket_key": "ABC-1",
        "files_changed": [],
    }
    generated = [
        Question(id="raw-1", text="Why this approach?", category=QuestionCategory.DESIGN_DECISION)
    ]
    posted: list[str] = []
    comments: list[GitHubComment] = []

    class FakeRegistry:
        def __enter__(self):
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def record_session(self, session) -> None:
            return None

        def record_question(self, **kwargs) -> None:
            return None

    class FakeGitHub(GitHubClient):
        def __init__(self, token: str) -> None:
            super().__init__(token)
            self._authenticated_login = "kojutsu-bot"

        def list_issue_comments(self, owner, repo, issue_number, *, per_page=100):
            return PagedResult(items=list(comments), truncated=False)

        def post_issue_comment(self, owner, repo, issue_number, body: str) -> GitHubComment:
            posted.append(body)
            comment = GitHubComment(
                id=1000 + len(comments),
                body=body,
                user=GitHubUser(login="kojutsu-bot"),
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
            comments.append(comment)
            return comment

    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(cli_module, "build_registry", lambda _settings: FakeRegistry())
    monkeypatch.setattr(cli_module, "GitHubClient", FakeGitHub)
    monkeypatch.setattr(
        cli_module,
        "generate_questions_for_pr",
        lambda **_kwargs: (generated, context),
    )

    assert runner.invoke(app, ["ask", "org/repo#7", "--apply"]).exit_code == 0

    question_id = derived_question_id(context["pr_url"], generated[0])
    comments[0] = comments[0].model_copy(
        update={"body": kojutsu_comment_body(question_id, "Why this approach, precisely?")}
    )

    reworded = runner.invoke(app, ["ask", "org/repo#7", "--apply"])

    assert reworded.exit_code == 0, reworded.output
    assert len(posted) == 1
    assert len(comments) == 1


def test_ask_plan_artifact_does_not_overwrite_existing_file(monkeypatch, tmp_path: Path) -> None:
    from kojutsu import cli_ask as cli_module
    from kojutsu.models import Question, QuestionCategory

    plan_path = tmp_path / "ask-plan.json"
    plan_path.write_text("do-not-overwrite", encoding="utf-8")
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(
        cli_module,
        "generate_questions_for_pr",
        lambda **_kwargs: (
            [Question(id="q1", text="Why?", category=QuestionCategory.DESIGN_DECISION)],
            {
                "pr_url": "https://github.com/org/repo/pull/7",
                "pr_number": 7,
                "repo": "org/repo",
                "owner": "org",
                "repo_name": "repo",
                "branch_name": "main",
                "jira_ticket_key": None,
                "files_changed": [],
            },
        ),
    )

    result = runner.invoke(app, ["ask", "org/repo#7", "--plan-file", str(plan_path)])

    assert result.exit_code == 1
    assert "already exists" in result.output
    assert plan_path.read_text(encoding="utf-8") == "do-not-overwrite"


def test_ask_rejects_apply_without_session_save(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "token")

    result = runner.invoke(app, ["ask", "org/repo#7", "--apply", "--no-save-session"])

    assert result.exit_code == 1
    assert "preview-only" in result.output


def test_ask_apply_checkpoints_session_and_comment_mapping(monkeypatch) -> None:
    from kojutsu import cli_ask as cli_module
    from kojutsu.models import Question, QuestionCategory

    events: list[str] = []
    records: list[dict[str, object]] = []
    monkeypatch.setenv("GITHUB_TOKEN", "token")

    class FakeRegistry:
        def __enter__(self):
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def record_session(self, session) -> None:
            events.append("session")
            assert session.context_url == "https://github.com/org/repo/pull/7"

        def record_question(self, **kwargs) -> None:
            events.append("question")
            records.append(kwargs)

    class FakeGitHub(ClosesLikeAClient):
        def __init__(self, token: str) -> None:
            assert token == "token"

        def list_issue_comments(self, owner: str, repo: str, pr_number: int) -> PagedResult:
            assert (owner, repo, pr_number) == ("org", "repo", 7)
            events.append("list")
            return PagedResult(items=[], truncated=False)

        def post_questions_as_pr_comments(
            self, owner, repo, pr_number, questions, *, existing_comments, on_comment
        ):
            assert events[-2:] == ["session", "list"]
            comment = SimpleNamespace(
                id=55,
                body="<!-- kojutsu:question:q-1 -->\n\nWhy this approach?",
                user=SimpleNamespace(login="bot"),
            )
            on_comment("q-1", "Why this approach?", comment)
            return [comment]

    def fake_generate(**kwargs):
        return (
            [
                Question(
                    id="q-1", text="Why this approach?", category=QuestionCategory.DESIGN_DECISION
                )
            ],
            {
                "pr_url": "https://github.com/org/repo/pull/7",
                "pr_number": 7,
                "repo": "org/repo",
                "owner": "org",
                "repo_name": "repo",
                "branch_name": "feature/ABC-1-x",
                "jira_ticket_key": "ABC-1",
                "files_changed": [],
            },
        )

    monkeypatch.setattr(cli_module, "generate_questions_for_pr", fake_generate)
    monkeypatch.setattr(cli_module, "build_registry", lambda _settings: FakeRegistry())
    monkeypatch.setattr(cli_module, "GitHubClient", FakeGitHub)

    result = runner.invoke(app, ["ask", "org/repo#7", "--apply"])

    assert result.exit_code == 0
    assert events == ["session", "list", "question"]
    assert records[0]["question_id"] == "q-1"
    assert records[0]["github_comment_id"] == 55


def test_packaged_asgi_entrypoint_serves_health() -> None:
    from kojutsu.asgi import app as asgi_app

    with TestClient(asgi_app) as client:
        response = client.get("/webhook/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_collect_requires_pr_spec() -> None:
    """Collect without PR exits with error."""
    result = runner.invoke(app, ["collect"])
    assert result.exit_code != 0


def test_collect_passes_author_association_to_answer_capture(monkeypatch, tmp_path) -> None:
    from kojutsu import cli_ask as cli_module
    from kojutsu.core.question_registry import SqliteQuestionRegistry
    from kojutsu.integrations.github import (
        answer_comment_body,
        kojutsu_comment_body,
    )

    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    registry.record_question(
        question_id="q1",
        github_comment_id=100,
        repo="org/repo",
        pr_number=1,
        pr_url="https://github.com/org/repo/pull/1",
        question_text="Why?",
        question_category="design_decision",
    )

    class FakeSink:
        def __init__(self) -> None:
            self.entries = []

        def store(self, entry) -> None:
            self.entries.append(entry)

    class FakeRuntime:
        def __init__(self) -> None:
            self.registry = registry
            self.sink = FakeSink()

        def __enter__(self):
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    comments = [
        SimpleNamespace(
            id=100,
            body=kojutsu_comment_body("q1", "Why?"),
            user=SimpleNamespace(login="bot"),
            created_at=datetime(2023, 1, 1, tzinfo=UTC),
            author_association="NONE",
        ),
        SimpleNamespace(
            id=201,
            body=answer_comment_body("q1", "Because X."),
            user=SimpleNamespace(login="dev"),
            created_at=datetime(2023, 1, 1, 0, 0, 1, tzinfo=UTC),
            author_association="MEMBER",
        ),
    ]

    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(
        cli_module.GitHubClient,
        "list_issue_comments",
        lambda self, owner, repo, number: PagedResult(items=comments, truncated=False),
    )
    fake_runtime = FakeRuntime()
    monkeypatch.setattr(cli_module, "build_runtime", lambda settings: fake_runtime)

    result = runner.invoke(app, ["collect", "org/repo#1"])

    assert result.exit_code == 0
    assert "Collected 1 new answer(s)." in result.output
    assert fake_runtime.sink.entries[0].metadata["github_author_association"] == "MEMBER"


def test_search_requires_tanseki() -> None:
    """Search exits with a deterministic error when Tanseki is unconfigured."""
    result = runner.invoke(app, ["search"])
    assert result.exit_code == 1
    assert "TANSEKI_URL" in result.output


def test_search_renders_mocked_results(monkeypatch) -> None:
    class FakeTansekiClient:
        def search(self, text, *, tags=None, frontmatter=None, limit=20):
            assert text == "authentication"
            assert frontmatter == {"repo": "org/repo"}
            assert limit == 1
            return [SimpleNamespace(id="doc-1")]

        def get_document(self, doc_id):
            assert doc_id == "doc-1"
            return SimpleNamespace(
                frontmatter={"repo": "org/repo", "pr": "7", "category": "design_decision"},
                content="Use short-lived tokens.",
            )

    monkeypatch.setenv("TANSEKI_URL", "https://tanseki.test")
    monkeypatch.setattr(
        "kojutsu.cli.TansekiClient.from_settings", lambda settings: FakeTansekiClient()
    )

    result = runner.invoke(app, ["search", "authentication", "--repo", "org/repo", "--limit", "1"])

    assert result.exit_code == 0
    assert "[org/repo #7] design_decision" in result.output
    assert "Use short-lived tokens." in result.output


def test_network_access_is_denied() -> None:
    with pytest.raises(RuntimeError, match="Unexpected network access"):
        socket.create_connection(("example.com", 443))


def test_relay_requires_tanseki(monkeypatch) -> None:
    """Relay exits with an error when TANSEKI_URL is unset."""
    monkeypatch.setenv("TANSEKI_URL", "")
    result = runner.invoke(app, ["relay"])
    assert result.exit_code == 1
    assert "TANSEKI_URL" in result.output


def test_outbox_reports_pending(monkeypatch, tmp_path) -> None:
    """The outbox command reports queued writes."""
    from kojutsu.core.outbox import TansekiOutbox

    path = tmp_path / "outbox.db"
    with TansekiOutbox(path) as queue:
        queue.enqueue("e1", {"id": "d1"})
    monkeypatch.setenv("TANSEKI_OUTBOX_PATH", str(path))

    result = runner.invoke(app, ["outbox"])
    assert result.exit_code == 0
    assert "pending: 1" in result.output
    assert "e1" in result.output


def test_outbox_dead_letters_list_and_requeue(monkeypatch, tmp_path) -> None:
    from kojutsu.core.outbox import TansekiOutbox
    from kojutsu.integrations.tanseki import TansekiPermanentError

    path = tmp_path / "outbox.db"
    with TansekiOutbox(path) as queue:
        queue.enqueue("dead-1", {"id": "d1"})
        queue.mark_failed("dead-1", TansekiPermanentError("invalid"))
    monkeypatch.setenv("TANSEKI_OUTBOX_PATH", str(path))

    listed = runner.invoke(app, ["outbox-dead-letters"])

    assert listed.exit_code == 0
    assert "dead-1" in listed.output
    assert "delivery-failed" in listed.output

    requeued = runner.invoke(app, ["outbox-requeue", "dead-1"])
    assert requeued.exit_code == 0
    with TansekiOutbox(path) as queue:
        assert queue.pending()[0].state == "captured-locally"


def test_outbox_retry_requeues_and_retries_immediately(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace

    calls: list[str] = []

    class FakeRuntime:
        def __enter__(self):
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def retry(self, entry_id: str):
            calls.append(f"retry:{entry_id}")
            return SimpleNamespace(sent=1, failed=0, dead_lettered=0)

    monkeypatch.setenv("TANSEKI_URL", "https://tanseki.test")
    monkeypatch.setenv("TANSEKI_OUTBOX_PATH", str(tmp_path / "outbox.db"))
    monkeypatch.setattr("kojutsu.cli_outbox.build_runtime", lambda _settings: FakeRuntime())

    result = runner.invoke(app, ["outbox-retry", "dead-1"])

    assert result.exit_code == 0
    assert calls == ["retry:dead-1"]
    assert "sent=1" in result.output


def test_outbox_cleanup_reports_removed_dead_letters(monkeypatch, tmp_path) -> None:
    from kojutsu.core.outbox import TansekiOutbox
    from kojutsu.integrations.tanseki import TansekiPermanentError

    path = tmp_path / "outbox.db"
    with TansekiOutbox(path, retention_days=0) as queue:
        queue.enqueue("dead-1", {"id": "d1"})
        queue.mark_failed("dead-1", TansekiPermanentError("invalid"))
    monkeypatch.setenv("TANSEKI_OUTBOX_PATH", str(path))

    result = runner.invoke(app, ["outbox-cleanup", "--older-than-days", "0"])

    assert result.exit_code == 0
    assert "removed: 1" in result.output
    with TansekiOutbox(path, retention_days=0) as queue:
        assert queue.dead_letters() == []


def test_status_reports_fields(monkeypatch, tmp_path) -> None:
    """The status command reports Tanseki/outbox/registry state."""
    monkeypatch.setenv("TANSEKI_URL", "")
    monkeypatch.setenv("TANSEKI_OUTBOX_PATH", str(tmp_path / "outbox.db"))

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "tanseki_url: (unset)" in result.output
    assert "outbox_pending: 0" in result.output
    assert "outbox_captured_locally: 0" in result.output
    assert "outbox_delivery_failed: 0" in result.output


def test_webhook_register_dry_run(monkeypatch) -> None:
    """cli_webhook imports cleanly and its commands are registered (regression guard)."""
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    result = runner.invoke(app, ["webhook-register", "org/repo", "--dry-run"])
    assert result.exit_code == 0
    assert "DRY RUN" in result.output


def test_serve_requires_webhook_secret_for_public_host(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "")

    result = runner.invoke(app, ["serve", "--host", "0.0.0.0"])

    assert result.exit_code == 1
    assert "GITHUB_WEBHOOK_SECRET" in result.output


def test_public_serve_without_registration_does_not_validate_bind_as_webhook_url(
    monkeypatch,
) -> None:
    import uvicorn

    import kojutsu.webhook as webhook_module

    captured: dict[str, object] = {}

    def create_app(*, webhook_url: str | None, cors_origins: list[str] | None):
        captured["webhook_url"] = webhook_url
        captured["cors_origins"] = cors_origins
        return object()

    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "secret")
    monkeypatch.setenv("GITHUB_WEBHOOK_REGISTER", "false")
    monkeypatch.setattr(webhook_module, "create_webhook_app", create_app)
    monkeypatch.setattr(uvicorn, "run", lambda *_args, **_kwargs: None)

    result = runner.invoke(app, ["serve", "--host", "0.0.0.0"])

    assert result.exit_code == 0, result.output
    assert captured == {"webhook_url": None, "cors_origins": None}


def test_webhook_register_requires_secret_outside_dry_run(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "")

    result = runner.invoke(app, ["webhook-register", "org/repo"])

    assert result.exit_code == 1
    assert "GITHUB_WEBHOOK_SECRET" in result.output


def test_public_serve_requires_explicit_public_webhook_url(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "secret")
    monkeypatch.setenv("GITHUB_WEBHOOK_REGISTER", "true")

    missing = runner.invoke(app, ["serve", "--host", "0.0.0.0"])
    local = runner.invoke(
        app,
        [
            "serve",
            "--host",
            "0.0.0.0",
            "--webhook-url",
            "http://localhost:8000/webhook/github",
        ],
    )

    assert missing.exit_code == 1
    assert "--webhook-url" in missing.output
    assert local.exit_code == 1
    assert "public webhook URL" in local.output


def test_public_webhook_requires_https_and_loopback_http_is_allowed() -> None:
    with pytest.raises(WebhookURLError, match="HTTPS"):
        validate_webhook_url("http://hooks.example.com/github")

    assert (
        validate_webhook_url("http://127.0.0.1:8000/webhook/github")
        == "http://127.0.0.1:8000/webhook/github"
    )


def test_search_caps_results_and_closes_client(monkeypatch) -> None:
    from types import SimpleNamespace

    from kojutsu import cli_search as cli_module

    class SearchClient:
        def __init__(self) -> None:
            self.limit = 0
            self.closed = False

        def close(self) -> None:
            self.closed = True

        def search_documents(self, _query, *, frontmatter, limit):
            self.limit = limit
            return [
                SimpleNamespace(
                    content="bounded answer",
                    frontmatter={"repo": "org/repo", "pr": "1", "category": "design_decision"},
                )
            ]

    fake = SearchClient()
    monkeypatch.setenv("TANSEKI_URL", "https://tanseki.test")
    monkeypatch.setattr(cli_module.TansekiClient, "from_settings", lambda _settings: fake)

    result = runner.invoke(app, ["search", "auth", "--limit", "1000"])

    assert result.exit_code == 0
    assert fake.limit == 50
    assert fake.closed is True
    assert "bounded answer" in result.output


def test_questions_reports_the_outstanding_capture_queue(monkeypatch, tmp_path) -> None:
    """The `questions` command is the read side of the question lifecycle."""
    from kojutsu.core.question_registry import SqliteQuestionRegistry

    path = tmp_path / "registry.db"
    registry = SqliteQuestionRegistry(path)
    for index in range(3):
        registry.record_question(
            question_id=f"q{index}",
            github_comment_id=100 + index,
            repo="org/repo",
            pr_number=index + 1,
            pr_url=f"https://github.com/org/repo/pull/{index + 1}",
            question_text="Why?",
            question_category="design_decision",
        )
    registry.record_question(
        question_id="qdone",
        github_comment_id=200,
        repo="org/repo",
        pr_number=9,
        pr_url="https://github.com/org/repo/pull/9",
        question_text="Why?",
        question_category="design_decision",
    )
    registry.mark_question_answered(200, 9001)
    registry.close()
    monkeypatch.setenv("KOJUTSU_REGISTRY_PATH", str(path))

    result = runner.invoke(app, ["questions"])
    assert result.exit_code == 0, result.output
    assert "outstanding: 3" in result.output
    assert "pending=3" in result.output
    assert "answered=1" in result.output
    for index in range(3):
        assert f"q{index} state=pending" in result.output
    assert "qdone" not in result.output

    filtered = runner.invoke(app, ["questions", "--status", "answered"])
    assert filtered.exit_code == 0
    assert "qdone state=answered" in filtered.output

    everything = runner.invoke(app, ["questions", "--status", "all"])
    assert everything.exit_code == 0
    assert "qdone state=answered" in everything.output
    assert "q0 state=pending" in everything.output

    by_repo = runner.invoke(app, ["questions", "--repo", "other/repo"])
    assert by_repo.exit_code == 0
    assert "no questions match" in by_repo.output

    limited = runner.invoke(app, ["questions", "--limit", "1"])
    assert limited.exit_code == 0
    assert limited.output.count("state=pending") == 1

    claimed = runner.invoke(app, ["questions", "--status", "claimed"])
    assert claimed.exit_code == 0
    assert "no questions match" in claimed.output


def test_questions_rejects_an_unknown_status_or_negative_limit(monkeypatch, tmp_path) -> None:
    from kojutsu.core.question_registry import SqliteQuestionRegistry

    path = tmp_path / "registry.db"
    SqliteQuestionRegistry(path).close()
    monkeypatch.setenv("KOJUTSU_REGISTRY_PATH", str(path))

    unknown = runner.invoke(app, ["questions", "--status", "invented"])
    assert unknown.exit_code == 1
    assert "Unknown status" in unknown.output

    negative = runner.invoke(app, ["questions", "--limit", "-1"])
    assert negative.exit_code == 1
    assert "non-negative" in negative.output
