"""Tests for the ``ask`` and ``collect`` orchestration in the CLI.

These two commands are the seams where the pieces meet: ``ask`` turns generated
questions into GitHub comments *and* registry rows, in that order, and ``collect``
turns a pull request's comments into stored records. The behaviour worth testing
is what a caller sees afterwards — what was posted, what was recorded, what
happens when the second half fails — rather than that either function was called.

The GitHub client is the real one over ``httpx.MockTransport``, so the marker
reconciliation under test is the reconciliation that runs in production. The
registry is the real SQLite one for the same reason: the idempotency claims these
tests make are claims about rows on disk, and a fake registry cannot fail the way
a real one does.

No network, no model. Question generation is replaced by a stub that hands back
questions with fresh ``uuid4`` ids on every call, which is what the real generator
does and what makes the idempotency question non-trivial.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
from typer.testing import CliRunner

from kojutsu.cli import app
from kojutsu.cli_ask import derived_question_id
from kojutsu.core.question_registry import SqliteQuestionRegistry
from kojutsu.integrations.github import (
    answer_comment_body,
    answer_comment_body_as_agent,
    extract_question_id_from_comment_body,
    kojutsu_comment_body,
)
from kojutsu.models import Question, QuestionCategory

# The MockTransport patch lives with the other GitHub client tests rather than
# being restated here: two copies of "how to fake GitHub" is two sets of fakes to
# keep in step with the client.
from test_github_client import patch_http

runner = CliRunner()

PR_URL = "https://github.com/org/repo/pull/7"


def _question(text: str, *, fresh_id: bool = True) -> Question:
    return Question(
        id="00000000-0000-4000-8000-000000000001" if not fresh_id else f"fresh-{text[:4]}",
        text=text,
        category=QuestionCategory.DESIGN_DECISION,
    )


CONTEXT = {
    "pr_url": PR_URL,
    "pr_number": 7,
    "repo": "org/repo",
    "owner": "org",
    "repo_name": "repo",
    "branch_name": "feature/ABC-1-x",
    "jira_ticket_key": "ABC-1",
    "files_changed": ["src/x.py"],
}


class FakeForge:
    """An in-memory GitHub issue, served over the real client's HTTP calls.

    ``posts`` records only what created a comment. A re-run that reconciles onto
    an existing comment appends nothing, which is the property under test.
    """

    def __init__(self, *, login: str = "kojutsu-bot") -> None:
        self.login = login
        self.comments: list[dict[str, Any]] = []
        self.posts: list[dict[str, Any]] = []
        self.fail_post: int | None = None
        self.fail_list = False
        self.next_id = 100

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/user":
            return httpx.Response(200, json={"login": self.login})
        if request.method == "GET":
            if self.fail_list:
                return httpx.Response(500, json={"message": "boom"})
            return httpx.Response(200, json=self.comments)
        if self.fail_post is not None and len(self.posts) >= self.fail_post:
            return httpx.Response(502, json={"message": "bad gateway"})
        body = json.loads(request.read())
        self.posts.append(body)
        self.next_id += 1
        created = {
            "id": self.next_id,
            "body": body["body"],
            "user": {"login": self.login},
            "created_at": "2024-01-01T00:00:00Z",
            "author_association": "OWNER",
        }
        self.comments.append(created)
        return httpx.Response(201, json=created)


def _stub_generation(
    monkeypatch, questions: list[Question], context: dict[str, Any] | None = None
) -> None:
    from kojutsu import cli_ask as cli_module

    monkeypatch.setattr(
        cli_module,
        "generate_questions_for_pr",
        lambda **_kwargs: (list(questions), dict(CONTEXT if context is None else context)),
    )


def _apply(*extra: str) -> Any:
    return runner.invoke(app, ["ask", "org/repo#7", "--apply", *extra])


def _recorded(registry_path: Path) -> list[dict[str, Any]]:
    registry = SqliteQuestionRegistry(registry_path)
    try:
        return registry.list_questions(repo="org/repo", pr_number=7)
    finally:
        registry.close()


# -- ask --------------------------------------------------------------------------


def _recorded_author(registry_path: Path, question_id: str) -> str | None:
    registry = SqliteQuestionRegistry(registry_path)
    try:
        row = registry.get_question_by_id(question_id)
    finally:
        registry.close()
    return None if row is None else row["question_author"]


def test_ask_apply_posts_a_marked_comment_and_records_it(monkeypatch, tmp_path: Path) -> None:
    """The whole path, end to end: a marker on the PR and a row in the registry."""
    forge = FakeForge()
    patch_http(monkeypatch, forge.handler)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    _stub_generation(monkeypatch, [_question("Why this approach?")])

    result = _apply()

    assert result.exit_code == 0, result.output
    assert len(forge.posts) == 1
    body = forge.posts[0]["body"]
    question_id = extract_question_id_from_comment_body(body)
    assert question_id is not None
    assert body.endswith("Why this approach?")

    rows = _recorded(tmp_path / "registry.db")
    assert [row["question_id"] for row in rows] == [question_id]
    assert rows[0]["identifier"] == forge.next_id
    assert rows[0]["question_text"] == "Why this approach?"
    assert rows[0]["pr_url"] == PR_URL
    # Recorded as the account that posted the comment, because that is what the
    # answer side later refuses to accept from the same account without an agent
    # claim. Getting it wrong here makes self-answering look like independent review.
    assert _recorded_author(tmp_path / "registry.db", question_id) == "kojutsu-bot"


def test_ask_rerun_does_not_double_post_when_the_generator_returns_new_ids(
    monkeypatch, tmp_path: Path
) -> None:
    """A re-run with fresh ids still reconciles, because the id is derived.

    This is the failure the ticket exists to stop. The generator hands out a new
    ``uuid4`` every time, so a marker carried from the previous run would not
    match and the second run would post the same question twice.
    """
    forge = FakeForge()
    patch_http(monkeypatch, forge.handler)
    monkeypatch.setenv("GITHUB_TOKEN", "token")

    first_ids = set()
    for _ in range(2):
        _stub_generation(monkeypatch, [_question("Why this approach?")])
        result = _apply()
        assert result.exit_code == 0, result.output

    assert len(forge.posts) == 1
    marker = extract_question_id_from_comment_body(forge.posts[0]["body"])
    assert marker == derived_question_id(PR_URL, _question("Why this approach?", fresh_id=False))
    assert marker not in first_ids
    assert len(_recorded(tmp_path / "registry.db")) == 1
    assert "0 posted, 1 already present" in result.output


def test_ask_recovers_when_a_crash_lost_the_record_but_not_the_comment(
    monkeypatch, tmp_path: Path
) -> None:
    """Post-then-record means the comment can outlive the row. The re-run repairs it.

    Ordered exactly as the ticket describes: the comment is on the pull request
    and the registry write is lost. The re-run must post nothing, and must end
    with the row the failed run never wrote.
    """
    from kojutsu import cli_ask as cli_module

    forge = FakeForge()
    patch_http(monkeypatch, forge.handler)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    state = {"fail": True}

    class RegistryThatLosesTheFirstWrite:
        """The real registry, except the first question write, which is dropped.

        Raising from inside ``on_comment`` is the shape the real failure takes:
        the comment is already on the pull request when the row is lost.
        """

        def __init__(self, path: Path) -> None:
            self._registry = SqliteQuestionRegistry(path)

        # Yields itself, not the delegate: the command holds whatever the context
        # manager yields, so yielding the inner registry would bypass the failure
        # being simulated.
        def __enter__(self) -> RegistryThatLosesTheFirstWrite:
            self._registry.__enter__()
            return self

        def __exit__(self, *exc: object) -> None:
            self._registry.__exit__(*exc)

        def record_session(self, session: Any) -> None:
            self._registry.record_session(session)

        def record_question(self, **kwargs: Any) -> None:
            if state["fail"]:
                state["fail"] = False
                raise RuntimeError("simulated crash between post and record")
            self._registry.record_question(**kwargs)

    registry_path = tmp_path / "registry.db"
    monkeypatch.setattr(
        cli_module, "build_registry", lambda _s: RegistryThatLosesTheFirstWrite(registry_path)
    )
    _stub_generation(monkeypatch, [_question("Why this approach?")])

    crashed = _apply()
    assert crashed.exit_code != 0
    assert len(forge.posts) == 1
    assert _recorded(registry_path) == []

    # Left unpatched: the re-run builds the real registry from settings, which is
    # the ordinary path.
    rerun = _apply()

    assert rerun.exit_code == 0, rerun.output
    assert len(forge.posts) == 1, "the comment from the failed run was posted a second time"
    rows = _recorded(registry_path)
    assert [row["question_id"] for row in rows] == [
        extract_question_id_from_comment_body(forge.posts[0]["body"])
    ]


def test_ask_posts_again_when_the_previous_comment_cannot_be_seen(
    monkeypatch, tmp_path: Path
) -> None:
    """A comment kojutsu can no longer read is not evidence the question was asked.

    The registry still holds the row from the first run. Suppressing the post on
    the strength of that row would leave the pull request with a marker pointing
    at a comment that is gone, and the reviewer with no question at all.
    """
    forge = FakeForge()
    patch_http(monkeypatch, forge.handler)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    _stub_generation(monkeypatch, [_question("Why this approach?")])

    assert _apply().exit_code == 0
    forge.comments.clear()

    result = _apply()

    assert result.exit_code == 0, result.output
    assert len(forge.posts) == 2
    assert "1 posted, 0 already present" in result.output
    # The row is refreshed onto the new comment rather than duplicated.
    rows = _recorded(tmp_path / "registry.db")
    assert len(rows) == 1
    assert rows[0]["identifier"] == forge.next_id


def test_ask_does_not_adopt_a_marker_copied_by_another_account(monkeypatch, tmp_path: Path) -> None:
    """Reconciliation is keyed on kojutsu's own comments, not on any comment.

    Without this, anybody who could post could suppress a question by copying its
    marker, and the reviewer would never be asked.
    """
    forge = FakeForge()
    patch_http(monkeypatch, forge.handler)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    marker = derived_question_id(PR_URL, _question("Why this approach?"))
    forge.comments.append(
        {
            "id": 7,
            "body": kojutsu_comment_body(marker, "Why this approach?"),
            "user": {"login": "someone-else"},
            "created_at": "2024-01-01T00:00:00Z",
        }
    )
    _stub_generation(monkeypatch, [_question("Why this approach?")])

    result = _apply()

    assert result.exit_code == 0, result.output
    assert len(forge.posts) == 1
    assert len(_recorded(tmp_path / "registry.db")) == 1


def test_ask_collapses_questions_that_are_actually_the_same_question(
    monkeypatch, tmp_path: Path
) -> None:
    """Two identical questions are one question, not two comments and two rows."""
    forge = FakeForge()
    patch_http(monkeypatch, forge.handler)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    _stub_generation(
        monkeypatch,
        [
            _question("Why this approach?"),
            Question(
                id="another-fresh-id",
                text="Why this approach?",
                category=QuestionCategory.DESIGN_DECISION,
            ),
        ],
    )

    result = _apply()

    assert result.exit_code == 0, result.output
    assert len(forge.posts) == 1
    assert len(_recorded(tmp_path / "registry.db")) == 1
    assert "Applied 1 question(s)" in result.output


def test_ask_collapses_a_question_whose_category_the_generator_changed(
    monkeypatch, tmp_path: Path
) -> None:
    """Category is not part of the identity, because the comment does not carry it.

    A posted comment is a marker and the text. Two registry rows on one comment
    would both sit pending with one answer between them, so the category being
    re-rolled by a model must not fork the question.
    """
    forge = FakeForge()
    patch_http(monkeypatch, forge.handler)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    _stub_generation(
        monkeypatch,
        [
            _question("Why this approach?"),
            Question(
                id="another-fresh-id",
                text="Why this approach?",
                category=QuestionCategory.EDGE_CASE,
            ),
        ],
    )

    result = _apply()

    assert result.exit_code == 0, result.output
    assert len(forge.posts) == 1
    assert len(_recorded(tmp_path / "registry.db")) == 1


def test_ask_asks_a_reworded_question_again(monkeypatch, tmp_path: Path) -> None:
    """A different sentence is a different question, and this says so out loud.

    The bound on marker-derived idempotency: kojutsu cannot tell that two
    different sentences want the same answer, and guessing wrong would silently
    drop a question. The reviewer gets asked twice instead, which is visible.
    """
    forge = FakeForge()
    patch_http(monkeypatch, forge.handler)
    monkeypatch.setenv("GITHUB_TOKEN", "token")

    _stub_generation(monkeypatch, [_question("Why this approach?")])
    assert _apply().exit_code == 0
    _stub_generation(monkeypatch, [_question("Why not the simpler approach?")])
    result = _apply()

    assert result.exit_code == 0, result.output
    assert len(forge.posts) == 2
    assert len(_recorded(tmp_path / "registry.db")) == 2


def test_ask_records_the_head_the_questions_were_asked_about(monkeypatch, tmp_path: Path) -> None:
    """The anchor has to reach the row, not just exist somewhere in the plumbing.

    A question is a claim about one diff, and this column is the only thing that
    says which. The registry stores it, the generator supplies it and the plan
    artifact carries it, but that is three opportunities to drop it silently, and
    a dropped anchor looks identical to a question that had none -- there is no
    error, only a row that quietly cannot be matched to a commit. So the assertion
    is on the row on disk, read back through the real registry.

    The second half is the trap worth naming: ``_AskPlanContext`` is
    ``extra="forbid"``, so a context key the plan does not declare is not
    ignored or defaulted, it is *rejected* -- and the artifact is written on a
    review run and read back on a later apply. Validating the generator's shape
    through the model here is what keeps the two halves of that contract agreeing.
    """
    from kojutsu.cli_ask import _AskPlanContext

    head_sha = "b" * 40
    context = {**CONTEXT, "head_sha": head_sha}

    # The artifact must accept what the generator publishes.
    assert _AskPlanContext.model_validate(context).head_sha == head_sha
    # ...and survive the JSON hop it actually makes between review and apply.
    round_tripped = _AskPlanContext.model_validate_json(
        _AskPlanContext.model_validate(context).model_dump_json()
    )
    assert round_tripped.head_sha == head_sha

    forge = FakeForge()
    patch_http(monkeypatch, forge.handler)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    _stub_generation(monkeypatch, [_question("Why this approach?")], context=context)

    result = _apply()

    assert result.exit_code == 0, result.output
    rows = _recorded(tmp_path / "registry.db")
    assert len(rows) == 1
    assert _recorded_head_sha(tmp_path / "registry.db", rows[0]["question_id"]) == head_sha


def _recorded_head_sha(registry_path: Path, question_id: str) -> str | None:
    """Read the anchor back the way a consumer would: by question id.

    Not through ``list_questions``: that enumeration is a deliberately fixed
    column list shared with the projection, and widening it is a separate change
    with its own blast radius. ``get_question_by_id`` is the row-level read, and
    it is where the anchor belongs until someone widens the list on purpose.
    """
    registry = SqliteQuestionRegistry(registry_path)
    try:
        row = registry.get_question_by_id(question_id)
    finally:
        registry.close()
    return None if row is None else row["head_sha"]


def test_ask_reports_a_failed_post_without_recording_the_question(
    monkeypatch, tmp_path: Path
) -> None:
    """A 502 must not leave a row pointing at a comment that does not exist."""
    forge = FakeForge()
    forge.fail_post = 0
    patch_http(monkeypatch, forge.handler)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    _stub_generation(monkeypatch, [_question("Why this approach?")])

    result = _apply()

    assert result.exit_code == 1
    assert "Unable to post questions to GitHub" in result.output
    assert _recorded(tmp_path / "registry.db") == []


def test_ask_records_the_questions_it_did_post_before_one_failed(
    monkeypatch, tmp_path: Path
) -> None:
    """A partial post is reported as partial, and what landed stays recorded.

    The session and the first question are durable. Reporting the run as a total
    failure would tell the operator to re-run it, and the re-run is harmless
    only because the first question reconciles.
    """
    forge = FakeForge()
    forge.fail_post = 1
    patch_http(monkeypatch, forge.handler)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    _stub_generation(
        monkeypatch,
        [_question("Why first question?"), _question("Why second question?")],
    )

    result = _apply()

    assert result.exit_code == 1
    assert len(forge.posts) == 1
    rows = _recorded(tmp_path / "registry.db")
    assert [row["question_text"] for row in rows] == ["Why first question?"]

    monkeypatch.setattr(forge, "fail_post", None)
    resumed = _apply()

    assert resumed.exit_code == 0, resumed.output
    assert len(forge.posts) == 2
    assert len(_recorded(tmp_path / "registry.db")) == 2
    assert "1 posted, 1 already present" in resumed.output


def test_ask_reports_an_unreadable_comment_list_rather_than_posting_blind(
    monkeypatch, tmp_path: Path
) -> None:
    """Reconciliation is only possible against a list that was actually read.

    Posting without it would post every question on every run — the exact
    duplicate this path is here to prevent — so a failed read has to stop the
    run instead of being treated as an empty pull request.
    """
    forge = FakeForge()
    forge.fail_list = True
    patch_http(monkeypatch, forge.handler)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    _stub_generation(monkeypatch, [_question("Why this approach?")])

    result = _apply()

    assert result.exit_code == 1
    assert "Unable to post questions to GitHub" in result.output
    assert forge.posts == []
    assert _recorded(tmp_path / "registry.db") == []


def test_ask_plan_artifact_ids_are_derived_and_apply_does_not_re_derive_them(
    monkeypatch, tmp_path: Path
) -> None:
    """The artifact is what gets applied, verbatim.

    Re-deriving on apply would make the reviewed artifact a description of the
    run rather than the thing being run, and would silently rewrite ids an
    operator has already seen.
    """
    from kojutsu import cli_ask as cli_module

    forge = FakeForge()
    patch_http(monkeypatch, forge.handler)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    plan_path = tmp_path / "plan.json"
    _stub_generation(monkeypatch, [_question("Why this approach?")])

    assert runner.invoke(app, ["ask", "org/repo#7", "--plan-file", str(plan_path)]).exit_code == 0
    artifact = json.loads(plan_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        cli_module,
        "generate_questions_for_pr",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("apply regenerated questions")),
    )

    result = runner.invoke(app, ["ask", "org/repo#7", "--apply", "--plan-file", str(plan_path)])

    assert result.exit_code == 0, result.output
    assert (
        extract_question_id_from_comment_body(forge.posts[0]["body"])
        == (artifact["questions"][0]["id"])
    )


# -- collect ----------------------------------------------------------------------


class RecordingSink:
    """A sink that keeps what it was given, so a test can read the record back."""

    def __init__(self, *, fail: bool = False) -> None:
        self.entries: list[Any] = []
        self.fail = fail

    def store(self, entry: Any) -> None:
        if self.fail:
            from kojutsu.integrations.tanseki import TansekiUnavailableError

            raise TansekiUnavailableError("store is unavailable")
        self.entries.append(entry)


def _runtime(registry_path: Path, sink: RecordingSink) -> Any:
    registry = SqliteQuestionRegistry(registry_path)

    class Runtime:
        def __init__(self) -> None:
            self.registry = registry
            self.sink = sink

        def __enter__(self) -> Runtime:
            registry.__enter__()
            return self

        def __exit__(self, *exc: object) -> None:
            registry.__exit__(*exc)

    return Runtime()


def _register_question(registry_path: Path, *, comment_id: int = 100) -> SqliteQuestionRegistry:
    registry = SqliteQuestionRegistry(registry_path)
    registry.record_question(
        question_id="q-registered",
        github_comment_id=comment_id,
        repo="org/repo",
        pr_number=7,
        pr_url=PR_URL,
        question_text="Why this approach?",
        question_category="design_decision",
        question_author="kojutsu-bot",
    )
    return registry


def _comment(
    identifier: int, body: str, login: str, *, association: str = "MEMBER", second: int = 0
) -> dict[str, Any]:
    return {
        "id": identifier,
        "body": body,
        "user": {"login": login},
        "created_at": f"2024-01-01T00:00:{second:02d}Z",
        "author_association": association,
    }


def _run_collect(monkeypatch, registry_path: Path, sink: RecordingSink, forge: FakeForge) -> Any:
    from kojutsu import cli_ask as cli_module

    patch_http(monkeypatch, forge.handler)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(
        cli_module, "build_runtime", lambda _settings: _runtime(registry_path, sink)
    )
    return runner.invoke(app, ["collect", "org/repo#7"])


def test_collect_stores_one_record_per_answer(monkeypatch, tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.db"
    _register_question(registry_path).close()
    sink = RecordingSink()
    forge = FakeForge()
    forge.comments = [
        _comment(100, kojutsu_comment_body("q-registered", "Why this approach?"), "bot"),
        _comment(201, answer_comment_body("q-registered", "Because X."), "dev", second=1),
    ]

    result = _run_collect(monkeypatch, registry_path, sink, forge)

    assert result.exit_code == 0, result.output
    assert "Collected 1 new answer(s)." in result.output
    assert len(sink.entries) == 1
    entry = sink.entries[0]
    assert entry.answer_text == "Because X."
    assert entry.metadata["question_id"] == "q-registered"
    assert entry.metadata["github_comment_id"] == 201
    assert entry.metadata["github_author_association"] == "MEMBER"
    assert entry.metadata["repo"] == "org/repo"


def test_collect_is_idempotent_across_runs(monkeypatch, tmp_path: Path) -> None:
    """Re-running ``collect`` on the same comments stores nothing the first run stored.

    This is the path the webhook and ``collect`` share, and it is the one that
    runs twice by accident: an operator retries a manual collect, or a webhook
    delivery is redelivered.
    """
    registry_path = tmp_path / "registry.db"
    _register_question(registry_path).close()
    sink = RecordingSink()
    forge = FakeForge()
    forge.comments = [
        _comment(100, kojutsu_comment_body("q-registered", "Why this approach?"), "bot"),
        _comment(201, answer_comment_body("q-registered", "Because X."), "dev", second=1),
    ]

    first = _run_collect(monkeypatch, registry_path, sink, forge)
    second = _run_collect(monkeypatch, registry_path, sink, forge)

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert "Collected 1 new answer(s)." in first.output
    assert "Collected 0 new answer(s)." in second.output
    assert len(sink.entries) == 1


def test_collect_ignores_comments_that_are_not_answers_to_a_question_it_asked(
    monkeypatch, tmp_path: Path
) -> None:
    """Ordinary conversation on a pull request is not knowledge."""
    registry_path = tmp_path / "registry.db"
    _register_question(registry_path).close()
    sink = RecordingSink()
    forge = FakeForge()
    forge.comments = [
        _comment(100, kojutsu_comment_body("q-registered", "Why this approach?"), "bot"),
        # Two people talking to each other, with no marker at all.
        _comment(201, "looks good to me", "dev", second=1),
        _comment(202, "yep, +1", "reviewer", second=2),
        _comment(203, answer_comment_body("q-registered", "Because X."), "dev", second=3),
        # Marked as an answer, but the question it names was never asked here.
        _comment(204, answer_comment_body("q-unknown", "Orphan."), "dev", second=4),
        # An agent marker alone does not make a comment an answer.
        _comment(205, "<!-- kojutsu:agent:reviewer -->\n\nlooks fine", "dev", second=5),
    ]

    result = _run_collect(monkeypatch, registry_path, sink, forge)

    assert result.exit_code == 0, result.output
    assert "Collected 1 new answer(s)." in result.output
    assert [entry.answer_text for entry in sink.entries] == ["Because X."]


def test_collect_stores_an_answer_from_an_account_with_no_standing(
    monkeypatch, tmp_path: Path
) -> None:
    """**The refusal this replaces was the policy that was wrong, measured.**

    ``collect`` used to drop an answer from anybody who was not ``OWNER``, ``MEMBER``
    or ``COLLABORATOR``, on the reasoning that it "is not evidence about the change".
    That set was the wrong filter for the thing it was doing: on t3code's PR #2829 it
    discarded 28 human comments to admit 21 bot ones, because a bot is by definition
    not a member or collaborator of anything and so lands on ``CONTRIBUTOR``. Over 400
    stored review captures re-fetched from the two corpora there were no
    ``CONTRIBUTOR`` or ``NONE`` reviewers at all. See
    ``kojutsu.allowlist.ADMIT_ALL_ASSOCIATIONS``.

    What an operator has left is the repository allowlist, which ``collect`` already
    checks before it reads anything, and the record: the association is stored on the
    answer, so a reader can see the reply came from an account with no standing in the
    repository, and ``independence`` says how far it is from checking itself.
    """
    registry_path = tmp_path / "registry.db"
    _register_question(registry_path).close()
    sink = RecordingSink()
    forge = FakeForge()
    forge.comments = [
        _comment(100, kojutsu_comment_body("q-registered", "Why this approach?"), "bot"),
        _comment(
            201,
            answer_comment_body("q-registered", "Because X."),
            "stranger",
            association="NONE",
            second=1,
        ),
    ]

    result = _run_collect(monkeypatch, registry_path, sink, forge)

    assert result.exit_code == 0, result.output
    assert "Collected 1 new answer(s)." in result.output
    assert len(sink.entries) == 1
    entry = sink.entries[0]
    assert entry.metadata["comment_author"] == "stranger"
    assert entry.metadata["github_author_association"] == "NONE", (
        "the fact the gate used to consume is still on the record, which is what a "
        "reader weighs in its place"
    )
    assert entry.metadata["comment_author_is_machine"] is False


def test_collect_reports_a_store_failure_and_holds_the_claim_for_the_next_run(
    monkeypatch, tmp_path: Path
) -> None:
    """A failed store must not consume the answer.

    The registry claims the answer before writing it and releases it when the
    write fails. If the claim were not released, one store outage would lose the
    answer permanently, and the re-run that would have recovered it would report
    the answer as already collected.
    """
    registry_path = tmp_path / "registry.db"
    _register_question(registry_path).close()
    forge = FakeForge()
    forge.comments = [
        _comment(100, kojutsu_comment_body("q-registered", "Why this approach?"), "bot"),
        _comment(201, answer_comment_body("q-registered", "Because X."), "dev", second=1),
    ]

    failing = _run_collect(monkeypatch, registry_path, RecordingSink(fail=True), forge)

    assert failing.exit_code == 1
    assert "Tanseki is unavailable" in failing.output
    assert "Collected" not in failing.output

    sink = RecordingSink()
    recovered = _run_collect(monkeypatch, registry_path, sink, forge)

    assert recovered.exit_code == 0, recovered.output
    assert "Collected 1 new answer(s)." in recovered.output
    assert [entry.answer_text for entry in sink.entries] == ["Because X."]


def test_collect_reports_an_unreadable_comment_list(monkeypatch, tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.db"
    _register_question(registry_path).close()
    sink = RecordingSink()
    forge = FakeForge()
    forge.fail_list = True

    result = _run_collect(monkeypatch, registry_path, sink, forge)

    assert result.exit_code == 1
    assert "Unable to read GitHub comments" in result.output
    assert sink.entries == []


def test_collect_reads_comments_in_creation_order(monkeypatch, tmp_path: Path) -> None:
    """An answer posted before its question is still matched to it.

    GitHub returns comments newest-first and the loop sorts by ``created_at``,
    so the parent is in the map before the reply is looked up. Without the sort,
    an answer that arrived out of order would be dropped as unparented and never
    retried, because it is already on the pull request.
    """
    registry_path = tmp_path / "registry.db"
    _register_question(registry_path).close()
    sink = RecordingSink()
    forge = FakeForge()
    forge.comments = [
        _comment(201, answer_comment_body("q-registered", "Because X."), "dev", second=9),
        _comment(100, kojutsu_comment_body("q-registered", "Why this approach?"), "bot"),
    ]

    result = _run_collect(monkeypatch, registry_path, sink, forge)

    assert result.exit_code == 0, result.output
    assert "Collected 1 new answer(s)." in result.output
    assert [entry.answer_text for entry in sink.entries] == ["Because X."]


def test_collect_refuses_the_asking_account_answering_its_own_question(
    monkeypatch, tmp_path: Path
) -> None:
    """The bot answering its own question is not independent review.

    Decided from ``question_author`` as stored at ask time, not from the current
    run's view of the pull request, so editing the question comment afterwards
    cannot change whether a stored answer counts.
    """
    registry_path = tmp_path / "registry.db"
    registry = SqliteQuestionRegistry(registry_path)
    registry.record_question(
        question_id="q-registered",
        github_comment_id=100,
        repo="org/repo",
        pr_number=7,
        pr_url=PR_URL,
        question_text="Why this approach?",
        question_category="design_decision",
        question_author="dev",
    )
    registry.close()
    sink = RecordingSink()
    forge = FakeForge()
    forge.comments = [
        _comment(100, kojutsu_comment_body("q-registered", "Why this approach?"), "bot"),
        _comment(201, answer_comment_body("q-registered", "I wrote it."), "dev", second=1),
    ]

    result = _run_collect(monkeypatch, registry_path, sink, forge)

    assert result.exit_code == 0, result.output
    assert "Collected 0 new answer(s)." in result.output
    assert sink.entries == []


def test_collect_records_a_self_answer_that_declares_an_agent(monkeypatch, tmp_path: Path) -> None:
    """An explicit machine claim is what makes a self-answer capturable at all.

    The record is then attributed to the named agent rather than to the account,
    so a reader can tell a declared principal from a person asserting their own
    review.
    """
    registry_path = tmp_path / "registry.db"
    registry = SqliteQuestionRegistry(registry_path)
    registry.record_question(
        question_id="q-registered",
        github_comment_id=100,
        repo="org/repo",
        pr_number=7,
        pr_url=PR_URL,
        question_text="Why this approach?",
        question_category="design_decision",
        question_author="dev",
    )
    registry.close()
    sink = RecordingSink()
    forge = FakeForge()
    forge.comments = [
        _comment(100, kojutsu_comment_body("q-registered", "Why this approach?"), "bot"),
        _comment(
            201,
            answer_comment_body_as_agent("q-registered", "Because X.", "reviewer", model="m"),
            "dev",
            second=1,
        ),
    ]

    result = _run_collect(monkeypatch, registry_path, sink, forge)

    assert result.exit_code == 0, result.output
    assert "Collected 1 new answer(s)." in result.output
    assert sink.entries[0].author == "reviewer"
    assert sink.entries[0].metadata["answered_by_agent"] == "reviewer"
    assert sink.entries[0].metadata["answered_by_model"] == "m"
