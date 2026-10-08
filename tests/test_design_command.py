"""The design phase on a command surface: what the pause does, and what the resume refuses.

The library tests (:mod:`tests.test_design_capture`, :mod:`tests.test_ticket_drafts`) pin
each phase's behaviour. What is missing is the wiring, and the wiring has exactly one
interesting property: **the pause is a state transition, not a blocking read.** These
tests are grouped by the claims that makes true.

- **No model, no network, no daemon.** Both model calls go through a seam
  (:func:`~kojutsu.cli_design._design_proposer` / :func:`~kojutsu.cli_design._design_reconciler`), and the
  capture collaborators are injected the way every other CLI test in this repository
  injects them -- by replacing one module attribute. Everything else is the real registry,
  the real approval ledger and the real ticket store, because the claims below are claims
  about those files' primary keys and a fake would agree with any implementation.
- **The first invocation records the block.** Not "creates no ticket" -- *records the
  block*, through the same `create_tickets_from_plan` call that would have created them.
  Asserting only the absence of tickets would pass for a command that consulted no gate.
- **The reconciler is not re-run on the resume.** This is the test that matters most, and
  it is built so that the wrong implementation cannot pass: the stubbed reconciler returns
  a *different* plan on a second call, so a resume that re-ran the model would either
  create the wrong tickets or be refused for a change nobody made. Both are failures, and
  the run must still succeed with the first plan's tickets.
- **Ambiguity is a refusal naming both candidates**, and a changed plan is a refusal
  before anything is written -- together those two are what makes "one recorded
  reconciliation" an enforced property rather than a convention.
- **The exit codes are distinguishable**, because a gated run that reported the error code
  would train an operator to ignore the command.
- **The pause prints something reviewable.** The last group is the one that stops the gate
  regressing into a rubber stamp: it asserts the goals, the decisions with their rejected
  alternatives, the discards with their reasons, and the drafts with their edges are all
  in the output, because a pause that prints "awaiting approval" over a digest has asked a
  person to take responsibility for a document nobody showed them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from kojutsu import cli_design as cli_module
from kojutsu.cli import app
from kojutsu.cli_design import DESIGN_GATED_EXIT_CODE
from kojutsu.core.design_capture import design_plan_digest, find_design_reconciliations
from kojutsu.core.design_plan import DesignPlan, parse_design_plan
from kojutsu.core.knowledge_sink import KnowledgeDeliveryOutcome, KnowledgeDeliveryStatus
from kojutsu.core.question_registry import SqliteQuestionRegistry
from kojutsu.core.ticket_drafts import SqlitePlanApprovalLedger, SqliteTicketSink

runner = CliRunner()

REPO = "org/repo"
TOPIC = "design-phase-recording"
RECONCILER = "reconciler"
APPROVER = "dana"
OTHER_APPROVER = "kai"

DECISION = "Reconcile from a principal that proposed nothing."
ALTERNATIVE = "Let the first proposer reconcile its own panel"
WHY_REJECTED = "A model judging its own proposal checks nothing."
DISCARDS_REASON = "It assumes a Redis dependency this repository does not have."
GOAL = "Make the design phase a recorded pipeline with an independent reconciler"


class RecordingSink:
    """Records what was stored. No network, no store, no clock worth asserting on."""

    def __init__(self) -> None:
        self.entries: list[Any] = []

    def store(self, entry: Any) -> KnowledgeDeliveryOutcome:
        self.entries.append(entry)
        return KnowledgeDeliveryOutcome(
            entry_id=entry.entry_id, status=KnowledgeDeliveryStatus.DELIVERED
        )


class FakeRuntime:
    """What :func:`kojutsu.runtime.build_runtime` hands the design flow, and nothing else.

    The design path needs a registry and a sink and does not talk to Tanseki, so this
    supplies a *real* registry at the configured path -- the proposals and the
    reconciliation are claimed through the actual table, which is the whole point of
    exercising this command against real stores.
    """

    def __init__(self, registry_path: str, sink: RecordingSink) -> None:
        self.registry = SqliteQuestionRegistry(registry_path)
        self.sink = sink

    def close(self) -> None:
        self.registry.close()


def _ticket(
    identifier: str, *, depends_on: tuple[str, ...] = (), **overrides: Any
) -> dict[str, Any]:
    draft: dict[str, Any] = {
        "id": identifier,
        "title": f"Implement {identifier} on the command surface",
        "description": f"Make {identifier} the thing the plan says it should be.",
        "acceptance_criteria": [
            f"Given {identifier}, when the command runs, then nothing else changes"
        ],
        "test_command": "uv run pytest -q",
        "reference_files": ["src/kojutsu/cli.py"],
        "labels": ["design-phase"],
        "priority": "P2",
        "depends_on": list(depends_on),
    }
    draft.update(overrides)
    return draft


def _plan(*drafts: dict[str, Any], goals: list[str] | None = None) -> dict[str, Any]:
    return {
        "goals": goals if goals is not None else [GOAL],
        "decisions": [
            {
                "summary": "Reconcile from a principal that proposed nothing",
                "rationale": DECISION,
                "alternatives_rejected": [
                    {"alternative": ALTERNATIVE, "why_rejected": WHY_REJECTED}
                ],
            }
        ],
        "tickets": list(drafts) or [_ticket("capture"), _ticket("resume", depends_on=("capture",))],
    }


def _payload(
    *,
    plan: dict[str, Any] | None = None,
    discarded: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "plan": plan if plan is not None else _plan(),
        "discarded": discarded if discarded is not None else [],
    }


class StubProposer:
    """One argument per principal, and a count of how often it was asked.

    The count is the point of the reconciler-equivalent below, and here it is just so a
    test can assert the panel was asked exactly once each.
    """

    def __init__(self, texts: dict[str, str] | None = None) -> None:
        self.texts = texts or {}
        self.calls: list[tuple[str, str]] = []

    def __call__(self, *, repo: str, topic: str, principal: str) -> str:
        self.calls.append((repo, principal))
        return self.texts.get(principal, f"{principal} argues for recording the whole topic.")


class StubReconciler:
    """A reconciler that returns a *different* plan every time it is called.

    Deliberately adversarial to the one behaviour this command must have. A model is
    nondeterministic, so a second call in the same run is exactly what "re-run the
    reconciler on resume" looks like from the outside -- and this stub makes that mistake
    produce a wrong ticket set rather than a passing test.
    """

    def __init__(self, payloads: list[dict[str, Any]] | None = None) -> None:
        self.payloads = payloads if payloads is not None else [_payload()]
        self.calls = 0

    def __call__(self, *, repo: str, topic: str, proposals: Any) -> dict[str, Any]:
        self.calls += 1
        payload = self.payloads[min(self.calls, len(self.payloads)) - 1]
        return json.loads(json.dumps(payload))


def _configure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, repository: str = REPO) -> None:
    """Point the command at isolated state, and allow ``repository``.

    Every path is a temporary one rather than the shipped default, so a test never writes
    into ``~/.kojutsu`` and never reads a developer's real ledger. The defaults themselves
    are pinned separately.
    """
    monkeypatch.setenv("KOJUTSU_REGISTRY_PATH", str(tmp_path / "registry.db"))
    monkeypatch.setenv("DESIGN_PLAN_APPROVAL_LEDGER_PATH", str(tmp_path / "approvals.db"))
    monkeypatch.setenv("DESIGN_TICKET_SINK_PATH", str(tmp_path / "tickets.db"))
    monkeypatch.setenv("GITHUB_WEBHOOK_ALLOWED_REPOSITORIES", repository)
    monkeypatch.setenv("LLM_MODEL", "opencode/model")


def _install(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    proposer: StubProposer | None = None,
    reconciler: StubReconciler | None = None,
    repository: str = REPO,
) -> tuple[StubProposer, StubReconciler, RecordingSink]:
    _configure(monkeypatch, tmp_path, repository=repository)
    sink = RecordingSink()
    monkeypatch.setattr(
        cli_module,
        "build_runtime",
        lambda settings: FakeRuntime(settings.kojutsu_registry_path, sink),
    )
    the_proposer = proposer or StubProposer()
    the_reconciler = reconciler or StubReconciler()
    monkeypatch.setattr(cli_module, "_design_proposer", lambda settings: the_proposer)
    monkeypatch.setattr(cli_module, "_design_reconciler", lambda settings: the_reconciler)
    return the_proposer, the_reconciler, sink


def _capture_argv(*proposers: str, reconciler: str = RECONCILER, topic: str = TOPIC) -> list[str]:
    argv = ["design", "--repo", REPO, "--topic", topic]
    for principal in proposers or ("proposer-a",):
        argv += ["--proposer", principal]
    argv += ["--reconciled-by", reconciler]
    return argv


def _resume_argv(*, approver: str = APPROVER, topic: str = TOPIC, extra: list[str] | None = None):
    return [
        "design",
        "--repo",
        REPO,
        "--topic",
        topic,
        "--approve",
        "--approved-by",
        approver,
        *(extra or []),
    ]


def _ledger(tmp_path: Path) -> SqlitePlanApprovalLedger:
    return SqlitePlanApprovalLedger(tmp_path / "approvals.db")


def _sink(tmp_path: Path) -> SqliteTicketSink:
    return SqliteTicketSink(tmp_path / "tickets.db")


def _registry(tmp_path: Path) -> SqliteQuestionRegistry:
    return SqliteQuestionRegistry(tmp_path / "registry.db")


def _gated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **kwargs: Any):
    """Run the capture invocation once, assert it gated, and hand back its collaborators."""
    proposer, reconciler, sink = _install(monkeypatch, tmp_path, **kwargs)
    result = runner.invoke(app, _capture_argv("proposer-a", "proposer-b"))
    assert result.exit_code == DESIGN_GATED_EXIT_CODE, result.output
    return result, proposer, reconciler, sink


# --- the first invocation: captures, reconciles, records the block, creates nothing ---


def test_the_first_invocation_records_the_block_and_creates_no_ticket(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The pause is a recorded gate decision, not a printed sentence.

    Asserting "no tickets" alone would pass for a command that consulted no gate at all,
    which is the whole thing the phase exists to prevent. So the assertion is the recorded
    decision as well: a *blocking* row, about this plan's digest, in the approval ledger.
    """
    result, _, _, sink = _gated(monkeypatch, tmp_path)

    assert "tickets-created: 0" in result.output
    with _sink(tmp_path) as tickets:
        assert tickets.list_tickets() == []
    with _ledger(tmp_path) as ledger:
        digest = design_plan_digest(_plan_artifact(result))
        decisions = ledger.list_decisions(plan_sha256=digest)
        assert decisions, "the gate decision has to be a row, not an inference"
        assert any(decision.verdict == "blocked" for decision in decisions)
        assert any("no recorded human approval" in decision.reason for decision in decisions)
        assert ledger.list_approvals(plan_sha256=digest) == []

    # And the proposals and the reconciliation are records, not transcript prose.
    with _registry(tmp_path) as registry:
        rows = registry.list_rationales(repo=REPO, limit=50)
    assert [row["declared_by"] for row in rows] == ["proposer-a", "proposer-b", RECONCILER]
    assert len(sink.entries) == 3


def _plan_artifact(result: Any) -> DesignPlan:
    """The plan the pause wrote, read back through the path the pause printed.

    Parsed rather than returned as text, so a caller cannot accidentally compare the
    digest of a dict against a digest of a model -- ``design_plan_digest`` reads
    ``model_dump_json`` and a raw dict has no such thing.
    """
    return parse_design_plan(json.loads(_artifact_path(result).read_text(encoding="utf-8")))


def _artifact_path(result: Any) -> Path:
    line = next(line for line in result.output.splitlines() if line.startswith("plan-artifact: "))
    return Path(line.split(": ", 1)[1])


# --- the pause prints something a person can actually review ------------------------


def test_the_pause_prints_the_thing_being_approved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The gate's value is entirely the content of this print, so this asserts the content.

    Every element checked here is one somebody reading the plan needs and one that a plan
    cannot recover for itself: what it is for, what was decided and *why*, what was
    rejected instead, what was thrown away of the panel, and what the plan will become
    including the edges between the drafts. A pause that printed "awaiting approval" over
    a digest would satisfy "stops at the gate" and gate nothing, and this test is what
    stops that regressing.
    """
    reconciler = StubReconciler(
        [
            _payload(
                discarded=[{"principal": "proposer-b", "revision": 1, "reason": DISCARDS_REASON}]
            )
        ]
    )
    result, _, _, _ = _gated(monkeypatch, tmp_path, reconciler=reconciler)

    output = result.output
    assert GOAL in output, "the goals are what the tickets have to serve"
    assert DECISION in output, "a decision without its rationale is an assertion"
    assert ALTERNATIVE in output and WHY_REJECTED in output, (
        "the rejected alternatives are the half a reconciliation usually loses, and an "
        "approval given without them approves a conclusion with no visible alternative"
    )
    assert DISCARDS_REASON in output and "proposer-b" in output, (
        "a discard is a decision worth recording; showing it is what tells the reader the "
        "panel was actually judged"
    )
    assert "Implement capture" in output, "the drafts are what the plan becomes"
    assert "depends on: capture" in output, "an edge shown as prose is not a dependency"
    assert "Given capture, when the command runs" in output
    assert "uv run pytest -q" in output, "the verification command is part of the review"
    assert "awaiting-approval" in output
    assert f"exit {DESIGN_GATED_EXIT_CODE}" in output


def test_a_panel_with_nothing_discarded_says_so_rather_than_printing_an_empty_section(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An empty section a reader has to interpret is worse than one that says it is empty."""
    result, _, _, _ = _gated(monkeypatch, tmp_path)

    assert "none; every proposal was read and none was discarded" in result.output


# --- the resume does not re-run the reconciler --------------------------------------


def test_the_resume_does_not_re_run_the_reconciler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The test this whole command is shaped around.

    The stubbed reconciler returns a *different* plan on its second call, so a resume that
    reached the model would produce the wrong tickets or be refused for a plan change
    nobody made -- and either way this assertion fails, which is the point of building the
    stub adversarially rather than harmlessly.

    Three things are asserted: the reconciler was asked exactly once across both
    invocations, the resume exited 0, and the tickets that exist are the first plan's.
    """
    other_plan = _plan(_ticket("something-else-entirely"))
    reconciler = StubReconciler([_payload(), _payload(plan=other_plan)])
    _, _, reconciler_used, _ = _gated(monkeypatch, tmp_path, reconciler=reconciler)
    assert reconciler_used.calls == 1

    result = runner.invoke(app, _resume_argv())

    assert result.exit_code == 0, result.output
    assert reconciler_used.calls == 1, "the resume reached the model"
    with _sink(tmp_path) as tickets:
        stored = tickets.list_tickets()
    assert [ticket.draft_id for ticket in stored] == ["capture", "resume"], (
        "the tickets that exist are the plan that was reconciled and reviewed"
    )
    assert "tickets-created: 2" in result.output


def test_the_resume_reuses_the_recorded_reconciliation_rather_than_a_second_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The mechanism behind that: the resume *finds* the record and spends it.

    Asserted on the ledger rather than only on the exit code, because "the run succeeded"
    and "the run succeeded by finding the reconciliation" are different claims, and a
    resume that re-derived an id instead of looking one up would pass the first.
    """
    result, _, _, _ = _gated(monkeypatch, tmp_path)
    recorded_id = next(
        line.split(": ", 1)[1]
        for line in result.output.splitlines()
        if line.startswith("reconciliation: ")
    )

    resumed = runner.invoke(app, _resume_argv())

    assert resumed.exit_code == 0, resumed.output
    assert f"reconciliation: {recorded_id}" in resumed.output
    with _registry(tmp_path) as registry:
        found = find_design_reconciliations(repo=REPO, topic=TOPIC, registry=registry)
    assert len(found) == 1
    assert found[0].entry_id == recorded_id


# --- refusals the resume owes the operator ------------------------------------------


def test_a_resume_with_no_recorded_reconciliation_is_a_refusal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ "Nothing found" must never be a success with zero tickets.

    A resume that reported no reconciliation, created nothing and exited 0 would be
    indistinguishable from a run that correctly decided there was no work -- and a script
    would report the design phase as done.
    """
    _install(monkeypatch, tmp_path)

    result = runner.invoke(app, _resume_argv())

    assert result.exit_code == 1
    assert "No recorded reconciliation" in result.output
    assert TOPIC in result.output and REPO in result.output, "the refusal names what it looked for"
    with _sink(tmp_path) as tickets:
        assert tickets.list_tickets() == []


def test_a_resume_refuses_two_reconciliations_naming_both(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ambiguity is a refusal, never a coin flip.

    Picking one of two recorded reconciliations is the same class of bug as a gateway
    choosing between two matching routes: it produces a well-formed result whose wrongness
    nothing downstream can detect. So both are named -- principal, revision and digest --
    because any one of the three is what an operator needs and printing the wrong one
    would leave them solving for the other.
    """
    _gated(monkeypatch, tmp_path)
    # A second judgement of the same topic, recorded the way the command tells you to make
    # one: a different reconciler at the next revision. This is the state the ticket warns
    # about, and it is reachable on purpose rather than by corruption.
    second_run = runner.invoke(
        app,
        [
            "design",
            "--repo",
            REPO,
            "--topic",
            TOPIC,
            "--proposer",
            "proposer-c",
            "--reconciled-by",
            "other-reconciler",
            "--revision",
            "2",
        ],
    )
    assert second_run.exit_code == DESIGN_GATED_EXIT_CODE, second_run.output

    result = runner.invoke(app, _resume_argv())

    assert result.exit_code == 1
    assert "2 recorded reconciliations" in result.output
    assert "by 'reconciler'" in result.output
    assert "by 'other-reconciler'" in result.output
    assert result.output.count("plan digest ") >= 2, "each candidate carries its own digest"
    with _sink(tmp_path) as tickets:
        assert tickets.list_tickets() == []


def test_a_second_capture_run_over_a_recorded_topic_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The gate is entered once per revision, and the refusal says how to enter the other.

    Silently reconciling again would be the bug the resume test is about, arriving from the
    capture side: a second plan under a first plan's record. So the run stops and names the
    two ways forward -- approve what is recorded, or record a genuinely different judgement
    at the next revision.
    """
    reconciler = StubReconciler([_payload(), _payload(plan=_plan(_ticket("other")))])
    _gated(monkeypatch, tmp_path, reconciler=reconciler)

    result = runner.invoke(app, _capture_argv("proposer-a", "proposer-b"))

    assert result.exit_code == 1
    assert "already recorded" in result.output
    assert "--approve" in result.output and "--revision" in result.output
    assert reconciler.calls == 1, "the reconciler was asked again before the refusal"


def test_a_resume_with_a_changed_plan_is_refused_before_anything_is_written(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A plan edited after its reconciliation is a different plan.

    This is the check the derived ticket id cannot make for itself: the ids are computed
    from the digest, so a changed plan derives *different* ids and would create a second,
    unapproved set that looks exactly like a first run of a new plan. Asserted on the
    ledger as well as the exit code, because "refused before anything was written" is the
    claim and an approval row would contradict it.
    """
    result, _, _, _ = _gated(monkeypatch, tmp_path)
    artifact = _artifact_path(result)
    tampered = json.loads(artifact.read_text(encoding="utf-8"))
    tampered["goals"] = ["A completely different goal, written after the reconciliation"]
    artifact.write_text(json.dumps(tampered), encoding="utf-8")
    # The artefact is read under an ownership-and-permission check, so the edit has to
    # leave it looking like the operator's own file or this test would prove the file
    # checks fire rather than the digest check.
    artifact.chmod(artifact.stat().st_mode & ~0o077)

    resumed = runner.invoke(app, _resume_argv())

    assert resumed.exit_code == 1
    assert "digests to" in resumed.output
    with _sink(tmp_path) as tickets:
        assert tickets.list_tickets() == []
    with _ledger(tmp_path) as ledger:
        for digest in _all_known_digests(result):
            assert ledger.list_approvals(plan_sha256=digest) == []


def _all_known_digests(result: Any) -> list[str]:
    """Both digests the operator could be holding: the recorded one and the edited one.

    The assertion below is that *neither* has an approval, so it has to cover both -- a run
    that recorded an approval against the edited digest and then refused would still have
    put a row in the ledger.
    """
    recorded = next(
        line.split(": ", 1)[1]
        for line in result.output.splitlines()
        if line.startswith("plan-digest: ")
    )
    edited = parse_design_plan(json.loads(_artifact_path(result).read_text(encoding="utf-8")))
    return [recorded, design_plan_digest(edited)]


# --- the reconciler must not be a proposer -----------------------------------------


@pytest.mark.parametrize("shape", ["kept", "discarded"])
def test_a_reconciler_that_proposed_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, shape: str
) -> None:
    """Both ways the constraint is attempted, and both are refused before the model runs.

    ``discarded`` is the loophole the phase closed: a reconciler permitted to bin its own
    argument without being checked against it could satisfy "I am not a proposer" by
    throwing the proposal away. The command asks the independence question *before* calling
    the reconciler, so it cannot even be attempted -- and that is why the reconciler stub's
    call count is asserted: the refusal costs no model call, and the recorder's own
    enforcement of the same rule (over discarded proposals included) is pinned separately in
    ``tests/test_design_capture.py``.
    """
    reconciler = StubReconciler(
        [
            _payload(
                discarded=(
                    [{"principal": "solo", "revision": 1, "reason": "It was wrong."}]
                    if shape == "discarded"
                    else []
                )
            )
        ]
    )
    proposer, reconciler_used, sink = _install(monkeypatch, tmp_path, reconciler=reconciler)

    result = runner.invoke(
        app,
        [
            "design",
            "--repo",
            REPO,
            "--topic",
            TOPIC,
            "--proposer",
            "solo",
            "--reconciled-by",
            "solo",
        ],
    )

    assert result.exit_code == 1
    assert "solo" in result.output
    assert reconciler_used.calls == 0, "the reconciler was called for a run that must be refused"
    with _sink(tmp_path) as tickets:
        assert tickets.list_tickets() == []
    assert proposer.calls == [(REPO, "solo")], "the proposal was captured before the refusal"
    assert len(sink.entries) == 1, "only the proposal; no reconciliation may be written"


# --- the allowlist -------------------------------------------------------------------


def test_a_repository_outside_the_allowlist_is_refused_naming_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same exposure as ``backfill-reviews``, same refusal, and it must name the setting.

    A design phase writes records into one repository's corpus. If it could name a
    repository capture may not, ``GITHUB_WEBHOOK_ALLOWED_REPOSITORIES`` would be a
    statement about the webhook handler rather than about kojutsu -- and the operator
    reading the refusal is the one who can fix it.
    """
    _install(monkeypatch, tmp_path, repository="someone/else")

    result = runner.invoke(app, _capture_argv("proposer-a"))

    assert result.exit_code == 1
    assert "someone/else" not in result.output or "not in" in result.output
    assert "GITHUB_WEBHOOK_ALLOWED_REPOSITORIES" in result.output
    assert "refused" in result.output
    with _sink(tmp_path) as tickets:
        assert tickets.list_tickets() == []


def test_a_resume_is_refused_for_a_repository_outside_the_allowlist_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The resume writes tickets, so it carries the same exposure as the capture half.

    Checking only the capture form would leave the command's most consequential path -- the
    one that actually creates work -- outside the scope capture is held to.
    """
    _install(monkeypatch, tmp_path, repository="someone/else")

    result = runner.invoke(app, _resume_argv())

    assert result.exit_code == 1
    assert "GITHUB_WEBHOOK_ALLOWED_REPOSITORIES" in result.output


# --- idempotency, derived rather than transmitted ------------------------------------


def test_a_second_run_after_approval_creates_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A retried approval must not produce a second copy of every ticket.

    Pinned against the real store because the claim is a claim about a primary key: ticket
    ids are ``sha256(plan_digest, draft_id)``, so the second run computes the same ids and
    the store refuses them.
    """
    _gated(monkeypatch, tmp_path)

    first = runner.invoke(app, _resume_argv())
    assert first.exit_code == 0, first.output
    assert "tickets-created: 2" in first.output

    second = runner.invoke(app, _resume_argv())

    assert second.exit_code == 0, second.output
    assert "tickets-created: 0" in second.output
    assert "tickets-already-present: 2" in second.output
    with _sink(tmp_path) as tickets:
        assert len(tickets.list_tickets()) == 2


def test_a_different_approver_is_a_second_approval_and_not_a_re_creation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The case a request-keyed idempotency token gets wrong, and the reason one is not used.

    Two people approving the same plan are two events, so the ledger holds two rows. The
    ticket set is *not* re-created, because a ticket's id is derived from the plan digest
    and the draft id and excludes the approver -- which is the whole claim that idempotency
    here is derived rather than transmitted.
    """
    _gated(monkeypatch, tmp_path)
    first = runner.invoke(app, _resume_argv(approver=APPROVER))
    assert first.exit_code == 0, first.output

    second = runner.invoke(app, _resume_argv(approver=OTHER_APPROVER))

    assert second.exit_code == 0, second.output
    assert "tickets-created: 0" in second.output
    assert "approvals-on-file-for-this-plan: 2" in second.output
    assert OTHER_APPROVER in second.output
    with _sink(tmp_path) as tickets:
        assert len(tickets.list_tickets()) == 2
    with _ledger(tmp_path) as ledger:
        digest = design_plan_digest(_plan_artifact_from_registry(tmp_path))
        approvers = {entry.approved_by for entry in ledger.list_approvals(plan_sha256=digest)}
    assert approvers == {APPROVER, OTHER_APPROVER}


def _plan_artifact_from_registry(tmp_path: Path) -> DesignPlan:
    """The plan the *ledger* points at, located the way the resume locates it.

    Derived from the recorded reconciliation's id rather than from the pause's output, so
    the assertion is about the artefact the resume would really read.
    """
    with _registry(tmp_path) as registry:
        found = find_design_reconciliations(repo=REPO, topic=TOPIC, registry=registry)
    assert len(found) == 1
    return parse_design_plan(
        json.loads((tmp_path / "design-plans" / f"{found[0].entry_id}.json").read_text("utf-8"))
    )


# --- exit codes -----------------------------------------------------------------------


def test_the_gated_exit_code_is_distinguishable_from_success_and_from_error() -> None:
    """The distinction a script depends on, pinned as numbers.

    0 is "the design phase ran" and 1 is "this failed"; a gated run is neither, because
    nothing was built and nothing failed. Reusing 1 would train an operator to ignore the
    command's failures, and reusing 0 would have a script report success while the plan
    sits unapproved.
    """
    assert DESIGN_GATED_EXIT_CODE == 3
    assert DESIGN_GATED_EXIT_CODE not in (0, 1)


def test_the_help_documents_the_gated_exit_code() -> None:
    """Documented in ``--help``, because an undocumented exit code is a private convention."""
    result = runner.invoke(app, ["design", "--help"])

    assert result.exit_code == 0
    assert str(DESIGN_GATED_EXIT_CODE) in result.output
    assert "gated" in result.output


def test_a_refusal_and_a_gate_are_different_exit_codes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Asserted through the command rather than only as constants.

    A constant can be right while both paths still return 1, which is the mistake that
    would survive every other test here.
    """
    _gated(monkeypatch, tmp_path)
    resumed = runner.invoke(app, _resume_argv())
    refused = runner.invoke(app, _resume_argv(extra=["--revision", "2"]))

    assert resumed.exit_code == 0
    assert refused.exit_code == 1
    assert refused.exit_code != DESIGN_GATED_EXIT_CODE


# --- the two settings -----------------------------------------------------------------


def test_both_design_stores_have_a_working_default() -> None:
    """Unset means "here is a store", so the command works with no configuration at all.

    The acceptance criterion for this pair of settings. A default that were empty would
    make the common case an error, and an error is the right answer for *empty* -- which is
    the other half of the design, tested below.
    """
    from kojutsu.config import Settings

    settings = Settings()

    assert settings.design_plan_approval_ledger_path.strip(), "a default is a location"
    assert settings.design_ticket_sink_path.strip()
    assert settings.design_plan_approval_ledger_path != settings.design_ticket_sink_path, (
        "the two stores are two files because a component that could both authorise and "
        "spend an authorisation could authorise itself"
    )


def test_an_empty_store_path_is_refused_rather_than_defaulted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unset and empty are deliberately different, and this is where they differ.

    ``[instances.pilot]`` exists to show why: for an allowlist, an empty value reads the
    same as an unset one and the difference is recorded by writing it down. For an approval
    ledger it cannot read the same, because a missing ledger has to be an error -- never a
    permission already granted. An empty string here is therefore refused rather than
    quietly replaced with the default, since quietly replacing it is how "not configured"
    would come to mean "the approval you asked for is not enforced".
    """
    _install(monkeypatch, tmp_path)
    monkeypatch.setenv("DESIGN_PLAN_APPROVAL_LEDGER_PATH", "")

    result = runner.invoke(app, _capture_argv("proposer-a"))

    assert result.exit_code == 1
    assert "design_plan_approval_ledger_path" in result.output, "the refusal names the setting"
    assert "Leave it unset for the default store" in result.output
    with _sink(tmp_path) as tickets:
        assert tickets.list_tickets() == []
    with _registry(tmp_path) as registry:
        assert find_design_reconciliations(repo=REPO, topic=TOPIC, registry=registry) == ()


# --- flag combinations that would otherwise silently do the wrong thing ----------------


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--approve", "--approved-by", "dana", "--proposer", "x"], "--proposer"),
        (["--approve", "--approved-by", "dana", "--reconciled-by", "r"], "--reconciled-by"),
        (["--approve", "--approved-by", "dana", "--revision", "2"], "--revision"),
        (["--approve", "--approved-by", "dana", "--note", "why"], "--note"),
    ],
)
def test_a_resume_refuses_capture_only_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: list[str], message: str
) -> None:
    """Silently ignoring a flag is how ``--approve --proposer x`` becomes a rubber stamp.

    Each flag has to mean something in the invocation it is given, or an operator who typed
    it believes something happened that did not.
    """
    _install(monkeypatch, tmp_path)

    result = runner.invoke(app, ["design", "--repo", REPO, "--topic", TOPIC, *argv])

    assert result.exit_code == 1
    assert message in result.output


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--proposer", "p", "--approved-by", "dana"], "--approved-by"),
        (["--proposer", "p", "--note", "why"], "--note"),
        (["--proposer", "p"], "--reconciled-by"),
        (["--reconciled-by", "r"], "at least one proposer"),
        (["--proposer", "p", "--proposer", " P ", "--reconciled-by", "r"], "same principal"),
        (["--proposer", "p", "--reconciled-by", "r", "--revision", "0"], "at least 1"),
    ],
)
def test_a_capture_run_refuses_flags_that_do_not_add_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: list[str], message: str
) -> None:
    """The capture half refuses the same way, for the same reason: nothing is ignored."""
    _install(monkeypatch, tmp_path)

    result = runner.invoke(app, ["design", "--repo", REPO, "--topic", TOPIC, *argv])

    assert result.exit_code == 1
    assert message in result.output
    with _sink(tmp_path) as tickets:
        assert tickets.list_tickets() == []


def test_a_malformed_reconciler_payload_is_refused_before_anything_is_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Model output enters through one validating call, and a bad plan stops the run.

    The cycle is the interesting fault: the document is well-formed field by field, so a
    caller catching only a validation error would treat it as fine.
    """
    cyclic = _plan(_ticket("a", depends_on=("b",)), _ticket("b", depends_on=("a",)))
    _install(monkeypatch, tmp_path, reconciler=StubReconciler([_payload(plan=cyclic)]))

    result = runner.invoke(app, _capture_argv("proposer-a"))

    assert result.exit_code == 1
    assert "cycle" in result.output
    with _registry(tmp_path) as registry:
        assert find_design_reconciliations(repo=REPO, topic=TOPIC, registry=registry) == ()


def test_a_discard_naming_an_unknown_proposal_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A decision recorded against something nobody proposed is a false record.

    The reconciler is allowed to claim whatever it likes; the recorder owns the rule. So
    this asserts the refusal survives the command rather than a pre-check in it -- which is
    also why there is only one implementation of "is this proposal one the reconciliation
    covers".
    """
    _install(
        monkeypatch,
        tmp_path,
        reconciler=StubReconciler(
            [_payload(discarded=[{"principal": "ghost", "revision": 1, "reason": "Wrong."}])]
        ),
    )

    result = runner.invoke(app, _capture_argv("proposer-a"))

    assert result.exit_code == 1
    assert "ghost" in result.output
    with _registry(tmp_path) as registry:
        assert find_design_reconciliations(repo=REPO, topic=TOPIC, registry=registry) == ()
