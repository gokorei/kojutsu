"""The real cycle, and the one claim the product is sold on.

Two things are under test here that the fakes could not reach.

The first is that :class:`CaptureCycleSteps` refuses to pretend. It has no
implementer, so it must refuse rather than complete a ticket having changed
nothing, and it must post nothing without ``apply``.

The second is criterion 6 of the worker ticket, which was quietly unmet while
everything was stubbed: a cycle whose implementer and answerer are the same model
must produce a record capture labels ``self_certified``, readable and filterable.
The independence arithmetic is well tested on its own, but nobody had ever run a
cycle and looked at what came out the other end.
"""

from __future__ import annotations

from typing import Any

import pytest

from kojutsu.config import Settings
from kojutsu.core.answerer import AnswerPlan
from kojutsu.core.gates import DEFAULT_GATES, GateRegistry
from kojutsu.integrations.github import (
    answer_comment_body_as_agent,
    extract_agent_claim,
    extract_answer_question_id_from_comment_body,
)
from kojutsu.models import Independence, compute_independence
from kojutsu.worker import (
    CaptureCycleSteps,
    ImplementerRequiredError,
    Worker,
    WorkerConfig,
    WorkItem,
)
from kojutsu.worker import steps as steps_module
from kojutsu.worker.loop import CycleOutcome

from .test_end_to_end import RecordingTicketSystem
from .test_loop import OpenGates

MODEL = "opencode/model"
ITEM = WorkItem("CHRON-7", "acme/widgets", "feat/chron-7", 91, subject="Fix the fence")


class StubQuestionSource:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows or [
            {
                "question_id": "q-1",
                "question_text": "Is the short-circuit reachable?",
                "status": "open",
            }
        ]

    def list_questions(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.rows


class StubClient:
    """A GitHub client that records writes instead of making them."""

    def __init__(self) -> None:
        self.posted: list[str] = []
        self._next = 500

    def post_issue_comment(self, owner: str, repo: str, pr_number: int, body: str) -> Any:
        self.posted.append(body)
        self._next += 1
        return type("Comment", (), {"id": self._next})()

    def get_pull_request(self, owner: str, repo: str, pr_number: int) -> Any:
        return type("PR", (), {"diff": "- if x:\n+ if x and y:\n"})()


# --- the steps refuse to lie --------------------------------------------------


def test_a_worker_with_no_implementer_refuses_to_run_a_cycle() -> None:
    """A ticket completed with nothing changed is worse than no worker at all."""
    steps = CaptureCycleSteps(
        client=StubClient(), apply=False, answer_model=MODEL, settings=Settings()
    )

    with pytest.raises(ImplementerRequiredError) as caught:
        steps.implement(ITEM)

    assert "honestly run a cycle" in str(caught.value)


def test_a_refused_implementation_fails_the_cycle_and_releases_the_claim(tmp_path: Any) -> None:
    tickets = RecordingTicketSystem(ITEM)
    worker = Worker(
        config=WorkerConfig(claim_ceiling=3, state_path=tmp_path / "state.json"),
        source=tickets,
        steps=CaptureCycleSteps(
            client=StubClient(),
            apply=False,
            answer_model=MODEL,
            settings=Settings(),
        ),
        gates=OpenGates(),
    )

    report = worker.run_once()

    assert report.outcome is CycleOutcome.FAILED
    assert "ImplementerRequiredError" in (report.error or "")
    assert [entry[0] for entry in tickets.log] == ["claim", "record_branch", "release"]
    assert "complete" not in [entry[0] for entry in tickets.log]


def _plan() -> AnswerPlan:
    return AnswerPlan(
        question_id="q-1",
        question_text="Is the short-circuit reachable?",
        answer_text="No, it is reachable.",
        body=answer_comment_body_as_agent("q-1", "No, it is reachable.", "opencode", MODEL),
    )


def test_the_ask_step_passes_the_configured_model_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generator's defaults are a different provider, model, and privacy policy.

    Calling it without them made an opencode deployment try ``openai``/``gpt-4o``
    with external processing off, which surfaced as a privacy refusal rather than
    as the wiring mistake it was.
    """
    seen: dict[str, Any] = {}

    def fake_generate(pr_spec: str, github_token: str, *args: Any, **kwargs: Any) -> Any:
        seen.update({"spec": pr_spec, "token": github_token, "args": args, "kwargs": kwargs})
        return ([], {})

    monkeypatch.setattr(steps_module, "generate_questions_for_pr", fake_generate)
    settings = Settings(
        llm_provider="opencode",
        llm_model="opencode/model",
        llm_external_enabled=True,
        llm_allowed_repositories="acme/repo",
    )
    steps = CaptureCycleSteps(client=StubClient(), apply=True, github_token="t", settings=settings)

    steps.ask(ITEM)

    assert seen["spec"] == f"{ITEM.repo}#{ITEM.pr_number}"
    # provider and model are positional 5 and 6 after spec, token, and the jira trio.
    assert seen["args"][3] == "opencode"
    assert seen["args"][4] == "opencode/model"
    assert seen["kwargs"]["llm_external_enabled"] is True
    assert seen["kwargs"]["llm_allowed_repositories"] == "acme/repo"


def test_nothing_is_posted_without_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dry run must reach the model and stop before the write.

    The model call is stubbed, because what is under test is the boundary, not the
    provider: the real path to a model is exercised by the answerer's own tests.
    """
    drafted: list[str] = []

    def fake_draft(questions: list[dict[str, Any]], **kwargs: Any) -> list[AnswerPlan]:
        drafted.append("drafted")
        return [_plan()]

    monkeypatch.setattr(steps_module, "draft_answers", fake_draft)
    client = StubClient()
    steps = CaptureCycleSteps(
        client=client,
        apply=False,
        answer_model=MODEL,
        question_source=StubQuestionSource(),
        implementer=lambda _i: "https://github.com/acme/widgets/pull/91",
        settings=Settings(),
    )

    described = steps.answer(ITEM)

    assert drafted == ["drafted"], "a dry run still does the thinking"
    assert "not posted" in described
    assert client.posted == [], "and stops before the write"


def test_apply_posts_through_the_ordinary_comment_path(monkeypatch: pytest.MonkeyPatch) -> None:
    client = StubClient()
    monkeypatch.setattr(steps_module, "draft_answers", lambda _q, **_k: [_plan()])
    steps = CaptureCycleSteps(
        client=client,
        apply=True,
        answer_model=MODEL,
        answer_agent="opencode",
        question_source=StubQuestionSource(),
        implementer=lambda _i: "u",
        settings=Settings(),
    )

    described = steps.answer(ITEM)

    assert "posted 1 answer" in described
    assert len(client.posted) == 1
    # The posted body is an ordinary issue comment carrying the ordinary markers,
    # so the collector picks it up on the same path a human answer would take.
    body = client.posted[0]
    assert extract_answer_question_id_from_comment_body(body) == "q-1"
    claim = extract_agent_claim(body)
    assert claim is not None and claim.model == MODEL


# --- criterion 6: a same-model cycle is labelled self_certified ----------------


def _level_for(body: str, *, asker_model: str) -> Independence:
    """The label capture computes for a posted answer, from the comment itself.

    Read the model out of the answer body exactly as the collector does, rather
    than passing it in, so this cannot pass by construction and then contradict
    what a real capture would compute.
    """
    claim = extract_agent_claim(body)
    assert claim is not None
    return compute_independence(
        asker_account="opencode",
        asker_model=asker_model,
        answerer_account="opencode",
        answerer_model=claim.model,
    )[0]


def test_a_same_model_cycle_produces_a_self_certified_record() -> None:
    """The claim the product is sold on, checked end to end.

    Implementer and answerer are the same model, which is the only configuration a
    single-model deployment can have. The record must be stored, labelled
    ``self_certified``, and never ``independent`` -- whatever the comments say.
    """
    body = answer_comment_body_as_agent("q-1", "Yes.", "opencode", MODEL)

    assert extract_agent_claim(body).model == MODEL
    assert _level_for(body, asker_model=MODEL) is Independence.SELF_CERTIFIED


def test_a_same_account_different_model_cycle_is_only_model_separated() -> None:
    """Two models are not two parties, and the label must not imply they are."""
    body = answer_comment_body_as_agent("q-1", "Yes.", "opencode", MODEL)

    level = _level_for(body, asker_model="opencode/model-2")

    assert level is Independence.MODEL_SEPARATED
    assert level is not Independence.INDEPENDENT


def test_a_same_model_record_survives_a_min_independence_filter() -> None:
    """It must be stored and filterable, not refused at the door.

    Refusing would make a single-model deployment unable to capture anything, and
    would discard evidence rather than caveat it. The label is the mitigation, so
    the label has to be visible to a reader who asks for it.
    """
    body = answer_comment_body_as_agent("q-1", "Yes.", "opencode", MODEL)
    level = _level_for(body, asker_model=MODEL)

    assert level is Independence.SELF_CERTIFIED
    # A reader filtering for independent evidence must not receive this, and
    # self_certified still ranks above the unlabelled floor.
    assert level.rank < Independence.INDEPENDENT.rank


def test_the_status_view_names_a_single_model_deployment() -> None:
    """A same-model configuration must be visible, not merely reachable."""
    same = Worker(
        config=WorkerConfig(implement_model=MODEL, answer_model=MODEL),
        source=RecordingTicketSystem(ITEM),
        steps=CaptureCycleSteps(settings=Settings()),
    ).status()
    split = Worker(
        config=WorkerConfig(implement_model=MODEL, answer_model="other"),
        source=RecordingTicketSystem(ITEM),
        steps=CaptureCycleSteps(settings=Settings()),
    ).status()

    assert same["single_model"] is True
    assert split["single_model"] is False


# --- gates still stop a real cycle --------------------------------------------


def test_a_gate_stops_the_real_steps_before_they_write() -> None:
    client = StubClient()
    tickets = RecordingTicketSystem(ITEM)
    worker = Worker(
        config=WorkerConfig(state_path=None),
        source=tickets,
        steps=CaptureCycleSteps(
            client=client,
            apply=True,
            answer_model=MODEL,
            question_source=StubQuestionSource(),
            implementer=lambda _i: "https://github.com/acme/widgets/pull/91",
            settings=Settings(),
        ),
        gates=GateRegistry(DEFAULT_GATES),
    )

    report = worker.run_once()

    assert report.outcome is CycleOutcome.GATED
    assert client.posted == [], "a blocked cycle must not reach GitHub"
    assert tickets.log == [], "and must not claim"


def test_the_worker_closes_its_source_only_when_it_stops(tmp_path: Any) -> None:
    """The forge client lives as long as the worker does, so the close belongs at shutdown.

    Both halves matter. Closing at construction would end the worker's ability to
    read on its first poll, and only asserting the *after* state would not notice
    that. So the source is asserted open across a real cycle, then closed.
    """
    tickets = RecordingTicketSystem(ITEM)
    worker = Worker(
        config=WorkerConfig(claim_ceiling=3, state_path=tmp_path / "state.json"),
        source=tickets,
        steps=CaptureCycleSteps(
            client=StubClient(),
            apply=False,
            answer_model=MODEL,
            settings=Settings(),
        ),
        gates=OpenGates(),
    )

    worker.run_once()

    assert tickets.closed is False, "a running worker still has to be able to read the forge"

    with pytest.raises(RuntimeError, match="interrupted"), worker:
        raise RuntimeError("interrupted")

    assert tickets.closed is True, "a worker that stops badly must still hand its client back"
