"""The evaluation harness: ground truth read from the store, and the numbers it does not claim.

Every one of these tests is about something the harness must *refuse* to do. It
must not score a thread the classifier could not be shown, must not blame a model
for a bound the harness applied, must not report a precision of 1.0 for a run that
produced nothing, and must not write a measurement whose inputs have changed since
the baseline. Each of those is a way a number gets quoted without its caveat, and
each is cheap to write down here and impossible to notice later.

The harness itself is a script and needs a live model, so nothing here calls one.
What is tested is the part that has no network in it: reading the archived thread,
deriving the truth from stored documents, blinding the markers, spotting the prompt
bound before spending a call, scoring a classification, naming contradictions,
flipping comments, and refusing to diff two measurements of different inputs.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from kojutsu.core.question_registry import stable_evaluation_entry_id
from kojutsu.core.tanseki_mapping import (
    build_evaluation_content,
    build_evaluation_frontmatter,
    evaluation_document_id,
    to_evaluation_upsert_payload,
)
from kojutsu.core.thread_classifier import (
    MAX_THREAD_CHARS,
    SingleKind,
    ThreadClassification,
    ThreadCoverage,
    ThreadPair,
    ThreadSingle,
)
from kojutsu.integrations.github import answer_comment_body, kojutsu_comment_body
from kojutsu.integrations.github_models import GitHubComment, GitHubUser
from kojutsu.models import (
    CaptureSource,
    EvaluationEntry,
    EvaluationTarget,
    QuestionCategory,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval_thread_classifier.py"

#: The harness lives in ``scripts/``, which is not an importable package, so it is
#: loaded by path. Importing it as a module rather than copying its logic is the
#: point: a test that re-implements the scoring it is checking proves nothing about
#: the scoring the operator runs.
_spec = importlib.util.spec_from_file_location("eval_thread_classifier", SCRIPT)
assert _spec is not None and _spec.loader is not None
harness = importlib.util.module_from_spec(_spec)
sys.modules["eval_thread_classifier"] = harness
_spec.loader.exec_module(harness)

REPO = "acme/widgets"
MODEL = "opencode/model"
START = datetime(2026, 9, 27, 23, 30, tzinfo=UTC)


def _comment(cid: int, body: str, *, minute: int = 0) -> GitHubComment:
    return GitHubComment(
        id=cid,
        body=body,
        user=GitHubUser(login="acme"),
        created_at=START + timedelta(minutes=minute),
        author_association="OWNER",
    )


def _thread() -> list[GitHubComment]:
    """Three anchored pairs, one unanswered question, and one unmarked statement."""
    return [
        _comment(1, kojutsu_comment_body("q1", "Why separate rows from events?")),
        _comment(2, answer_comment_body("q1", "They carry different evidence."), minute=1),
        _comment(3, kojutsu_comment_body("q2", "What happens on a crash?"), minute=2),
        _comment(4, answer_comment_body("q2", "The gap is accepted."), minute=3),
        _comment(5, kojutsu_comment_body("q3", "Why is the lock advisory?"), minute=4),
        _comment(6, answer_comment_body("q3", "Because peers can ignore it."), minute=5),
        _comment(7, kojutsu_comment_body("q4", "Nobody answered this one."), minute=6),
        _comment(8, "A standalone remark nobody asked for.", minute=7),
    ]


def _truth() -> Any:
    return harness.GroundTruth(
        pairs=((1, 2), (3, 4), (5, 6)),
        documents={2: "doc-a", 4: "doc-b", 6: "doc-c"},
    )


def _classification(
    items: list[ThreadPair | ThreadSingle],
    *,
    anchored: tuple[ThreadPair, ...] = (),
    coverage: dict[str, Any] | None = None,
) -> ThreadClassification:
    """A classification built by hand, with the coverage its own items describe.

    ``placed_comment_ids`` is the set the coverage checks itself against, so a test
    classification reports the same number of accounted comments the module would.
    Building it any other way would test the harness against a coverage the module
    cannot produce.
    """
    placed = [comment_id for pair in anchored for comment_id in pair.comment_ids] + [
        comment_id
        for item in items
        for comment_id in (item.comment_ids if isinstance(item, ThreadPair) else (item.comment_id,))
    ]
    defaults: dict[str, Any] = {
        "comments": len(placed),
        "anchored_pairs": len(anchored),
        "anchored_questions": 0,
        "inferred_pairs": sum(1 for item in items if isinstance(item, ThreadPair)),
        "clarifications": sum(
            1
            for item in items
            if isinstance(item, ThreadSingle) and item.kind is SingleKind.CLARIFICATION
        ),
        "unrelated": sum(
            1
            for item in items
            if isinstance(item, ThreadSingle) and item.kind is SingleKind.UNRELATED
        ),
        "declined_pairs": 0,
        "conflicts": 0,
        "unattributable_lines": 0,
        "model_calls": 1,
        "placed_comment_ids": frozenset(placed),
    }
    return ThreadClassification(
        repo=REPO,
        pr_number=1,
        model=MODEL,
        input_comment_ids=frozenset(placed),
        items=tuple(items),
        anchored=anchored,
        coverage=ThreadCoverage(**{**defaults, **(coverage or {})}),
    )


def _pair(question: int, answer: int) -> ThreadPair:
    return ThreadPair(
        question_comment_id=question,
        answer_comment_id=answer,
        inferred_question="why?",
        confidence=0.9,
        anchored=False,
    )


def _single(comment_id: int, kind: SingleKind, reason: str | None = None) -> ThreadSingle:
    return ThreadSingle(comment_id=comment_id, kind=kind, declined_reason=reason)


# --- ground truth comes from the store, and refuses a denominator it did not expect


def test_the_truth_is_the_stored_pairings_and_nothing_else() -> None:
    """Marker -> comment, joined to the answer comment the store recorded.

    Derived through the same ``setdefault`` rule ``resolve_anchored_pairs`` uses, so
    a duplicated marker cannot put the truth and the classifier on different
    questions. Were this to prefer the later duplicate while the anchored path
    preferred the first, every miss would look like a model error.
    """
    markers = harness.question_comment_ids(_thread())
    assert markers == {"q1": 1, "q2": 3, "q3": 5, "q4": 7}
    truth = _truth()
    assert truth.pairs == ((1, 2), (3, 4), (5, 6))
    assert truth.comment_ids == frozenset({1, 2, 3, 4, 5, 6})
    assert truth.questions == frozenset({1, 3, 5})
    assert truth.answers == frozenset({2, 4, 6})


def test_the_truth_digest_moves_when_the_store_moves() -> None:
    """Otherwise a ``--compare`` reports a changed denominator as a model regression."""
    before = _truth().digest()
    after = harness.GroundTruth(pairs=((1, 2), (3, 4))).digest()
    assert before != after
    assert _truth().digest() == before, "the same pairs must digest the same"


class _StubDocument:
    def __init__(self, document_id: str, frontmatter: dict[str, Any]) -> None:
        self.id = document_id
        self.frontmatter = frontmatter


class _StubClient:
    """The smallest store that can answer the question the harness asks of one."""

    def __init__(self, documents: dict[str, dict[str, Any]]) -> None:
        self.documents = documents
        self.listed: list[str] = []

    def list_documents(self, *, limit: int, collection: str = "") -> list[str]:
        self.listed = sorted(self.documents)
        return self.listed

    def get_document(self, document_id: str, collection: str = "") -> _StubDocument | None:
        frontmatter = self.documents.get(document_id)
        return None if frontmatter is None else _StubDocument(document_id, frontmatter)


def _stored(count: int) -> dict[str, dict[str, Any]]:
    """Stored frontmatter for ``count`` of the thread's three anchored answers."""
    by_marker = {"q1": 2, "q2": 4, "q3": 6}
    return {
        f"{REPO}/pr-1/answer-{index}": {
            "question_id": marker,
            "github_comment_id": str(answer),
            "capture_source": "webhook",
        }
        for index, (marker, answer) in enumerate(list(by_marker.items())[:count])
    }


def test_the_expected_number_of_records_is_asserted_rather_than_adopted() -> None:
    """A store holding six records must not quietly become a recall over six.

    The harness is written against eleven. A store that disagrees has changed
    underneath the measurement, which is a fact to be told, not a denominator to
    absorb: a recall figure over a different truth set is a different measurement
    wearing the same name.
    """
    client = _StubClient(_stored(harness.EXPECTED_GROUND_TRUTH))
    monkey = pytest.MonkeyPatch()
    monkey.setattr(harness, "EXPECTED_GROUND_TRUTH", 3)
    try:
        truth = harness.read_ground_truth(
            client,
            repo=REPO,
            pr_number=1,
            comments=_thread(),  # type: ignore[arg-type]
        )
        assert truth.pairs == ((1, 2), (3, 4), (5, 6))
    finally:
        monkey.undo()

    monkey = pytest.MonkeyPatch()
    monkey.setattr(harness, "EXPECTED_GROUND_TRUTH", 11)
    try:
        with pytest.raises(harness.EvaluationError, match="11"):
            harness.read_ground_truth(
                client,
                repo=REPO,
                pr_number=1,
                comments=_thread(),  # type: ignore[arg-type]
            )
    finally:
        monkey.undo()


def test_a_record_naming_a_question_the_thread_does_not_carry_is_refused() -> None:
    """Otherwise the truth and the thread are two different threads."""
    documents = _stored(3)
    key = f"{REPO}/pr-1/answer-0"
    documents[key]["question_id"] = "not-in-this-thread"
    client = _StubClient(documents)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(harness, "EXPECTED_GROUND_TRUTH", 3)
    try:
        with pytest.raises(harness.EvaluationError, match="no comment in the thread carries"):
            harness.read_ground_truth(
                client,
                repo=REPO,
                pr_number=1,
                comments=_thread(),  # type: ignore[arg-type]
            )
    finally:
        monkey.undo()


def test_a_record_that_is_not_anchored_is_reported_not_rejected() -> None:
    """A mislabelled record is a finding about the store, not a reason to stop.

    Raised on, the harness would discard a whole measurement because one document
    says something odd about itself. Reported, the operator decides. A ``collect``
    or ``asserted`` record in the anchored namespace is exactly the case the
    ``structure`` axis exists to surface, and this is one more surface for it.
    """
    documents = _stored(3)
    documents[f"{REPO}/pr-1/answer-0"]["capture_source"] = "asserted"
    client = _StubClient(documents)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(harness, "EXPECTED_GROUND_TRUTH", 3)
    try:
        truth = harness.read_ground_truth(
            client,
            repo=REPO,
            pr_number=1,
            comments=_thread(),  # type: ignore[arg-type]
        )
    finally:
        monkey.undo()
    assert truth.unanchored_document_ids == (f"{REPO}/pr-1/answer-0",)
    assert truth.pairs == ((1, 2), (3, 4), (5, 6))


def test_only_the_answer_namespace_is_read_as_truth() -> None:
    """An evaluation record must never be scored as one of the truths it measures.

    Selection is by path, not by ``capture_source``, because the path is the only
    thing that says "this document is a captured Q&A" without the document's own
    account of itself being taken at face value. The harness writes its report
    under ``evaluation/`` for exactly this reason.
    """
    documents = _stored(3)
    documents[f"{REPO}/pr-1/evaluation/evaluation-v1-x"] = {
        "capture_source": "asserted",
        "evaluation_target": "model",
    }
    client = _StubClient(documents)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(harness, "EXPECTED_GROUND_TRUTH", 3)
    try:
        truth = harness.read_ground_truth(
            client,
            repo=REPO,
            pr_number=1,
            comments=_thread(),  # type: ignore[arg-type]
        )
    finally:
        monkey.undo()
    assert truth.pairs == ((1, 2), (3, 4), (5, 6))


# --- blinding removes the pairing markers and nothing else


def test_blinding_removes_the_pairing_markers_and_keeps_the_agent_marker() -> None:
    """The blinded thread is the one a repository Kojutsu never ran would present.

    The ``kojutsu:agent:`` marker stays: it is a claim about who wrote the
    comment and it is in the text a real reader of the thread would see. Removing it
    would make this thread *less* like an unmarked repository, and blinding the
    classifier to authorship would test a different thing than the one asked about.
    """
    body = "<!-- kojutsu:answer:q1 -->\n\n<!-- kojutsu:agent:opencode -->\n\nBecause."
    blinded = harness.strip_pairing_markers(body)
    assert "kojutsu:answer" not in blinded
    assert "kojutsu:agent:opencode" in blinded
    assert "Because." in blinded
    assert "\n\n\n" not in blinded


def test_blinding_leaves_no_marker_the_anchored_path_can_see() -> None:
    """A half-blinded thread produces a correct-looking zero for the wrong reason.

    If any question marker survived, the anchored path would own those comments,
    the model would never be sent them, and the measurement would score the
    classifier on a thread it was not shown. The harness refuses that before it
    spends a model call.
    """
    blinded = harness.blind_markers(_thread())
    assert not [comment.id for comment in blinded if "kojutsu:question" in comment.body]
    assert not [comment.id for comment in blinded if "kojutsu:answer" in comment.body]
    assert len(blinded) == len(_thread())


# --- the prompt bound is a fact about the input, and it is found before the call


def test_a_thread_over_the_prompt_bound_is_named_before_any_model_is_called() -> None:
    """The finding this harness exists to make, stated as a test.

    The builder clips a thread at ``MAX_THREAD_CHARS``, ``classify_thread`` then
    requires the model to account for *every* comment including the ones it was
    never shown, and the resulting refusal blames the model. Reading the prompt
    first turns an unexplained refusal into a nameable bound.
    """
    long_thread = [
        _comment(index, f"Comment {index}. " + "x" * 900, minute=index) for index in range(20)
    ]
    fit = harness.prompt_fit(long_thread)

    assert fit.prompt_chars > MAX_THREAD_CHARS
    assert not fit.complete
    assert fit.omitted_comment_ids, "a thread over the bound must lose comments"
    assert set(fit.omitted_comment_ids) == {c.id for c in long_thread} - set(fit.sent_comment_ids)
    described = fit.describe()
    assert str(MAX_THREAD_CHARS) in described
    assert "cannot classify this thread" in described


def test_a_thread_inside_the_bound_is_reported_as_fitting() -> None:
    fit = harness.prompt_fit(_thread())
    assert fit.complete
    assert "within the" in fit.describe()


def test_the_recall_ceiling_is_what_a_smaller_input_could_have_reached() -> None:
    """A recall of 9/11 with two answers never shown is not a model that missed two.

    The ceiling is the number of established pairings whose *both* comments were
    measured, and it is the only thing that distinguishes "recovered everything it
    could" from "recovered two thirds of the thread".
    """
    truth = _truth()
    whole = _thread()
    assert harness.recall_ceiling(truth, whole) == 3
    without = [comment for comment in whole if comment.id not in {5, 6}]
    assert harness.recall_ceiling(truth, without) == 2


def test_a_comment_the_prompt_cannot_see_is_still_readable_in_the_report() -> None:
    """A blind regex would report "every comment sent" and hide the whole finding.

    If the builder's per-comment format ever moves, nothing parses, every comment
    reads as omitted, and the harness says the thread is unsendable. Loud beats
    quiet here: a harness that cannot see which comments it sent has lost the one
    fact it needs, and must not paper over it with an empty match.
    """
    fit = harness.prompt_fit(_thread())
    assert set(fit.sent_comment_ids) == {comment.id for comment in _thread()}


# --- scoring: two recalls, and no precision out of an empty set


def test_recall_counts_the_anchored_path_and_the_model_separately() -> None:
    """Crediting a dict with eleven correct answers is how this number goes wrong.

    On a fully anchored thread the anchored path resolves every pairing in code and
    the model is never called, so "recall over every pairing held" is 1.0 while
    "recall of the model's own pairings" is 0.0. Both are true and only reporting
    one of them is a lie.
    """
    anchored = (
        ThreadPair(
            question_comment_id=1,
            answer_comment_id=2,
            inferred_question="q",
            confidence=1.0,
            anchored=True,
        ),
    )
    classification = _classification([], anchored=anchored)
    scored = harness.score(classification, _truth(), index=1)

    assert scored.recovered == ((1, 2),)
    assert scored.recovered_by_model == ()
    assert scored.recall == pytest.approx(1 / 3)
    assert scored.model_recall == 0.0
    assert scored.precision is None, "no pairs were produced, so precision is undefined"


def test_a_run_that_produced_no_pairs_has_no_precision_rather_than_a_perfect_one() -> None:
    """A classifier that refused to pair anything would otherwise score perfectly.

    Defaulting undefined precision to 1.0 is the single easiest way for this
    harness to produce the tidiest possible result from the worst possible
    behaviour.
    """
    classification = _classification(
        [_single(1, SingleKind.CLARIFICATION), _single(2, SingleKind.UNRELATED)]
    )
    scored = harness.score(classification, _truth(), index=1)

    assert scored.precision is None
    assert scored.recall == 0.0
    assert "precision_model_only" in scored.as_dict()
    assert scored.as_dict()["precision_model_only"] is None


def test_precision_is_measured_against_the_pairings_and_not_the_comments() -> None:
    """Three pairs, two of them real: 0.67, and the third is named for a person.

    A pair is scored as a pair. Counting it against the comments it covers would
    halve the figure for a reason that has nothing to do with whether it was right.
    """
    classification = _classification(
        [
            _pair(1, 2),
            _pair(3, 4),
            _pair(7, 8),
            _single(5, SingleKind.CLARIFICATION),
            _single(6, SingleKind.UNRELATED),
        ],
        coverage={"inferred_pairs": 3, "clarifications": 1, "unrelated": 1, "comments": 8},
    )
    scored = harness.score(classification, _truth(), index=1)

    assert scored.recovered_by_model == ((1, 2), (3, 4))
    assert scored.unmatched_inferred == ((7, 8),)
    assert scored.precision == pytest.approx(2 / 3)
    assert scored.recall == pytest.approx(2 / 3)


def test_a_refused_run_is_a_zero_with_the_reason_attached() -> None:
    """Not dropped, not retried, and not a division of nothing by nothing.

    Retrying until a run succeeds would select for the draws the model happens to
    get right, which is the one thing a measurement must not do to its own sample.
    And a refusal that scored zero recall against zero missed pairs would report
    nothing-over-nothing as a perfect score.
    """
    truth = _truth()
    scored = harness.refused_score(2, "the model did not account for comment id(s) 8", truth)

    assert scored.refused is not None
    assert scored.recall == 0.0
    assert scored.missed == truth.pairs
    assert scored.precision is None
    assert scored.as_dict()["refused"] == "the model did not account for comment id(s) 8"


def test_a_refusal_still_reports_the_pairs_the_anchored_path_resolved() -> None:
    """The anchored path is code, so what it resolved is known even when the run died."""
    scored = harness.refused_score(1, "refused", _truth(), anchored=((1, 2),))

    assert scored.recovered == ((1, 2),)
    assert scored.missed == ((3, 4), (5, 6))
    assert scored.recovered_by_model == ()


# --- contradictions are named, not counted


def test_a_pairing_a_person_established_elsewhere_is_named_against_its_record() -> None:
    """The question was asked, and the store says what answered it.

    Aggregated into a share this becomes a fraction; named, it is a comment id and
    a document a person can go and read.
    """
    classification = _classification(
        [
            _pair(1, 8),
            _single(2, SingleKind.CLARIFICATION),
            _single(3, SingleKind.CLARIFICATION),
            _single(4, SingleKind.CLARIFICATION),
            _single(5, SingleKind.CLARIFICATION),
            _single(6, SingleKind.CLARIFICATION),
            _single(7, SingleKind.CLARIFICATION),
        ],
        coverage={"inferred_pairs": 1, "clarifications": 6, "unrelated": 0, "comments": 8},
    )
    scored = harness.score(classification, _truth(), index=1)
    found = harness.contradictions([scored], _truth())

    assert len(found) == 1
    assert found[0]["kind"] == "question_of_known_pair_paired_to_a_new_answer"
    assert found[0]["question_comment_id"] == 1
    assert found[0]["answer_comment_id"] == 8
    assert "answered by 2" in found[0]["conflicts_with"]
    assert "doc-a" in found[0]["conflicts_with"]


def test_a_real_answer_presented_as_a_question_is_named_separately() -> None:
    """Using a stored answer to manufacture a second conversation is its own defect.

    It is listed as its own entry rather than concatenated onto the other reasons,
    because a reader triaging by comment id wants them separately and one entry
    with two reasons reads as one problem that is somehow twice as bad.
    """
    classification = _classification(
        [
            _pair(2, 7),
            _single(1, SingleKind.CLARIFICATION),
            _single(3, SingleKind.CLARIFICATION),
            _single(4, SingleKind.CLARIFICATION),
            _single(5, SingleKind.CLARIFICATION),
            _single(6, SingleKind.CLARIFICATION),
        ],
        coverage={"inferred_pairs": 1, "clarifications": 5, "unrelated": 0, "comments": 7},
    )
    scored = harness.score(classification, _truth(), index=1)
    found = harness.contradictions([scored], _truth())

    kinds = {item["kind"] for item in found}
    assert "answer_of_known_pair_used_as_a_question" in kinds
    assert all(item["conflicts_with"] for item in found)


def test_a_pairing_with_nothing_behind_it_is_still_listed() -> None:
    """Not a contradiction of a record -- but entirely the model's, so it is named.

    The truth set contains no record of a comment that correctly has no pairing, so
    this is the case a reader has to adjudicate and the one most easily lost inside
    a share.
    """
    classification = _classification(
        [
            _pair(7, 8),
            _single(1, SingleKind.CLARIFICATION),
            _single(2, SingleKind.CLARIFICATION),
            _single(3, SingleKind.CLARIFICATION),
            _single(4, SingleKind.CLARIFICATION),
            _single(5, SingleKind.CLARIFICATION),
            _single(6, SingleKind.CLARIFICATION),
        ],
        coverage={"inferred_pairs": 1, "clarifications": 6, "unrelated": 0, "comments": 8},
    )
    scored = harness.score(classification, _truth(), index=1)
    found = harness.contradictions([scored], _truth())

    assert [item["kind"] for item in found] == ["no_end_in_the_anchored_set"]
    assert found[0]["run"] == 1


def test_the_same_contradiction_in_two_runs_is_reported_once_per_run() -> None:
    """Run-indexed, because "one comment, three runs" and "three comments" differ."""
    classification = _classification(
        [
            _pair(1, 8),
            _single(2, SingleKind.CLARIFICATION),
            _single(3, SingleKind.CLARIFICATION),
            _single(4, SingleKind.CLARIFICATION),
            _single(5, SingleKind.CLARIFICATION),
            _single(6, SingleKind.CLARIFICATION),
            _single(7, SingleKind.CLARIFICATION),
        ],
        coverage={"inferred_pairs": 1, "clarifications": 6, "unrelated": 0, "comments": 8},
    )
    truth = _truth()
    runs = [harness.score(classification, truth, index=index) for index in (1, 2, 3)]

    found = harness.contradictions(runs, truth)
    assert sorted(item["run"] for item in found) == [1, 2, 3]


def test_a_pairing_the_anchored_path_resolved_without_a_record_is_named() -> None:
    """The ``as-captured`` run's version of the same check, and not a formality.

    The anchored path is code with no model in it, so a disagreement between what it
    resolves and what the store recorded is a bookkeeping defect rather than a
    measurement -- and it would be invisible inside a recall figure that came out at
    1.0 for the boring reason that a dict looked itself up.
    """
    thread = [
        _comment(1, kojutsu_comment_body("q", "What happens on a crash?")),
        _comment(2, answer_comment_body("q", "The gap is accepted."), minute=1),
        _comment(3, answer_comment_body("q", "The gap is accepted, at length."), minute=2),
    ]
    disagreements = harness.anchored_disagreements(_truth(), thread)

    assert len(disagreements) == 1
    assert disagreements[0]["kind"] == "anchored_pair_with_no_stored_record"
    assert disagreements[0]["question_comment_id"] == 1
    assert disagreements[0]["answer_comment_id"] in {2, 3}


# --- variance: the flips are the finding, and a refused run is not an ambiguity


def test_the_comments_that_flip_are_named_with_what_each_run_said() -> None:
    truth = _truth()
    first = _classification(
        [
            _pair(1, 2),
            _single(3, SingleKind.CLARIFICATION),
            _single(4, SingleKind.CLARIFICATION),
            _single(5, SingleKind.CLARIFICATION),
            _single(6, SingleKind.CLARIFICATION),
            _single(7, SingleKind.CLARIFICATION),
            _single(8, SingleKind.CLARIFICATION),
        ],
        coverage={"inferred_pairs": 1, "clarifications": 6, "unrelated": 0, "comments": 8},
    )
    second = _classification(
        [
            _single(1, SingleKind.CLARIFICATION),
            _single(2, SingleKind.CLARIFICATION),
            _pair(3, 4),
            _single(5, SingleKind.CLARIFICATION),
            _single(6, SingleKind.CLARIFICATION),
            _single(7, SingleKind.CLARIFICATION),
            _single(8, SingleKind.UNRELATED),
        ],
        coverage={"inferred_pairs": 1, "clarifications": 5, "unrelated": 1, "comments": 8},
    )
    runs = [harness.score(first, truth, index=1), harness.score(second, truth, index=2)]
    flipping = {item["comment_id"] for item in harness.flips(runs)}

    assert flipping == {1, 2, 3, 4, 8}
    assert 5 not in flipping, "a comment both runs placed alike is not a finding"


def test_a_refused_run_does_not_make_every_comment_look_ambiguous() -> None:
    """One call failing is not twenty-five ambiguous comments.

    A refused run placed nothing, so every comment disagrees with it. Counting that
    as instability reports a classifier that is incoherent when what happened is
    that one of three calls returned an incomplete list -- and it buries the handful
    of comments that genuinely move.
    """
    truth = _truth()
    agreeing = [
        _pair(1, 2),
        _single(3, SingleKind.CLARIFICATION),
        _single(4, SingleKind.CLARIFICATION),
        _single(5, SingleKind.CLARIFICATION),
        _single(6, SingleKind.CLARIFICATION),
        _single(7, SingleKind.CLARIFICATION),
        _single(8, SingleKind.CLARIFICATION),
    ]
    runs = [
        harness.score(_classification(agreeing), truth, index=1),
        harness.score(_classification(agreeing), truth, index=2),
        harness.refused_score(3, "the model did not account for comment id(s) 8", truth),
    ]

    assert harness.flips(runs) == []
    assert [share["stability"] for share in harness.pairwise_stability(runs)] == [1.0]
    assert harness.scored_runs(runs) == runs[:2]


def test_the_outcome_spread_is_reported_and_not_just_the_last_run() -> None:
    """A distribution without its range invites a reader to treat the sample as behaviour."""
    truth = _truth()
    runs = [
        harness.score(
            _classification(
                [
                    _pair(1, 2),
                    _single(3, SingleKind.CLARIFICATION),
                    _single(4, SingleKind.CLARIFICATION),
                    _single(5, SingleKind.CLARIFICATION),
                    _single(6, SingleKind.CLARIFICATION),
                    _single(7, SingleKind.CLARIFICATION),
                    _single(8, SingleKind.CLARIFICATION),
                ],
                coverage={"inferred_pairs": 1, "clarifications": 6, "unrelated": 0, "comments": 8},
            ),
            truth,
            index=1,
        ),
        harness.score(
            _classification(
                [
                    _pair(1, 2),
                    _pair(3, 4),
                    _single(5, SingleKind.CLARIFICATION),
                    _single(6, SingleKind.CLARIFICATION),
                    _single(7, SingleKind.CLARIFICATION),
                    _single(8, SingleKind.UNRELATED),
                ],
                coverage={"inferred_pairs": 2, "clarifications": 3, "unrelated": 1, "comments": 8},
            ),
            truth,
            index=2,
        ),
    ]
    distribution = harness.outcome_distribution(runs)

    assert distribution["inferred_pairs"]["per_run"] == [1, 2]
    assert distribution["inferred_pairs"]["min"] == 1
    assert distribution["inferred_pairs"]["max"] == 2
    assert distribution["unrelated"]["per_run"] == [0, 1]
    assert distribution["declined_pairs"]["per_run"] == [0, 0]


# --- baseline: comparable, or refused


def _payload(**overrides: Any) -> dict[str, Any]:
    base = {
        "repo": REPO,
        "pr": 1,
        "model": MODEL,
        "thread_mode": "marker-blind",
        "archive_digest": "abc123",
        "ground_truth_digest": "def456",
        "aggregates": {"recovered_mean": 5.0, "invented_pairs_mean": 1.0},
    }
    return {**base, **overrides}


def test_a_baseline_of_a_different_input_is_not_diffed_at_all() -> None:
    """A changed denominator produces a delta that looks exactly like a regression.

    Refused rather than reported as a large improvement or a large regression: the
    operator has to re-baseline deliberately, because the only thing this harness
    can say about two different inputs is that they are different.
    """
    for key, value in (
        ("archive_digest", "changed"),
        ("ground_truth_digest", "changed"),
        ("model", "opencode-go/other"),
        ("thread_mode", "as-captured"),
    ):
        with pytest.raises(harness.EvaluationError, match=key):
            harness.compare(_payload(), _payload(**{key: value}))


def test_a_regression_in_pair_recovery_is_named() -> None:
    """Visible without anyone having to remember the previous number."""
    rows = harness.compare(
        _payload(), _payload(aggregates={"recovered_mean": 2.0, "invented_pairs_mean": 1.0})
    )
    recovered = next(row for row in rows if row["metric"] == "recovered_mean")

    assert recovered["baseline"] == 5.0
    assert recovered["current"] == 2.0
    assert recovered["delta"] == -3.0
    assert recovered["regression"] is True


def test_a_fall_in_invented_pairings_is_not_called_a_regression() -> None:
    """Fewer fabricated pairings is the classifier getting better, not worse.

    The regression table carries a direction rather than a bare list of names,
    because "did it drop" is backwards for half the metrics a classifier is judged
    on. Getting this the wrong way round is how a harness ends up rewarding a
    classifier for inventing more -- and how a report ends up congratulating a
    regression.
    """
    rows = harness.compare(
        _payload(), _payload(aggregates={"recovered_mean": 5.0, "invented_pairs_mean": 0.0})
    )
    invented = next(row for row in rows if row["metric"] == "invented_pairs_mean")

    assert invented["delta"] == -1.0
    assert "regression" not in invented


def test_a_rise_in_invented_pairings_is_a_regression() -> None:
    """The opposite failure, and the one this whole programme exists to prevent."""
    rows = harness.compare(
        _payload(), _payload(aggregates={"recovered_mean": 5.0, "invented_pairs_mean": 4.0})
    )
    invented = next(row for row in rows if row["metric"] == "invented_pairs_mean")

    assert invented["delta"] == 3.0
    assert invented["regression"] is True


def test_a_baseline_of_an_unrecognised_shape_is_refused_rather_than_diffed(tmp_path: Path) -> None:
    """A harness that does not understand the record cannot read a number out of it."""
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"format": "something/else", "aggregates": {}}), encoding="utf-8")

    with pytest.raises(harness.EvaluationError, match=r"not a kojutsu\.thread-classifier-eval"):
        harness.read_baseline(path)


def test_a_missing_baseline_is_said_to_be_missing_rather_than_treated_as_empty(
    tmp_path: Path,
) -> None:
    """An absent baseline is not a baseline of zeroes, and the two are not comparable."""
    with pytest.raises(harness.EvaluationError, match="cannot read the baseline"):
        harness.read_baseline(tmp_path / "nothing-here.json")


# --- the archive reader refuses what it cannot score


def test_a_comment_with_no_numeric_id_refuses_the_thread(tmp_path: Path) -> None:
    """A comment that cannot be matched to a stored id is not scoreable.

    Skipping it would be the quieter failure and the worse one: the classifier would
    never be shown the comment, and a thread the model was not shown looks exactly
    like a thread the model handled.
    """
    archive = tmp_path / "pr-1.json"
    archive.write_text(
        json.dumps(
            {
                "comments": [
                    {
                        "id": "IC_noNumericId",
                        "author": {"login": "someone"},
                        "body": "hello",
                        "createdAt": "2026-09-27T23:30:10Z",
                        "url": "https://github.com/o/r/pull/1#discussion_r1",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(harness.EvaluationError, match="no issuecomment id"):
        harness.load_thread(archive)


def test_the_thread_is_read_in_the_order_the_classifier_refuses_against(tmp_path: Path) -> None:
    """Sorted by ``(created_at, id)``, because a reversed pair is refused on that order."""
    archive = tmp_path / "pr-1.json"
    archive.write_text(
        json.dumps(
            {
                "comments": [
                    {
                        "id": "node-2",
                        "author": {"login": "someone"},
                        "body": "second",
                        "createdAt": "2026-09-27T23:31:10Z",
                        "url": "https://github.com/o/r/pull/1#issuecomment-2",
                    },
                    {
                        "id": "node-1",
                        "author": {"login": "someone"},
                        "body": "first",
                        "createdAt": "2026-09-27T23:30:10Z",
                        "url": "https://github.com/o/r/pull/1#issuecomment-1",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    assert [comment.id for comment in harness.load_thread(archive)] == [1, 2]


# --- the stored report: a measurement of a model, never a review record


def _entry(**overrides: Any) -> EvaluationEntry:
    base = {
        "entry_id": stable_evaluation_entry_id(
            repo=REPO, pr_number=1, subject=MODEL, measurement="thread-classifier-accuracy"
        ),
        "repo": REPO,
        "pr_number": 1,
        "subject": MODEL,
        "measurement": "thread-classifier-accuracy",
        "result_text": "recall 0.45",
        "scope": "One thread. One model. One run configuration.",
    }
    return EvaluationEntry(**{**base, **overrides})


def test_a_measurement_cannot_claim_to_have_been_captured() -> None:
    """The same rule as a rationale, for a stronger reason.

    Nothing signed this. The numbers were produced in-process by the same class of
    component whose output they grade, so a harness able to label its own report
    ``webhook`` would be a harness able to assert that a measurement about a model
    is review evidence from a trusted thread.
    """
    with pytest.raises(ValueError, match="always asserted"):
        _entry(capture_source=CaptureSource.WEBHOOK)


def test_a_measurement_must_state_what_its_numbers_can_and_cannot_stand_for() -> None:
    """A figure whose scope is blank is a number nobody can weigh."""
    with pytest.raises(ValueError, match="what its numbers can and cannot stand for"):
        _entry(scope="   ")


def test_a_measurement_must_name_what_it_measured() -> None:
    """The target is the load-bearing fact, and it has exactly one value.

    A measurement whose subject is ambiguous is worse than none, because it gets
    compared against a number about something else. The enum is the guard -- a
    measurement of a repository or of a person is not constructible, so it cannot be
    filed beside eleven review records and read as a twelfth -- and the model-level
    check behind it is what says so in words rather than in a validation trace.
    """
    assert [member.value for member in EvaluationTarget] == ["model"]
    with pytest.raises(ValueError, match="model"):
        _entry(target="repository")
    assert _entry().target is EvaluationTarget.MODEL


def test_an_evaluation_never_lands_in_the_answer_namespace() -> None:
    """The path is the first thing a reader meets, and it is the whole defence here.

    Unlike a rationale or a clarification, nothing in the body of an answer says
    otherwise, so a precision figure filed beside the eleven real records looks
    like a twelfth fact about the pull request.
    """
    document_id = evaluation_document_id(_entry())

    assert "/evaluation/" in document_id
    assert f"{REPO}/pr-1/{_entry().entry_id}" != document_id
    assert document_id.startswith(f"{REPO}/pr-1/evaluation/")


def test_the_stored_document_says_what_it_measured_and_who() -> None:
    """``evaluation_target`` is what the read path keys on, and it must be present.

    Registered in ``check_docs.py``'s read-path key list and surfaced by
    ``search_knowledge``, so a reader served this document through MCP can tell a
    measurement from a finding about the code.
    """
    frontmatter = build_evaluation_frontmatter(_entry())

    assert frontmatter["evaluation_target"] == EvaluationTarget.MODEL.value
    assert frontmatter["evaluated_model"] == MODEL
    assert frontmatter["capture_source"] == CaptureSource.ASSERTED.value
    assert frontmatter["tags"] == ["evaluation", "model"]
    assert "category" not in frontmatter, (
        "no QuestionCategory can express 'a measurement of a model', and the nearest "
        "fit would be a false claim about the repository"
    )
    assert "structure" not in frontmatter, (
        "the report's own question and answer were written by the harness; the "
        "model-inferred material is named by evaluated_model"
    )


def test_the_stored_document_carries_its_limits_in_the_header_and_the_body() -> None:
    """A scope that only appears in the prose is a scope nobody meets in time.

    The header carries it so a listing shows it, and it has to be a single line: a
    frontmatter value containing a newline is rejected by the store, which is right,
    because every anchored record beside it is single-line.
    """
    entry = _entry(scope="First paragraph.\n\nSecond paragraph, with more to it.")
    frontmatter = build_evaluation_frontmatter(entry)
    content = build_evaluation_content(entry)

    assert "\n" not in frontmatter["evaluation_scope"]
    assert frontmatter["evaluation_scope"] == "First paragraph. Second paragraph, with more to it."
    assert "## What these numbers can and cannot stand for" in content
    assert "First paragraph.\n\nSecond paragraph" in content


def test_the_result_comes_before_the_attribution() -> None:
    """A model id in the first line of a report reads as a certification.

    Asserted on the body, not the whole document: the frontmatter is a header every
    reader skims, and naming the model there is the point -- it is what stops the
    report being read as a finding about the repository. What must not happen is a
    reader meeting the model's name before the numbers in the prose.
    """
    content = build_evaluation_content(_entry())
    body = content.split("---", 2)[-1]

    assert body.index("## Result") < body.index("## Subject")
    assert body.index("recall 0.45") < body.index(MODEL)


def test_a_re_measurement_updates_one_document_rather_than_accumulating_them() -> None:
    """The id is derived from the subject and the question, never from the numbers.

    A digest over the result would give every harness invocation a new id, the store
    would fill with near-identical rows, and the history of the measurement would
    stop being readable as a series. The history lives in the baseline file, which
    is version-controlled.
    """
    first = _entry(result_text="recall 0.45")
    second = _entry(result_text="recall 0.91")

    assert first.entry_id == second.entry_id
    assert to_evaluation_upsert_payload(first)["id"] == to_evaluation_upsert_payload(second)["id"]


def test_a_different_question_about_the_same_model_is_a_different_record() -> None:
    """Accuracy and refusal rate are different questions and must not be averaged."""
    accuracy = _entry()
    refusals = EvaluationEntry(
        entry_id=stable_evaluation_entry_id(
            repo=REPO, pr_number=1, subject=MODEL, measurement="thread-classifier-refusal-rate"
        ),
        repo=REPO,
        pr_number=1,
        subject=MODEL,
        measurement="thread-classifier-refusal-rate",
        result_text="one run in three refused",
        scope="One thread. One model.",
    )
    assert accuracy.entry_id != refusals.entry_id


def test_a_measurement_of_nothing_is_refused_at_the_identity() -> None:
    """An id derived from an empty subject would collide with every other empty one."""
    with pytest.raises(ValueError, match="must name what it measured"):
        stable_evaluation_entry_id(repo=REPO, pr_number=1, subject="  ", measurement="m")
    with pytest.raises(ValueError, match="must name the measurement"):
        stable_evaluation_entry_id(repo=REPO, pr_number=1, subject="m", measurement="")


def test_a_measurement_never_carries_a_category() -> None:
    """Guard on the guard: the enum is not widened to make an evaluation fit."""
    assert QuestionCategory.DESIGN_DECISION != EvaluationTarget.MODEL
    assert not hasattr(QuestionCategory, "MODEL_EVALUATION"), (
        "an evaluation is not a retrospective finding about a change, and adding a "
        "category for it would file a measurement as a decision"
    )


def test_the_stored_report_counts_the_comments_measured_and_not_the_runs() -> None:
    """The first number a reader meets, and it is about the input, not the sample.

    It read "3 of 27 comments were measured" -- the number of runs that returned a
    classification, which is also 3 -- on a run that measured 25. A reader who
    believes it concludes the harness threw away twenty-two comments, which is
    nearly the opposite of what happened.
    """
    truth = _truth()
    runs = [
        harness.score(
            _classification(
                [
                    _pair(1, 2),
                    _single(3, SingleKind.CLARIFICATION),
                    _single(4, SingleKind.CLARIFICATION),
                    _single(5, SingleKind.CLARIFICATION),
                    _single(6, SingleKind.CLARIFICATION),
                    _single(7, SingleKind.CLARIFICATION),
                    _single(8, SingleKind.CLARIFICATION),
                ],
                coverage={"inferred_pairs": 1, "clarifications": 6, "unrelated": 0, "comments": 8},
            ),
            truth,
            index=1,
        ),
        harness.refused_score(2, "the model did not account for comment id(s) 8", truth),
    ]
    aggregates = {
        "recall_all_pairs_mean": 0.09,
        "recall_model_only_mean": 0.09,
        "precision_model_only_mean": 1.0,
        "flipping_comments": 0,
    }
    rendered = harness.render_result(
        truth=truth,
        runs=runs,
        aggregates=aggregates,
        ceiling=3,
        omitted=(8, 9),
        comments_measured=7,
        thread_comments=9,
    )

    assert "7 of 9 comments were measured" in rendered
    assert "3 of 9" not in rendered, "the run count must not stand in for the comment count"
    assert "highest recall available here is 3 of 3" in rendered
    assert "runs refused outright: 1 of 2" in rendered
    assert "REFUSED" in rendered, "a refused run is shown as refused, not as zeroes"


def test_a_single_run_is_refused_before_anything_is_measured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One sample cannot measure variance, so the harness declines rather than report.

    A single run is not a small measurement, it is a different kind of claim: it
    reads as a deterministic verdict on a classifier that is stochastic. The measured
    pair recovery across six runs of this thread ranged from 4 to 7, and run 1 against
    run 2 agreed on 8 of 25 comments, so a one-run report here would be a number
    presented as a fact that the next run contradicts.

    The guard sits immediately after argument parsing, ahead of the store and the
    provider, so this can be exercised without either: a refused measurement must not
    be able to touch the store on its way out.
    """
    baseline = tmp_path / "baseline.json"
    monkeypatch.setattr(
        sys, "argv", ["eval_thread_classifier.py", "--runs", "1", "--baseline", str(baseline)]
    )

    assert harness.main() == 2
    assert not baseline.exists(), "a measurement below the floor wrote a baseline"
    assert "cannot measure variance" in capsys.readouterr().err


def test_three_runs_is_the_floor_and_it_is_reached_by_passing() -> None:
    """The floor is MIN_RUNS itself, not a number near it that drifts over time."""
    assert harness.MIN_RUNS == 3
