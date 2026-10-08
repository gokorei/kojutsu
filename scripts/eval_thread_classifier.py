#!/usr/bin/env python3
"""Measure the thread classifier against the pairings a person actually established.

This is a script and not a test on purpose. It needs a live model, and a network
call in the suite is a network call CI skips -- so it would go green forever
without ever running, and the number in the design note would be the number from
whichever run somebody happened to paste in. The suite proves the refusal paths
with canned responses; this is the only thing in the repository that puts a real
model in front of them.

**Ground truth is read from the store, never written down here.** The eleven
anchored records in the Tanseki collection were captured through the marker-anchored
path, so their pairings are not in dispute. A fixture would be a second copy of a
fact the store already holds, and the day the capture path changed the fixture
would go on asserting the old answer with nothing failing. Reading the truth back
out of the store is what makes it impossible for the evaluation to drift from what
the system produced.

**Two threads are classified, and the difference between them is the finding.**

- ``as-captured`` is the thread exactly as the forge has it, markers and all.
  Every comment in this one carries a marker, so the anchored path owns all
  twenty-seven, the model is sent nothing, and ``model_calls`` is 0. That is the
  production path working, and it is also why the model's accuracy cannot be read
  off it: the model is never asked.
- ``marker-blind`` is the same thread with the ``kojutsu:question:`` and
  ``kojutsu:answer:`` markers removed. This is the thread a repository where
  Kojutsu never ran would present, which is the only situation the classifier
  exists for, and the only configuration in which recall and precision against the
  anchored set mean anything.

Reporting one and not the other would be the tidier result and a dishonest one. The
anchored path recovering eleven of eleven is a fact about a ``dict``; it says
nothing about the model whose accuracy this ticket exists to establish.

**Read the prompt before spending a model call on it.** On this thread that is the
difference between a finding and a mystery: the thread is over
``MAX_THREAD_CHARS``, so the builder clips two comments out, and the coverage check
then refuses the run for failing to account for comments the model was never shown.
See :class:`PromptFit`. The harness reports the bound instead of measuring a thread
it cannot classify, and ``--measure-what-fits`` is how you ask for the smaller
measurement on purpose.

**A number from one thread is a data point and is labelled as one** in the stored
report, in the baseline, and in the body of this file. One thread, one repository,
one reviewer, one model, one configuration. Across the nine samples taken so far
the recall ranged 0.00-0.64 and the precision 0.00-1.00, which is a statement about
how hard the classifier finds the draw rather than about the classifier. See
``docs/design-review/thread-classifier.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kojutsu.allowlist import repository_allowed
from kojutsu.config import Settings
from kojutsu.core.question_registry import stable_evaluation_entry_id
from kojutsu.core.tanseki_mapping import to_evaluation_upsert_payload
from kojutsu.core.thread_classifier import (
    MAX_THREAD_CHARS,
    ThreadClassification,
    ThreadClassificationError,
    build_thread_prompt,
    classify_thread,
    outcomes_by_comment,
    resolve_anchored_pairs,
)
from kojutsu.integrations.github import (
    KOJUTSU_ANSWER_PREFIX,
    KOJUTSU_MARKER_PREFIX,
    KOJUTSU_MARKER_SUFFIX,
    extract_question_id_from_comment_body,
)
from kojutsu.integrations.github_models import GitHubComment, GitHubUser
from kojutsu.integrations.tanseki import TansekiClient
from kojutsu.models import CaptureSource, EvaluationEntry

ROOT = Path(__file__).resolve().parents[1]

#: Where the archived thread is read from. An archive rather than the API for one
#: reason only: reading it back needs no ``GITHUB_TOKEN``, so the measurement can be
#: re-taken by anyone with the store and the archive, and a measurement that needs a
#: credential is a measurement that quietly stops being re-taken.
#:
#: A path that does not exist rather than one that happened to exist on the machine
#: that first ran this. The baseline records which archive a measurement came from
#: and its digest, so the figure stays checkable; a default pointing into somebody
#: else's home directory is neither reproducible nor portable, and failing with
#: "no such file" says which input is missing.
DEFAULT_ARCHIVE = Path("thread-archive/pr-1.json")

#: The store the records were captured into, and the collection within it. Defaults
#: match ``Settings.tanseki_collection``'s own default so the script and the capture
#: path cannot disagree about where the truth lives.
DEFAULT_TANSEKI_URL = "http://localhost:8099"
DEFAULT_COLLECTION = "kojutsu-real"
DEFAULT_REPO = "acme/widgets"
DEFAULT_PR = 1

#: The number of anchored records the store is expected to hold. Asserted rather
#: than inferred: an evaluation that quietly measures a store holding six records
#: because six were deleted would report a recall against six and call it a
#: result. This is the count the ticket was written against, and a store that
#: disagrees has changed underneath the measurement, which is a fact to be told
#: rather than a denominator to be absorbed.
EXPECTED_GROUND_TRUTH = 11

#: The listing is bounded rather than paginated. The answer namespace holds eleven
#: documents and a store holding enough to overflow this is a store whose truth set
#: is not the one this measurement was written against -- which is reported above as
#: a mismatch rather than silently truncated, so a bound well above the expected
#: count fails loudly instead of quietly scoring a prefix of the truth.
MAX_GROUND_TRUTH_DOCUMENTS = 500

DEFAULT_RUNS = 3
#: A single sample is not a variance measurement. Three is the floor rather than a
#: preference: it is the smallest number of runs that can distinguish "the same
#: answer twice" from "the same answer every time", which is the only claim the
#: per-comment agreement below is making.
MIN_RUNS = 3

#: A placeholder rather than a working default, for the same reason the archive
#: path above is: a model id that names one operator's provider does not belong
#: in published source. `--model` overrides it, and an unset provider fails by
#: naming the providers that do have credentials.
DEFAULT_MODEL = "opencode/model"
DEFAULT_BASELINE = ROOT / "docs" / "design-review" / "thread-classifier-baseline.json"

#: The pairing markers, matched from the constants the anchored path extracts them
#: with rather than written out again. A hand-written copy of the spelling is a
#: second definition of what a marker is, and the day one of the two moves this
#: script would "blind" a thread whose markers are still there -- the one failure
#: this measurement cannot detect from its own output, because a thread that is
#: still anchored produces a clean run with a model that was never called.
_PAIRING_MARKER = re.compile(
    re.escape(KOJUTSU_MARKER_PREFIX)
    + r"[^>]*"
    + re.escape(KOJUTSU_MARKER_SUFFIX)
    + "|"
    + re.escape(KOJUTSU_ANSWER_PREFIX)
    + r"[^>]*"
    + re.escape(KOJUTSU_MARKER_SUFFIX)
)

#: GitHub's own numeric comment id is in the permalink fragment; the archive's
#: ``id`` field is the GraphQL node id, which is opaque and is not what the store
#: recorded in ``github_comment_id``.
_COMMENT_ID_IN_URL = re.compile(r"issuecomment-(\d+)")

#: The version of this harness's own output format. Bumped when the baseline's
#: shape changes, so ``--compare`` refuses to diff a baseline it does not
#: understand rather than reporting a field-level difference that means nothing.
BASELINE_FORMAT = "kojutsu.thread-classifier-eval/1"

Pair = tuple[int, int]


class EvaluationError(RuntimeError):
    """The measurement cannot be taken, and no number will be reported.

    Raised rather than degraded into a partial result. Half a recall figure
    computed over a store that held a different number of records than expected is
    not a smaller measurement, it is a different one wearing the same name, and it
    is exactly the shape of number that later gets quoted without its caveat.
    """


# --------------------------------------------------------------------------- inputs


def strip_pairing_markers(body: str) -> str:
    """Remove the two markers the anchored path keys on, and nothing else.

    The ``kojutsu:agent:`` marker stays. It is a claim about who wrote the
    comment, it is in the text a real reader of the thread would see, and a
    repository that had never run Kojutsu would not have it -- but removing it
    would make this thread *less* like a real unmarked repository rather than more,
    and blinding the classifier to authorship would test a different thing than the
    one this ticket asks about.

    Blank lines left behind by a removed marker are collapsed so the blinded body
    is the comment a person would have read, not a comment with a hole in it. The
    length difference is why the two threads are not the same input and why the
    report states which one produced a number.
    """
    without_markers = _PAIRING_MARKER.sub("", body)
    return re.sub(r"\n{3,}", "\n\n", without_markers).strip()


def load_thread(archive: Path) -> list[GitHubComment]:
    """Read the archived PR thread into comments the classifier accepts.

    Thread order is ``(created_at, id)`` rather than the archive's array order,
    because that is the order :func:`classify_thread` sorts into and the order in
    which an answer-not-before-its-question refusal is decided. Sorting here too
    means the id list this script prints and the one the classifier refused against
    are the same list.
    """
    try:
        payload = json.loads(archive.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read the archived thread at {archive}: {exc}") from exc
    raw_comments = payload.get("comments")
    if not isinstance(raw_comments, list) or not raw_comments:
        raise EvaluationError(f"{archive} holds no comments to classify")

    comments: list[GitHubComment] = []
    for raw in raw_comments:
        url = str(raw.get("url") or "")
        match = _COMMENT_ID_IN_URL.search(url)
        if match is None:
            # Not a warning and not a skip. A comment whose numeric id cannot be
            # read cannot be matched against a stored ``github_comment_id``, so it
            # is a comment that silently stops being scoreable -- and a comment the
            # model is never shown is not the same measurement as one it is.
            raise EvaluationError(f"comment {raw.get('id')!r} has no issuecomment id in {url!r}")
        author = raw.get("author") or {}
        comments.append(
            GitHubComment(
                id=int(match.group(1)),
                body=str(raw.get("body") or ""),
                user=GitHubUser(login=str(author.get("login") or "unknown")),
                created_at=datetime.fromisoformat(str(raw["createdAt"]).replace("Z", "+00:00")),
                author_association=raw.get("authorAssociation"),
            )
        )
    if len({comment.id for comment in comments}) != len(comments):
        raise EvaluationError(f"{archive} contains a duplicate comment id")
    return sorted(comments, key=lambda comment: (comment.created_at, comment.id))


def blind_markers(comments: list[GitHubComment]) -> list[GitHubComment]:
    """The same comments with the pairing markers taken out of the bodies."""
    return [
        comment.model_copy(update={"body": strip_pairing_markers(comment.body)})
        for comment in comments
    ]


#: What :func:`kojutsu.core.thread_classifier.build_thread_prompt` writes ahead of
#: each body. Mirrors the f-string in the builder rather than sharing it, because the
#: builder returns a string and not the set it decided to send -- so the only honest
#: way to know which comments reached the model is to read the prompt the model will
#: actually receive.
#:
#: The failure mode if the builder's format moves is loud rather than silent: nothing
#: parses, every comment reads as omitted, and the harness reports the thread as
#: unsendable. A regex that quietly matched nothing and reported "every comment sent"
#: would be the opposite, and much worse.
_SENT_COMMENT_IN_PROMPT = re.compile(r"\[comment (\d+) \|")


@dataclass(frozen=True)
class PromptFit:
    """Which comments the classifier's own prompt bound can actually carry.

    ``MAX_THREAD_CHARS`` is 12,000 and this thread is 13,608. The builder clips a
    thread of long comments rather than sending it whole, marks the clip *in the
    prompt* so the model knows, and then :func:`classify_thread` requires the model to
    account for every comment it was given -- including the ones it was never shown.
    The run is therefore refused, and the refusal is reported as "the model did not
    account for comment id(s) N, M", which blames the model for a bound this code
    applied to itself.

    So the harness reads the prompt before spending a model call on it. On this
    thread that turns an unexplained refusal into a nameable fact: two comments were
    never sent, and the classifier cannot classify this thread at all.
    """

    prompt_chars: int
    sent_comment_ids: tuple[int, ...]
    omitted_comment_ids: tuple[int, ...]

    @property
    def complete(self) -> bool:
        return not self.omitted_comment_ids

    def describe(self) -> str:
        if self.complete:
            return (
                f"the classifier's prompt for this thread is {self.prompt_chars} "
                f"characters, within the {MAX_THREAD_CHARS}-character bound, and all "
                f"{len(self.sent_comment_ids)} comments fit"
            )
        omitted = ", ".join(str(comment_id) for comment_id in self.omitted_comment_ids)
        return (
            f"the classifier's prompt for this thread would be {self.prompt_chars} "
            f"characters, over the {MAX_THREAD_CHARS}-character bound: "
            f"{len(self.omitted_comment_ids)} comment(s) are clipped out before the "
            f"model sees them ({omitted}). The module marks the clip in the prompt and "
            "then requires the model to account for every comment including the ones it "
            "was never shown, so the run is refused and the refusal reads as a model "
            "error. This is a bound the classifier applied to itself, and it means the "
            "classifier cannot classify this thread at all"
        )


def prompt_fit(comments: list[GitHubComment]) -> PromptFit:
    """Read the prompt the classifier would send and report what it carries."""
    prompt = build_thread_prompt(comments)
    sent = tuple(sorted({int(match) for match in _SENT_COMMENT_IN_PROMPT.findall(prompt)}))
    supplied = {comment.id for comment in comments}
    return PromptFit(
        prompt_chars=len(prompt),
        sent_comment_ids=sent,
        omitted_comment_ids=tuple(sorted(supplied - set(sent))),
    )


def recall_ceiling(truth: GroundTruth, comments: list[GitHubComment]) -> int:
    """How many established pairings the measured comments could possibly contain.

    Reported next to the recall figure whenever the harness measures less than the
    whole thread. A recall of 9/11 with two of the thread's comments never measured
    is not a model that recovered 9 of 11; it is a model that recovered 9 of the 9 it
    was shown, and the two are only distinguishable if the ceiling is on the page.
    """
    measured = {comment.id for comment in comments}
    return sum(1 for question, answer in truth.pairs if {question, answer} <= measured)


# ------------------------------------------------------------------- ground truth


def question_comment_ids(comments: list[GitHubComment]) -> dict[str, int]:
    """Map each question marker to the comment that carried it.

    First occurrence wins, which is :func:`resolve_anchored_pairs`' own rule rather
    than a choice made here. The two have to agree: were this script to prefer a
    later duplicate while the anchored path preferred the first, the ground truth
    and the pairing the classifier recovered in code would be scored against
    different questions, and the miss would look like a model error.
    """
    by_marker: dict[str, int] = {}
    for comment in comments:
        marker = extract_question_id_from_comment_body(comment.body)
        if marker:
            by_marker.setdefault(marker, comment.id)
    return by_marker


@dataclass(frozen=True)
class GroundTruth:
    """The pairings a person established, as the store recorded them."""

    pairs: tuple[Pair, ...]
    #: Comment id -> the document it came from, so a reader can go and look at the
    #: record rather than take the number's word for which comments were real.
    documents: dict[int, str] = field(default_factory=dict)
    #: Anchored records that stated no capture source, or claimed a source other
    #: than a signed delivery or an authenticated read. Empty in a healthy store
    #: and reported rather than raised, because a mislabelled record is a finding
    #: about the store and this is a measurement of a model.
    unanchored_document_ids: tuple[str, ...] = ()

    @property
    def questions(self) -> frozenset[int]:
        return frozenset(question for question, _ in self.pairs)

    @property
    def answers(self) -> frozenset[int]:
        return frozenset(answer for _, answer in self.pairs)

    @property
    def comment_ids(self) -> frozenset[int]:
        return self.questions | self.answers

    def digest(self) -> str:
        """A digest of the pairs themselves, so a changed truth set is visible.

        Without it a ``--compare`` against a baseline recorded when the store held
        a different set of records would report a recall difference and the reader
        would have no way to tell a model regression from a changed denominator.
        """
        preimage = ";".join(f"{question}->{answer}" for question, answer in sorted(self.pairs))
        return hashlib.sha256(preimage.encode()).hexdigest()[:16]


def read_ground_truth(
    client: TansekiClient, *, repo: str, pr_number: int, comments: list[GitHubComment]
) -> GroundTruth:
    """Read the anchored pairings out of the store.

    Selection is by the answer namespace -- ``<repo>/pr-<n>/answer-<id>`` -- rather
    than by a ``capture_source`` or a title, because the namespace is the only thing
    that says "this document is a captured Q&A" without a caller having to believe
    the document's own account of itself. A clarification, a rationale or an
    evaluation filed under the same pull request is excluded by the filter rather
    than by a heuristic, and an evaluation record written by this script can never
    be scored as one of the truths it is measuring against.
    """
    prefix = f"{repo}/pr-{pr_number}/answer-"
    stored = [
        document_id
        for document_id in client.list_documents(limit=MAX_GROUND_TRUTH_DOCUMENTS)
        if document_id.startswith(prefix)
    ]
    if len(stored) != EXPECTED_GROUND_TRUTH:
        raise EvaluationError(
            f"the store holds {len(stored)} anchored record(s) under {prefix!r}, and "
            f"this measurement was written against {EXPECTED_GROUND_TRUTH}. Either the "
            "capture path changed or the wrong collection is being read, and a recall "
            "figure over a different denominator is a different measurement, not a "
            "smaller one"
        )

    markers = question_comment_ids(comments)
    known_ids = {comment.id for comment in comments}
    pairs: list[Pair] = []
    documents: dict[int, str] = {}
    unanchored: list[str] = []
    for document_id in stored:
        document = client.get_document(document_id)
        if document is None or not document.frontmatter:
            raise EvaluationError(f"the store listed {document_id!r} but would not return it")
        frontmatter = document.frontmatter
        question_id = str(frontmatter.get("question_id") or "").strip()
        answer_raw = str(frontmatter.get("github_comment_id") or "").strip()
        if not question_id or not answer_raw.isdigit():
            raise EvaluationError(
                f"{document_id!r} names no question_id and github_comment_id, so the "
                "pairing it recorded cannot be reconstructed; the comment ids the "
                "store keeps are what the markers in the thread resolve to"
            )
        if str(frontmatter.get("capture_source") or "").strip() != CaptureSource.WEBHOOK.value:
            unanchored.append(document_id)
        question_comment_id = markers.get(question_id)
        answer_comment_id = int(answer_raw)
        if question_comment_id is None:
            raise EvaluationError(
                f"{document_id!r} names question {question_id}, which no comment in the "
                "thread carries; the truth and the thread are not the same thread"
            )
        if {question_comment_id, answer_comment_id} - known_ids:
            raise EvaluationError(
                f"{document_id!r} pairs comments {question_comment_id} and "
                f"{answer_comment_id}, and at least one is not in the thread being "
                "classified"
            )
        pairs.append((question_comment_id, answer_comment_id))
        documents[answer_comment_id] = document_id
    if len(set(pairs)) != len(pairs):
        raise EvaluationError("the store recorded the same pairing twice under two documents")
    return GroundTruth(
        pairs=tuple(sorted(pairs)),
        documents=documents,
        unanchored_document_ids=tuple(sorted(unanchored)),
    )


# -------------------------------------------------------------------- measurement


@dataclass(frozen=True)
class RunScore:
    """One sample of the classifier on one thread, scored against the truth.

    Two recalls, because there are two producers and conflating them is the whole
    way to a dishonest number here:

    - :attr:`recovered` counts every pairing the run holds, anchored or inferred.
      That is what the system would put in the store for this thread.
    - :attr:`recovered_by_model` counts only the pairings the *model* produced. On
      a thread whose comments all carry markers this is zero while
      :attr:`recovered` is eleven, because the anchored path resolved all of them in
      code and the model was never called. Reporting the first as "the classifier's
      recall" would credit a dict with eleven correct answers.
    """

    index: int
    coverage: dict[str, Any]
    #: Every pairing the run holds, anchored first.
    all_pairs: tuple[Pair, ...]
    #: Every pairing the run holds that a person established.
    recovered: tuple[Pair, ...]
    #: Every pairing the *model* produced that a person established.
    recovered_by_model: tuple[Pair, ...]
    #: Established pairings the run does not hold at all.
    missed: tuple[Pair, ...]
    #: Pairings the run holds that no stored record established, split by producer
    #: because an extra anchored pairing is a bookkeeping disagreement and an extra
    #: inferred one is a claim a model invented.
    unmatched_anchored: tuple[Pair, ...]
    unmatched_inferred: tuple[Pair, ...]
    clarifications: tuple[int, ...]
    unrelated: tuple[int, ...]
    declined: tuple[int, ...]
    #: Set when the classifier refused the run outright, carrying the reason. A
    #: refused run is a real outcome with real zeroes: the module returned nothing,
    #: so the caller stored nothing, and scoring that as a run which happened to
    #: produce no pairs would read as a model that declined everything rather than
    #: a classifier that returned no result at all.
    refused: str | None = None
    #: The classifier's own per-comment placements, kept whole so the flip analysis
    #: can be recomputed from the record rather than from a summary of it.
    placements: dict[int, str] = field(default_factory=dict)

    @property
    def inferred(self) -> tuple[Pair, ...]:
        return self.recovered_by_model + self.unmatched_inferred

    @property
    def recall(self) -> float:
        return len(self.recovered) / (len(self.recovered) + len(self.missed))

    @property
    def model_recall(self) -> float:
        return len(self.recovered_by_model) / (len(self.recovered) + len(self.missed))

    @property
    def precision(self) -> float | None:
        """Of the pairs the model produced, the share a person established.

        ``None`` rather than 1.0 when the run produced no pairs at all. Precision
        over an empty set is undefined, and defaulting it to 1.0 would let a
        classifier that refused to pair anything score a perfect precision -- the
        tidiest possible-looking result, produced by exactly the failure this whole
        group exists to catch.
        """
        inferred = self.inferred
        if not inferred:
            return None
        return len(self.recovered_by_model) / len(inferred)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run": self.index,
            "refused": self.refused,
            "coverage": self.coverage,
            "recall_all_pairs": round(self.recall, 4),
            "recall_model_only": round(self.model_recall, 4),
            "precision_model_only": None if self.precision is None else round(self.precision, 4),
            "pairs_held": len(self.all_pairs),
            "recovered": len(self.recovered),
            "recovered_by_model": len(self.recovered_by_model),
            "missed": len(self.missed),
            "unmatched_anchored": len(self.unmatched_anchored),
            "unmatched_inferred": len(self.unmatched_inferred),
            "clarifications": len(self.clarifications),
            "unrelated": len(self.unrelated),
            "declined": len(self.declined),
        }


def refused_score(
    index: int, reason: str, truth: GroundTruth, *, anchored: tuple[Pair, ...] = ()
) -> RunScore:
    """A run the classifier refused, scored as the zero it actually is.

    ``anchored`` is passed through because a refusal can happen *after* the anchored
    path has already resolved pairs in code. The module raises from its own coverage
    check, and the pairs it had resolved are a real part of what the run produced --
    they were determined without a model, so they are known even though the run as a
    whole returned nothing. Nothing inferred is carried, because there was no
    inference to carry.

    ``missed`` is the whole truth set, not empty. The run recovered nothing, and a
    refusal that scored zero recall against zero missed pairs would report a
    division of nothing by nothing as a perfect score.
    """
    known = set(truth.pairs)
    return RunScore(
        index=index,
        coverage={
            "comments": 0,
            "accounted": 0,
            "complete": False,
            "anchored_pairs": len(anchored),
            "anchored_questions": 0,
            "inferred_pairs": 0,
            "clarifications": 0,
            "unrelated": 0,
            "declined_pairs": 0,
            "conflicts": 0,
            "unattributable_lines": 0,
            "model_calls": 0,
        },
        all_pairs=anchored,
        recovered=tuple(pair for pair in anchored if pair in known),
        recovered_by_model=(),
        missed=tuple(pair for pair in truth.pairs if pair not in set(anchored)),
        unmatched_anchored=tuple(pair for pair in anchored if pair not in known),
        unmatched_inferred=(),
        clarifications=(),
        unrelated=(),
        declined=(),
        refused=reason,
    )


def score(classification: ThreadClassification, truth: GroundTruth, *, index: int) -> RunScore:
    """Score one classification against the anchored pairings.

    Precision here is *not* accuracy, and the difference is not a caveat. The truth
    set is the pairings somebody established, and it says nothing about the comments
    it does not mention -- the store holds no record of a comment that legitimately
    has no pairing. So a pairing outside the eleven might be a fabrication or might
    be a correct reading of a comment nobody marked. The score therefore treats
    every unmatched pair as something a person has to adjudicate and the script
    lists each one by comment id, rather than folding them into a share that reads
    as a verdict. Calling that share "precision" and stopping there is how an unfair
    number gets quoted.
    """
    inferred = tuple(
        (pair.question_comment_id, pair.answer_comment_id) for pair in classification.inferred_pairs
    )
    anchored = tuple(
        (pair.question_comment_id, pair.answer_comment_id) for pair in classification.anchored
    )
    known = set(truth.pairs)
    held = (*anchored, *inferred)
    placements = {
        comment_id: f"{kind}:{detail or '-'}"
        for comment_id, (kind, detail) in sorted(outcomes_by_comment(classification).items())
    }
    return RunScore(
        index=index,
        coverage=classification.coverage.as_dict(),
        all_pairs=held,
        recovered=tuple(pair for pair in held if pair in known),
        recovered_by_model=tuple(pair for pair in inferred if pair in known),
        missed=tuple(pair for pair in truth.pairs if pair not in set(held)),
        unmatched_anchored=tuple(pair for pair in anchored if pair not in known),
        unmatched_inferred=tuple(pair for pair in inferred if pair not in known),
        clarifications=tuple(single.comment_id for single in classification.clarifications),
        unrelated=tuple(single.comment_id for single in classification.unrelated),
        declined=tuple(
            single.comment_id
            for single in classification.singles
            if single.declined_reason is not None
        ),
        placements=placements,
    )


#: The ways one run can contradict a record somebody else wrote. Named rather than
#: counted, because a count of "2 contradictions" is a number nobody can act on and
#: two comment ids are.
CONTRADICTION_KINDS = (
    # The comment a person asked is paired to a different comment. The stored record
    # names one answer for that question; this is not it.
    "question_of_known_pair_paired_to_a_new_answer",
    # A comment the store holds as somebody's *answer* is presented as the question,
    # so a real answer is being used to manufacture a second conversation under it.
    "answer_of_known_pair_used_as_a_question",
    # The comment somebody answered is paired to a question that was not the one it
    # answered, so a real answer is attached to a question nobody asked.
    "answer_of_known_pair_paired_to_a_new_question",
    # Neither end is in the truth set. Not a contradiction of a record -- nothing
    # disagrees -- but the pairing is entirely the model's, and it is listed here so
    # it cannot be lost among the shares.
    "no_end_in_the_anchored_set",
    # The anchored path resolved a pairing the store holds no record of. Not a model
    # error at all, and reported separately from the rest for that reason: it is a
    # bookkeeping disagreement between two halves of the capture path.
    "anchored_pair_with_no_stored_record",
)


def contradictions(runs: list[RunScore], truth: GroundTruth) -> list[dict[str, Any]]:
    """Name every pairing a run produced that a stored record disagrees with.

    One entry per run per kind, and each names the comment ids and the record it
    disagrees with. Aggregating them into a recall or a precision figure is the
    specific thing this must not do: a fabricated pairing is not a fraction of a
    score, it is a record that would be filed next to eleven real ones and read as a
    twelfth, and the only useful output is the comment ids a human can go and check.

    A pairing can be wrong in more than one way at once -- a known answer used as a
    question *and* attached to a new question -- so each way is its own entry rather
    than one entry with the reasons concatenated. A reader triaging by comment id
    wants the reasons separately, and a single entry with two reasons reads as one
    problem that is somehow twice as bad.
    """
    found: list[dict[str, Any]] = []
    known = set(truth.pairs)
    for run in runs:
        for question, answer in run.inferred:
            if (question, answer) in known:
                continue
            reasons: list[tuple[str, str]] = []
            if question in truth.questions:
                established = next(pair[1] for pair in truth.pairs if pair[0] == question)
                reasons.append(
                    (
                        "question_of_known_pair_paired_to_a_new_answer",
                        f"the store holds {question} answered by {established} "
                        f"({truth.documents.get(established, 'unknown document')})",
                    )
                )
            if question in truth.answers:
                reasons.append(
                    (
                        "answer_of_known_pair_used_as_a_question",
                        f"the store holds {question} as somebody's answer, and this "
                        "pairing presents it as the thing that was asked",
                    )
                )
            if answer in truth.answers:
                reasons.append(
                    (
                        "answer_of_known_pair_paired_to_a_new_question",
                        f"the store holds {answer} as the answer to "
                        f"{next(pair[0] for pair in truth.pairs if pair[1] == answer)}",
                    )
                )
            if not reasons:
                reasons.append(
                    (
                        "no_end_in_the_anchored_set",
                        "neither comment appears in any stored anchored record",
                    )
                )
            for kind, detail in reasons:
                if kind not in CONTRADICTION_KINDS:
                    raise EvaluationError(
                        f"contradiction kind {kind!r} is not declared in "
                        "CONTRADICTION_KINDS, so a new shape of disagreement would reach "
                        "the report unlabelled"
                    )
                found.append(
                    {
                        "run": run.index,
                        "kind": kind,
                        "question_comment_id": question,
                        "answer_comment_id": answer,
                        "conflicts_with": detail,
                    }
                )
    return found


def anchored_disagreements(
    truth: GroundTruth, comments: list[GitHubComment]
) -> list[dict[str, Any]]:
    """Pairs the anchored path resolves in code that the store holds no record of.

    This is the ``as-captured`` run's version of the same check, and it is not a
    formality. The anchored path is code with no model in it, so any disagreement
    between what it resolves and what the store recorded is a bookkeeping defect
    rather than a measurement, and it would be invisible inside a recall figure that
    came out at 1.0 for the boring reason that a dict looked itself up.
    """
    resolved = resolve_anchored_pairs(comments)
    stored = set(truth.pairs)
    disagreements: list[dict[str, Any]] = []
    for pair in resolved.pairs:
        if (pair.question_comment_id, pair.answer_comment_id) in stored:
            continue
        sibling = next(
            (answer for question, answer in sorted(stored) if question == pair.question_comment_id),
            None,
        )
        disagreements.append(
            {
                "kind": "anchored_pair_with_no_stored_record",
                "question_comment_id": pair.question_comment_id,
                "answer_comment_id": pair.answer_comment_id,
                "conflicts_with": (
                    f"the store holds no record of this pairing; the same question is "
                    f"recorded as answered by {sibling}"
                    if sibling is not None
                    else "the store holds no record of this pairing and no other answer "
                    "to the same question"
                ),
            }
        )
    return disagreements


def scored_runs(runs: list[RunScore]) -> list[RunScore]:
    """The runs that produced a classification, dropping the refused ones.

    A refused run placed no comment, so every comment disagrees with it and it would
    make the flip list the whole thread. That is not a finding about ambiguity, it
    is one run failing, and reporting it as twenty-five ambiguous comments would
    bury the handful that genuinely move.
    """
    return [run for run in runs if run.refused is None]


def flips(runs: list[RunScore]) -> list[dict[str, Any]]:
    """Comments the scoring runs did not place identically, with what each said.

    The genuinely ambiguous ones, and the most useful thing this harness produces.
    A comment that is a pair in every run is settled; a comment that alternates
    between a pair and a clarification is not, and the alternation is the finding
    rather than noise to be averaged away. Each is listed with the body so a reader
    can decide whether the ambiguity is in the comment or in the model.
    """
    comparable = scored_runs(runs)
    comment_ids = sorted({comment_id for run in comparable for comment_id in run.placements})
    out: list[dict[str, Any]] = []
    for comment_id in comment_ids:
        per_run = {run.index: run.placements.get(comment_id, "absent") for run in comparable}
        if len(set(per_run.values())) > 1:
            out.append({"comment_id": comment_id, "placements": per_run})
    return out


def outcome_distribution(runs: list[RunScore]) -> dict[str, dict[str, Any]]:
    """Per-outcome counts, and the range each one moved across the runs.

    The range is the point. A single run's distribution is a sample; the spread
    across samples is how much of it was the thread and how much was the draw, and
    a distribution reported without it invites a reader to treat the sample as the
    classifier's behaviour.
    """
    keys = (
        "inferred_pairs",
        "clarifications",
        "unrelated",
        "declined_pairs",
        "anchored_pairs",
        "anchored_questions",
        "conflicts",
        "unattributable_lines",
    )
    distribution: dict[str, dict[str, Any]] = {}
    for key in keys:
        values = [int(run.coverage.get(key, 0)) for run in runs]
        distribution[key] = {
            "min": min(values),
            "max": max(values),
            "per_run": values,
        }
    return distribution


def pairwise_stability(runs: list[RunScore]) -> list[dict[str, Any]]:
    """Agreement for every pair of runs that both produced a classification.

    Refused runs are excluded for the reason :func:`flips` excludes them. A run that
    placed nothing agrees with nothing, so including it would report a stability of
    0.00 for a thread where the two runs that answered each other agreed on 17 of 25
    comments -- and a stability figure of zero is a claim that the classifier is
    incoherent, not that one call failed.
    """
    shares: list[dict[str, Any]] = []
    comparable = scored_runs(runs)
    for left_index, left in enumerate(comparable):
        for right in comparable[left_index + 1 :]:
            shared = sorted(set(left.placements) | set(right.placements))
            agreed = sum(
                1
                for comment_id in shared
                if left.placements.get(comment_id) == right.placements.get(comment_id)
            )
            shares.append(
                {
                    "runs": [left.index, right.index],
                    "comments": len(shared),
                    "agreements": agreed,
                    "stability": round(agreed / len(shared), 4) if shared else 1.0,
                }
            )
    return shares


# ----------------------------------------------------------------------- baseline


def digest_of(path: Path) -> str:
    """A digest of an input file, so a re-run can prove it read the same bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def build_baseline(payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap a measurement in the record that makes a later run comparable."""
    return {"format": BASELINE_FORMAT, "recorded_at": datetime.now(UTC).isoformat(), **payload}


def read_baseline(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvaluationError(f"cannot read the baseline at {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"the baseline at {path} is not readable JSON: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("format") != BASELINE_FORMAT:
        raise EvaluationError(
            f"the baseline at {path} is not a {BASELINE_FORMAT} record, so this run "
            "cannot be compared against it; a shape this harness does not recognise is "
            "not a number to diff"
        )
    return raw


#: Metrics that can regress, and the direction a *good* change moves them in.
#:
#: Deliberately two entries, and deliberately carrying a sign rather than a bare
#: name. A direction-less list can only express "this fell", which is backwards for
#: half the metrics a classifier is judged on: fewer pairings a person established
#: is a regression, and *more* pairings nobody established is also a regression --
#: they are opposite failures, and a single "did it drop" rule would call the second
#: one an improvement. Getting this backwards is how a harness ends up rewarding a
#: classifier for inventing more.
#:
#: Nothing about the volume of clarifications is here. A run that produces more of
#: them is a run that declined more, and treating that as a failure would push the
#: harness towards scoring tidiness, which is the thing this whole programme is
#: built against.
REGRESSION_DIRECTIONS = {
    # Recovering more of what a person established is the job, so more is better.
    "recovered_mean": +1,
    # Emitting pairings nobody established is the failure the whole axis exists to
    # prevent, so fewer is better -- which is why this one carries a minus.
    "invented_pairs_mean": -1,
}


def compare(baseline: dict[str, Any], current: dict[str, Any]) -> list[dict[str, Any]]:
    """Diff two measurements, and name the ones that moved against the design.

    Comparability is checked before any number is diffed. A baseline recorded
    against a different thread, a different truth set or a different model is a
    measurement of something else, and diffing it produces a delta that looks
    exactly like a regression and is not one.
    """
    for key in ("archive_digest", "ground_truth_digest", "model", "thread_mode"):
        if baseline.get(key) != current.get(key):
            raise EvaluationError(
                f"the baseline and this run disagree about {key!r} "
                f"({baseline.get(key)!r} vs {current.get(key)!r}), so a difference "
                "between them would be a change of inputs and not a change of "
                "behaviour; no comparison is reported"
            )
    left = baseline.get("aggregates") or {}
    right = current.get("aggregates") or {}
    rows: list[dict[str, Any]] = []
    for metric in sorted(set(left) | set(right)):
        before, after = left.get(metric), right.get(metric)
        if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
            rows.append({"metric": metric, "baseline": before, "current": after})
            continue
        delta = float(after) - float(before)
        row = {
            "metric": metric,
            "baseline": round(float(before), 4),
            "current": round(float(after), 4),
            "delta": round(delta, 4),
        }
        direction = REGRESSION_DIRECTIONS.get(metric)
        if direction is not None and delta * direction < 0:
            row["regression"] = True
        rows.append(row)
    return rows


# -------------------------------------------------------------------------- report


def scope_statement(
    truth: GroundTruth,
    mode: str,
    runs: int,
    *,
    ceiling: int | None = None,
    omitted: tuple[int, ...] = (),
) -> str:
    """The limits, in the harness's own words, for the stored record and the printout.

    Required rather than optional, and this is where they live. One thread, one
    repository, one reviewer, one model, one configuration, and a truth set that
    says nothing about the comments it does not mention. A precision figure from a
    single thread is a data point, and a report that reads like a property of the
    classifier is the over-claim every document in ``docs/design-review/`` was
    written to refuse.

    A partial measurement adds its own limit, and it is the first paragraph rather
    than the last: a recall of 9/11 on a thread whose other two answers were never
    shown to the model reads as a model that missed two things unless the ceiling is
    stated at the top.
    """
    partial = ""
    if omitted:
        dropped = ", ".join(str(comment_id) for comment_id in omitted)
        partial = (
            f"THIS IS NOT A MEASUREMENT OF THE WHOLE THREAD. The classifier's own prompt "
            f"bound clipped {len(omitted)} comment(s) out of the thread before the model "
            f"saw them ({dropped}), so the highest recall this measurement could have "
            f"reported is {ceiling} of {len(truth.pairs)} and every figure below is on a "
            "smaller input than the thread. The classifier as built cannot classify this "
            "thread at all; the numbers describe what it does with the part it accepts.\n\n"
        )
    return (
        f"{partial}"
        f"Measured on one thread: {len(truth.pairs)} pairings a person established were "
        f"read back out of the store as the truth, covering {len(truth.comment_ids)} of "
        f"the thread's comments. One repository, one reviewer, one model, one "
        f"configuration, {runs} sample(s) in the '{mode}' configuration. "
        "This is a data point, not a property of the classifier.\n\n"
        "What it cannot show: the truth set is the pairings somebody marked, so it "
        "contains no record of a comment that correctly has no pairing. A pairing "
        "outside the set is therefore a pairing that must be read by a person, and this "
        "report lists each one by comment id rather than counting it as an error.\n\n"
        "What it cannot show: one thread is one author writing to himself, with no "
        "disagreement between parties. A classifier measured on it has been asked to "
        "resolve ambiguity that is unusually low, and nothing here says what it does on "
        "a thread where two people disagree.\n\n"
        "What it cannot show: one model, a handful of times. Confidence numbers move "
        "between runs even where the structural decision does not, so a single sample's "
        "figures are not a distribution. The per-run spread in this report is the honest "
        "width of the measurement, and it is narrow because the sample is small.\n\n"
        "What it is not: a review of the pull request it ran against. Nothing in this "
        "record is evidence about that repository's code, and a good number here says "
        "nothing about whether the review was right."
    )


def render_result(
    *,
    truth: GroundTruth,
    runs: list[RunScore],
    aggregates: dict[str, Any],
    ceiling: int | None,
    omitted: tuple[int, ...],
    comments_measured: int,
    thread_comments: int,
) -> str:
    """The stored document's result section, as prose a reader can act on.

    A report whose result section is a JSON blob has pushed the reading back onto
    whoever opens it, and the reader is a person deciding how much to trust a
    classifier. The numbers are the same ones the baseline holds; this is the
    reading of them, and it leads with the shape because the shape is what the
    design predicted and the shape is what a reader has to judge.
    """
    refused = [run.index for run in runs if run.refused is not None]
    lines = [
        f"Ground truth: {len(truth.pairs)} pairings a person established, read back out "
        f"of the store, covering the {len(truth.comment_ids)} comments they involve.",
        "",
    ]
    if ceiling is not None:
        lines += [
            f"**This is not a measurement of the whole thread.** {comments_measured} of "
            f"{thread_comments} comments were measured: the classifier's prompt bound "
            f"clipped {len(omitted)} of them out before the model saw them, so the "
            f"highest recall available here is {ceiling} of {len(truth.pairs)}.",
            "",
        ]
    lines += [
        "| run | pairs held | by model | recovered | missed | invented | clarifications "
        "| unrelated | released |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        if run.refused is not None:
            lines.append(
                f"| {run.index} | REFUSED | REFUSED | 0 | {len(truth.pairs)} | — | — | — | — |"
            )
            continue
        lines.append(
            f"| {run.index} | {len(run.all_pairs)} | {len(run.inferred)} | "
            f"{len(run.recovered)} | {len(run.missed)} | {len(run.unmatched_inferred)} | "
            f"{len(run.clarifications)} | {len(run.unrelated)} | {len(run.declined)} |"
        )
    precision = aggregates["precision_model_only_mean"]
    lines += [
        "",
        f"- recall over every pairing held: {aggregates['recall_all_pairs_mean']:.2f} mean",
        f"- recall of the model's own pairings: {aggregates['recall_model_only_mean']:.2f} mean",
        "- precision of the model's pairings against the anchored set: "
        f"{precision if precision is None else f'{precision:.2f}'}",
        f"- runs refused outright: {len(refused)} of {len(runs)}"
        + (f" (run {', '.join(str(index) for index in refused)})" if refused else ""),
        f"- comments the scoring runs placed differently: {aggregates['flipping_comments']}",
    ]
    return "\n".join(lines)


def build_report_entry(
    *,
    repo: str,
    pr_number: int,
    model: str,
    measurement: str,
    result_text: str,
    scope: str,
    metadata: dict[str, Any],
) -> EvaluationEntry:
    """Assemble the stored record. Never anything but ``asserted``.

    The id is derived from the subject and the question rather than from the
    numbers, so a re-run updates one document instead of leaving a new one per
    harness invocation. The history that would preserve is in the baseline file,
    which is version-controlled, rather than in a store that grows a row every time
    somebody re-measures.
    """
    return EvaluationEntry(
        entry_id=stable_evaluation_entry_id(
            repo=repo, pr_number=pr_number, subject=model, measurement=measurement
        ),
        repo=repo,
        pr_number=pr_number,
        subject=model,
        measurement=measurement,
        result_text=result_text,
        scope=scope,
        metadata=metadata,
    )


# --------------------------------------------------------------------------- print


def _quote(body: str, *, limit: int = 400) -> str:
    text = " ".join(body.split())
    if len(text) <= limit:
        return text
    return text[:limit] + f"… [{len(body)} chars in the comment]"


def print_report(
    *,
    model: str,
    mode: str,
    comments: list[GitHubComment],
    truth: GroundTruth,
    runs: list[RunScore],
    contradictions_found: list[dict[str, Any]],
    extra_disagreements: list[dict[str, Any]],
    ceiling: int | None = None,
) -> None:
    """Print the measurement in the order a reader needs it, including the parts that hurt.

    Every section here is one somebody could otherwise leave out. The distribution
    is what shows whether the pairs were forced, the contradictions are named rather
    than averaged, and the unrelated comments are printed *in full* because a
    comment the classifier dropped is a comment a reader cannot go and look at, and
    "nothing valuable was dropped" is a judgement that has to be made in public.
    """
    print(f"\n=== thread classifier evaluation · {mode} · {model} ===")
    print(
        f"thread: {len(comments)} comment(s) shown to the classifier; "
        f"ground truth: {len(truth.pairs)} anchored pair(s) "
        f"covering {len(truth.comment_ids)} comment(s)"
    )
    if ceiling is not None:
        print(
            f"RECALL CEILING {ceiling}/{len(truth.pairs)}: {len(truth.pairs) - ceiling} of "
            "the established pairings could not have been recovered from this input, "
            "because one of their comments was never sent to the model"
        )
    if truth.unanchored_document_ids:
        print(
            "WARNING: these stored records do not carry capture_source=webhook, so they "
            "are being scored as truth on their own say-so: "
            + ", ".join(truth.unanchored_document_ids)
        )
    bodies = {comment.id: comment.body for comment in comments}

    for run in runs:
        coverage = run.coverage
        if run.refused is not None:
            print(
                f"\nrun {run.index}: REFUSED — the classifier returned no classification "
                f"at all, so this run contributes nothing to the store and nothing to "
                f"recall. Reason: {run.refused}"
            )
            continue
        precision = (
            "n/a (the model produced no pairs)" if run.precision is None else f"{run.precision:.2f}"
        )
        print(
            f"\nrun {run.index}: pairs held={len(run.all_pairs)} "
            f"(anchored {coverage.get('anchored_pairs')}, inferred "
            f"{len(run.inferred)}) clarifications={len(run.clarifications)} "
            f"unrelated={len(run.unrelated)} declined={coverage.get('declined_pairs')} "
            f"conflicts={coverage.get('conflicts')} model_calls={coverage.get('model_calls')}"
        )
        print(
            f"        recall over every pairing held {run.recall:.2f} "
            f"({len(run.recovered)}/{len(truth.pairs)}); recall of the model's own "
            f"pairings {run.model_recall:.2f} ({len(run.recovered_by_model)}/"
            f"{len(truth.pairs)}); precision against the anchored set {precision}"
        )
        if run.missed:
            print(f"        not held at all: {_pairs(run.missed)}")
        if run.unmatched_inferred:
            print(f"        model produced, nobody established: {_pairs(run.unmatched_inferred)}")
        if run.unmatched_anchored:
            print(
                f"        anchored path produced, store holds no record: "
                f"{_pairs(run.unmatched_anchored)}"
            )
        if run.declined:
            print(
                f"        released rather than paired, with a reason attached: {list(run.declined)}"
            )

    distribution = outcome_distribution(runs)
    print("\noutcome spread across runs (min–max of each counter):")  # noqa: RUF001
    for key, values in distribution.items():
        if values["min"] == values["max"]:
            print(f"  {key}: {values['per_run']}")
        else:
            print(f"  {key}: {values['per_run']} (moved {values['min']}–{values['max']})")  # noqa: RUF001

    stability = pairwise_stability(runs)
    refused = [run.index for run in runs if run.refused is not None]
    if refused:
        print(
            f"stability below compares only the runs that returned a classification; "
            f"run(s) {refused} were refused and agree with nothing by definition"
        )
    if not stability:
        print("stability: no two runs produced a classification to compare")
    for share in stability:
        print(
            f"runs {share['runs'][0]}/{share['runs'][1]}: {share['agreements']} of "
            f"{share['comments']} comments placed identically "
            f"(stability {share['stability']:.2f})"
        )

    flipping = flips(runs)
    print(f"\ncomments the scoring runs did not place identically: {len(flipping)}")
    for flip in flipping:
        placements = ", ".join(
            f"run {index}: {value}" for index, value in sorted(flip["placements"].items())
        )
        print(f"  comment {flip['comment_id']}: {placements}")
        print(f"    text: {_quote(bodies[flip['comment_id']])}")

    everything_flagged = contradictions_found + extra_disagreements
    print(f"\nCONTRADICTIONS with a stored anchored record: {len(everything_flagged)}")
    if not everything_flagged:
        print("  none: every pairing produced is one a person established")
    for item in everything_flagged:
        print(
            f"  [run {item.get('run', '-')}] {item['kind']}: "
            f"{item['question_comment_id']} -> {item['answer_comment_id']} "
            f"conflicts with {item['conflicts_with']}"
        )

    dropped = sorted({comment_id for run in runs for comment_id in run.unrelated})
    print(
        f"\nUNRELATED, no record produced ({len(dropped)} comment(s) across "
        f"{len(runs)} run(s)) — read these before believing the drop was right:"
    )
    if not dropped:
        print("  none. A thread where nothing is ever dropped is a thread where the")
        print("  third option is not being exercised, which is the finding, not a pass.")
    for comment_id in dropped:
        runs_dropping = [run.index for run in runs if comment_id in run.unrelated]
        # The store already knows which of these are worth something, so the
        # harness says so rather than leaving the reader to go and look. A drop is
        # invisible in precision by construction -- it is a loss, not a fabrication
        # -- and "which of these did we already have" is the one question that turns
        # the list from a wall of text into a triage.
        print(
            f"  comment {comment_id} (dropped by run(s) {runs_dropping}): {_standing(comment_id, truth)}"
        )
        print(f"    {_quote(bodies[comment_id], limit=700)}")


def _standing(comment_id: int, truth: GroundTruth) -> str:
    """What the store already holds about a comment, in one line.

    The three answers are not equally weighted and the wording says which is which.
    A comment the store holds as half of a real record being called unrelated is not
    a judgement call; it is a record the classification would have contradicted by
    silence.
    """
    if comment_id in truth.questions:
        return "the store holds this as the question anchoring a real record"
    if comment_id in truth.answers:
        return "the store holds this as the answer in a real record"
    return "the store holds nothing about this comment"


def _pairs(pairs: list[Pair] | tuple[Pair, ...]) -> str:
    return ", ".join(f"{question}->{answer}" for question, answer in pairs)


# -------------------------------------------------------------------------- main


def classify_repeatedly(
    comments: list[GitHubComment],
    truth: GroundTruth,
    *,
    repo: str,
    pr_number: int,
    model: str,
    runs: int,
) -> list[RunScore]:
    """Classify the same thread ``runs`` times and score each sample.

    Sequentially and in one process on purpose. Runs are what measure the draw, and
    running them concurrently would have them contend for one sandbox home and one
    provider, which turns a measurement of the model into a measurement of the queue.

    A run the classifier *refuses* is scored as a refusal, not dropped and not
    retried. The module refuses partial classifications by design, and that refusal
    is the run's output: a caller gets no classification, so the record of what
    happened is "the classifier produced nothing", and scoring that as a run which
    produced no pairs and declined nothing is exactly right. Retrying until a run
    succeeds would quietly select for the draws the model happens to get right, which
    is the one thing a measurement must not do to its own sample.
    """
    scores: list[RunScore] = []
    for index in range(1, runs + 1):
        print(f"  run {index}/{runs}…", end=" ", flush=True)
        try:
            classification = classify_thread(
                comments, repo=repo, pr_number=pr_number, model=model, timeout_seconds=240.0
            )
        except ThreadClassificationError as exc:
            print(f"REFUSED: {exc}")
            # The anchored path is deterministic code, so what it resolved is known
            # even though the run as a whole returned nothing. Reporting a refused
            # run as holding nothing would discard real work the run did do.
            anchored = tuple(
                (pair.question_comment_id, pair.answer_comment_id)
                for pair in resolve_anchored_pairs(comments).pairs
            )
            scores.append(refused_score(index, str(exc), truth, anchored=anchored))
            continue
        print(
            f"{len(classification.inferred_pairs)} pair(s), "
            f"{len(classification.clarifications)} clarification(s), "
            f"{len(classification.unrelated)} unrelated"
        )
        scores.append(score(classification, truth, index=index))
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure the thread classifier against the anchored pairings in the store."
    )
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--pr", type=int, default=DEFAULT_PR)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--tanseki-url", default=DEFAULT_TANSEKI_URL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--mode",
        choices=("as-captured", "marker-blind"),
        default="marker-blind",
        help=(
            "as-captured classifies the thread with its markers intact, which on this "
            "thread means the anchored path owns every comment and the model is never "
            "called; marker-blind removes the pairing markers so the model's own "
            "accuracy can be read. Default marker-blind because that is the "
            "configuration the classifier exists for."
        ),
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="diff this run against the stored baseline instead of replacing it",
    )
    parser.add_argument(
        "--measure-what-fits",
        action="store_true",
        help=(
            "when the classifier's own prompt bound clips comments out of the thread, "
            "measure only the comments it would send and state the recall ceiling that "
            "implies. This is a different and smaller measurement than the one asked "
            "for, and the report says so on every line it is on"
        ),
    )
    parser.add_argument("--no-store", action="store_true", help="do not write the Tanseki report")
    parser.add_argument("--json", action="store_true", help="also print the baseline JSON")
    args = parser.parse_args()

    if args.runs < MIN_RUNS:
        print(
            f"--runs {args.runs} cannot measure variance; {MIN_RUNS} is the floor and "
            "one sample is a number, not a distribution",
            file=sys.stderr,
        )
        return 2

    settings = Settings()
    if not repository_allowed(args.repo, settings):
        print(
            f"{args.repo!r} is not in GITHUB_WEBHOOK_ALLOWED_REPOSITORIES, so the "
            "repository allowlist refuses to classify a thread from it. The allowlist "
            "is kojutsu.allowlist and this script does not have a second copy "
            "of it: a measurement of a repository the system would not capture is not "
            "a measurement of the system.",
            file=sys.stderr,
        )
        return 2

    try:
        captured = load_thread(args.archive)
        comments = captured if args.mode == "as-captured" else blind_markers(captured)
        still_anchored = [
            comment.id
            for comment in comments
            if extract_question_id_from_comment_body(comment.body)
        ]
        if args.mode == "marker-blind" and still_anchored:
            raise EvaluationError(
                f"comment(s) {still_anchored} still carry a question marker after "
                "blinding, so the model would see an anchored thread the classifier "
                "then declines to classify and this measurement would report a "
                "correct-looking zero for the wrong reason"
            )

        with TansekiClient(args.tanseki_url, collection=args.collection, timeout=20.0) as client:
            if not client.health():
                raise EvaluationError(
                    f"the Tanseki store at {args.tanseki_url} did not answer /health, so the "
                    "ground truth cannot be read. Start the store first: an evaluation "
                    "that cannot see the records it is measuring against reports "
                    "nothing rather than a fixture"
                )
            truth = read_ground_truth(client, repo=args.repo, pr_number=args.pr, comments=captured)

        fit = prompt_fit(comments)
        omitted = fit.omitted_comment_ids
        if omitted and not args.measure_what_fits:
            print(
                f"\nREFUSING TO MEASURE: {fit.describe()}\n\n"
                "Nothing is reported and no baseline is written, because the only "
                "numbers available here would be a harness that had quietly thrown "
                "comments away before measuring -- which is the failure this whole "
                "programme is about, committed by the thing meant to detect it.\n\n"
                f"Pass --measure-what-fits to measure the {len(fit.sent_comment_ids)} "
                "comment(s) the classifier can actually accept. That is a smaller "
                "measurement with a lower ceiling, and every number in it will carry "
                "the ceiling.",
                file=sys.stderr,
            )
            return 1
        ceiling: int | None = None
        if omitted:
            keep = set(fit.sent_comment_ids)
            comments = [comment for comment in comments if comment.id in keep]
            ceiling = recall_ceiling(truth, comments)
            print(f"\n{fit.describe()}")
            print(
                f"measuring the {len(comments)} comment(s) that fit; the recall ceiling "
                f"is therefore {ceiling} of the {len(truth.pairs)} established pairings, "
                "because the omitted comments are answers that were never shown to the "
                "model"
            )
        else:
            print(f"\n{fit.describe()}")

        runs = classify_repeatedly(
            comments,
            truth,
            repo=args.repo,
            pr_number=args.pr,
            model=args.model,
            runs=args.runs,
        )
    except EvaluationError as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # the harness reports, it does not crash
        print(
            f"evaluation failed: {type(exc).__name__}: {exc}. No numbers are reported: "
            "a measurement that could not be taken is not a measurement of zero.",
            file=sys.stderr,
        )
        return 1

    refused = [run for run in runs if run.refused is not None]
    if len(refused) == len(runs):
        print(
            f"\nall {len(runs)} run(s) were refused by the classifier, so there is no "
            "measurement to record. No baseline is written: a baseline of zeroes is a "
            "number a later run will diff against, and it would be diffing against a "
            "classifier that was never asked anything.",
            file=sys.stderr,
        )
        for run in refused:
            print(f"  run {run.index}: {run.refused}", file=sys.stderr)
        return 1

    found = contradictions(runs, truth)
    disagreements = anchored_disagreements(truth, captured) if args.mode == "as-captured" else []
    print_report(
        model=args.model,
        mode=args.mode,
        comments=comments,
        truth=truth,
        runs=runs,
        contradictions_found=found,
        extra_disagreements=disagreements,
        ceiling=ceiling,
    )

    model_calls = sum(int(run.coverage.get("model_calls", 0)) for run in runs)
    recovered = [len(run.recovered) for run in runs]
    by_model = [len(run.recovered_by_model) for run in runs]
    invented = [len(run.unmatched_inferred) for run in runs]
    precisions = [run.precision for run in runs if run.precision is not None]
    aggregates = {
        "recovered_mean": round(sum(recovered) / len(recovered), 4),
        "recovered_min": min(recovered),
        "recovered_max": max(recovered),
        "recovered_by_model_mean": round(sum(by_model) / len(by_model), 4),
        "invented_pairs_mean": round(sum(invented) / len(invented), 4),
        "recall_all_pairs_mean": round(sum(run.recall for run in runs) / len(runs), 4),
        "recall_model_only_mean": round(sum(run.model_recall for run in runs) / len(runs), 4),
        "precision_model_only_mean": (
            None if not precisions else round(sum(precisions) / len(precisions), 4)
        ),
        "runs_refused": len(refused),
        "runs_that_produced_no_pairs": sum(1 for run in runs if run.precision is None),
        "model_calls": model_calls,
        "flipping_comments": len(flips(runs)),
        "contradictions": len(found),
    }
    scope = scope_statement(truth, args.mode, args.runs, ceiling=ceiling, omitted=omitted)
    payload = build_baseline(
        {
            "repo": args.repo,
            "pr": args.pr,
            "model": args.model,
            "thread_mode": args.mode,
            "archive": str(args.archive),
            "archive_digest": digest_of(args.archive),
            "collection": args.collection,
            "comments": len(comments),
            "thread_comments": len(captured),
            "omitted_comment_ids": list(omitted),
            "recall_ceiling": ceiling,
            "ground_truth_digest": truth.digest(),
            "ground_truth_pairs": [list(pair) for pair in truth.pairs],
            "runs": [run.as_dict() for run in runs],
            "aggregates": aggregates,
            "outcome_distribution": outcome_distribution(runs),
            "pairwise_stability": pairwise_stability(runs),
            "flips": flips(runs),
            "contradictions": found + disagreements,
            "unrelated_comment_ids": sorted(
                {comment_id for run in runs for comment_id in run.unrelated}
            ),
            "scope": scope,
        }
    )

    if args.compare:
        try:
            rows = compare(read_baseline(args.baseline), payload)
        except EvaluationError as exc:
            print(f"\ncomparison failed: {exc}", file=sys.stderr)
            return 1
        print(f"\n=== against the baseline at {args.baseline} ===")
        regressions = 0
        for row in rows:
            marker = "  REGRESSION" if row.get("regression") else ""
            if row.get("regression"):
                regressions += 1
            if "delta" in row:
                print(
                    f"  {row['metric']}: {row['baseline']} -> {row['current']} "
                    f"({row['delta']:+g}){marker}"
                )
            else:
                print(f"  {row['metric']}: {row['baseline']} -> {row['current']}")
        print(f"  {regressions} regression(s)")
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 1 if regressions else 0

    args.baseline.parent.mkdir(parents=True, exist_ok=True)
    args.baseline.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nbaseline written to {args.baseline}")

    if not args.no_store:
        entry = build_report_entry(
            repo=args.repo,
            pr_number=args.pr,
            model=args.model,
            measurement=f"thread-classifier-accuracy/{args.mode}",
            result_text=render_result(
                truth=truth,
                runs=runs,
                aggregates=aggregates,
                ceiling=ceiling,
                omitted=omitted,
                comments_measured=len(comments),
                thread_comments=len(captured),
            ),
            scope=scope,
            metadata={
                "repo": args.repo,
                "pr_number": args.pr,
                "collection": args.collection,
                "archive_digest": payload["archive_digest"],
                "ground_truth_digest": truth.digest(),
                "thread_mode": args.mode,
                "thread_comments": len(captured),
                "comments_measured": len(comments),
                "omitted_comment_ids": list(omitted),
                "recall_ceiling": ceiling,
                "baseline_path": str(args.baseline),
                "contradiction_count": len(found) + len(disagreements),
                "unrelated_comment_ids": payload["unrelated_comment_ids"],
            },
        )
        with TansekiClient(args.tanseki_url, collection=args.collection, timeout=20.0) as client:
            client.upsert_document(to_evaluation_upsert_payload(entry))
        print(f"report stored in {args.collection} as {entry.entry_id}")

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
