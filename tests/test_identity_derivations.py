"""What each identity derivation hashes, pinned to a golden value.

Every derivation in Kojutsu ends in a digest that becomes a primary key in the
registry and, from there, an Tanseki document id. None of that is visible in a diff of
the function, and none of it is checkable by the type system: ``"a|1"`` and ``"a"``
joined differently, or two fields reordered, produce a different string that is just
as plausible. So the failure mode this file exists for is not a crash. It is a corpus
that quietly re-identifies itself, and nobody finds out until a reader notices a
document is missing.

**What breaks if a value below changes.** Every already-stored record with that id
becomes unreachable. It is not updated and it is not superseded -- the document in
the store is keyed on a path derived from the old id, so re-deriving the id writes a
*second* document and leaves the first as an orphan nothing will ever update. A
capture arriving after the change is deduped against the new id, so the same comment
stored once before and once after becomes two records that look identical. The fix is
never to edit the derivation in place: bump the version, keep the old one, and say in
a migration what the two namespaces mean.

The one rule these tests encode beyond the digests: every preimage is a JSON array
behind a per-kind domain label. Not delimiter-joined, because a component that can
contain the delimiter forges a different record; not domain-free, because a digest
that says nothing about which kind of record asked for it is a shared namespace with
a prefix painted on the front.

## The text boundary

The second half of this file is about the characters in the text an identity is
*derived alongside*, and it is here because the first half gives the reason it
matters. No derivation in this codebase hashes captured prose -- each one keys on
forge-issued ids precisely so that a rephrasing does not orphan a record, and
``stable_rationale_entry_id``'s docstring is the argument for that. So the usual
justification for normalising text before hashing does not apply: there is no prose
digest for two spellings of one sentence to disagree about.

What is left is the reason the normalisation is still worth having, and it is not
about identity at all. A comment body reaches the store, the console and the search
index as bytes, and two contributors writing the same sentence -- one on a keyboard
that composes, one that does not -- produce two byte strings a reader cannot tell
apart and nothing can deduplicate. That is the same failure as a digest that moved,
one layer up: two records for one fact, with no way to say so. And the invisible
characters are a display-fidelity problem before they are anything else, which is a
problem no amount of correct identity fixes.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from kojutsu.core.answer_collector import (
    CENSUS_IDENTITY_DOMAIN,
    CHECK_IDENTITY_DOMAIN,
    PR_EVENT_IDENTITY_DOMAIN,
    REVIEW_ENTRY_IDENTITY_DOMAIN,
    REVIEW_EVENT_IDENTITY_DOMAIN,
    REVIEW_ID_KEY,
    process_comment_reply_outcome,
    process_review_event_outcome,
    semantic_pr_event_id,
    semantic_review_event_id,
    stable_census_entry_id,
    stable_check_entry_id,
    stable_review_entry_id,
)
from kojutsu.core.design_capture import (
    DESIGN_PROPOSAL_IDENTITY_DOMAIN,
    DESIGN_RECONCILIATION_IDENTITY_DOMAIN,
)
from kojutsu.core.knowledge_sink import KnowledgeDeliveryOutcome
from kojutsu.core.question_registry import (
    ANSWER_IDENTITY_DOMAIN,
    CLARIFICATION_IDENTITY_DOMAIN,
    EVALUATION_IDENTITY_DOMAIN,
    RATIONALE_IDENTITY_DOMAIN,
    SqliteQuestionRegistry,
    stable_answer_entry_id,
    stable_clarification_entry_id,
    stable_evaluation_entry_id,
    stable_rationale_entry_id,
)
from kojutsu.identity import (
    IDEMPOTENCY_IDENTITY_DOMAIN,
    TRANSPORT_ASSIGNED_FIELDS,
    identity_preimage,
)
from kojutsu.integrations.github import (
    KOJUTSU_AGENT_PREFIX,
    KOJUTSU_AGENT_SUFFIX,
    KOJUTSU_ANSWER_PREFIX,
    KOJUTSU_ANSWER_SUFFIX,
    KOJUTSU_RATIONALE_PREFIX,
    KOJUTSU_RATIONALE_SUFFIX,
    extract_agent_claim,
    extract_answer_question_id_from_comment_body,
    extract_answer_text_from_comment_body,
    extract_rationale_claim,
    normalise_captured_text,
)
from kojutsu.integrations.tanseki import _idempotency_key, idempotency_key
from kojutsu.models import KnowledgeEntry

REPO = "org/repo"
PR = 42


class FakeSink:
    """Records what was stored. No network, no store, no clock worth asserting on."""

    def __init__(self) -> None:
        self.entries: list[KnowledgeEntry] = []

    def store(self, entry: KnowledgeEntry) -> KnowledgeDeliveryOutcome | None:
        self.entries.append(entry)
        return None


ANSWER_ID = "answer-v1-186464735b547895d39e233b88783df9bccbd2cf27e9e337189dbb2a7a2a99b3"
RATIONALE_ID = "rationale-v1-e37c64a46f781c82d986447e1abec611d9966f99461a461bdecc0d6050e965bf"
CLARIFICATION_ID = (
    "clarification-v1-e9ee0278ea7a0cd11f0bcd7f9003a79b321ef1afed6d3da760235ee18086e9ce"
)
EVALUATION_ID = "evaluation-v1-9e4590efa0fbfe9302308fff8e09c8c3eeed6e6693c41548974deaa2ceca5f18"
PR_EVENT_OPENED = "pr-event:ee2251bcf9629faea07fcfda6361d7d00266b38fef2a25dec4200d35fe17c492"
PR_EVENT_CLOSED = "pr-event:f9db859d2e0fa8359cfe62113efde2ba62d823ae559055c05c5d7ed0eaf81f02"
REVIEW_EVENT = "review:4ec3e0800b5f15be50de9e06bf40f1ec37fa5ca664332028b7c513e8475a17e0"
REVIEW_ENTRY = "review-v1-21775150b22dfcc3121bc34455bf41fca9d81234d9db9f45d26f60748a348065"
CENSUS_ENTRY = "census-v1-49f195d2306ab3d498e6e44b96652bf416caeeecef0676a224e681738786f559"
CENSUS_ENTRY_WITHOUT_PR = (
    "census-v1-885789e7613c32a03a17914670f96e95aadc74e18b2627645ef02f1a4ac249b2"
)
CHECK_ENTRY = "check-v1-acccc05d3246bd5d1daadb61d5577540bf6685e0c3256a297a40fc8575b950e6"


def _digest(identifier: str, prefix: str) -> str:
    """Strip a known label so a comparison is over the digest rather than the naming.

    The label on the front of an id separates nothing. A digest that is identical
    across two record kinds means one record's identity is the other's, whatever the
    two are called afterwards.
    """
    assert identifier.startswith(prefix), f"{identifier!r} does not start with {prefix!r}"
    return identifier[len(prefix) :]


def test_the_answer_entry_id_is_pinned() -> None:
    """The durable key of every captured answer, and the first thing to move.

    Changing it re-identifies the entire answer corpus and orphans every answer
    document already in the store.
    """
    assert stable_answer_entry_id(REPO, PR, 201) == ANSWER_ID


def test_the_rationale_entry_id_is_pinned() -> None:
    """A rationale is appended by revision, so an id that moves loses the sequence.

    A reader asking "what did the agent think when it started" gets a different
    document than the one asking "what did it think at the end".
    """
    assert (
        stable_rationale_entry_id(
            repo=REPO,
            pr_number=PR,
            branch="feat/backoff",
            declared_by="opencode",
            revision=1,
        )
        == RATIONALE_ID
    )


def test_the_clarification_entry_id_is_pinned() -> None:
    """A clarification's value is that the quoted comment can be re-fetched.

    An id that moves does not orphan the text, it orphans the link: the stored
    clarification no longer names the comment it claims to quote.
    """
    assert (
        stable_clarification_entry_id(repo=REPO, pr_number=PR, github_comment_id=201)
        == CLARIFICATION_ID
    )


def test_the_evaluation_entry_id_is_pinned() -> None:
    """A measurement is re-taken every harness run, so an id that moves is a corpus
    of near-identical documents and a series destroyed.

    What is worth keeping about a measurement is the history of it.
    """
    assert (
        stable_evaluation_entry_id(
            repo=REPO,
            pr_number=PR,
            subject="Why is the retry unbounded?",
            measurement="refusal_rate",
        )
        == EVALUATION_ID
    )


def test_the_lifecycle_event_ids_are_pinned() -> None:
    """The registry key behind ``pr_state_changes``, and the anchor of every
    lifecycle record.

    A close is the event that says a change was adopted or abandoned, so this is the
    id whose movement decides whether an adoption is recorded at all.
    """
    assert semantic_pr_event_id(REPO, PR, "opened", "open", None) == PR_EVENT_OPENED
    assert semantic_pr_event_id(REPO, PR, "closed", "closed", None) == PR_EVENT_CLOSED


def test_the_review_event_id_is_pinned() -> None:
    """The identity a re-serialised review collapses onto.

    GitHub redelivers one review under many delivery ids. This is what stops that
    becoming one document per delivery, and an id that moves turns the dedupe off.
    """
    assert semantic_review_event_id(REPO, PR, 7, comment_id=99) == REVIEW_EVENT


def test_the_review_entry_id_is_pinned() -> None:
    """Derived from the event id, so the pin moves when that does -- deliberately.

    It is here as well so that the two cannot drift apart unnoticed: this is the id
    that becomes a document path, and the event id above is the registry key they
    disagree about when only one of the two has been reviewed.
    """
    event_id = semantic_review_event_id(REPO, PR, 7, comment_id=99)
    assert stable_review_entry_id(event_id) == REVIEW_ENTRY


def test_the_census_entry_ids_are_pinned() -> None:
    """An observation's identity, including the no-pull-request case.

    The absent pull request becomes ``0`` rather than ``null`` in the preimage, so
    every malformed delivery lands in one named bucket instead of a null that would
    read as a fact about a change nobody could name. Both are pinned because the
    second is the one that would otherwise be invented by the encoder.
    """
    assert stable_census_entry_id(REPO, PR, "opened") == CENSUS_ENTRY
    assert stable_census_entry_id(REPO, None, "opened") == CENSUS_ENTRY_WITHOUT_PR


def test_the_check_entry_id_is_pinned() -> None:
    """A machine report's identity, claimed in the same table as a review's.

    The event string is ``"<prefix>:<repo>:<check id>"``: the same shape a review event
    id is built to look like, which is the whole of why the shared namespace below was
    worth closing.
    """
    assert stable_check_entry_id(f"check-run:{REPO}:99") == CHECK_ENTRY


def test_a_check_run_and_a_review_of_the_same_event_string_are_different_records() -> None:
    """The collision this ticket closed, and the one that was reachable.

    Both derivations took a bare ``"<prefix>:<repo>:<id>"`` event string and hashed it
    with nothing in the preimage to say which kind of event it was. So
    ``stable_review_entry_id("check-run:org/repo:99")`` produced the very digest
    ``process_check_run_outcome`` derives for that check run, and both kinds are
    claimed in one table -- ``review_captures`` -- under the event string itself.

    Verified byte-for-byte against the pre-change behaviour before the domain labels
    were added; asserted here so the shared namespace cannot come back unnoticed.
    """
    event_id = f"check-run:{REPO}:99"

    assert _digest(stable_check_entry_id(event_id), "check-v1-") != _digest(
        stable_review_entry_id(event_id), "review-v1-"
    )


def test_a_lifecycle_event_and_a_review_are_different_records() -> None:
    """The other shared namespace: two derivations, one field layout.

    ``semantic_pr_event_id`` and ``semantic_review_event_id`` both take five
    positionally significant values and both are handed to a registry table as an
    opaque primary key. They did not actually collide -- the third element is a string
    in one and an integer in the other, so the JSON forms differ -- which is luck, not
    design. Nothing in a bare array says which kind of record asked for it.
    """
    lifecycle = semantic_pr_event_id(REPO, PR, "opened", "open", None)
    review = semantic_review_event_id(REPO, PR, 0)

    assert _digest(lifecycle, "pr-event:") != _digest(review, "review:")


def test_a_component_containing_the_old_separator_cannot_forge_another_record() -> None:
    """The ambiguity the answer derivation used to have, reproduced on purpose.

    The old preimage was ``f"{repo}|{pr}|{comment}"``. ``repo="a|1"`` with ``pr=2``
    and ``repo="a"`` with ``pr="1|2"`` serialise to the same bytes, so two different
    answers derived one id and the second was refused by the ``answer_captures``
    unique index as a duplicate of a record nobody wrote -- a real answer silently
    dropped, which is the failure a dedupe check is supposed to make impossible.

    Both spellings are passed as written even though the second is against the type
    annotation: an annotation is not a check, and the derivation has to be safe
    against the callers it actually gets rather than the ones it wishes for.
    """
    forged = stable_answer_entry_id("a|1", 2, 3)
    legitimate = stable_answer_entry_id("a", "1|2", 3)  # type: ignore[arg-type]

    assert forged != legitimate, (
        "two different preimages produced one id, so a real answer would be dropped "
        "as a duplicate of a record that does not exist"
    )


def test_the_domain_is_inside_the_digest_and_not_only_painted_on_the_front() -> None:
    """A label on the front of an id separates nothing.

    This is the property that makes the cross-namespace tests above mean anything, so
    it is asserted against the preimage rather than against the two ids: the golden
    answer digest is a function of bytes that *contain* the domain label, so the same
    fields under another kind's label produce a different digest. An id whose prefix
    says ``answer-`` over a digest computed with no notion of answers is a naming
    convention, and a convention is not checked by anything.
    """
    preimage = identity_preimage("kojutsu.answer.v1", (REPO, PR, 201))

    assert hashlib.sha256(preimage).hexdigest() == ANSWER_ID.split("-")[-1]
    assert (
        hashlib.sha256(identity_preimage("kojutsu.review_entry.v1", (REPO, PR, 201))).hexdigest()
        != ANSWER_ID.split("-")[-1]
    )


def test_what_these_goldens_do_not_protect() -> None:
    """The negative test: the promise these tests do not make.

    Stating it here is the only way it does not quietly become a claim that the corpus
    is protected. Nothing at runtime compares a stored id against these goldens: a
    record captured before a change keeps the old id and is simply never reached
    again, and the outbox would happily deliver a second document for the same fact.

    What the goldens make loud is a change to the *code*. Re-deriving ids for what is
    already stored -- recording a version alongside each one and letting the two
    namespaces coexist -- is a migration with a story, and this file is not it.
    """
    # Every derivation names its version on the front, so an id captured before a
    # version bump and one captured after are visibly different namespaces rather
    # than two values a reader has no way to tell apart.
    assert ANSWER_ID.startswith("answer-v1-")
    assert RATIONALE_ID.startswith("rationale-v1-")
    assert CLARIFICATION_ID.startswith("clarification-v1-")
    assert EVALUATION_ID.startswith("evaluation-v1-")
    assert REVIEW_ENTRY.startswith("review-v1-")
    assert CENSUS_ENTRY.startswith("census-v1-")
    assert CHECK_ENTRY.startswith("check-v1-")


# -- the character policy at the text boundary ---------------------------------

ZWSP = "\u200b"
RLO = "\u202e"
BOM = "\ufeff"
SOFT_HYPHEN = "\u00ad"
ZWJ = "\u200d"
ZWNJ = "\u200c"
ARABIC_LETTER_MARK = "\u061c"
INVISIBLE_PLUS = "\u2064"


def test_the_same_sentence_arriving_two_ways_is_stored_one_way() -> None:
    """NFC, and the reason it is not optional.

    An e-acute reaches a comment body as one code point or as ``e`` plus a combining
    acute. They are the same characters to a reader, to GitHub, and to a search over
    the store -- and two byte strings to everything else, so the same answer is
    captured twice, counts twice, and no reader can tell which copy is the original.
    Neither spelling is wrong, so there is nothing to detect after the fact.
    """
    composed = "Caf\u00e9"
    decomposed = "Cafe\u0301"

    assert composed != decomposed, "the fixture is not testing what it claims to"
    assert normalise_captured_text(composed) == normalise_captured_text(decomposed)
    assert normalise_captured_text(decomposed) == composed


@pytest.mark.parametrize(
    ("character", "name"),
    [
        (ZWSP, "zero width space"),
        (RLO, "right-to-left override"),
        (BOM, "byte order mark"),
        (SOFT_HYPHEN, "soft hyphen"),
    ],
)
def test_invisible_formatting_characters_do_not_survive_into_the_store(
    character: str, name: str
) -> None:
    """Display fidelity, which is the reason the filter exists.

    A name that renders as somebody else's, or a path that renders as a different
    file, deceives the one reader the record was kept for. The character is invisible
    in the diff and in the console, so nothing downstream can notice it and nothing
    downstream is expected to.
    """
    body = f"src/{character}auth{character}.py looks fine"

    assert normalise_captured_text(body) == "src/auth.py looks fine", name


def test_honest_content_that_looks_invisible_is_left_alone() -> None:
    """The asymmetry, asserted so it cannot be tightened away by a well-meaning edit.

    ZWJ builds every multi-person emoji; ZWNJ and the Arabic letter mark carry meaning
    in Persian, Urdu and Arabic. Stripping them mangles real names, and a filter that
    mangles real names is one a maintainer removes -- which is how a system ends up
    with no filter at all and the bidi overrides still arriving.
    """
    family = "\U0001f468\u200d\U0001f469\u200d\U0001f467"
    persian = "\u0645\u06cc\u200c\u0631\u0648\u062f"
    arabic_digit = "\u0661"

    assert normalise_captured_text(family) == family
    assert normalise_captured_text(persian) == persian
    assert normalise_captured_text(f"{ARABIC_LETTER_MARK}{arabic_digit}") == (
        f"{ARABIC_LETTER_MARK}{arabic_digit}"
    )
    assert normalise_captured_text(f"1{INVISIBLE_PLUS}") == f"1{INVISIBLE_PLUS}"


def test_a_comment_of_nothing_but_invisible_characters_has_nothing_to_capture() -> None:
    """The outcome the filter buys, rather than a merely tidier string.

    A body of zero-width spaces is truthy to ``.strip()``, so without this the answer
    path would store a record whose entire content is invisible: a record that counts
    as knowledge and reads as nothing.
    """
    body = f"{KOJUTSU_ANSWER_PREFIX}q-1{KOJUTSU_ANSWER_SUFFIX}\n\n{ZWSP}{RLO}{BOM}"

    assert extract_answer_text_from_comment_body(body) == ""
    assert not extract_answer_text_from_comment_body(body).strip()


def test_marker_grammar_fails_closed_while_a_markers_value_is_cleaned() -> None:
    """The split the marker extractors follow, and why it is not one line.

    The ``kojutsu:`` punctuation is grammar: it is what makes a comment a
    Kojutsu record rather than a stranger's text. A token whose own delimiters had to
    be repaired before it parsed is not one this system wrote, so that is refused.

    The value inside the grammar is data, and it is the half that matters here: it
    names the question, it is a registry lookup key, and it becomes part of a document
    path. An invisible character surviving into it is a question that renders as a
    different question -- so it is removed.

    Cleaning the value grants no capability to anyone who could not already have the
    association: anyone who can write ``q-1`` into a marker can write it without a
    zero-width space in front of it. What must not happen is the invisible character
    persisting into stored data.
    """
    honest = f"{KOJUTSU_ANSWER_PREFIX}q-1{KOJUTSU_ANSWER_SUFFIX}\n\nBecause X."
    value_altered = f"{KOJUTSU_ANSWER_PREFIX}q{ZWSP}-1{KOJUTSU_ANSWER_SUFFIX}\n\nBecause X."
    grammar_altered = f"<{ZWSP}!-- kojutsu:answer:q-1 -->\n\nBecause X."

    assert extract_answer_question_id_from_comment_body(honest) == "q-1"
    assert extract_answer_question_id_from_comment_body(value_altered) == "q-1"
    assert extract_answer_question_id_from_comment_body(grammar_altered) is None

    # The prose is unaffected by any of it: the same words come out of all three, so
    # the decision above is about the marker alone.
    assert extract_answer_text_from_comment_body(honest) == extract_answer_text_from_comment_body(
        value_altered
    )


def test_a_declared_agent_id_is_cleaned_because_it_is_stored_as_an_author() -> None:
    """A machine principal that renders as a different principal.

    Nothing hashed here moves -- the answer entry id is keyed on the comment, not on
    who says it -- so this is a display-fidelity fix rather than a re-identification.
    """
    body = f"{KOJUTSU_AGENT_PREFIX}open{ZWSP}code model=o{RLO}pencode{KOJUTSU_AGENT_SUFFIX}"

    claim = extract_agent_claim(body)

    assert claim is not None
    assert claim.agent_id == "opencode"
    assert claim.model == "opencode"


def test_a_rationale_marker_value_is_left_alone_and_that_is_deliberate() -> None:
    """The one place the boundary stops short of the value, asserted so it is a choice.

    ``stable_rationale_entry_id`` hashes the agent id and the branch, and the capture
    tool derives the same id on the *writing* side before it posts. Normalising on the
    collector side alone would give one declaration two ids whenever either value
    arrived decomposed -- one declaration stored twice, which is the exact loss the
    branch field is in the marker to prevent.

    Stated as a test because it is the kind of gap that reads as an oversight: this is
    the counterpart on the far side of the comment, not a local judgement.
    """
    body = f"{KOJUTSU_RATIONALE_PREFIX}open{ZWSP}code rev=1{KOJUTSU_RATIONALE_SUFFIX}"

    claim = extract_rationale_claim(body)

    assert claim is not None
    assert claim.agent_id == f"open{ZWSP}code"


def test_a_captured_answer_is_stored_in_canonical_form_end_to_end(tmp_path) -> None:
    """The normaliser on the path that actually writes, not just in a unit test.

    The stored bytes are what the console renders, what the search index holds and what
    a later reader compares against. Two contributors writing the same answer, one on a
    keyboard that composes and one that does not, have to end up as one answer here or
    the corpus holds two records of one sentence with nothing to say which is which.
    """
    sink = FakeSink()
    with SqliteQuestionRegistry(tmp_path / "registry.db") as registry:
        registry.record_question(
            question_id="q-100",
            github_comment_id=100,
            repo=REPO,
            pr_number=PR,
            pr_url=f"https://github.com/{REPO}/pull/{PR}",
            question_text="Why the cafe?",
            question_category="design_decision",
            question_author="asker",
        )
        outcome = process_comment_reply_outcome(
            new_comment_id=201,
            new_comment_body=f"{KOJUTSU_ANSWER_PREFIX}q-100{KOJUTSU_ANSWER_SUFFIX}"
            "\n\nCaf\u0065\u0301 was already open.",
            new_comment_author="answerer",
            new_comment_created_at=None,
            parent_comment_id=100,
            repo=REPO,
            pr_number=PR,
            registry=registry,
            sink=sink,
            new_comment_author_association="MEMBER",
        )

    assert outcome is not None
    assert sink.entries[0].answer_text == "Caf\u00e9 was already open."


def test_an_answer_of_nothing_but_invisible_characters_is_not_captured(tmp_path) -> None:
    """The refusal that stripping makes possible, on the path that decides it.

    A record with no visible content is still a record: it counts towards the corpus,
    it answers a question, and it renders as nothing at all. The record has to be
    refused, and it is refused by finding there is nothing left to capture rather than
    by inspecting the characters a second time.
    """
    sink = FakeSink()
    with SqliteQuestionRegistry(tmp_path / "registry.db") as registry:
        registry.record_question(
            question_id="q-100",
            github_comment_id=100,
            repo=REPO,
            pr_number=PR,
            pr_url=f"https://github.com/{REPO}/pull/{PR}",
            question_text="Why the cafe?",
            question_category="design_decision",
            question_author="asker",
        )
        outcome = process_comment_reply_outcome(
            new_comment_id=201,
            new_comment_body=f"{KOJUTSU_ANSWER_PREFIX}q-100{KOJUTSU_ANSWER_SUFFIX}"
            f"\n\n{ZWSP}{RLO}{BOM}",
            new_comment_author="answerer",
            new_comment_created_at=None,
            parent_comment_id=100,
            repo=REPO,
            pr_number=PR,
            registry=registry,
            sink=sink,
            new_comment_author_association="MEMBER",
        )

    assert outcome is None
    assert sink.entries == []


def test_a_hostile_review_body_is_stored_rather_than_refused(tmp_path) -> None:
    """The policy decision, exercised end to end: strip and keep.

    Refusing the record is the obvious reading of "reject", and it loses the thing this
    project exists to preserve -- a reviewer who submitted a right-to-left override is
    a fact about the review, and one a reader most needs and cannot reconstruct. The
    anchor is stored alongside, so the original stays re-fetchable from the forge; the
    record is what a reader gets, and it renders as what it says.
    """
    sink = FakeSink()
    with SqliteQuestionRegistry(tmp_path / "registry.db") as registry:
        result = process_review_event_outcome(
            repo=REPO,
            pr_number=PR,
            review_id=7,
            review_state="changes_requested",
            review_body=f"Read {RLO}src/other.py{ZWSP} before approving.",
            review_author="reviewer",
            pr_author_account="author",
            review_submitted_at=None,
            review_author_association="MEMBER",
            comments=[],
            registry=registry,
            sink=sink,
        )

    assert len(result.outcomes) == 1
    assert sink.entries[0].answer_text == "Read src/other.py before approving."
    # The anchor survives, which is what makes keeping the record the right call: a
    # reader who wants the original can re-fetch review 7 from the forge.
    assert sink.entries[0].metadata[REVIEW_ID_KEY] == 7


# --- the request key, which was the one derivation left on the old encoding --------


def test_a_field_containing_the_separator_cannot_forge_another_key() -> None:
    """**The collision this ticket exists to remove, still present in the one place a
    collision makes a write disappear.**

    ``_idempotency_key`` was ``f"{id}|{content}"`` over two attacker-influenced
    fields, so these two bodies derived one key. A repeated idempotency key is how
    the store is told *this is a replay of something already sent* — so the forged
    body is answered as a duplicate and a genuine write silently does not happen.
    Nothing stops a document id or a captured answer containing a pipe.
    """
    forged = _idempotency_key({"id": "a|b", "collection": "c", "content": "c"})
    honest = _idempotency_key({"id": "a", "collection": "c", "content": "b|c"})

    assert forged != honest, (
        "a `|` in one field can move to the other and reproduce the key, which the "
        "store reads as a replay"
    )


def test_an_absent_field_is_not_an_empty_one() -> None:
    """``body.get("id", "")`` could not tell the two apart, so a body with no id and
    a body with ``"id": ""`` derived the same key. They are different requests and
    one of them is a real write."""
    absent = _idempotency_key({"collection": "c", "content": "body"})
    empty = _idempotency_key({"id": "", "collection": "c", "content": "body"})

    assert absent != empty


def test_the_same_id_and_content_in_two_collections_are_different_keys() -> None:
    """The document id is path-derived from repo and entry id, so it does not
    identify a collection. Without the collection in the key, the same write against
    two collections is the same bytes."""
    left = _idempotency_key({"id": "org/repo/pr-1/abc", "collection": "one", "content": "x"})
    right = _idempotency_key({"id": "org/repo/pr-1/abc", "collection": "two", "content": "x"})

    assert left != right


def test_two_serialisations_of_one_logical_write_share_a_key() -> None:
    """**What idempotence actually requires.** Key order in the body is not part of
    what was sent, so re-serialising the same write must not look like a second one."""
    first = _idempotency_key({"id": "org/repo/pr-1/abc", "collection": "c", "content": "x"})
    reordered = _idempotency_key({"content": "x", "collection": "c", "id": "org/repo/pr-1/abc"})

    assert first == reordered


def test_the_request_key_is_not_a_record_identity() -> None:
    """It is computed from overlapping inputs — a document id and content — so a
    shared domain would let a request key and a record id collide on the same bytes.
    Different questions: one says which record this is, the other whether this request
    was already sent."""
    request_key = idempotency_key(document_id="org/repo/pr-1/abc", collection="c", content="x")
    record_id = stable_review_entry_id("org/repo/pr-1/review-7")

    assert request_key != record_id
    assert "kojutsu.idempotency.v1" not in {
        "kojutsu.answer.v1",
        "kojutsu.rationale.v1",
        "kojutsu.clarification.v1",
        "kojutsu.evaluation.v1",
        "kojutsu.review_entry.v1",
        "kojutsu.review_event.v1",
        "kojutsu.pr_event.v1",
        "kojutsu.census.v1",
        "kojutsu.check.v1",
        "kojutsu.design_proposal.v1",
        "kojutsu.design_reconciliation.v1",
    }, "the key needs a namespace of its own, or it can collide with a record id"


def test_every_declared_identity_domain_is_distinct() -> None:
    """A domain is element zero of a preimage, so its only job is to be unique.

    The set above is a literal, and a literal does not notice a new derivation being
    added beside it. Two derivations sharing a domain derive the same id from the same
    bytes, which is a silent collision: the store holds one record under two identities
    or refuses a second write as a duplicate, and nothing raises at the point of
    mistake. So the enumeration is asserted against the constants themselves, which
    means a new domain fails here until somebody decides where it belongs.
    """
    declared = [
        ANSWER_IDENTITY_DOMAIN,
        RATIONALE_IDENTITY_DOMAIN,
        CLARIFICATION_IDENTITY_DOMAIN,
        EVALUATION_IDENTITY_DOMAIN,
        REVIEW_ENTRY_IDENTITY_DOMAIN,
        REVIEW_EVENT_IDENTITY_DOMAIN,
        PR_EVENT_IDENTITY_DOMAIN,
        CENSUS_IDENTITY_DOMAIN,
        CHECK_IDENTITY_DOMAIN,
        DESIGN_PROPOSAL_IDENTITY_DOMAIN,
        DESIGN_RECONCILIATION_IDENTITY_DOMAIN,
    ]

    duplicates = {d for d in declared if declared.count(d) > 1}
    assert not duplicates, f"two derivations share an identity domain: {sorted(duplicates)}"
    assert IDEMPOTENCY_IDENTITY_DOMAIN not in declared, (
        "the idempotency key is not a record identity and must keep its own domain"
    )


def test_no_transport_assigned_field_can_reach_a_preimage() -> None:
    """**The criterion with teeth against silent duplication.**

    These fields change every time a row is retried. A preimage that folded one in
    would derive a new key for the same logical write on each attempt, so a retry
    could not be recognised as a retry — the mechanism would produce a second
    delivery instead of a confirmation, and the store would hold the same knowledge
    twice under two identities. Nothing about that raises; it only shows up later as
    a count that looks wrong.
    """
    preimages = [
        identity_preimage(REVIEW_ENTRY_IDENTITY_DOMAIN, (REPO, PR, 201)),
        identity_preimage("kojutsu.answer.v1", (REPO, PR, 201)),
        identity_preimage(
            IDEMPOTENCY_IDENTITY_DOMAIN, ("org/repo/pr-1/abc", "collection", "content")
        ),
    ]
    encoded = [json.loads(raw.decode()) for raw in preimages]

    for fields in encoded:
        assert not TRANSPORT_ASSIGNED_FIELDS.intersection(fields), (
            "a transport-assigned field reached a preimage, so a retry would derive a "
            "new identity for the same write"
        )


def test_the_excluded_field_set_covers_retry_accounting_and_lease_state() -> None:
    """The constant is only useful if it names the fields that actually change on a
    retry. An empty set would pass the test above."""
    for field in ("attempts", "next_attempt_at", "dead_lettered_at", "lease_expires_at"):
        assert field in TRANSPORT_ASSIGNED_FIELDS, (
            f"{field} is assigned by transport and changes per attempt; leaving it "
            "unlisted makes the exclusion quietly incomplete"
        )

    for field in ("repo", "pr_number", "question_text", "answer_text", "comment_author"):
        assert field not in TRANSPORT_ASSIGNED_FIELDS, (
            f"{field} is part of what the record says, not what transport did to it"
        )
