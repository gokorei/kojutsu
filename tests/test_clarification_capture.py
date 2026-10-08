"""The three gaps the merge agents reported, each now closed and pinned.

1. Nothing constructed a ``ClarificationEntry`` from a live comment, and which
   comments count was an unmade policy decision. The decision is now made,
   stated, and overridable.
2. Backfill could not reach a pull request's *review* thread -- verdicts and inline
   diff comments -- even though the collector for them has existed since 9QFC2PTV.
3. ``structure`` is omitted from frontmatter when anchored, so absence resolves to
   anchored. That is sound but it is the *reader* supplying it, so the payload now
   says whether the store actually stated it.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from kojutsu.allowlist import ADMIT_ALL_ASSOCIATIONS, association_admitted
from kojutsu.config import Settings
from kojutsu.core.clarification_collector import (
    MIN_CLARIFICATION_BODY_CHARS,
    ClarificationPolicy,
    clarification_from_comment,
    collect_clarifications,
)
from kojutsu.core.question_registry import build_registry, stable_clarification_entry_id
from kojutsu.core.tanseki_mapping import build_clarification_frontmatter
from kojutsu.core.text_hygiene import REFUSED_RE, SANITISATION_KEY
from kojutsu.integrations.github_models import GitHubComment, GitHubUser
from kojutsu.models import CaptureSource, RecordStructure

NOW = datetime(2026, 3, 4, 12, 0, tzinfo=UTC)


def _registry(tmp_path: Path):
    return build_registry(Settings(kojutsu_registry_path=str(tmp_path / "r.sqlite")))


def _comment(
    cid: int,
    body: str,
    *,
    login: str = "davy",
    association: str | None = "OWNER",
    account_type: str | None = None,
) -> GitHubComment:
    return GitHubComment(
        id=cid,
        body=body,
        user=GitHubUser(login=login, type=account_type),
        created_at=NOW,
        author_association=association,
    )


class RecordingSink:
    def __init__(self) -> None:
        self.entries: list[Any] = []

    def store(self, record: Any) -> Any:
        self.entries.append(record)
        return record


# --- 1. the clarification capture path ----------------------------------------


def test_a_plain_trusted_comment_becomes_a_clarification(tmp_path: Path) -> None:
    """The gap: the record was storable but nothing ever built one."""
    sink = RecordingSink()
    comment = _comment(
        1,
        "The narrow window is deliberate: we accept one missing audit row per crash "
        "landing inside it, and nothing in v0.1 depends on that row.",
    )

    stored = collect_clarifications(
        repo="o/r",
        pr_number=7,
        comments=[comment],
        registry=_registry(tmp_path),
        sink=sink,  # type: ignore[arg-type]
    )

    assert len(stored) == 1
    entry = stored[0]
    assert entry.statement == comment.body, (
        "a clean quotation must be byte-for-byte the comment; the policy removes the "
        "characters that cannot render as what they say and nothing else"
    )
    assert entry.github_comment_id == 1
    assert entry.author == "davy"
    assert entry.author_association == "OWNER"
    assert entry.capture_source is CaptureSource.COLLECT
    assert sink.entries == [entry]


def test_a_short_acknowledgement_is_not_a_clarification(tmp_path: Path) -> None:
    """ "+1" is not a statement. The floor is length, and length is all it is."""
    sink = RecordingSink()

    stored = collect_clarifications(
        repo="o/r",
        pr_number=7,
        comments=[_comment(1, "LGTM")],
        registry=_registry(tmp_path),
        sink=sink,  # type: ignore[arg-type]
    )

    assert stored == []
    assert sink.entries == []


def test_a_strangers_clarification_is_stored_and_readable_as_a_strangers(
    tmp_path: Path,
) -> None:
    """**The exclusion this replaces was the wrong filter, and it removed real prose.**

    On this record kind the author is *anyone who can comment on a pull request*, so
    the association was standing in for trust where what it actually measured was
    membership -- and on a public project a bot is by definition not a member or
    collaborator of anything, so the excluded ``CONTRIBUTOR`` bucket is where the
    automated reviewers live. Measured on t3code's PR #2829, the old default threw
    away 28 human comments and no bots at all.

    What replaced the filter is that the record says who said it. A drive-by comment
    is now stored, and it is legible as a drive-by comment because the association is
    on the entry and the automation flag is beside it. The repository allowlist is
    what decides whether a stranger's comment can cause a write at all, and that is
    unchanged.
    """
    sink = RecordingSink()
    body = "I think this whole approach is wrong and always has been, for what it is worth."

    stored = collect_clarifications(
        repo="o/r",
        pr_number=7,
        comments=[_comment(1, body, login="drive-by", association="NONE")],
        registry=_registry(tmp_path),
        sink=sink,  # type: ignore[arg-type]
    )

    assert len(stored) == 1
    entry = sink.entries[0]
    assert entry.author == "drive-by"
    assert entry.author_association == "NONE", "the fact the gate used to consume is on the record"
    assert entry.metadata["comment_author_is_machine"] is False


def test_a_clarification_records_whether_the_quoting_account_was_automated(
    tmp_path: Path,
) -> None:
    """This path has the object, so it gets GitHub's own report rather than the
    convention -- which is the only one of the four writers that can, and the reason
    ``GitHubUser`` parses ``type`` at all. The login here follows no convention, so a
    suffix-only implementation would have recorded a bot as a person."""
    sink = RecordingSink()
    body = "The retry budget is three because the provider documents that limit."

    collect_clarifications(
        repo="o/r",
        pr_number=7,
        comments=[
            _comment(
                1,
                body,
                login="a-reviewer-with-no-suffix",
                association="CONTRIBUTOR",
                account_type="Bot",
            )
        ],
        registry=_registry(tmp_path),
        sink=sink,  # type: ignore[arg-type]
    )

    assert sink.entries[0].metadata["comment_author_is_machine"] is True
    frontmatter = build_clarification_frontmatter(sink.entries[0])
    assert frontmatter["comment_author_is_machine"] is True, (
        "the stored document is what a reader opens, so a flag that only reaches the "
        "in-memory entry is not the reader's tool it is meant to be"
    )
    assert frontmatter["github_author_association"] == "CONTRIBUTOR"


def test_a_registered_answer_is_never_re_recorded_as_a_clarification(tmp_path: Path) -> None:
    """The whole point: a comment already counted as an answer must not appear again
    under a heading that reads as "nobody asked about this"."""
    registry = _registry(tmp_path)
    registry.record_question(
        question_id="q-1",
        repo="o/r",
        pr_number=7,
        pr_url="https://github.com/o/r/pull/7",
        question_text="Is the window deliberate?",
        question_category="design_decision",
        github_comment_id=100,
    )
    registry.mark_question_answered(100, 200)
    sink = RecordingSink()
    answer_text = "Yes, deliberately. One missing audit row per crash is the accepted cost."

    stored = collect_clarifications(
        repo="o/r",
        pr_number=7,
        comments=[_comment(200, answer_text)],
        registry=registry,
        sink=sink,  # type: ignore[arg-type]
    )

    assert stored == [], "an answer is already captured; re-recording it double-counts"
    assert sink.entries == []


def test_the_policy_is_a_value_not_a_hard_coded_rule(tmp_path: Path) -> None:
    """An operator can widen it without editing the collector."""
    sink = RecordingSink()
    body = "short but substantive: the lock is advisory between processes."
    default = ClarificationPolicy()
    narrow = ClarificationPolicy(associations=frozenset({"OWNER", "MEMBER"}))

    assert default.admits(_comment(1, body)) is True
    assert default.admits(_comment(1, body, association="NONE")) is True, (
        "the default admits every association; see "
        "kojutsu.allowlist.ADMIT_ALL_ASSOCIATIONS before changing it back"
    )
    assert narrow.admits(_comment(1, body, association="NONE")) is False, (
        "an explicit set means only those, which is how an operator narrows"
    )
    assert narrow.admits(_comment(1, body, association="OWNER")) is True

    stored = collect_clarifications(
        repo="o/r",
        pr_number=7,
        comments=[_comment(1, body, association="OWNER")],
        registry=_registry(tmp_path),
        sink=sink,  # type: ignore[arg-type]
        policy=narrow,
    )
    assert len(stored) == 1


def test_an_empty_policy_set_admits_nothing_and_is_not_the_same_as_no_policy(
    tmp_path: Path,
) -> None:
    """**``None`` and ``frozenset()`` are opposite answers and used to be read as one
    thing.** The gate was ``if association not in self.associations`` against a
    default set, so the ambiguity lived in the default rather than the comparison --
    and the review path spelled the same decision as
    ``authorized_associations or DEFAULT``, which turned "store nothing" into "store
    everything" with a green run summary. An operator who narrows to nothing must get
    silence, and an operator who says nothing must get everything.
    """
    sink = RecordingSink()
    body = "short but substantive: the lock is advisory between processes."
    admit_nothing = ClarificationPolicy(associations=frozenset())
    unconstrained = ClarificationPolicy(associations=None)

    assert admit_nothing.admits(_comment(1, body, association="OWNER")) is False
    assert unconstrained.admits(_comment(1, body, association="NONE")) is True

    stored = collect_clarifications(
        repo="o/r",
        pr_number=7,
        comments=[_comment(1, body, association="OWNER")],
        registry=_registry(tmp_path),
        sink=sink,  # type: ignore[arg-type]
        policy=admit_nothing,
    )

    assert stored == [], "an empty set is a real narrowing, not an absent one"
    assert sink.entries == []


def test_the_default_policy_is_inherited_from_the_one_definition() -> None:
    """The policy is not re-spelled here. It was declared twice -- once in this module's
    import, once in ``rationale_collector`` -- and a policy that exists in two places
    is a policy whose two places get edited on different days. Asserted against the
    constant rather than against a literal so this fails if someone re-declares it."""
    assert ClarificationPolicy().associations is ADMIT_ALL_ASSOCIATIONS
    assert association_admitted("CONTRIBUTOR") is True
    assert association_admitted(None, frozenset()) is False, "empty narrows to nothing"
    assert association_admitted("NONE", frozenset({"MEMBER"})) is False


def test_identity_is_the_comment_so_rerunning_does_not_duplicate(tmp_path: Path) -> None:
    comment = _comment(1, "The retry budget is three because the provider documents that limit.")
    first = clarification_from_comment(comment, repo="o/r", pr_number=7)
    reworded = clarification_from_comment(
        _comment(1, "The retry budget is three; the provider documents that limit."),
        repo="o/r",
        pr_number=7,
    )
    other = clarification_from_comment(comment, repo="o/r", pr_number=8)

    assert first.entry_id == reworded.entry_id, "reworded text is the same comment"
    assert first.entry_id != other.entry_id, "a different PR is a different record"


def test_an_agent_marker_is_recorded_but_never_inferred() -> None:
    declared = _comment(
        1,
        "<!-- kojutsu:agent:opencode model=opencode/model -->\n"
        "Left the lazy recovery path in place rather than adding a sweeper timer.",
    )
    silent = _comment(2, "Left the lazy recovery path in place rather than adding a sweeper.")

    marked = clarification_from_comment(declared, repo="o/r", pr_number=7)
    unmarked = clarification_from_comment(silent, repo="o/r", pr_number=7)

    assert marked.authored_by_agent == "opencode"
    assert marked.authored_by_model == "opencode/model"
    assert unmarked.authored_by_agent is None, "absence of a marker is recorded as absence"
    assert unmarked.authored_by_model is None


# --- 3. absence is distinguishable from a stated value -------------------------


def test_the_payload_says_whether_the_store_actually_stated_the_structure() -> None:
    """Anchored is omitted from frontmatter by design, so a reader would otherwise
    be unable to tell "nobody claimed a structure" from "anchored, and someone said
    so" -- and the second is the only one the store actually asserts."""
    from kojutsu.core.question_registry import stable_clarification_entry_id
    from kojutsu.core.tanseki_mapping import to_clarification_upsert_payload
    from kojutsu.dev_console import capture_of
    from kojutsu.models import UNKNOWN_MODEL, ClarificationEntry

    def flatten(frontmatter: dict[str, Any]) -> dict[str, Any]:
        payload = to_clarification_upsert_payload(
            ClarificationEntry(
                entry_id=stable_clarification_entry_id(
                    repo="o/r", pr_number=1, github_comment_id=5
                ),
                repo="o/r",
                pr_number=1,
                statement="s" * 60,
                author="davy",
                author_association="OWNER",
                github_comment_id=5,
                capture_source=CaptureSource.COLLECT,
                captured_at=NOW,
                metadata={} if not frontmatter else {"structure_inferred_by_model": UNKNOWN_MODEL},
                **({"structure": RecordStructure.INFERRED} if frontmatter.get("structure") else {}),
            )
        )
        body = payload["content"]

        class Doc:
            id = payload["id"]
            updated_at = "2026-03-04T12:00:00Z"
            revision = 1

        Doc.content = body
        Doc.frontmatter = payload["frontmatter"]
        return capture_of(_OneDoc(Doc()), payload["id"])  # type: ignore[arg-type]

    anchored = flatten({})
    inferred = flatten({"structure": "inferred"})

    assert anchored["structure"] == "anchored"
    assert anchored["structure_stated"] is False, "nothing was asserted; absence resolved it"
    assert inferred["structure"] == "inferred"
    assert inferred["structure_stated"] is True, "the store made this claim"


class _OneDoc:
    def __init__(self, doc: Any) -> None:
        self._doc = doc

    def get_document(self, _id: str) -> Any:
        return self._doc


def test_a_corrupt_structure_value_is_reported_as_unknown_not_anchored() -> None:
    """A renamed or corrupted value must never read as a confirmed pairing."""
    from kojutsu.models import structure_of

    assert structure_of("nonsense") is None
    assert structure_of(None) is RecordStructure.ANCHORED


# --- 2. the review thread is reachable -----------------------------------------


def test_review_endpoints_are_read_only_and_paginated() -> None:
    """The new client methods issue GETs and walk pages, like their neighbours."""
    import inspect

    from kojutsu.integrations.github import GitHubClient

    for name in ("list_pull_request_reviews", "list_pull_request_review_comments"):
        source = inspect.getsource(getattr(GitHubClient, name))
        assert '"GET"' in source, f"{name} must not issue a mutating request"
        for verb in ('"POST"', '"PATCH"', '"PUT"', '"DELETE"'):
            assert verb not in source, f"{name} must not issue {verb}"
        assert "while True:" in source, f"{name} must page rather than read one page"


def test_a_backfill_reports_the_new_counts_separately() -> None:
    """Clarifications and reviews are different kinds of record and must not be
    folded into a count called 'answers'."""
    from kojutsu.core.backfill import BackfillCoverage

    coverage = BackfillCoverage(
        repository="o/r",
        since="2026-01-01",
        until="2026-02-01",
        pull_requests=1,
        search_total_count=1,
        truncated=False,
        prs_read=1,
        prs_failed=0,
        comments_read=4,
        answers_captured=2,
        rationales_captured=1,
        clarifications_captured=3,
        reviews_captured=5,
    )

    assert coverage.as_dict()["clarifications_captured"] == 3
    assert coverage.as_dict()["reviews_captured"] == 5
    summary = coverage.summary()
    assert "3 clarification(s)" in summary
    assert "5 review(s)" in summary


def test_the_stub_routes_review_comments_apart_from_issue_comments() -> None:
    """Both paths end in '/comments'. A suffix-only route answers a review request
    with the thread, which is a fixture that lets a real bug pass."""
    import tests.test_github_range as range_tests

    transport = range_tests.RecordingTransport(review_comments={3: [{"id": 1}]})
    issue = range_tests.RecordingTransport(comments={3: [{"id": 2}]})

    def request_for(transport_: Any, path: str) -> Any:
        import httpx

        return transport_(httpx.Request("GET", f"https://api.github.com{path}"))

    review = request_for(transport, "/repos/o/r/pulls/3/comments")
    thread = request_for(issue, "/repos/o/r/issues/3/comments")

    assert review.json() == [{"id": 1}]
    assert thread.json() == [{"id": 2}]


@pytest.mark.parametrize(
    "directory",
    [None],
)
def test_collector_module_imports_without_a_store(directory: str | None) -> None:
    """Importing the collector must not open anything; it is a pure module."""
    import kojutsu.core.clarification_collector as module

    assert module.MIN_CLARIFICATION_BODY_CHARS > 0
    with tempfile.TemporaryDirectory() as unused:
        assert unused


# --- 4. the quotation, and the character policy applied to it -----------------
#
# This is the one record kind whose author is *any human who can comment on a pull
# request*, and the one whose docstring calls it explicitly a quotation — so the
# argument for applying the policy here had to be made rather than inherited. It is
# in ``clarification_collector``'s module docstring; what these pin is the decision
# in **every** direction, because the failure mode of each is a different wrong
# answer rather than an error.
#
#   - reverting to storing the body untouched fails
#     ``test_a_display_deceptive_quotation_is_stored_clean_and_says_so``;
#   - widening the refused set past the enumeration fails
#     ``test_the_wording_a_quotation_owes_its_reader_survives_this_pass`` — and that
#     is the one protecting the *quotational fidelity* half of the argument, since
#     the whole case for removing anything rests on the set containing no content;
#   - dropping the note while keeping the removal fails the first test too, which is
#     the point: a silent edit of a quotation is the dishonesty, not the filtering.

BELL = ""
RLO = "‮"  # noqa: PLE2502 - fixture data for the bidi filter
WORD_JOINER = "⁠"
ZWJ = "‍"
ZWNJ = "‌"

#: The content a filter one step too strict eats, and the reason the decision above
#: is cheap rather than a trade-off. Every one of these is a character the refused
#: set excludes by name, and every one of them appears in a real pull request
#: comment often enough that a filter which took them would mangle a contributor.
FAMILY = "\U0001f468" + ZWJ + "\U0001f469" + ZWJ + "\U0001f467"
PERSIAN = "مي" + ZWNJ + "رود"
ENGLAND_FLAG = "\U0001f3f4\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f"
#: U+061C attaches an Arabic-Indic digit to the digit before it, and U+2064 glues a
#: number together in accounting. Both are format characters, so the obvious
#: implementation -- refuse every ``Cf`` -- takes them, and neither is decoration.
ARABIC_INDIC_ACCOUNTING = "١٢" + "؜" + "٣٤" + "⁤" + "٥٦"  # noqa: PLE2502 - fixture data for the bidi filter


def test_a_display_deceptive_quotation_is_stored_clean_and_says_so() -> None:
    """A trusted account says something that renders as something else.

    The record still exists, the wording is intact, every refused character is gone,
    and the note names them — because a hostile comment is evidence *of* a hostile
    comment, and the only thing worse than storing the attack is storing it without
    saying what was done to it.
    """
    body = (
        "I would push back on the narrow window here, and I mean that literally."
        f"{RLO}gnit收aW{WORD_JOINER}{WORD_JOINER}{BELL}"
    )

    entry = clarification_from_comment(_comment(1, body), repo="o/r", pr_number=7)

    # The wording the author wrote, with the display controls taken out of it.
    assert (
        entry.statement
        == "I would push back on the narrow window here, and I mean that literally.gnit收aW"
    )
    assert REFUSED_RE.search(entry.statement) is None

    # And it says what it took, in code points rather than by presence, so the note
    # cannot reintroduce what it reports.
    note = entry.metadata[SANITISATION_KEY]
    assert note.startswith("removed 4 characters before storing: ")
    assert "U+202E RIGHT-TO-LEFT OVERRIDE (x1)" in note
    assert "U+0007 unnamed Cc (x1)" in note
    assert f"U+{ord(WORD_JOINER):04X} WORD JOINER (x2)" in note
    assert REFUSED_RE.search(note) is None


def test_the_wording_a_quotation_owes_its_reader_survives_this_pass() -> None:
    """The half that makes the decision above cheap, and the half most likely to be "fixed".

    The argument for removing characters from a *quotation* is that the refused set
    holds no propositional content — no word, digit or meant punctuation. These five
    are the characters that do hold content, and they are excluded from the refused
    set by name. If somebody ever adds one of them, this fails and the argument for
    sanitising a clarification collapses into the trade-off it was said not to be.

    See ``PRESERVED_INVISIBLE_CHARACTERS`` in :mod:`kojutsu.core.text_hygiene` for
    why each is here; the failure mode of getting this wrong is not a broken filter,
    it is *no* filter, and nothing about that looks like a bug at runtime.
    """
    body = (
        f"Reviewed with {FAMILY}; the name is {PERSIAN}. {ENGLAND_FLAG} is the right flag, "
        f"and the invoice total is {ARABIC_INDIC_ACCOUNTING} for the quarter."
    )

    entry = clarification_from_comment(_comment(1, body), repo="o/r", pr_number=7)

    assert entry.statement == body, (
        "a quotation must lose no content; this is what makes removing the display "
        "controls a zero-cost edit rather than a trade against fidelity"
    )
    assert SANITISATION_KEY not in entry.metadata, (
        "absence is the claim that the stored wording is the comment's own bytes, and "
        "a comment nothing had to be taken out of is exactly that case"
    )


def test_removing_characters_cannot_re_identify_a_clarification() -> None:
    """Why this is safe here, and the check that makes it safe.

    ``stable_clarification_entry_id`` derives from ``(repo, pr_number, comment.id)``
    and never from the text — a digest over the statement would orphan the earlier
    record on every rephrasing and collapse two people who independently said the
    same thing into one, losing the agreement, which is the finding. So editing the
    prose cannot move a document, orphan one, or produce a second record of one
    comment.
    """
    clean = clarification_from_comment(
        _comment(1, "The narrow window is the accepted cost for v0.1, deliberately."),
        repo="o/r",
        pr_number=7,
    )
    attacked = clarification_from_comment(
        _comment(1, f"The narrow window is the accepted cost for v0.1, deliberately.{RLO}"),
        repo="o/r",
        pr_number=7,
    )

    assert attacked.entry_id == clean.entry_id
    assert attacked.entry_id == stable_clarification_entry_id(
        repo="o/r", pr_number=7, github_comment_id=1
    )


def test_a_quotation_that_is_only_display_controls_is_not_stored_and_does_not_raise(
    tmp_path: Path,
) -> None:
    """The failure this ordering exists to prevent.

    Forty-plus characters of nothing a reader can see clear the length floor, so
    without the post-policy check this reaches ``ClarificationEntry``, which refuses
    an empty statement — and an attacker's comment would abort a collector loop. That
    is the one way a hostile comment gets to decide when capture runs.

    Refused rather than stored, because there is no quotation left to store: this is
    not the policy discarding a record, it is the record not existing.
    """
    sink = RecordingSink()
    hostile = _comment(1, RLO * 60)

    assert len(hostile.body.strip()) >= MIN_CLARIFICATION_BODY_CHARS
    assert ClarificationPolicy().admits(hostile) is False

    stored = collect_clarifications(
        repo="o/r",
        pr_number=7,
        comments=[hostile],
        registry=_registry(tmp_path),
        sink=sink,  # type: ignore[arg-type]
    )

    assert stored == []
    assert sink.entries == []


def test_the_policy_does_not_change_which_comments_are_admitted(tmp_path: Path) -> None:
    """The gate's *selection* is still a judgement about the comment as it arrived.

    The length floor stays on the raw body, because that is what it was chosen to
    reason about — drop "+1", keep anything with a clause in it. Only the emptiness
    rule moved, and only because it is the one question that cannot be answered
    until the policy has run. A comment with real words and one stray override is
    still a clarification, and it is still selected on its length.
    """
    body = f"The narrow window is the accepted cost for v0.1, deliberately.{RLO}"
    sink = RecordingSink()

    assert ClarificationPolicy().admits(_comment(1, body)) is True
    stored = collect_clarifications(
        repo="o/r",
        pr_number=7,
        comments=[_comment(1, body)],
        registry=_registry(tmp_path),
        sink=sink,  # type: ignore[arg-type]
    )

    assert len(stored) == 1
    assert stored[0].statement == "The narrow window is the accepted cost for v0.1, deliberately."
    assert sink.entries[0].metadata[SANITISATION_KEY].startswith("removed 1 character ")


def test_the_agent_marker_is_read_before_the_policy_touches_the_body() -> None:
    """The ordering constraint, on the one path where both operations exist.

    ``extract_agent_claim`` parses the ``kojutsu:`` grammar *verbatim*, so a marker
    whose own punctuation had to be repaired in order to parse fails closed rather
    than becoming a claim this system did not write. Sanitising the body first would
    repair exactly that punctuation and defeat the check, which is why the marker is
    read above the ``sanitise`` call rather than from its output.

    Pinned because the two calls sit four lines apart and the wrong order is silent:
    the record is still stored, still correct-looking, and the fail-closed behaviour
    it was written for simply never fires.
    """
    body = f"<!-- kojutsu:agent:opencode model=opencode/model -->\n{BELL}{RLO}The retry budget is three."
    entry = clarification_from_comment(_comment(1, body), repo="o/r", pr_number=7)

    # The marker was read from the raw body, so the declaration survived...
    assert entry.authored_by_agent == "opencode"
    assert entry.authored_by_model == "opencode/model"
    # ...and the prose beside it was still cleaned.
    assert entry.statement.endswith("The retry budget is three.")
    assert REFUSED_RE.search(entry.statement) is None


def test_a_caller_supplied_metadata_key_is_kept_alongside_the_note() -> None:
    """The thread classifier passes a fact about *selection*; the note is about the text.

    Two different questions, and merging them would drop whichever lost. The note is
    written last so it cannot be shadowed by a caller's copy of the same key: only
    this module knows what it actually removed.
    """
    entry = clarification_from_comment(
        _comment(1, f"The narrow window is the accepted cost.{BELL}"),
        repo="o/r",
        pr_number=7,
        metadata={"thread_classified_by_model": "opencode/model"},
    )

    assert entry.metadata["thread_classified_by_model"] == "opencode/model"
    assert "U+0007" in entry.metadata[SANITISATION_KEY]
