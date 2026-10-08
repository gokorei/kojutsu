"""What the design phase refuses, and what the ledger holds afterwards.

The tests are grouped by the claim each one defends, and the grouping is the argument:
a test that only exercised the happy path would pass for an implementation that
appends prose to a list.

- **Capture makes a proposal evidence.** A proposal is a record with a principal, a
  model and a re-derivable id, not a paragraph. The identity tests pin the digest,
  because a moved derivation is the one failure here that is invisible in a diff and
  orphans every document already written.
- **The reconciliation refuses to grade itself.** This is the ticket's central claim,
  so it gets the most tests and the ones written against the ways it fails *quietly*:
  a case difference, a discard used as a loophole, a proposal that was never captured.
- **Nothing is stored until the plan parses.** A malformed plan must leave the ledger
  exactly as it was, which is asserted by counting claims rather than by asserting the
  absence of an exception.
- **Re-running is a no-op.** Pinned against the **real** ``SqliteQuestionRegistry``
  rather than a fake, because the idempotency claim is a claim about that table's
  unique index and a fake would agree with any implementation.
- **What the enforcement is worth is pinned as a limitation, not as a feature.** The
  last group exists so the honest sentence cannot be quietly deleted, and so nobody
  later "fixes" the check by making it look stronger than it is.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from kojutsu.core import design_capture
from kojutsu.core.design_capture import (
    DESIGN_DISCARDED_KEY,
    DESIGN_PLAN_DIGEST_KEY,
    DESIGN_PROPOSAL_IDENTITY_DOMAIN,
    DESIGN_PROPOSAL_IDS_KEY,
    DESIGN_ROLE_KEY,
    DESIGN_ROLE_PROPOSAL,
    DESIGN_ROLE_RECONCILIATION,
    INDEPENDENCE_LIMITATION_NOTE,
    MAX_DESIGN_PROPOSALS,
    DesignProposal,
    DiscardedProposal,
    ReconcilerNotIndependentError,
    capture_design_proposal,
    check_reconciler_independence,
    design_plan_digest,
    reconcile_design_proposals,
    stable_design_proposal_id,
    stable_design_reconciliation_id,
)
from kojutsu.core.design_plan import DesignPlanError
from kojutsu.core.knowledge_sink import KnowledgeDeliveryOutcome, KnowledgeDeliveryStatus
from kojutsu.core.question_registry import SqliteQuestionRegistry
from kojutsu.core.text_hygiene import REFUSED_RE, SANITISATION_KEY
from kojutsu.identity import identity_preimage
from kojutsu.models import CaptureSource, RationaleChannel, RationaleSource

REPO = "org/repo"
TOPIC = "design-phase-recording"
MODEL = "opencode/model"
PROPOSAL_TEXT = "Capture every proposal as a record, then reconcile from outside the panel."
DECISION = "Record the reconciliation, including what it discarded."

RLO = "\u202e"
BELL = "\u0007"

#: The two derivations, pinned as goldens. Same property as
#: ``tests/test_identity_derivations.py`` holds for the other namespaces, asserted here
#: because these domains are registered in that file's enumeration -- which this file
#: does not own and therefore cannot edit, so the pin lives where the derivation does.
PROPOSAL_ID = "design-proposal-v1-ad4a03ed6118b2d567adcf581d075f9c515886fcd631292e4c25bc72da43945b"
RECONCILIATION_ID = (
    "design-reconciliation-v1-010f56b8e89d13adbafe33fae3ed7a66e755fa9a4a0e88b4bcb8b2694fa72f35"
)


class FakeSink:
    """Records what was stored. No network, no store, no clock worth asserting on."""

    def __init__(self, outcome: KnowledgeDeliveryOutcome | None = None) -> None:
        self.entries: list[Any] = []
        self.outcome = outcome

    def store(self, entry: Any) -> KnowledgeDeliveryOutcome | None:
        self.entries.append(entry)
        return self.outcome


def _capture(
    *,
    registry: Any,
    sink: FakeSink | None = None,
    repo: str = REPO,
    topic: str = TOPIC,
    principal: str = "proposer-a",
    model: str | None = MODEL,
    text: str = PROPOSAL_TEXT,
    revision: int = 1,
):  # type: ignore[no-untyped-def]
    sink = sink or FakeSink()
    outcome = capture_design_proposal(
        repo=repo,
        topic=topic,
        principal=principal,
        model=model,
        text=text,
        registry=registry,
        sink=sink,
        proposed_at=datetime(2026, 10, 1, tzinfo=UTC),
        revision=revision,
    )
    return outcome, sink


def _plan(**overrides: Any) -> dict[str, Any]:
    plan: dict[str, Any] = {
        "goals": ["Make the design phase a recorded pipeline with an independent reconciler"],
        "decisions": [
            {
                "summary": "Reconcile from a principal that proposed nothing",
                "rationale": DECISION,
                "alternatives_rejected": [
                    {
                        "alternative": "Let the first proposer reconcile its own panel",
                        "why_rejected": "A model judging its own proposal checks nothing.",
                    }
                ],
            }
        ],
        "tickets": [
            {
                "id": "capture",
                "title": "Capture every proposal as a record",
                "description": "Each proposer's output reaches the ledger with provenance.",
                "acceptance_criteria": [
                    "Given a plan topic, when proposals are requested, then every proposal "
                    "is captured as a record with its own provenance"
                ],
                "test_command": "uv run pytest -q tests/test_design_capture.py",
                "reference_files": ["src/kojutsu/core/design_capture.py"],
                "labels": ["design-phase"],
                "priority": "P2",
            }
        ],
    }
    plan.update(overrides)
    return plan


def _reconcile(
    *,
    registry: Any,
    proposals: list[DesignProposal],
    sink: FakeSink | None = None,
    reconciled_by: str = "reconciler",
    model: str | None = "other/model",
    plan: dict[str, Any] | None = None,
    discarded: list[DiscardedProposal] | None = None,
    topic: str = TOPIC,
    revision: int = 1,
):  # type: ignore[no-untyped-def]
    sink = sink or FakeSink()
    outcome = reconcile_design_proposals(
        repo=REPO,
        topic=topic,
        reconciled_by=reconciled_by,
        model=model,
        proposals=proposals,
        plan_payload=_plan() if plan is None else plan,
        registry=registry,
        sink=sink,
        discarded=discarded or (),
        reconciled_at=datetime(2026, 10, 2, tzinfo=UTC),
        revision=revision,
    )
    return outcome, sink


@pytest.fixture
def registry(tmp_path: Path):  # type: ignore[no-untyped-def]
    with SqliteQuestionRegistry(tmp_path / "registry.db") as opened:
        yield opened


# --- capture: a proposal is a record, and its identity is pinned -----------------


def test_the_design_proposal_id_is_pinned() -> None:
    """The first thing to move, and the one nothing notices when it does.

    Moving it re-identifies every stored proposal and orphans every document already
    written under the old id, while a capture arriving after the change is deduped
    against the new one -- so one proposal stored before and after becomes two records
    that look identical. The fix is a version bump and a migration, never an edit here.
    """
    assert (
        stable_design_proposal_id(repo=REPO, topic=TOPIC, declared_by="proposer-a", revision=1)
        == PROPOSAL_ID
    )


def test_the_design_reconciliation_id_is_pinned() -> None:
    """A second namespace, pinned so a proposal and a reconciliation cannot be one id.

    Both are stated reasons over the same shape of anchor, so nothing but the domain
    label keeps them apart -- and a prefix painted on the front of a digest is a naming
    convention rather than a property of the digest.
    """
    assert (
        stable_design_reconciliation_id(
            repo=REPO, topic=TOPIC, reconciled_by="reconciler", revision=1
        )
        == RECONCILIATION_ID
    )


def test_the_domain_is_inside_the_digest_and_not_only_painted_on_the_front() -> None:
    """The property that makes the two pins mean anything.

    A prefix is a label; the domain is an element of the preimage. If it were only the
    label, the same anchor under the other kind's domain would produce one digest and
    the two record kinds would be one namespace wearing two prefixes -- the collision
    ``REVIEW_ENTRY_IDENTITY_DOMAIN`` documents for checks and reviews.
    """
    proposal = identity_preimage(DESIGN_PROPOSAL_IDENTITY_DOMAIN, (REPO, TOPIC, "proposer-a", "1"))

    assert proposal != identity_preimage(
        "kojutsu.design_reconciliation.v1", (REPO, TOPIC, "proposer-a", "1")
    )
    assert b"kojutsu.design_proposal.v1" in proposal


def test_a_proposal_is_stored_with_its_principal_model_and_role(registry: Any) -> None:
    outcome, sink = _capture(registry=registry)

    assert outcome.captured is True
    assert len(sink.entries) == 1
    entry = sink.entries[0]
    assert entry.declared_by == "proposer-a"
    assert entry.declared_model == MODEL
    assert entry.rationale_text == PROPOSAL_TEXT
    assert entry.source is RationaleSource.DECLARED
    # No forge behind a local design proposal, so the channel must not claim one.
    assert entry.channel is RationaleChannel.CAPTURE_SERVER
    assert entry.metadata[DESIGN_ROLE_KEY] == DESIGN_ROLE_PROPOSAL
    assert entry.entry_id == stable_design_proposal_id(
        repo=REPO, topic=TOPIC, declared_by="proposer-a", revision=1
    )


def test_a_stored_proposal_is_never_capture_evidence(registry: Any) -> None:
    """It was produced by a model that had just read the repository.

    That is the strongest reason this record kind must stay ``ASSERTED``: it is a
    model reading attacker-influenceable text and writing prose about what it found.
    Promotable provenance here would let a later reader treat a model's own argument
    as evidence, which is the failure ``RationaleEntry`` already refuses.
    """
    _, sink = _capture(registry=registry)

    assert sink.entries[0].capture_source is CaptureSource.ASSERTED


def test_an_unstated_model_is_recorded_as_unstated(registry: Any) -> None:
    """A reconciler reading a fabricated model would weigh the proposal wrongly."""
    _, sink = _capture(registry=registry, model=None)

    assert sink.entries[0].declared_model is None


def test_two_topics_from_one_principal_are_two_records(registry: Any) -> None:
    """The reason the topic rides in the branch column, and the reason it has to.

    ``rationale_captures`` carries its unique index over the anchor columns and nothing
    else. Had the topic been left out of both the id and the claim, this second capture
    would be refused by that index as a duplicate of the first -- and the proposal
    would be lost with nothing in the ledger to say a capture was ever attempted.
    """
    first, _ = _capture(registry=registry, topic="topic-one")
    second, _ = _capture(registry=registry, topic="topic-two")

    assert first.entry_id != second.entry_id
    assert len(_stored(registry)) == 2


def test_a_rephrased_proposal_is_the_same_record_not_a_second_one(registry: Any) -> None:
    """The identity excludes the text, for the rationale derivation's reason.

    A digest over the prose would orphan the earlier document on every rewording and
    leave a reader with two records of one argument and no way to say which is which.
    """
    _capture(registry=registry, text=PROPOSAL_TEXT)
    outcome, sink = _capture(registry=registry, text=PROPOSAL_TEXT + " And gate on it.")

    assert outcome.captured is False
    assert len(sink.entries) == 0
    assert len(_stored(registry)) == 1


def test_a_second_revision_records_what_it_supersedes(registry: Any) -> None:
    """Appended, never overwritten: the first argument is often the more useful half."""
    first, _ = _capture(registry=registry)
    _second, sink = _capture(registry=registry, revision=2, text="Changed my mind.")

    assert sink.entries[0].revises == first.entry_id
    assert sink.entries[0].revision == 2
    assert len(_stored(registry)) == 2


def _stored(registry: Any) -> list[dict[str, Any]]:
    return registry.list_rationales(repo=REPO, limit=100)


# --- the independence constraint, enforced ---------------------------------------


def test_a_reconciler_that_proposed_cannot_reconcile(registry: Any) -> None:
    """The ticket's central constraint, in the case it exists for.

    It is refused *before* the plan is read: an unentitled reconciler must not get to
    parse a document and then decide the constraint was not applicable.
    """
    outcome, _ = _capture(registry=registry, principal="solo")
    proposal = outcome.proposal

    with pytest.raises(ReconcilerNotIndependentError) as caught:
        _reconcile(registry=registry, proposals=[proposal], reconciled_by="solo")

    assert "solo" in str(caught.value)
    assert len(_stored(registry)) == 1, "only the proposal may have been written"


@pytest.mark.parametrize("spelling", ["SOLO", " solo ", "Solo"])
def test_the_comparison_is_case_and_whitespace_insensitive(registry: Any, spelling: str) -> None:
    """The accidental-and-sloppy half the check is actually worth.

    A plain ``!=`` would pass every one of these, and they are the overwhelmingly
    likely way the constraint gets violated by accident: one process runs several
    agents and the identifier is right there in the environment.
    """
    outcome, _ = _capture(registry=registry, principal="solo")

    with pytest.raises(ReconcilerNotIndependentError):
        _reconcile(registry=registry, proposals=[outcome.proposal], reconciled_by=spelling)


def test_every_conflict_is_reported_at_once(registry: Any) -> None:
    """One round of repair, for the reason ``DesignPlanError`` collects its problems.

    The reconciler principal is configuration, so the fix is one edit; reporting one
    conflict per run would make a three-proposer panel take three runs to notice one
    mistake.
    """
    first, _ = _capture(registry=registry, principal="solo", revision=1)
    second, _ = _capture(registry=registry, principal="solo", revision=2)

    with pytest.raises(ReconcilerNotIndependentError) as caught:
        _reconcile(
            registry=registry,
            proposals=[first.proposal, second.proposal],
            reconciled_by="solo",
        )

    assert len(caught.value.conflicts) == 2
    assert "2 conflicts" in str(caught.value)


def test_discarding_your_own_proposal_does_not_excuse_you(registry: Any) -> None:
    """The loophole the check closes by running over the whole proposal set.

    If only the *retained* proposals were compared, a reconciler could satisfy the
    constraint by throwing its own proposal away -- and the record would then show a
    discarded proposal from the principal that reconciled it, which is the exact
    self-approval the ticket is about.
    """
    outcome, _ = _capture(registry=registry, principal="solo")

    with pytest.raises(ReconcilerNotIndependentError):
        _reconcile(
            registry=registry,
            proposals=[outcome.proposal],
            reconciled_by="solo",
            discarded=[DiscardedProposal(proposal=outcome.proposal, reason="It was wrong.")],
        )


def test_an_independent_reconciler_is_accepted_and_the_record_says_what_it_checked(
    registry: Any,
) -> None:
    """The limitation travels in the record, not only in the source.

    A reader of the stored document months later sees "independence enforced" and will
    read it as a verification unless the record says what it is. That sentence is the
    difference between an auditable record and a claim this system cannot back.
    """
    first, _ = _capture(registry=registry, principal="proposer-a")
    second, _ = _capture(registry=registry, principal="proposer-b")

    outcome, sink = _reconcile(registry=registry, proposals=[first.proposal, second.proposal])

    assert outcome.captured is True
    entry = sink.entries[0]
    assert entry.metadata[DESIGN_ROLE_KEY] == DESIGN_ROLE_RECONCILIATION
    assert entry.declared_by == "reconciler"
    assert "differs from all 2 proposer principal" in entry.rationale_text
    assert INDEPENDENCE_LIMITATION_NOTE in entry.rationale_text


def test_the_check_can_be_asked_before_a_plan_exists() -> None:
    """The same code, exposed, so a caller can ask the question earlier.

    Two implementations of a constraint is how one of them ends up being the one that
    ships, so this is a seam onto the enforced check rather than a second copy of it.
    """
    assert check_reconciler_independence(
        "reconciler", [DesignProposal(principal="proposer-a")]
    ) == ("different posting accounts",)

    with pytest.raises(ReconcilerNotIndependentError):
        check_reconciler_independence("proposer-a", [DesignProposal(principal="proposer-a")])


def test_what_the_check_cannot_do_is_stated_in_the_module_and_pinned(registry: Any) -> None:
    """The negative test: the promise this enforcement does not make.

    An ``agent_id`` is self-declared and never verified, so the check compares two
    strings the participants typed about themselves. It catches the accidental and the
    sloppy. It cannot catch an agent that wants to grade its own homework and only has
    to spell its name differently -- and the only honest answer to that is to say so
    where the next maintainer will read it, not to imply a guarantee the code does not
    have.
    """
    module_docstring = design_capture.__doc__ or ""
    assert "self-declared and never verified" in module_docstring
    assert "What it does not buy: anything about a determined self-approval" in (module_docstring)

    # And it is genuinely true of the code: two principals that differ as strings pass.
    outcome, _ = _capture(registry=registry, principal="proposer-a")

    assert check_reconciler_independence("proposer-a-2", [outcome.proposal]) == (
        "different posting accounts",
    ), (
        "the enforcement is self-assertion, and this test exists so nobody reads a "
        "stronger guarantee into it"
    )


# --- the plan is validated before anything is stored ------------------------------


def test_a_malformed_plan_is_rejected_with_its_structured_error(registry: Any) -> None:
    """Named problems, not a partial application.

    The cycle is the interesting fault: the document is well-formed field by field, so
    a caller catching only ``ValidationError`` would treat it as fine and go on to
    create tickets from a graph that cannot be ordered.
    """
    outcome, _ = _capture(registry=registry)
    cyclic = _plan()
    cyclic["tickets"].append(
        {
            "id": "second",
            "title": "Implement second",
            "description": "Depends on the other one.",
            "acceptance_criteria": ["Given a cycle, when ordered, then it is refused"],
            "priority": "P2",
            "depends_on": ["capture"],
        }
    )
    cyclic["tickets"][0]["depends_on"] = ["second"]

    with pytest.raises(DesignPlanError) as caught:
        _reconcile(registry=registry, proposals=[outcome.proposal], plan=cyclic)

    assert "cycle" in str(caught.value)
    assert len(_stored(registry)) == 1, "a refused plan must leave the ledger untouched"


def test_every_plan_problem_is_reported_not_the_first(registry: Any) -> None:
    """One repair pass, for the reason the schema module argues for it."""
    outcome, _ = _capture(registry=registry)
    broken = _plan(goals=[], decisions=[])

    with pytest.raises(DesignPlanError) as caught:
        _reconcile(registry=registry, proposals=[outcome.proposal], plan=broken)

    assert len(caught.value.problems) == 2


# --- the reconciliation records what it discarded ---------------------------------


def test_a_discard_is_recorded_with_its_reason_in_prose_and_in_metadata(
    registry: Any,
) -> None:
    """The half a reader most often needs, kept twice for two different readers.

    The prose reaches the stored document; the structured copy is what a query would
    filter on. They are written from the same sanitised string in the same call, so
    they cannot drift into disagreeing about what was discarded.
    """
    kept, _ = _capture(registry=registry, principal="proposer-a")
    dropped, _ = _capture(registry=registry, principal="proposer-b")

    discard = DiscardedProposal(
        proposal=dropped.proposal, reason="It assumes a Redis dependency this repo lacks."
    )
    outcome, sink = _reconcile(
        registry=registry, proposals=[kept.proposal, dropped.proposal], discarded=[discard]
    )

    entry = sink.entries[0]
    assert outcome.discarded == (discard,)
    stored = entry.metadata[DESIGN_DISCARDED_KEY][0]
    assert stored["principal"] == "proposer-b"
    assert stored["reason"] == discard.reason
    assert stored["entry_id"] == dropped.entry_id
    assert discard.reason in entry.rationale_text
    assert set(entry.metadata[DESIGN_PROPOSAL_IDS_KEY]) == {kept.entry_id, dropped.entry_id}


def test_a_reconciliation_with_nothing_discarded_says_so(registry: Any) -> None:
    """An empty section a reader has to interpret is worse than one that says it is empty."""
    _outcome, _ = _capture(registry=registry)
    kept, _ = _capture(registry=registry)

    reconciliation, sink = _reconcile(registry=registry, proposals=[kept.proposal])

    assert reconciliation.discarded == ()
    assert "none was discarded" in sink.entries[0].rationale_text


def test_a_discard_naming_an_uncaptured_proposal_is_refused(registry: Any) -> None:
    """A decision recorded against something nobody proposed is a false record."""
    kept, _ = _capture(registry=registry)

    with pytest.raises(ValueError, match="not among the proposals"):
        _reconcile(
            registry=registry,
            proposals=[kept.proposal],
            discarded=[
                DiscardedProposal(
                    proposal=DesignProposal(principal="ghost"), reason="It was wrong."
                )
            ],
        )


def test_a_discard_reason_is_required(registry: Any) -> None:
    """A discard with no reason is the decision this record exists to prevent losing."""
    kept, _ = _capture(registry=registry)

    with pytest.raises(ValueError, match="without a reason"):
        _reconcile(
            registry=registry,
            proposals=[kept.proposal],
            discarded=[DiscardedProposal(proposal=kept.proposal, reason="   ")],
        )


def test_an_oversized_discard_reason_is_refused_not_truncated(registry: Any) -> None:
    """A cut sentence argues the wrong way without saying so."""
    kept, _ = _capture(registry=registry)

    with pytest.raises(ValueError, match="at most"):
        _reconcile(
            registry=registry,
            proposals=[kept.proposal],
            discarded=[DiscardedProposal(proposal=kept.proposal, reason="x" * 1_001)],
        )


def test_the_record_names_the_plan_it_produced(registry: Any) -> None:
    """The seam a ticket-creation phase checks, and the reason this record is needed.

    The plan carries no provenance on purpose, so a reader of the reconciliation could
    not otherwise tell which document it was made from -- and a later phase creating
    tickets has to be able to refuse a plan no reconciliation ever described.
    """
    kept, _ = _capture(registry=registry)
    outcome, sink = _reconcile(registry=registry, proposals=[kept.proposal])

    digest = sink.entries[0].metadata[DESIGN_PLAN_DIGEST_KEY]
    assert digest == outcome.plan_sha256
    assert digest in sink.entries[0].rationale_text


# --- idempotency, against the real registry --------------------------------------


def test_recapturing_a_proposal_stores_nothing_twice(registry: Any) -> None:
    """The claim, not an invented key, is what says this."""
    first, sink = _capture(registry=registry)
    second, second_sink = _capture(registry=registry, text="Something else entirely.")

    assert second.entry_id == first.entry_id
    assert second.captured is False
    assert second_sink.entries == []
    assert len(sink.entries) == 1
    assert len(_stored(registry)) == 1


def test_reconciling_twice_records_one_reconciliation(registry: Any) -> None:
    """A retried approval must not produce a second judgement in the ledger."""
    kept, _ = _capture(registry=registry)
    other, _ = _capture(registry=registry, principal="proposer-b")

    first, sink = _reconcile(registry=registry, proposals=[kept.proposal, other.proposal])
    second, second_sink = _reconcile(registry=registry, proposals=[kept.proposal, other.proposal])

    assert first.entry_id == second.entry_id
    assert second.captured is False
    assert second_sink.entries == []
    assert len(sink.entries) == 1
    assert second.detail is not None and "already exists" in second.detail


def test_a_second_reconciliation_is_appended_rather_than_overwritten(registry: Any) -> None:
    """Revision 2, for the reason ``stable_design_reconciliation_id`` explains.

    The proposal set cannot be the discriminator because the registry's uniqueness index
    knows only the anchor columns -- so a second reconciliation of one topic by one
    principal has to take the next revision, and each record says what it covered so
    the two can be compared afterwards.
    """
    kept, _ = _capture(registry=registry)

    first, _ = _reconcile(registry=registry, proposals=[kept.proposal])
    second, sink = _reconcile(registry=registry, proposals=[kept.proposal], revision=2)

    assert second.entry_id != first.entry_id
    assert sink.entries[0].revises == first.entry_id
    assert len(_stored(registry)) == 3


def test_a_reconciliation_refuses_a_proposal_that_was_never_captured(registry: Any) -> None:
    """Proposals must be evidence, and a well-formed id proves nothing about that.

    Without this, a caller that mistyped a revision records a reconciliation whose whole
    provenance points at a record that does not exist, and nothing downstream can tell
    because the id is perfectly formed.
    """
    captured, _ = _capture(registry=registry)
    never_captured = DesignProposal(principal="proposer-b", model=MODEL)

    with pytest.raises(ValueError, match="not captured as records"):
        _reconcile(registry=registry, proposals=[captured.proposal, never_captured])

    assert len(_stored(registry)) == 1


def test_a_reconciliation_needs_at_least_one_proposal(registry: Any) -> None:
    """A plan with no argument behind it is a wish list, which the schema also refuses."""
    with pytest.raises(ValueError, match="at least one captured proposal"):
        _reconcile(registry=registry, proposals=[])


def test_the_panel_is_bounded(registry: Any) -> None:
    """The human gate reads this record once, so its length is the gate's strength.

    Refused rather than truncated, and not bounded by the plan's own
    ``MAX_PLAN_TICKETS``: the gate approves the reconciliation *and* the plan, and this
    is the half that says what was thrown away, which is the half nobody will read
    twice.
    """
    too_many = [
        DesignProposal(principal=f"proposer-{index}") for index in range(MAX_DESIGN_PROPOSALS + 1)
    ]

    with pytest.raises(ValueError, match=f"at most {MAX_DESIGN_PROPOSALS}"):
        _reconcile(registry=registry, proposals=too_many)


# --- refusals on the inputs, and the character policy -----------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"principal": "  "}, "must name itself"),
        ({"principal": "x" * 65}, "at most 64 characters"),
        ({"principal": f"open{RLO}code"}, "must not contain"),
        ({"model": " "}, "either stated or unstated"),
        ({"model": "m" * 129}, "at most 128 characters"),
        ({"topic": ""}, "name the topic"),
        ({"topic": "t" * 256}, "at most 255 characters"),
        ({"revision": 0}, "at least 1"),
    ],
)
def test_a_refused_identity_input_is_refused_rather_than_stored(
    registry: Any, kwargs: dict[str, Any], message: str
) -> None:
    """Configuration reaching a derived identity has to be bounded and clean.

    ``principal``, ``model`` and ``topic`` are all hashed, so the alternative to a
    refusal -- stripping the offending character -- would give two values that differ
    only by an invisible glyph one id and lose one of the records. There is nothing to
    re-fetch a principal to check it against, so there is no honest repair available.
    """
    with pytest.raises(ValueError, match=message):
        _capture(registry=registry, **kwargs)

    assert _stored(registry) == []


def test_a_proposal_of_nothing_but_invisible_characters_is_refused(registry: Any) -> None:
    """A record that counts as an argument and reads as nothing.

    Same rule as the rationale path and for the same reason: the emptiness is found
    after the policy has run, rather than by inspecting the characters a second time.
    """
    with pytest.raises(ValueError, match="character policy"):
        _capture(registry=registry, text=f"{RLO}{BELL}")

    assert _stored(registry) == []


def test_an_oversized_proposal_is_refused_rather_than_truncated(registry: Any) -> None:
    """A cut argument stored under its own principal is a different one."""
    with pytest.raises(ValueError, match="at most 8000 characters"):
        _capture(registry=registry, text="x" * 8_001)

    assert _stored(registry) == []


def test_a_hostile_proposal_is_stored_clean_and_says_so(registry: Any) -> None:
    """Strip and keep, the decision the whole character policy is built on.

    Dropping the record would satisfy "the control character is not stored" and lose
    the fact that a model was fed something hostile. The note is the half that makes it
    checkable: without it, silence would look like a fidelity claim.
    """
    _, sink = _capture(registry=registry, text=f"Use the cache.{RLO}{BELL}")

    entry = sink.entries[0]
    assert entry.rationale_text == "Use the cache."
    assert REFUSED_RE.search(entry.rationale_text) is None
    assert entry.metadata[SANITISATION_KEY] == (
        f"removed 2 characters before storing: U+0007 unnamed Cc (x1), "
        f"U+{ord(RLO):04X} RIGHT-TO-LEFT OVERRIDE (x1)"
    )


def test_rewriting_a_proposal_cannot_re_identify_a_record_already_stored(
    registry: Any,
) -> None:
    """Why a wider policy is safe on the prose and would not be safe on the identity.

    The id is derived from ``(repo, topic, principal, revision)`` and never from the
    text, so removing a character from the argument cannot move a document, orphan one,
    or produce a second record of one proposal.
    """
    _, sink = _capture(registry=registry, text=f"Use the cache.{RLO}")

    assert sink.entries[0].entry_id == stable_design_proposal_id(
        repo=REPO, topic=TOPIC, declared_by="proposer-a", revision=1
    )


# --- delivery outcomes ------------------------------------------------------------


def test_a_dead_lettered_proposal_releases_the_claim(registry: Any) -> None:
    """A claim left held by a delivery that will never happen is a proposal that
    cannot be retried until its lease expires."""
    outcome, _ = _capture(
        registry=registry,
        sink=FakeSink(
            KnowledgeDeliveryOutcome(
                entry_id="x", status=KnowledgeDeliveryStatus.DEAD_LETTERED, detail="refused"
            )
        ),
    )

    rows = _stored(registry)
    assert outcome.captured is True
    assert rows[0]["status"] == "retryable"


def test_a_sink_returning_nothing_is_uncertain_rather_than_delivered(registry: Any) -> None:
    """The claim still completes: an uncertain delivery is not a failed one."""
    outcome, _ = _capture(registry=registry, sink=FakeSink(None))

    assert outcome.delivery is not None
    assert outcome.delivery.queued is True
    assert _stored(registry)[0]["status"] == "completed"


# --- what a later phase has to recompute for itself -------------------------------


def test_the_plan_digest_is_a_function_of_the_document(registry: Any) -> None:
    """Deterministic over one document, and blind to everything else.

    This is the seam phase 3 uses to refuse creating tickets from a plan no recorded
    reconciliation describes, so the property it needs is that the digest identifies
    *the plan* -- not that it identifies the event. Two reconciliations of different
    panels that produced the same plan carry the same digest on purpose: the question a
    reader asks of the record is "which document was this a judgement about", and two
    panels agreeing is a finding rather than an id collision.
    """
    kept, _ = _capture(registry=registry)
    first, _ = _reconcile(registry=registry, proposals=[kept.proposal])
    second, _ = _reconcile(registry=registry, proposals=[kept.proposal], revision=2)
    different, _ = _reconcile(
        registry=registry,
        proposals=[kept.proposal],
        revision=3,
        plan=_plan(goals=["A different goal entirely"]),
    )

    assert first.plan_sha256 == second.plan_sha256
    assert first.plan_sha256 == design_plan_digest(first.plan)
    assert different.plan_sha256 != first.plan_sha256
