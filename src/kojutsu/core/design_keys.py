"""The vocabulary of the design phase's record metadata, in one place.

Six string constants: four metadata keys and the two roles. They name what a design
proposal and a design reconciliation carry in
:attr:`~kojutsu.models.RationaleEntry.metadata`, and two modules need those names --
:mod:`kojutsu.core.design_capture`, which writes them, and
:mod:`kojutsu.core.tanseki_mapping`, which projects them onto stored frontmatter.

**This module exists because the dependency cannot run either of those ways.**
``design_capture`` owns the claim/store tail through ``knowledge_sink``, and
``knowledge_sink`` imports ``tanseki_mapping`` for the upsert payloads -- so
``tanseki_mapping`` importing ``design_capture`` would close a cycle
(``tanseki_mapping`` -> ``design_capture`` -> ``knowledge_sink`` ->
``tanseki_mapping``), and ``design_capture`` importing ``tanseki_mapping`` would do
the same in reverse. A third spelling of the keys inside ``tanseki_mapping`` would
avoid the cycle and create the drift these constants exist to prevent: two spellings
of one key is a document that filters on one of them and silently misses the other.

So the keys live here, importing nothing from either consumer, and both import from
here. A future key joins this list rather than being spelled at its two use sites,
for the same reason.
"""

from __future__ import annotations

#: Metadata key naming which of the two design record kinds this is.
#:
#: A reader of the ledger has to be able to tell a proposer's argument from the
#: judgement made about it. ``declared_by`` alone does not: both are
#: :class:`~kojutsu.models.RationaleEntry`, both are stated by a principal, and the
#: one that reconciled is not distinguished from the ones that were reconciled by
#: anything structural. The prefix on the id separates them to a reader who is
#: already looking at the id and to nobody else.
#:
#: **This key now reaches the stored document**, projected by
#: :func:`kojutsu.core.tanseki_mapping.build_rationale_frontmatter` -- that loss is
#: what used to be recorded here, and the projection is what closed it. The local
#: half of the story is unchanged: ``rationale_captures`` stores no metadata, so
#: :func:`~kojutsu.core.design_capture.find_design_reconciliations` still separates
#: the two record kinds by :data:`~kojutsu.core.design_capture.RECONCILIATION_ID_PREFIX`
#: on the id. The frontmatter is what reaches the *store*, not the registry, and
#: the prefix stays load-bearing until the registry grows a metadata column.
DESIGN_ROLE_KEY = "design_role"

#: Metadata key holding the plan a reconciliation produced, as a digest of the plan
#: document. See :func:`kojutsu.core.design_capture.design_plan_digest`.
DESIGN_PLAN_DIGEST_KEY = "design_plan_digest"

#: Metadata key holding the entry ids of the proposals one reconciliation covered.
DESIGN_PROPOSAL_IDS_KEY = "design_proposal_ids"

#: Metadata key holding, per discarded proposal, its entry id, the principal that
#: made it and the reason it was not taken.
DESIGN_DISCARDED_KEY = "design_discarded"

#: The two roles, as the strings written under :data:`DESIGN_ROLE_KEY`. A pair of
#: constants rather than an enum because nothing filters on these yet and a closed
#: vocabulary this phase cannot enforce from the read side is a set with a
#: documentation obligation attached to it.
DESIGN_ROLE_PROPOSAL = "proposal"
DESIGN_ROLE_RECONCILIATION = "reconciliation"

__all__ = [
    "DESIGN_DISCARDED_KEY",
    "DESIGN_PLAN_DIGEST_KEY",
    "DESIGN_PROPOSAL_IDS_KEY",
    "DESIGN_ROLE_KEY",
    "DESIGN_ROLE_PROPOSAL",
    "DESIGN_ROLE_RECONCILIATION",
]
