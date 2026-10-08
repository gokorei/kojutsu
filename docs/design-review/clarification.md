# Clarifications: the statement nobody asked for

## What this record is

A review thread contains statements that answer no question. The pull request
owner says *"this is deliberate, the narrow window is the accepted cost for
v0.1"*. A reviewer says *"fair criticism, and it should be recorded as a known
weakness of this test."* Often the most valuable sentence in the thread, and
usually unprompted.

`KnowledgeEntry` cannot hold either. It requires `question_text`, and all six
`QuestionCategory` values are retrospective — `design_decision`, `trade_off`,
`edge_case`, `system_event` and the rest all name something decided. Forcing an
unprompted statement into that shape is the exact mirror of the bug that made a
rationale read as a conclusion: it produces a record whose structure asserts that
a question was asked and a category applied, when neither was.

So `ClarificationEntry` is a separate model with **no question text field at all**
and no category. The absence is the design; an empty string would not do, because
a reader could not then tell *"no question was asked"* (a fact about the record)
from *"the question text was lost on the way in"* (a defect in this system). The
model forbids unknown fields for the same reason: a caller porting a
`KnowledgeEntry` across has its `question_text` refused rather than silently
dropped, and a clarification stops being one at the moment someone attaches a
question to it.

## The trust runs opposite to a rationale's

This is the part worth stating twice, because the two records are neighbours and
their trust profiles are inverses of each other.

| | Rationale | Clarification |
|---|---|---|
| What it is | a stated reason for a decision | a quotation from a comment |
| Is there a question? | no | no |
| Delivery behind it | none | a signed provider delivery, or an authenticated API read |
| Checkable anchor | nothing | the comment id, re-fetchable by anyone |
| `capture_source` | permanently `ASSERTED` | `COLLECT` or `WEBHOOK` |
| What an empty field would mean | — | the reason for the empty `question_text` above |

A rationale is *permanently asserted* because nothing was signed. The forge
verifies who posted a comment, never which model drafted it or what it decided, so
there is no anchor that would make a stated reason checkable — see
[`rationale.md`](rationale.md).

A clarification has exactly the anchor the rationale lacks. It is a quotation of
real text, read from a real comment, attributed to a real account, carrying the
association the forge reported, and any reader can re-fetch the comment and check
the words. So it is held to `capture_anchor_gaps` exactly like an answer, and it
is stored as evidence. `ASSERTED` is *refused* rather than defaulted: a record
that quotes nobody is a typed claim, which is what `RationaleEntry` is for, and
letting this one be written as an assertion would give the same sentence two
different levels of trust depending on which model happened to hold it.

**A clarification is real evidence that simply has no question attached to it.**
That is the whole of it, and the sentence is worth repeating because the failure
mode is symmetrical and common: reading it as a conclusion, and reading it as
nothing. On the read path the first shows as a capture with an empty
"uncategorized" row, and the second shows as an "unverified" badge on a record
that anybody can check. Both are the model this record exists to prevent.

## Identity is the comment, not the text

`stable_clarification_entry_id` derives from `(repo, pr_number,
github_comment_id)` over a length-prefixed, domain-labelled preimage, and is
pinned to a golden value in `tests/test_clarification_record.py`. The comment is
the only part of the record a reader can re-fetch, so it is the only part that
can serve as an identity. Two consequences, both intended:

- **Reworded text on the same comment is the same record.** A digest over the
  text would make every rephrasing an unrelated record and orphan the one already
  stored. Comments get edited; the id does not move.
- **A different comment is a different record even when the text is identical.** So
  "two people independently said this" stays countable. A text digest would
  collapse them into one and lose the agreement — which is the finding.

Revisions are deliberately not modelled, unlike a rationale's. A comment is edited
in place, so the stored text is the text as it stood at capture time and the live
comment remains the record of truth. Appending a revision would imply the earlier
wording still existed somewhere to be superseded, and it does not.

## What is deliberately not claimed

- **A clarification raises no `independence` level.** `Independence` answers "who
  was in a position to disagree about a change", and a clarification has no asker
  to compare against — nobody asked. The record states no level, so the read path
  treats it as below every `min_independence` threshold, and counts the exclusion
  so that "we showed you none" is never mistaken for "there was nothing". That is
  the same resting place as a rationale, reached for a different reason: not
  because nothing is verifiable, but because the question that level answers does
  not apply.
- **A declared agent and model are self-assertions.** On exactly the terms as an
  answer: read from the comment's own marker, absent when the marker is, and never
  verified by anything in the platform. A bot's clarification is surfaced on the
  same read surface as a bot's answer, or the record is unreadable — a reader who
  cannot see a machine wrote it is reading a machine's claim as a person's.
- **A clarification does not stand in for the answer it was not an answer to.** It
  is stored beside the capture, in its own path namespace, under its own heading
  on the dashboard, and never on a question-category axis. A store that let the
  two interleave would let a reader attribute a stated weakness to a design
  decision, which is the confusion the whole programme exists to prevent.

## What would have to be true for this to be wrong

- If unprompted statements turn out to be mostly restatements of the diff rather
  than additions to it, the record is a transcript of a review thread, which is
  already the forge's job. The test is whether a clarification carries anything a
  diff and its comments could not have been read from without it — the deliberate
  trade-off, the accepted weakness, the reason nobody objected to either.
- If a clarification is allowed to become a general "note" field — a decision
  recorded after the fact, a status update, a to-do — then the retrospective
  category problem returns in a new place, because each of those *is* about a
  decision and belongs in a `KnowledgeEntry` with a question and a category. The
  test is whether the record still means "a person said this and nobody asked".
- If the agent marker on a clarification turns out to be a reliable identity, the
  caveat above is still correct but the cost of the restriction falls. It should
  not be relaxed on that basis: the marker is written by the account being
  described, so reliability is a property of the bot, not of the platform.
