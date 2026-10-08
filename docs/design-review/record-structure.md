# Record structure: an inferred pairing is never a captured one

## The gap

`CaptureSource` is the trust axis and it answers one question: **where did this
text come from.** `webhook` means a signed provider delivery, `collect` means an
authenticated read of the provider API, `asserted` means someone typed it in.

It is a good axis, and it is not enough. It cannot also answer **was this
structure established or inferred.** Those are different questions, and until
inferred question/answer pairings and clarification records exist nothing has
needed the second one.

The moment something can pair a question with an answer, the record is ambiguous
in a way the store cannot express. It can have a real delivery id, a real comment
id, a real capture timestamp — every anchor `CaptureSource` asks for — and still
be a record that *nobody asked*: a model matched the two sides. Served through
`search_knowledge`, that record is a row of text, and a row of text is what a
genuine conversation looks like too. A reader choosing which of the two to trust
has nothing to choose on.

The failure is not that the inferred record is stored. It is that it is
**indistinguishable**. The same shape a real capture has, in the only field the
reader was told to consult.

## The decision

A second, independent axis on `KnowledgeEntry`: `structure`, with two values.

| | `anchored` | `inferred` |
|---|---|---|
| Means | a question was asked and an answer captured, and the record joins them | a model inferred the pairing |
| Checkable | no — but the record says so | no — and the record says so |
| Requires | nothing | the model that inferred it |

`inferred` is refused at construction unless the record names the inferring model,
because an unattributed guess is exactly what this axis exists to distinguish from
a capture: the label would be there and the reason for it would not.

## The default is the strict one, for the same reason `CaptureSource` defaults to `asserted`

An unmarked record reads as `anchored`. That looks like the permissive choice and
is not, because the burden falls where the evidence is: a producer that inferred a
pairing must say so, and nothing has to be proved by a producer that did not
infer one. Defaulting the other way would require every ordinary capture to name
an inferrer, and reading `inferred` as the default on the read path would label
every document written before the axis existed as a guess — which is false, since
nothing in the capture path could infer a pairing then.

What the default does **not** do is detect an inference that was never labelled.
Nothing can, and any claim that it does would be the over-claim this whole
directory exists to prevent. The axis makes the *declared* case readable; it does
not make the undeclared case detectable. Stated as a test
(`test_an_inferred_record_is_refused_without_the_model_that_inferred_it`, and the
`structure_of` resolution tests) so it cannot quietly become a promise.

## What it does not do, and why that is the point

**It is not on `RationaleEntry`.** A rationale is a statement, not a pair. There is
no question and answer for a model to have matched up, so a `structure` field
there could only ever hold one value — and a field that can only hold one value
invites a reader to believe the thing has a structure to declare at all. Whether
a rationale's reasoning was declared or reconstructed is already `source` on that
model, and that is the honest place for it.

**It does not borrow the capture axis's anchors.** An inferred record is refused
no delivery id and no comment id. It is a different claim about a different thing,
and if it had to claim a capture it did not have, the two axes would start standing
in for one another — the outcome both were separated to prevent.

**It is not written onto records that already exist.** `anchored` is omitted from
the frontmatter rather than written out, because every record in the store today
is anchored and stamping the label on all of them would re-identify stored
documents to add a claim their writers never made
([`identity-and-limits.md`](identity-and-limits.md) records the same rule for
identity derivations). Absence resolves to `anchored` on read, so the value that
is omitted is exactly the value a reader recovers. A golden test pins the bytes of
a real anchored record, before and after, so the next person who decides
"provenance should always be written" finds the argument rather than a diff.

**It does not detect inference.** See above. The honest form of this feature is
"an inferred record cannot be read as a captured one", not "no inferred record is
ever stored".

## Reading it back

`search_knowledge` grew `anchored_only`, alongside `min_independence`, and it
behaves the way that filter does: exclusions are counted and named, never dropped
in silence, because "nothing here mentions this" is a much stronger claim than the
store supports ([`read-path.md`](read-path.md)). The provenance block states the
structure and names the model in the same place the agent is already reading, and
two anomalies cover the read-time cases the write-time validator cannot: a
structure nobody can read, and an `inferred` naming no model.

Unrecognised values are excluded by `anchored_only` rather than defaulted. A filter
whose purpose is protecting a reader from a misread should fail towards showing
too little.

## What would have to be true for this to be wrong

- If inferred pairings are never actually produced — if every question/answer
  pairing really does come from a thread somebody answered in — then this is a
  field with one reachable value, and the honest response is to delete it rather
  than to keep it as a placeholder. The same applies to
  `clarification records`: if they are always anchored, they do not need the axis.
- If the inferring model is not a meaningful unit — if the same name covers
  several deployments, or a name can be reused across a retrieval boundary — then
  naming it satisfies the letter of the rule and none of its purpose. A reader
  would be able to see a guess and still be unable to weigh it.
- If requiring the model name causes producers to record `unknown` (as
  `declared_by_model` already can) then the rule is a naming ceremony rather than
  a constraint, and the value it adds is close to the value of the `inferred` label
  alone. That is a measurable outcome, not a hypothetical one: the count of
  `unknown` names on inferred records is the number to watch.
- If a reader is never given the option to filter, the label is decoration. This
  is the part of the claim that is easiest to lose in a refactor — a badge nobody
  can filter on, and a filter nobody counts, look identical from the code.
