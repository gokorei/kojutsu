# Decision rationale: stating why, without capturing how

## What this programme is

Kojutsu captures Q&A: a question generated from a change, an answer harvested
from a comment. The answer is the *conclusion*. What is missing is the *reason*
that drove the implementation — why the change looks the way it does, decided by
the agent that wrote it and stated by that same agent once the work is done.

So this programme records a **stated reason for a decision**, per change, from the
principal that made the decision. Not a transcript. Not a reconstruction of intent
by a later reader. A declaration.

## What it deliberately does not capture, and why

The obvious adjacent feature is model-reported chain-of-thought: Anthropic
thinking blocks, OpenAI `reasoning_content`, opencode `reasoning` stream events.
All of it is available, and none of it is captured. Two reasons, either of which
would be sufficient.

**It is not a faithful account of anything.** Providers document this themselves.
Thinking blocks are text the model generates about its own process; they are not
a recording of that process, and they are frequently summarised or elided. Storing
them under a heading called "reasoning" would be a claim the store cannot back,
which is the specific defect class [the design review was written about](README.md).

**It is the highest-value prompt-injection persistence surface available.** Every
Kojutsu prompt carries untrusted pull request or ticket text. A model's
reasoning quotes that prompt back nearly verbatim. Capture that, and an attacker
who can open a pull request has laundered their own text into the knowledge store
carrying a delivery id and a capture timestamp — text that later agents will read
through `search_knowledge` as *review evidence from a trusted thread*. The
instruction is not a secret, so `_reject_likely_secrets` does not catch it. The
MCP evidence envelope (`_render_tanseki_document`) bounds the damage on retrieval but
does not undo the forgery at write time.

The existing behaviour is therefore correct and deliberate. `_extract_text` in
`src/kojutsu/integrations/opencode.py` keeps only `type == "text"` events and
discards the rest of the stream, and `tests/test_opencode_provider.py` pins that
for tool events. **A future change that widens that filter is a security
regression, not a feature.** Stating it here is what keeps a well-meaning
maintainer from "fixing" the omission.

What replaces it is not a mind-reading attempt. An agent can be *asked* what it
decided and why, and the answer is a claim — bounded, labelled, attributable, and
in the case of this system never a check on correctness.

## The three claims, each of which could be false

These are stated as claims rather than as documentation because the practice
adopted by [the review this directory came from](README.md) is *assert the claim
or drop it*. A limitation in a docstring is a comment; a limitation in a test
survives the next refactor and the next person who upgrades a word like "verified"
by one degree. Each has a corresponding test in `tests/test_provenance.py`.

### A rationale never raises a record's independence

An agent that wrote the code and then explains it is the textbook self-certified
case: same account, same model, and additionally the author. `compute_independence`
already classifies it that way and this programme does not touch it.

"I did it and here's why" is a claim about **intent**, not a check on
**correctness**. [`Independence`](../github-seam.md) already says its levels
describe only *who was positioned to disagree*, and this is where that sentence
earns its keep: the one record in the store with the strongest claim to know the
reason is the one record that has checked nothing. The most confident rationale is
the least verified, and the label must not move because the record is unusually
informative.

This would be wrong if a first-hand rationale were ever *combined* with a genuinely
separate party — a human confirming what the agent said it did. That is not a
rationale and does not raise the rationale's level; it is a second record, with
its own independence, and the two are stored separately.

### A rationale is never evidence

There is no provider delivery behind a stated reason. Nobody signed anything; the
forge verified only *who posted a comment*, never which model drafted its text —
the caveat that `AgentClaim` already carries in its own docstring. So a rationale
is permanently `asserted`, and `capture_anchor_gaps` returns no anchors for that
source, which means rationale records sit below every `min_independence` threshold
by default.

**That is the correct resting place, not a defect to engineer around.** A reader
who asked for independent evidence would rather see nothing than see a
self-declared reason and assume it had been checked. Excluding it is the honest
outcome, and — per the read-path rule in [`read-path.md`](read-path.md) — an
excluded result is counted and named, so "we did not show you any" is never
mistaken for "there was nothing".

### A stated reason and an inferred one are never the same record

Review is a **standalone step from intent**, and this is the decision that shapes
the storage model. An implementation produces two rationales from two principals at
two times, and they are stored as two records with two labels:

| | Implementation rationale | Review rationale |
|---|---|---|
| Principal | the agent that did the work | a reviewer, from the diff |
| Basis | its own session | inference over untrusted text |
| Value | first-hand | a genuine second opinion |
| Source label | `declared` | `reconstructed` |

Merging them would let a reconstruction stand in for a recollection, or the
reverse, and the reader could not tell which they had. Fusing them is the failure
this whole axis exists to prevent.

**Keeping them apart is what makes the interesting comparison possible.** When a
declared rationale says the retry was for idempotency and a reconstructed one says
it was for rate limiting, that gap is the finding: the intent is not visible in the
diff. It is also the only place the manufactured-consensus risk becomes measurable
rather than merely described — see [`github-seam.md`](../github-seam.md) on what an
unattended loop built on agreeable answers does. Two rationales that coincide
across principals on a change where they should differ is the signal to look for.

The comparison is not automatically a corroboration. Two rationales from the same
principal on the same model agreeing with each other is a restatement, and the
report says so rather than presenting it as a second opinion.

## What would have to be true for this to be wrong

- If declared rationales turn out to be as generic as reconstructed ones — if
  agents asked to justify their work produce fluent filler — then the first-hand
  label is a claim about provenance that the content does not support, and the
  programme has manufactured exactly the consensus it set out to avoid. The
  declaration clause is written against this (it must name rejected alternatives
  and omissions, and make "I am not sure" an expected answer), and the honest test
  is whether declared rationales carry information a diff-derived one could not.
- If the divergence between the two rationales is noise rather than signal, the
  comparison in the final group is a confident wrong answer, which is worse than no
  report. It is built last for that reason.
- If storing agent self-assertions at all is judged too much, the entire programme
  is unnecessary: the existing system records reviewer conclusions and does not
  need to be extended to be useful. What is lost is specifically the implementer's
  own account of a choice the diff cannot explain.
