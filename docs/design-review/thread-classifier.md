# Thread classification: the third option is the design

## The gap

GitHub's reply tree is not a record of what a comment responds to. In a five-deep
thread the fourth comment is often about the first, and the human replying to
another human frequently does not know which comment they are answering either —
that ambiguity is the normal condition of review, not a defect in the review or in
the tool.

So a thread has to be read, and read by something that did not attend the review.
On a repository where Kojutsu never ran there are no `kojutsu:question:`
markers at all, which means the only evidence available is the text — and a model
given the text has to decide, for every comment, one of three things:

1. it answers an earlier comment,
2. it is a standalone clarification nobody asked for, or
3. it relates to nothing and should be dropped.

Nothing in the store had a place to put the answer. `KnowledgeEntry` requires a
`question_text` under one of six retrospective categories, so option 2 was the bug
fixed by [`clarification.md`](clarification.md) and option 3 had nowhere to go at
all: a comment that is not worth holding was stored anyway, because there was no
outcome that said so.

## The decision

The third option exists, and a classifier is allowed to use it. A comment the
model places alone with no question attached is one of two things, and the store
can tell them apart: a clarification, which is a real quotation worth holding; or
unrelated, which produces **no record at all**.

**The uncomfortable part is that this makes the numbers worse.** A classifier that
must emit a pair for every comment produces the tidier, larger, more impressive
result. It is also wrong in exactly the way this system exists to prevent: a
fabricated question attached to a real person's real answer is the most persuasive
unverified record the store could hold, because the answer text is genuine and
nothing downstream looks wrong. The expected yield on a real thread is therefore
*fewer* pairs than forced pairing, more clarifications, and a tail of unrelated
comments — and that distribution is the honest one.

`ThreadCoverage` carries the counts that show when it has gone wrong:
`declined_pairs` and `unrelated` above all. A run where every unanchored comment
landed in a pair and nothing was dropped is a run that forced its pairs, and those
are the two numbers a reviewer should look at before believing a result.

## Two hard rules, both enforced in code

**An existing anchor always wins.** A comment carrying `kojutsu:question:`, and
an answer carrying its `kojutsu:answer:` marker, are resolved by the existing
anchored path and removed from the model's input *before it is called*. Showing a
model a correct answer and asking it to infer the question produces an inference
we did not need and did not want: the model copies the question it can see, and the
store then holds a copy presented with the confidence of a marker. A test asserts
the anchored comment ids and bodies are *absent from the prompt*, not merely
ignored in the response, because those are different bugs.

**The model must be able to decline.** A pairing claimed below
`MIN_PAIR_CONFIDENCE`, one pointing backwards in thread order, and one where a
comment answers itself are all refused here rather than in the prompt, because a
model asked to be careful is careful until it is not and the failure it produces is
the persuasive one. A refused pairing releases **both** comments it named to stand
alone, with the reason attached. That is deliberate: a refusal is a statement about
the relationship between two comments, not about either comment's worth keeping,
and dropping them would turn the honest outcome into a silent loss.

## Three things it will not do quietly

- **It will not store a partial classification.** The union of pair and single
  comment ids must equal the input set; a comment id the caller never supplied is a
  model error, not a result. A comment silently dropped looks identical to one
  nobody read, which is why the check is at the boundary rather than in a comment.
- **It will not go around the sandbox.** The call goes through `complete_task`, so
  attacker-controlled comment bodies reach the provider through the adapter where
  the opencode sandbox lives. A direct litellm call would put a stranger's PR
  comment into an unsandboxed process.
- **It will not present a stochastic answer as a deterministic one.** Every result
  names its model and reports itself as a single sample, and
  `compare_classifications` is how two samples are held against each other.
  Measuring many is sibling ticket GD4A8B25's job; this module only records what
  ran.

## The narrower vocabulary, again

`_ALLOWED_CLASSIFIER_CATEGORIES` is `{design_decision, trade_off, edge_case}` —
identical to the rationale allowlist and for the same reason it is narrower than
`_ALLOWED_QUESTION_CATEGORIES`. `DOMAIN_KNOWLEDGE` and `DEPENDENCY` are excluded
because they invite the model to volunteer a domain fact in order to fill the slot,
and here that fact would be filed as **the question a real person asked**. A
category the classifier may not name is a class of fabrication the store cannot
hold.

This is the third place a vocabulary has been narrowed for a specific producer
(`SYSTEM_EVENT` for questions, then the rationale allowlist, then this one). The
pattern is the point: not every category in the vocabulary is one a model should
be permitted to assert, and a test states each narrowing so a later widening is a
decision rather than a drift.

## One thing that differs from the diff path, on purpose

`build_questions_prompt` *refuses* a whole batch over one likely credential in the
diff. A thread is a dozen strangers' comments, and refusing there would let any one
commenter make every other thread on the repository unclassifiable by typing a
token-shaped string — a denial of capture handed to the least privileged account
in the conversation. So thread bodies are redacted and the batch is kept. The
property that matters, that the credential does not leave the process, still holds.

## The record it produces

An inferred pair becomes a `KnowledgeEntry` with `structure: inferred`, the
inferring model in `metadata`, and the reconstructed question labelled as
**RECONSTRUCTED — NOBODY ASKED THIS** in the document *body*. The body label is the
part that is easy to skip: frontmatter is a header a reader skims past, and a
question copied out of this document into a ticket or a design note carries only
the prose. The label has to travel with it.

The capture axis is set independently. The answer half genuinely was read from the
forge and the comment id is in the metadata, so a reader can re-fetch the
quotation — and the pairing still was not established by anybody. The two axes
answer different questions and are set by different evidence, which is what
[`record-structure.md`](record-structure.md) separated them for.

An anchored pair is never stored here. The markers already put it through the
existing answer path, and a second write would be two documents for one person's
words.

## What would have to be true for this to be wrong

- **If the model pairs almost everything, the third option is decoration.** The
  `unrelated` and `declined_pairs` counts are the check, and a run where they are
  both zero across a real corpus is evidence the task clause has been softened or
  the confidence floor is too low. Raise the floor before loosening the prompt.
- **If a confidence floor produces systematically more clarifications than
  anyone wants**, the floor is miscalibrated for the model in use. The number to
  watch is the declined-pair count against the inferred-pair count; a ratio near
  one is a classifier that has stopped pairing rather than one that is careful.
- **If comment ids alone turn out to be too little context** — and on a busy
  thread they may be — the fix is more context in the prompt (author, timestamp,
  reply structure), not a second pass. Two passes would make the result a
  function of a chain of samples and would hide the variance this module exists to
  expose.
- **If truncation ever matters**, a thread clipped at 12,000 characters is being
  classified from partial text and a model can easily misread a half-comment. The
  truncation is marked in the prompt for that reason; if it starts showing up in
  results, the bound is the thing to fix rather than the mark.
- **If an inferred pairing is never actually read as a capture**, the `inferred`
  label and the body notice are doing work that a filter would do better. What
  would falsify it is a reader quoting the reconstructed question without the label
  — which is exactly why the label is in the body and not only in the frontmatter.

## What a real run looks like

### On the fixture thread, with canned responses

Three runs of the ten-comment fixture thread through the sandboxed provider, as
evidence that the shape is the one the design predicts and not the one forced
pairing produces:

| outcome | count |
|---|---|
| anchored pairs (markers, model never saw them) | 1 |
| inferred pairs | 2 |
| clarifications | 2 |
| unrelated, no record | 2 |
| declined pairs | 0 |
| comments unaccounted | 0 |

Two pairs out of eight unanchored comments, and a quarter of the thread dropped.
The `declined_pairs` count is zero here because the fixture's two pairings are
genuine and the model was confident about both — the decline path is proven by a
canned response in the tests rather than by hoping a model hesitates.

The three runs agreed on every outcome (`stability: 1.00`) while the confidence
numbers moved (0.92, 0.92, 0.93 on one pair; 0.92, 0.95, 0.92 on the other). So
even here the *numeric* confidence is noisier than the structural decision.

### On the real thread, against the pairings the store holds

`scripts/eval_thread_classifier.py`, six samples of
`opencode/model` in two independent three-run sets over the
twenty-seven-comment thread of `acme/widgets#1`. The ground truth is
the eleven anchored records read back out of the Tanseki collection, not a fixture, so
the evaluation cannot drift from what the capture path produced. The second set is the
recorded run and is in
[`thread-classifier-baseline.json`](thread-classifier-baseline.json); re-take it with
the command in the section below.

**The finding that came first is about this module rather than the model: the
classifier cannot classify this thread at all.** The thread is 13,608 characters.
`MAX_THREAD_CHARS` is 12,000. `build_thread_prompt` clips it, marks the clip in the
prompt, and `classify_thread` then requires the model to account for *every*
comment including the two that were never sent to it — comments `5894503310` and
`5894503555`, which are the answers to two of the eleven established pairs. The run
is refused, and the refusal reads `the model did not account for comment id(s)
5894503310, 5894503555`, which charges the model for a bound this code applied to
itself.

That is the "if truncation ever matters" case above, and it has started happening.
**The bound is the thing to fix, not the mark** — but it is a behaviour change to
the classifier, so it is not made here. A measurement that fixed the thing it
measured and then reported the result would be reporting a number about a
classifier that had just been tuned to suit the data.

`MAX_THREAD_COMMENTS` is 40 and `MAX_THREAD_CHARS` fits about twenty-four comments
of this length, so the two bounds disagree with each other: the module claims a
forty-comment ceiling it cannot prompt for. **That inconsistency is the defect**,
and the numbers below are on a 25-comment input with a recall ceiling of 9 of 11,
not on this thread.

#### The recorded run, on the 25 comments the classifier can accept

| outcome | run 1 | run 2 | run 3 |
|---|---|---|---|
| inferred pairs | 8 | **refused** | 8 |
| clarifications | 9 | — | 1 |
| unrelated, no record | 0 | — | 8 |
| declined pairs | 0 | — | 0 |
| conflicts (a comment placed twice) | 0 | — | 3 |
| established pairings recovered | 0 | 7 | 7 |
| precision against the anchored set | 0.875 | — | 0.875 |

- **Recall 0.42 mean**, counting the refused run as the zero it is — 0.64 and 0.64
  of eleven in the two runs that answered, 0.78 of the nine the input made reachable
  in each. One run of three was refused outright: *the model did not account for
  comment id(s) 5861225138*. A refusal is a run that produced nothing, and scoring it
  as a run which happened to recover nothing would be reporting a coverage failure as
  a model result.
- **Precision 0.875** in both runs that answered, and one mismatched pairing per run,
  named below.
- **Pairwise stability 0.68**, the only comparable pair — a refused run has no
  placements to hold still — and 8 of 25 comments placed differently. This is the
  opposite of the fixture's `1.00`.
- **The decline path is exercised, but rarely.** `declined_pairs` was 0 in every run
  of this sample set and 2 in one run of the other, so a real model has crossed the
  `MIN_PAIR_CONFIDENCE` floor on this thread exactly once in six attempts. That is
  evidence the path works, and nowhere near enough of it to call the threshold
  calibrated.

#### The variance is larger than the number

Two independent three-run sample sets, same thread, same model, same input. Both are
in the repository; the second is the recorded baseline:

| sample | inferred pairs per run | recovered per run | precision per run | recall of the 9 reachable |
|---|---|---|---|---|
| A | 6, 6, 8 | 4, 4, 7 | 0.667, 0.667, 0.875 | 0.44, 0.44, 0.78 |
| B (recorded) | 8, refused, 8 | 7, 0, 7 | 0.875, —, 0.875 | 0.78, —, 0.78 |

Across six samples: **recall of the reachable set 0.44–0.78, precision 0.667–0.875,
inferred pairs 6–8, clarifications 1–13, unrelated 0–10, one run refused outright,
and `declined_pairs` non-zero in exactly one.** Pairwise stability within a set was
0.32, 0.56 and 0.60 in A and 0.68 in B, so two runs of the same thread on the same
day agreed on as few as 8 of 25 comments. The mean recovered per set is 5.00 and
4.67 — a difference of a third of a pairing, which is smaller than the difference
between the best and worst run inside either set.

**The honest reading is that this classifier has no accuracy figure.** Reporting the
recorded set's 0.42 without the other set would be a number with a six-sample spread
behind it of 0.44 to 0.78 on recall and 0.32 to 0.68 on stability, and the spread is
the finding. The recorded set is in the baseline because a re-run has to have
something to compare against; it is not a claim about what the classifier does.

Two things this measurement cannot settle, both because one thread of one reviewer
will not support them: whether the low stability is the classifier or the thread —
every question here was asked by the same person who wrote the answers, which is the
hardest case for pairing and an easy one for a model to bluff — and whether 0.78 recall
of the reachable set is good or bad, since nothing here says how many of the nine
pairings a human reading the thread cold would have reconstructed.

#### Which comments flip, and why that is useful

Eight of twenty-five in the recorded run, and 18 of 25 across the six. They are not
random. Two clusters:

- **The questions the store holds as the anchor of a real record.** `5860835793`
  and `5860835876` are a clarification in one run of A and a pair in the other two.
  `5886792798` is a clarification, an unrelated and a pair across three runs. These
  are the same comments that appear in the contradictions list: the model is not
  sure whether a comment in this thread is half of a conversation or a remark, and
  the thread gives it a great deal of latitude to be wrong either way, because every
  question was asked by the same person who wrote the answers.
- **The long bot-written review findings in the last third of the thread**, several
  thousand characters each. `5891489025` is the answer to a real question in the
  store and was a pair in one run, an unrelated in another, and a contradiction in a
  third.

#### The unrelated list, and my judgement on it

Eleven distinct comments across the six runs, no record produced. Full text is in the
script's output and the baseline; what matters is which ones the store already
holds:

| comment | what the store holds | my judgement |
|---|---|---|
| `5886792079` | the question anchoring a stored record | **bad drop** |
| `5886792333` | the question anchoring a stored record | **bad drop** |
| `5886792547` | the question anchoring a stored record | **bad drop** |
| `5886792798` | the question anchoring a stored record | **bad drop** |
| `5886793012` | the question anchoring a stored record | **bad drop** |
| `5891488763` | half of a stored record, as its answer | **bad drop** |
| `5891489025` | half of a stored record, as its answer | **bad drop** |
| `5894503097` | half of a stored record, as its answer | **bad drop** |
| `5860835972` | nothing — an unanswered question | defensible |
| `5860836186` | nothing — an unanswered question | defensible |
| `5861224811` | nothing — an unanswered question | defensible |

**Eight of the eleven are comments the store already holds as parts of real records.**
Five are the questions that anchor a stored Q&A and three are stored answers. In the
marker-blind configuration — the configuration for a repository Kojutsu never
ran on — calling them `unrelated` means the store would gain nothing from a comment
that already anchors something, and the pairing those records rest on would have no
counterpart in the new classification.

**The classifier is too eager to prune on this thread, and a precision figure cannot
show it.** A drop is not a fabrication, so precision is blind to it by
construction: the recorded set's precision of 0.875 says nothing about the eight
records it would have thrown away, and neither would a precision of 1.00. The task
clause's rule 1 — *prefer `unrelated` over a pair you are not sure about* — is doing
more work than it was written to do. The model reaches for `unrelated` to express
uncertainty about whether a comment belongs to a conversation at all, and a
clarification would have recorded the text instead. That is a prompt problem, and the
fix the design
already names is in the wrong direction: the clause should say that `unrelated` is
for a comment that is about nothing, not for a comment whose relationship to the
thread is unclear.

#### The contradictions, individually

Eight entries across six runs, which is three distinct pairings. They are not eight
problems:

1. `5860836084 → 5860841941`, in **five of the six runs**. **A false alarm in
   substance, a real one on paper.** Question `29ddd2e7` was answered twice in this
   thread — once by the human (`5860841941`) and once by the agent (`5860877070`) —
   and the store holds only the agent's. The model paired the question to the *other*
   real answer, which is a defensible reading of the text. It is counted against
   precision because the harness scores against what the store holds, and that is the
   right default: the store's silence about a second answer is a gap in the truth set,
   not a fabrication by the model. It also means **the truth set is incomplete**, the
   real ceiling is below 11, and the single most persistent pairing in this
   measurement is the one the scoring most likely punishes unfairly. Any precision
   figure here is therefore a floor, not a central estimate.
2. `5886792798 → 5891489025` (one run, on two counts). **A real contradiction.** The
   store holds `5891489025` as the answer to `5886792547` and `5894503555` as the
   answer to `5886792798`. The model put a stored answer under a different stored
   question.
3. `5861224811 → 5861229111` (one run). **A real contradiction.** `5861229111` is the
   stored answer to `5861225037`. This pairing attaches a real answer to the wrong
   question, which is the failure the whole design exists to prevent.

So: **two real contradictions across six runs, both of the same shape** — a real
answer moved onto the wrong question. Neither is a hallucinated comment id, and
neither is a reversed pair; the model is reading real text and attaching it to a real
but incorrect question. That is a more tractable failure than inventing a question,
and it is still a fabricated question attached to a real person's real answer.

The frequency is the part worth keeping. One pairing recurred in five of six runs
while the two genuine contradictions appeared once each, so the most *consistent*
thing this classifier does on this thread is the one the truth set cannot judge
fairly — and the errors a reader would actually be harmed by are the rare ones.

### A second defect the measurement found

`ThreadCoverage.accounted` used to add up its counters, and one anchored question
answered by two comments resolved to two pairs sharing a question end. The thread
was then one comment *over*-accounted, `complete` was false, and
`classify_thread` **refused a classification of a thread the markers accounted for
completely** — naming its own invariant as the fault. This is the real
`widgets#1` thread, not a synthetic case. `accounted` now counts the distinct
comments placed, which is what "exactly once" was always about, and the check is
made against the set that was placed rather than a figure the run derived about
itself. Stated as a test
(`test_a_question_answered_twice_is_still_one_comment_accounted_for_once`) so it
cannot quietly come back.

## What these numbers can and cannot stand for

One thread. One repository. One reviewer — one author writing to himself, with no
disagreement between parties, which is the least ambiguous review conversation the
classifier could be pointed at. One model, `opencode/model`, six times,
in one configuration, in two sample sets that do not agree with each other.
Twenty-five of twenty-seven comments, the other two clipped by a bound in this
module. Scored against eleven pairings, which is an incomplete truth set holding no
record of any comment that correctly has no pairing, and which contains at least one
pairing it cannot see because the question behind it was answered twice.

**So: this is a data point, and a weak one.** Not a precision figure for the
classifier, not a recall figure for the classifier, and not evidence about
`widgets`. Across six samples the classifier's recall of the reachable set
ranged from 0.44 to 0.78, its precision from 0.667 to 0.875, and its stability
between two runs of the same thread from 0.32 to 0.68 — so a reader who wants a
claim about this classifier needs a corpus: many threads, several authors, real
disagreement, an adjudicated answer for *every* comment rather than eleven, and
enough repetitions per thread that the between-sample spread is smaller than the
effect being measured. None of that exists yet, and the honest statement is that the
classifier's accuracy on a real thread is currently unmeasured.

The numbers are worse than the fixture's on every axis, which is the point of running
them. The fixture reported 2 pairs, 2 clarifications, 2 unrelated and `stability:
1.00` from a ten-comment thread with canned responses; the real thread produces 6 to
8 pairs, 1 to 13 clarifications, 0 to 10 unrelated, one refusal in six, and a
stability that varies with the sample. **The fixture's shape was not evidence about
anything.**

## Re-running it

```bash
GITHUB_WEBHOOK_ALLOWED_REPOSITORIES=acme/widgets \
  uv run python scripts/eval_thread_classifier.py \
    --mode marker-blind --measure-what-fits
```

Needs the opencode CLI and its credential, and the Tanseki store answering on
`localhost:8099`. Not a test, on purpose: a network call in the suite is a call CI
skips, so it would go green forever without ever running and the number in this
document would be the number from whichever run somebody pasted in. `--compare`
diffs against the baseline and exits non-zero on a regression; it refuses to compare
two measurements whose inputs differ, because a changed denominator produces a delta
that looks exactly like a regression. The regression table carries a *direction*
rather than a bare list of metrics, because a rise in invented pairings is a
regression and a fall is an improvement, and a harness that cannot tell those apart
ends up rewarding a classifier for inventing more.

Default mode is `marker-blind`, and the reason is in the module docstring: every
comment on this thread carries a marker, so `--mode as-captured` hands all
twenty-seven to the anchored path, the model is never called, and `model_calls` is
0. That run is worth taking — it is how the fact that the production path never
reaches the model on a fully marked thread stays visible — but its recall over every
pairing held is 1.00 (a dict looking itself up) and its recall of the model's own
pairings is 0.00. It also reports one disagreement between the anchored path and
the store: the anchored path resolves `5860836084 → 5860841941`, for which no
document exists, because that question was answered twice.

