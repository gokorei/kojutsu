# Chapters and prologue: spending the question budget on the change

## The gap

Question generation used to be one prompt over one diff. `build_questions_prompt`
carried the whole diff, truncated at `MAX_DIFF_CHARS`, and asked for one to six
questions. `_max_questions_for_diff` sized the budget from the diff length and the
file count, but nothing decided **where** the questions landed.

That is a coverage defect disguised as a sampling one. A diff is ordered by
filename, so the first fifteen thousand characters are not a representative
sample of the change — they are the part git sorted first. A lockfile bump and a
vendored directory compete with the decision the reviewer needed to record, and
the questions that come back cluster wherever the budget happened to be spent.
The reviewer receives six isolated questions about a forty-file change with no
statement of which part each belongs to, and no way to see that whole areas were
never asked about.

The pull request description was fetched and then discarded. `_fetch_pr_context`
returned the PR, `_resolve_jira_key` used its body only to regex a ticket key, and
the author's own account of what the change was for never reached the model.

## The decision

Two pieces of orientation, both supplied before any question is asked.

**A chapter map.** `kojutsu.chapters` splits the unified diff at its
`diff --git` headers, groups the files by the directory that names them, and
excludes support files: lockfiles, generated and vendored output, binaries, and
pure renames. Each reviewable group becomes a chapter (`c1`, `c2`, …) with its own
share of the diff budget. Support files are named separately in the prompt with
the reason they were excluded, so their absence from the chapters is visible.

**A prologue.** The pull request title and body, plus the Jira fields already
fetched, are rendered as the author's account of the change. The prompt says
explicitly that the prologue is not itself a subject for a question; it is there
to help decide which chapters carry a decision worth recording.

The question budget is then split across chapters. `_max_questions_per_chapter`
allows one per chapter by default, and a second only when the chapters are at most
half as numerous as the questions. The cap is requested in the prompt and enforced
again in the parser, because an instruction is a preference and the parser is the
guarantee.

## Why the chaptering is deterministic

Chapter names are derived from paths. Asking a model to name the chapters — which
is what a story-mode narrative does — would produce a more pleasant result and a
less auditable one. The grouping would differ between runs, and a chapter that
silently swallowed half the diff would be indistinguishable from one that did
not. A path group can be recomputed from the same diff and checked.

Path grouping is also the only part of this that can be wrong in a way a test can
show. `chapter_title` is a small function with named inputs, and the support
classification is a list of names, suffixes, and diff markers, each with a reason
attached. Neither claims to understand the change.

## Why a chapter with no question is reported

The parser attaches each question to the chapter it names. When at least one
question is anchored, the generator warns about every reviewable chapter that
received none. That is the same rule as the read path: an answer shortened by a
bound must be visible as shortened, because a reader's conclusion from a truncated
result is much stronger than the evidence supports.

Nothing is reported when every question arrived unlabelled. Coverage of that
response is unknown, and listing every chapter as uncovered would state a fact the
parser does not have. The absence of a label is a formatting miss, not a claim
that those areas were skipped.

## What the prologue is not

It is not evidence. It is the author's description of their own change, which is a
claim, and the questions it helps produce are questions, not findings. The
prologue is input context for generation; it is not stored as a knowledge record
and carries no `capture_source`. A model-authored narrative of a pull request
would be a different artefact with a different trust question, and this change
does not create one.

## What it does not do, and why that is the point

**It does not guarantee the model obeys the labels.** A response that ignores the
chapter format, or invents a chapter label, still yields its questions: the line
is read through the flat form, or the question is kept unassigned. Dropping a
question over a formatting slip would let the format cost a capture, which is the
wrong trade. The consequence is that anchoring is best-effort, and the coverage
warning is silent when no question was anchored.

**It does not make a chapter a semantic unit.** `src/kojutsu/core` is a directory,
not a concept. Two unrelated decisions can live in one chapter, and one decision
can span two. The chapter map is a reading order and a budget divider, not a claim
about how the change decomposes.

**It does not read a support file's content.** A generated file that happens to
contain a decision is still excluded. That is a deliberate cost: the alternative —
judging each file's content to decide whether it is noise — is the model-authored
classification this design rejects, and it would put the noisiest files back into
the budget whenever the judgement was wrong.

## What would make it wrong

If chapter groups routinely split a decision that a reviewer would call one
change, the title function is wrong and should group by a different path depth. If
the model reliably ignores the chapter labels, the anchoring is decoration and the
coverage warning never fires; that would show up as a run of prompts whose
responses never contain a `c` label. If support classification excludes a file
whose diff genuinely carried the rationale — a checked-in generated client with a
hand-written header, say — the reason list is the place to reconsider, not the
classification rule as a whole.
