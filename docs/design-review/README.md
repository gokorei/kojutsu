# Design review: hardening kojutsu

The reasoning behind a round of hardening work, recorded here so it does not
depend on a scratch repository staying on disk.

## Why this exists

Kojutsu was exercised end to end against a purpose-built showcase project
(`acme/widgets`, a throwaway demo whose implementation turned out
to be more rigorous than most production code). Running against a real review
surfaced four defects that no amount of unit testing had found, because each one
is a claim the documentation made and the code did not back:

- a component described as durable that acknowledged writes before they reached
  disk;
- a read path that told a caller "nothing found" when it had in fact dropped
  everything it found;
- a prompt-injection fence made of constants that stored text could forge;
- a public API parameter that was accepted, documented by its name, and never
  read.

None of these are exotic. They are the ordinary result of a system whose prose
runs ahead of its code, which is why the practice adopted here is *assert the
claim or drop it* rather than *add more features*.

## What these documents are

Reasons, not instructions. Each one records why a decision was made and what
would have to be true for it to be wrong, so the decision can be revisited
without re-deriving the reasoning. Implementation specifics belong in the ticket
and in the code.

| Document | Covers |
|---|---|
| [`durability.md`](durability.md) | What "durable" has to mean before it can be claimed, and why copying a live database file is not backing it up |
| [`read-path.md`](read-path.md) | Why a bounded answer must never read as a complete one, and why a fence made of constants is not a fence |
| [`read-log.md`](read-log.md) | What a read log may record, why it is a local file rather than knowledge, and why it starts on a date it cannot move |
| [`identity-and-limits.md`](identity-and-limits.md) | What a content digest does and does not prove, and how to keep a limitation from becoming an implied promise |
| [`rationale.md`](rationale.md) | Why stated decision rationale is captured and chain-of-thought is not, and the three claims that keep a self-asserted reason from reading as evidence |
| [`record-structure.md`](record-structure.md) | Why a second axis answers "was this pairing established or inferred", and what it does not claim |
| [`clarification.md`](clarification.md) | Why a human statement that answered no question gets its own record, and why its trust runs opposite to a rationale's |
| [`thread-classifier.md`](thread-classifier.md) | Why a comment thread has three outcomes and the third is the design, and why an uncertain model is allowed to decline |
| [`open-questions.md`](open-questions.md) | The reasoning behind the work not yet done |

## Provenance and licensing

The reasoning above was developed against `acme/widgets`, which is
licensed Apache-2.0. Kojutsu is AGPL-3.0-only. Apache-2.0 into AGPL-3.0 is a
compatible direction, so the ideas transfer; no source was copied verbatim, and
these documents record reasoning rather than implementation.

That repository is a throwaway showcase and may be deleted. Nothing in this
directory depends on it existing.
