# Durability: what the word has to mean

## The claim

Kojutsu's README says writes are "captured durably in a local outbox first",
and the whole dead-letter and retry design leans on it. Before this round the
outbox and the registry set one pragma between them — a busy timeout — and ran
with SQLite's default journal and synchronous level.

The default synchronous level is `NORMAL`. Under it, a commit is not guaranteed
to have reached stable storage when SQLite reports success. So the system
acknowledged captures it had not actually made durable, and the retry machinery
designed to protect against store outages could not protect against the machine
going down.

This was found by reading the code against the prose, not by a test. No test
failed, because nothing asserted the claim.

## Why the pragmas are read back

`PRAGMA journal_mode=WAL` is a no-op that reports the *pre-existing* mode when the
conversion cannot happen — most commonly on a filesystem without working
shared-memory support. A statement that returns successfully and quietly did
nothing is the exact failure mode these pragmas exist to prevent, so applying
them is not enough; the resulting value is read back and checked.

The same applies to the synchronous level, which is per-connection rather than
per-file: applying it once at construction would leave every other connection
running at the default. This is why the pragmas live in one function that every
open path calls, rather than being sprinkled at construction sites.

Startup fails loudly rather than degrading. Continuing without
`synchronous=FULL` would reinstate precisely the gap being closed, and it would do
so invisibly.

## Why the pragmas are asserted directly as well

An early draft of this document claimed a crash test could not catch a missing
`synchronous=FULL`, on the grounds that the operating system's page cache
absorbs the write and the machine survives anyway. **That claim was wrong, and
measuring it is what corrected it.** Reverting the outbox to `synchronous=NORMAL`
and re-running `tests/test_durability.py` fails all fifteen tests, including the
acknowledged-enqueue case, because a child committing hundreds of entries in a
tight loop leaves the page cache under enough pressure for the gap to show.

The claim is not that a crash test *cannot* catch this. It is that **whether it
does depends on kernel scheduling, page-cache pressure and filesystem**, which
makes it a property of the machine rather than of the code. The direct pragma
assertion catches it deterministically, in microseconds, on any machine, and
tells you which pragma is wrong. The crash suite catches the class of failure
the pragma cannot show — a torn write, a stranded lease, a lost acknowledgement —
and on this machine catches the pragma regression too, which is a bonus rather
than the reason to keep it.

Keep both. Assert the mechanism, and exercise the behaviour.

## Why copying a local file is not backing it up

In WAL mode the data lives partly in a `-wal` sibling. A copy of the main file
taken while a writer is live opens without error and is missing the most recent
writes. The failure is completely silent, which is what makes it dangerous: the
operator believes they have a backup.

`tests/test_sqlite_durability.py` asserts this as a measurement rather than
stating it as advice, because a measured fact survives being ignored in a way
that a warning does not.

Note the asymmetry that makes it worse: after a *clean close*, SQLite checkpoints
the log into the main file, so a copy taken then is complete. The hazard is
specifically a copy of a live file, which is also what a well-meaning `cp` in a
runbook does.

## Why delivery retries are unbounded, and the registry's are not

The outbox deliberately never dead-letters on an attempt count. A transient
failure means the store was unavailable, not that the capture was unwanted, and
the queued row is the only copy of something a human typed. A ceiling would
convert a temporary outage into permanent loss of knowledge, which is the one
outcome this project exists to prevent.

The question registry does the opposite, and moves a claim to a terminal `failed`
state. That is not an inconsistency: a question claim is a lease on work that can
be re-derived from the PR, while a captured answer cannot be re-derived at all.
The two ceilings are different decisions about different things.

An attempt count *was* exposed on the outbox constructor, was never read, and has
been removed. A parameter named `max_attempts` is a promise; an inert one is worse
than no parameter, because an operator reading the signature concludes they
configured a bound when they did not.

The visibility an operator actually needs is preserved: the attempt count is still
recorded and still reported by `kojutsu outbox`. The difference is that
kojutsu will not make that decision silently, and will not discard the
knowledge first.

## Why migrations carry checksums

Kojutsu's migrations were already forward-only, which is the hard part. What
was missing is the ability to notice a migration edited after release: the
existing validation proves the schema *looks* right, not that the same migration
produced it. A checksum per migration closes that gap for a few dozen lines.

The add-only rule matters more than the checksum. A reversible migration invites
a path that rewrites stored history, and kojutsu has no state worth rolling
back to — the outbox is a spool and Tanseki is the record.

## What this does not buy

None of the above makes kojutsu's local state authoritative. It is a spool and
a cache in front of a remote store, and a checksum over one local file is a
checksum with a history, not a signature. See
[`identity-and-limits.md`](identity-and-limits.md) for the boundary, and
[`../../README.md`](../../README.md) for the operational constraints that already
existed.
