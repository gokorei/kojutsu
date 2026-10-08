# Identity, integrity, and the limits of both

## What an identity derivation has to be

Kojutsu deduplicates at four independent layers and they work. Underneath them
sit two hashes, and both are underspecified in a way that can quietly turn a
dedupe check into a coin flip.

`stable_answer_entry_id` joins its inputs with a delimiter and hashes the result.
No domain label, so the identity namespaces in the codebase are not separated.
Delimiter-joined rather than length-prefixed, so the preimage is ambiguous if a
component ever contains the delimiter. And nothing pins the output, so a refactor
that changes a field silently changes every future id.

`_idempotency_key` is worse, for two independent reasons. It hashes the
**already-serialised** request body, so two builds of the same logical Q&A with
different key ordering produce different keys. And it reads the id with a default,
so a missing key and an empty-string key produce the same key — a real collision,
not a theoretical one.

The requirement underneath is one sentence: **a digest taken over whatever the
encoder happened to emit is not an identity.** Derive from a named, ordered field
tuple, length-prefix the components, give each derivation its own domain label,
and pin the output to a golden value so a refactor cannot move it unnoticed.

The domain label does double duty. It separates namespaces, and — because the
label is part of the preimage — it means a future change to how the digest is
computed produces a different value rather than one that looks comparable and is
not.

### These are persisted, so this is a version bump and not a patch

The entry id is a durable `UNIQUE` key in the registry, and the Tanseki
idempotency key is server-side state. Changing a derivation re-identifies existing
records, and the Tanseki document id is path-derived from the entry id — so
rewriting stored ids would orphan documents that are already in the store.

The answer is therefore **not** to re-identify existing rows. Version the
derivation, record the version alongside each stored id, and let the two coexist.
Doing this once, deliberately, while the tables are small, is much cheaper than
doing it later.

## Character policy: refuse less than you think

Third-party text arrives decomposed. `é` may reach kojutsu as one code point or
as `e` plus a combining acute; these are the same text to a human and to GitHub,
and different byte strings, so they produce different digests. Normalising before
hashing is what makes one logical record have one identity.

The same characters matter for a second, unrelated reason: a right-to-left
override or a bidirectional embedding control in a stored comment makes evidence
*display* as something it is not. The comment appears to say one thing and renders
as another, to whoever reads it later. That is a display-fidelity control, and it
is worth having independently of any digest.

The part that is easy to get wrong is what **not** to refuse. Joiners and emoji tag
sequences break honest content — real names, real scripts — and a character filter
that blocks legitimate content gets switched off, which is worse than not having
one. The refused set is the invisible and bidirectional formatting characters.
The allowed set includes everything that can legitimately appear in a human name.
That asymmetry is deliberate and belongs in a docstring, because a well-meaning
maintainer will otherwise "fix" it by tightening the filter.

And the record is always kept. A hostile comment is evidence; stripping the
dangerous characters and recording that it happened is correct, while refusing to
store it destroys exactly the thing this project exists to preserve.

## What a content digest proves, and what it does not

The useful integrity property kojutsu can honestly claim is narrow:

> A record whose stored content no longer matches what kojutsu sent is visible
> to an operator.

A digest recorded at capture time, checked later against a re-derived body, gives
exactly that. It is worth having, and it is worth being precise about what it is
not.

It is **drift detection, not tamper-evidence.** The digest lives in kojutsu's
local registry, so a party who can write both the store and the registry defeats
it. It is not a signature, because kojutsu holds no secret an attacker could
not also reach.

And it is only possible if the body is **reproducible**, which is the constraint
that shapes the design. A digest taken over the rendered body silently invalidates
itself the next time the body rendering changes. So the preimage is the semantic
content — question, answer, title, author — not the rendered form, and the
derivation version is part of it.

Entries captured before the feature existed have no digest. That is a missing
fact, not a negative one, and must be reported as unverified rather than as a
divergence. A system that reports "unverified" as "failed" trains its operator to
ignore it.

## Why the hash chain was rejected

A hash chain over the local files would verify green while proving only that a
spool file has not been edited since its last row was written. That is a
statement about a queue, not about captured knowledge — and it invites exactly
the over-claim the reference project is careful to avoid in its own system, where
the chain is explicitly described as detecting corruption and never as
tamper-evidence, on the grounds that an adversary who can write the file
recomputes the chain and it verifies.

Rejection reasons, recorded so they are not re-litigated:

- The outbox and registry are a rebuildable spool. Tanseki is the record. A chain
  over a spool defends against a threat the storage model already excludes.
- It would verify green while meaning something other than what a reader would
  take it to mean, and then require the caveat to be repeated in five places
  forever.
- Same for backup/restore of the local files: copying a spool is not backing up a
  knowledge base.
- Same for a derived full-text index: Tanseki *is* the index, and it is remote, so
  the entire problem class of keeping a derived artefact in sync with its source
  does not exist here.
- Same for a replay fingerprint: it would require an ordered, complete,
  revision-stamped listing from the store, and the seam returns only a total. A
  digest over an unordered response is a determinism claim kojutsu cannot
  support.

None of these are bad ideas. They are answers to a different storage model, and
carrying them over would mean carrying the model too.

## State a limitation and it stops being a limitation

The single most valuable practice found in this review is a test that asserts a
*negative* — that something is **not** detected — with the comment "stating it as
a test is the only way it does not quietly become an implied promise."

A limitation in a docstring is a comment; a limitation in a test survives the
next refactor, the next reviewer, and the next person who upgrades a word like
"durable" or "verified" by one degree.

Kojutsu already has this instinct in
`tests/test_provenance.py::test_detail_does_not_promote_an_entry_to_evidence`,
which reproduces the exact failure mode the provenance model exists to prevent.
The practice should generalise to every claim the documentation makes, and a doc
that outruns its code should be corrected — the reference project has a live
branch doing exactly that, for a claim about audit-row atomicity that the code did
not honour.
