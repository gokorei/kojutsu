# Open questions: reasoning behind the work not yet done

Each item records **why** the work is worth doing and what would make it wrong.
The steps belong in the ticket; the reasoning belongs here, so it survives the
ticket being closed.

## Bounded input at the webhook edge

`MAX_COMMENT_CHARS` bounds the comment text, but nothing bounds the request body
before it is buffered, HMAC-verified and parsed. Unbounded input is a
denial-of-service primitive, not a formatting choice.

The non-obvious part is *where* the bound goes. In an async framework, a handler
that has received a parsed body has already paid the memory cost, so a check
inside the handler is decoration. The ceiling has to be enforced before the body
is materialised, which means a different layer than the one holding the rest of
the webhook logic.

The ceiling must be set from a measurement of what real deliveries look like, not
from a guess, and it must sit *above* the comment ceiling rather than at it: a
comment already stored near the limit must remain capturable, or the bound
becomes a way to make existing records uncapturable.

## Outbox growth has no ceiling

Rows accumulate without limit while a store is unreachable. A quota that is
quietly resolved by deleting data is not a quota, so the honest options are a
refusal or a loud warning — not a background trim.

The subtlety is that a refusal must not apply to an entry that is *already
stored*. Refusing to re-confirm something that is already durable leaves the
producer unable to tell "not stored" from "stored, and you already have it", and a
webhook that answers that ambiguity with a retry produces duplicate GitHub
comments. This is a correctness property, not a nicety, and it is the reason a
quota needs more thought than a counter.

## Read-time validation of capture provenance

The validator that refuses a captured record it cannot verify runs once, at
construction. It is never re-applied, so a document in the store claiming
`capture_source: webhook` with no `delivery_id` is served today as verified
evidence.

The write-time check is the right check — a record must earn the right to read as
evidence — and it is mature. What is missing is that the store is a separate
service, so what is written and what is later read are not guaranteed to be the
same thing. Re-validating on read is cheap and closes a real gap.

A document that fails must still be **served**. It is real stored content and
discarding it would lose knowledge; it should be served and flagged, exactly as
the character policy serves a hostile comment after neutralising it.

## A caller may want evidence only

`capture_source` is recorded and surfaced but never acted on. An agent receives an
asserted record framed identically to a signed webhook capture, differing only in
which provenance keys are present.

The fix is a filter, and the discipline is what not to change: **default off, and
no down-ranking.** Asserted records are legitimate — the pilot corpus is full of
agent-authored records an operator will want to find. A caller should be able to
ask for evidence only and to see which kind it got; asserted content must not
become hard to find. And an excluded-not-empty result must be distinguishable
from a genuine absence, for the same reason a budget-exhausted result is (see
[`read-path.md`](read-path.md)).

## An agent has no cheap way to orient itself

To learn whether the store is reachable, what it may read, or whether Tanseki is
configured at all, an agent must run a real search and interpret the failure. An
unconfigured store and an empty store produce different failures that the agent has
to know how to distinguish.

A status call fixes this, and it must stay cheap: no search, no traversal, no
per-document calls, and no repository names outside the allowlist. It should
return the allowlist in its **exact** configured spelling, because once case
folding is removed an agent that guesses `Owner/Name` for an allowlist of
`owner/name` is denied with no way to discover its mistake.

## The read-only surface is a convention, not a structure

*Partly resolved. See the end of this section.*

Nothing prevents a future change from adding a write tool to the MCP server. The only
thing standing in the way today is a reviewer noticing.

What is enforceable, and worth doing cheaply: freeze the tool table so it cannot
grow by side effect, and assert in a test that every tool is read-only and that the
module imports no write path. That test fails when someone adds a write tool,
which turns the review from "did you notice?" into "why did you change this?".

What is *not* enforceable, and must not be claimed: a stdio server with a single
trust domain has no caller identity, so "read-only" here means "this build exposes
no mutation tool", not "no caller can mutate".

### Resolution

`tests/test_mcp_server.py` now asserts all three things, and they pass: every tool
on `kojutsu-knowledge` carries read-only annotations, the tool table is exactly
`{search_knowledge, get_knowledge_entry}`, and the read server's source contains
none of the write paths — `post_issue_comment`, the rationale marker builder,
`claim_rationale`, `KnowledgeSink`, or `TansekiOutbox`.

A write surface now exists, and it is a **separate server**
(`mcp_server/capture_server.py`, `kojutsu-capture`) rather than a tool added to
the read one. That placement is the substance of the fix. A single server with a
read and a write tool would leave "read-only" a property of the tool table, which is
the convention this section was about; two servers leave it a property of which
process an agent is connected to.

The limit above still holds, and is now restated in that module's docstring and
asserted by a test: the capture server cannot know who invoked it. What can be
claimed is narrow — this build exposes one tool, it posts only on an allow-listed
repository with the capture process's token, and a declaration is a self-assertion
by the account that posted it. That is not the same claim as "no caller can induce
a write", and nothing in the module asserts it is.

What remains unenforced is the *caller*: a host with write access to
`kojutsu-capture` is a host that can induce a comment. That is a property of who
is given which server, not of this codebase.

### One later addition, and the property it had to keep

The read server now records what it was asked and what it answered, in a local
file (`kojutsu.core.read_log`, and
[`read-log.md`](read-log.md) for why it is a file and not the corpus). That is the
first change to add an import to this module since the assertions above, and it is
the case they were written for: the tempting move was to keep the events
somewhere kojutsu already writes to, which is a write path on a read server and
would have made every one of these tests false.

Instead the log is a stdlib-only module that cannot reach the store even by
accident, and the same argument applies to who read it: the event has no caller
field at all, and a test asserts that no key in it reads as an identity. The limit
above still holds and is unchanged — what can be claimed is that this build
records a retrieval, never who performed it.

