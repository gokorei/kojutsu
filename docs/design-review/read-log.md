# The read log: a retrieval event is not knowledge

The thesis is that captured knowledge changes what a later reader does. Between
the write and any possible change sits one step the system does not observe: the
read. `search_knowledge` and `get_knowledge_entry` were correct about what they
returned and silent about the fact that they had been called at all, so the one
causal edge in the thesis was unobserved at both ends.

This document records what the read log holds, and — more usefully — what it
must never be made to hold.

## Why the log is a local file, and neither of the other two places

It is not in Tanseki. A retrieval event is not knowledge, and a store containing
one would claim to hold something it does not. That is the same flattening as
recording a declared rationale as an empty `uncategorized` row would be:
a category of thing that is not what it is, and a count that
includes it. The corpus is the answer to "what do we know"; a read log is the
answer to "was the answer used", and a store that contains both cannot answer
either question honestly.

It is not in the SQLite registry either, and this one is about concurrency
rather than category. The registry is the single-writer capture path: the dedupe
claim, the lease, the provenance record. A read would take a write lock on it to
write down that it did not write. Reads are the one operation that must never be
made slower, or less available, by the machinery that observes it.

So the log is an append-only JSON Lines file next to the other local state. The
corpus's contents are unchanged by who read what, and deleting the log cannot lose
knowledge, which is the property the outbox's durability argument is about.

## What may go in it, and what may not

The minimum that supports the claim: the tool, the caller's own query, the
filters they stated, how many entries the answer carried, whether a bound or a
filter left anything out, the outcome, and the time.

Not the document bodies. Not the snippets. Not the evidence fence. The renderer
in `mcp_server/server.py` is the only code that can see that content and the log
module never receives it, so there is no code path by which stored third-party
text reaches a line of the file. A read log that quoted what was read would be a
second copy of the knowledge, held somewhere with a different reader and a
different retention story, and the two would drift.

The recorded query is bounded to 200 characters against the tool's own 10 000. The
bound is a privacy decision rather than a formatting one: a read log about a
repository is close to an activity record about a person, and enough of a query
to recognise what was asked is not enough to reconstruct the prompt that carried
it. The truncation is marked, because an answer shortened by a bound has to be
visible as shortened — a consumer reading the first 200 characters believes it
is reading the whole question.

The log is written escaped, in ASCII, for the reason
[`identity-and-limits.md`](identity-and-limits.md) gives for evidence: a
bidirectional control inside a query would otherwise let a line *display* as
something it is not. It has a second consequence worth knowing: a query
containing a character no UTF-8 file can hold is recorded rather than dropped.

## There is no caller, and no field may imply one

The MCP server is stdio with a single trust domain. It cannot know who asked, and
nothing added later can make it know. So an event records what the caller *said*
— the repository it named, the ticket key it asked for, the limit it set — in a
field called `caller_claims`, and never in a field that reads as an identity.

The naming is the enforcement. A repository name supplied by a caller is a
statement by that caller, exactly as `answered_by_model` is a statement by the
account that posted a rationale: trustworthy as far as the claim, which is
nothing. Put it in a field called `principal` and every later reader of the log
will treat a self-assertion as a fact, and the error is invisible because nothing
about the record looks wrong. A test asserts that no key in an event, and no key
inside the claims, reads as an identity.

`get_knowledge_entry` records an **empty** claim set. The caller named an
identifier, not a scope, and the stored document knows which repository it
belongs to — but recording that would be kojutsu's own statement filed under
the caller's name. The identifier is recorded as the query, because it is the
caller's own argument and the caller could have named it with no store at all.

## A refusal is not the same absence as an empty result

Every terminal path of both tools produces exactly one event, and the outcome is
a closed set: `served`, `no_results`, `budget_exhausted`, `denied`, `rejected`,
`unconfigured`, `failed`. Closed because this is the field an operator triages
on, and a field that can hold any string is a field no query can rely on.

The cases that must not collapse into each other:

- **Denied.** Kojutsu declined. The answer is "you may not look here", which
  says something about the policy and nothing about the store.
- **Rejected.** The caller sent something kojutsu would not look at. Also a
  refusal, and a different event, because one is kojutsu declining and the
  other is a malformed request.
- **Unconfigured.** There is no store. Nothing was retrieved and nothing was
  denied, and a report that folded this into either of the above would be
  describing a policy decision that never happened.
- **`budget_exhausted`.** The store held matches and the response budget carried
  none of them. Its own outcome, because recording this as `no_results` says the
  store was empty — the exact defect
  [`read-path.md`](read-path.md) is about, committed to the operator's log where
  it would be read by someone deciding whether the store is worth keeping.
- **`no_results` with exclusions.** A search that found entries and showed none
  of them, because the caller's own threshold removed them or the store returned
  something cross-repository, records the reasons alongside the count. A count
  of zero with a count of exclusions is not a count of zero.

## Retention is a bound, and a removal is never silent

Two bounds, both enforced on every recorded event rather than on a timer: the
newest `READ_LOG_MAX_ENTRIES` lines, and anything older than
`READ_LOG_MAX_AGE_DAYS`. A timer would make the bound aspirational — correct on
the day it fires and unbounded until then, and `open-questions.md` already
rejects a quota that is quietly resolved by a background trim. The cost is
bounded by the entry count, so the check stays in the low milliseconds at the
default of ten thousand.

Anything removed is reported on the server's stderr, with the count and the
bounds that caused it. A behavioural record that shrinks without a word leaves an
operator reasoning about a gap they cannot see, and a gap that cannot be seen is
the same absence this log exists to remove.

A line whose timestamp cannot be read is removed rather than kept for ever: an
entry whose age cannot be *shown* to be inside the bound cannot be defended as
inside it, and the removal is announced like any other.

There is no setting that turns the bound off. An operator who wants a longer
history widens it. A bound that can be switched off by one environment variable
is a preference, and the default is chosen as if the log would be left running:
seven days, which is long enough to see whether a change was informed by a read
and short enough that the file does not become a durable activity trail.

## Off by default, and the date that cannot move

`READ_LOG_ENABLED` defaults to `false`. Enabling it fixes a start date, and
nothing can move that date earlier: reads that already happened left no trace and
are gone. The discarded facts and the unread registry are both about data that
exists and is not exposed, and both can be exposed retroactively. This one
cannot.

So the flag is off by default, the log has no start date until somebody sets it,
and any report built on it states the date rather than drawing a trend line that
implies one. A report that says "reads since the log was enabled" is accurate. A
report that says "reads" is claiming a history that does not exist.

## What this does not do

**It does not measure anything.** A count of reads is not a value judgement.
Nothing in kojutsu compares one principal's reads to another's, and a
consumer that wants to has to bring its own justification and, ideally, its own
consent story. This is the same line
[`rationale.md`](rationale.md) draws about a stated reason: a record that looks
like a score is a score, whatever the field is called.

**It cannot say the retrieval mattered.** It establishes that a retrieval
happened. It does not establish that the retrieval changed anything, and linking
a read to a subsequent outcome needs a counterfactual this project does not
have. A report that treats a read count as evidence of value has made the same
mistake as a rationale that reads as a correctness check.

**It attributes nothing.** See above. If a future change adds a field that names
who read what, it should be read as adding an activity record about a person to a
knowledge tool, and it should have to argue for that separately.

**It is not durable.** Appends are not fsynced. A crash can lose the last few
events, which is the right trade: the outbox is where a crash must not lose
something, and a read that pays a disk flush on every call is a read that can be
made to fail. Telemetry that can fail the read it measures has made the store
harder to use in order to observe it.

## What would make this wrong

A pilot whose question is "did the agents use the knowledge?" gets a real answer
here, bounded and starting on a known date. A pilot whose question is "which agent
uses the knowledge more?" is asking for a ranking of people's activity, and this
log will not give it to them without a change that should be argued for on its own
terms. If that is the only question being asked, the honest answer is that
kojutsu should not answer it — not that kojutsu should log harder.
