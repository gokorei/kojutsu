# The read path: bounded answers and unforgeable fences

Two defects, both about the same failure: telling the caller something that is
not true.

## A bounded answer must not read as a complete one

`search_knowledge` renders documents into a response bounded by
`MAX_SEARCH_RESPONSE_CHARS`. Documents that did not fit were skipped silently,
and if none fit, the tool returned the string `No knowledge entries found.`

That is the worst possible output for the situation. "No knowledge entries found"
is a claim about the *store*. The tool actually knew about entries and had chosen
not to show them, and it reported the store's state instead of its own. A caller
cannot distinguish the two, and neither can it recover: the search is over.

Three things make it worse than a single bad string.

**The skips came from the middle of a rank-ordered list,** so the retained set
was not a prefix of relevance. A caller receiving three of ten results had no way
to know they were the top three.

**No number was reported.** kojutsu had per-document truncation
(`content_truncated`, `content_length_chars`) and per-value truncation, so a
caller could see that *a* document had been clipped. It could not see that
*most of the answer* had been dropped, because nothing counted the drops.

**The single-document case is easy to miss when reasoning about it.** The
per-document content cap is 20 000 characters and the response budget is
100 000, so one document cannot normally exceed the budget on its own. It can,
though: the payload is JSON-encoded with `ensure_ascii=True`, so non-ASCII
content expands to six bytes per character in the wire form while the cap counts
characters. That is the only way one document outgrows the response budget, which
is exactly why the case deserved a test rather than an argument.

The fix keeps the two bounds that already existed and makes the third visible:

- The retained set is a prefix of the rank order, so what the caller gets is the
  most relevant part rather than an arbitrary subset.
- Every exclusion is counted and reported, in a note outside the evidence fence —
  it is kojutsu's framing, not evidence.
- A fully-exhausted budget is reported as *budget exhaustion with the matching
  entry ids*, not as an empty store. Naming the entries lets the caller narrow
  with `jira_ticket_key` and actually recover the answer.

The general principle: **an answer shortened by a bound must be visible as
shortened.** The reason is that a caller's conclusion from a truncated result is
much stronger than the evidence supports — "nothing here mentions this" is a very
different claim when ten things matched and three were shown.

## Truncate, never refuse — and keep the distinction

An oversized *input* is refused; an oversized *answer* is answered with a smaller
answer. The caller chooses the window (`limit`, `jira_ticket_key`), so refusing
would only teach it to ask again with a bigger one. This distinction was already
correct and was preserved.

## A fence made of constants is not a fence

Third-party text is wrapped in `=== UNTRUSTED_EVIDENCE_BEGIN/END ===` markers
before being handed to an agent, with a `trust: untrusted` label and an explicit
instruction not to follow instructions inside.

The markers were constants. Any contributor with comment access can write the
literal closing marker into a review comment, and the fence closes early —
everything after the injected marker reads as kojutsu's own framing rather
than as evidence. A prompt-injection attempt has every reason to do exactly this.

The JSON wrapping means content cannot break out *structurally*, which is why
this was easy to miss. But the marker lines are plain text outside the JSON, and
a literal string in the payload is not a structural escape.

The fix is a per-response nonce from a CSPRNG, generated **after** the payload is
assembled. The property that matters follows from that ordering: stored content
cannot contain a marker that closes its own block, because reproducing the marker
would require a value that did not exist when the content was written. The nonce
is announced to the reader, so a genuine closing marker is distinguishable from
a forgery.

The rendered result is then inspected before it is emitted. A response whose
framing did not survive rendering is refused under its own error code, because a
desynchronised fence is worse than no fence — it mislabels the remainder as ours.

### One consequence worth knowing about

A random nonce in the output makes raw-substring assertions on rendered text
unreliable: a 32-character hex nonce can contain a two-character document id by
coincidence, often enough to matter. Tests assert on the *parsed* provenance
rather than on substrings. This is a better test regardless of the nonce, but the
nonce is what exposed it.

## Scope: one definition, and case folding that is deliberately not tightened

The scope is genuinely good and was not rewritten: default-deny, `repo`
mandatory, the wildcard never treated as authorisation, filters pushed to the
store, and error strings that do not echo the offending repository.

Two things were found. One was a real defect nobody had spotted; the other was a
recommendation from the reference project that does **not** transfer.

**The defect: capture and read authorised different things.** Two functions
parsed the same `GITHUB_WEBHOOK_ALLOWED_REPOSITORIES` with different acceptance
criteria. The webhook applied no token validation at all; the MCP server required
a pattern. The write scope was therefore *looser* than the read scope, so
kojutsu could capture a review from a repository and then refuse to read it
back — a review that existed and was unreachable. Both now call one function in
`kojutsu.allowlist`, so the two surfaces cannot disagree by construction
rather than by discipline. The predicate fails closed on a malformed
configuration, so a typo denies rather than raising mid-request.

**The reference project's advice would have made it worse.** It holds that an
allowlist must not fold case, on the reasoning that folding is "a
case-insensitive allowlist with extra steps". That is right for a system with no
forge, where an operator both writes and reads the token and no external
canonical form exists. kojutsu is not that: repository names arrive from
GitHub, which defines them as case-insensitive, and a false denial silently drops
a real review that nobody notices is missing.

The asymmetry is the argument. Folding when it should not have accepts a
repository the operator did not name — a real risk, and the price of agreeing
with the platform rather than imposing a stricter spelling than the platform has.
Not folding when it should have loses a review. Those are not symmetric, and the
allowlist is on the side that loses data.

What makes this easy to get wrong later is that kojutsu does *not* fold
everywhere. `list_questions` matches its `repo` filter exactly, so the work index
stays usable and a mis-cased filter shows less work rather than another
repository's questions mixed into a page. **Authorisation folds; enumeration does
not.** Both halves are pinned by tests in `tests/test_allowlist.py`, precisely
because the pair looks like an inconsistency and the next reader's instinct will
be to make them agree.

The remaining gap, recorded rather than papered over: `get_knowledge_entry` must
fetch the document and *then* check its stored repository. The caller named the
identifier, so it is entitled to be told the answer is not in the repository it
asked about — but the content crosses the network before the policy decision,
because Tanseki's `:get` accepts only `{id, collection}` and offers no server-side
repository filter. Tightening that means changing the store's contract, not this
call.

## The same bound has to be visible to the operator, not only to the caller

`kojutsu.core.read_log` records one event per call, including a search whose
answer was shortened. The accounting is the one already in this document — the
retained set is a prefix of the rank order, and every exclusion is counted and
given one of a fixed set of reasons — so the event says how many entries came
back, how many did not, and why.

That is the same defect seen from the other side. An answer that dropped nine of
ten matches now says so, and a *record* of that answer that reported `no_results`
would put the false claim back in front of the person deciding whether the store
is worth keeping. It gets its own outcome, `budget_exhausted`, distinct from both
`served` and `no_results`, for exactly that reason.

The reasoning behind the log's contents, its refusals, its retention bound, and
the start date it cannot move is in [`read-log.md`](read-log.md).

## What the read path still cannot promise

Fencing is a *presentation* control, not a sandbox. It makes stored text
identifiable as evidence to a reader; it does not stop a determined reader, and it
does not stop a store from returning documents for repositories that were never
requested. The client-side re-filter is defence in depth against the latter and
is worth keeping, but it is a second line, not the boundary — the boundary is the
exact-match scope enforced before a result is rendered.
