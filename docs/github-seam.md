# GitHub seam

Kojutsu reaches GitHub over the REST API, and the only code that does is
`src/kojutsu/integrations/github.py`. This document is the contract for that
seam: what is called, what the token may do, and how the two halves of capture
(the process that reads and comments, and the sandboxed process that reads
untrusted text) are kept apart.

## What Kojutsu actually calls

Every GitHub request the pipeline makes, and nothing else. **Three modules issue
them**, which is worth knowing before adding a fourth:

- `integrations/github.py` — the capture pipeline, and `kojutsu backfill`;
- `integrations/webhook_client.py` — hook registration, only when
  `GITHUB_WEBHOOK_REGISTER=true`;
- `core/backfill_reviews_client.py` — `kojutsu backfill-reviews` only.

The last is separate because `github.py` is the *capture* client: it constructs
requests for the live pipeline and for a by-date-range ingest, and folding a third
kind of read into it would put history reconstruction on the path every webhook
delivery depends on. The reviews client is separate, read-only, and reached only
by its own command.

| Call | Purpose | Access |
| --- | --- | --- |
| `GET /user` | Confirm the token works and identify it | read |
| `GET /repos/{owner}/{repo}/pulls/{n}` | PR title, body, head ref | read |
| `GET /repos/{owner}/{repo}/pulls/{n}` (`Accept: application/vnd.github.v3.diff`) | The diff the question generator reads | read |
| `GET /repos/{owner}/{repo}/pulls/{n}/files` | Changed-file names | read |
| `GET /search/issues` | Enumerate pull requests *created* in a date range — `kojutsu backfill` | read |
| `GET /repos/{owner}/{repo}/pulls/{n}` | One pull request, for `kojutsu backfill-reviews` | read |
| `GET /repos/{owner}/{repo}/pulls/{n}/reviews` | Review history — `kojutsu backfill-reviews` | read |
| `GET /repos/{owner}/{repo}/pulls/{n}/reviews/{review_id}/comments` | A review's inline comments — `kojutsu backfill-reviews` | read |
| `GET /repos/{owner}/{repo}/issues/{n}/comments` | Resolve a question marker to a real comment, and read issue comments in `backfill-reviews` | read |
| `POST /repos/{owner}/{repo}/issues/{n}/comments` | Post a generated question, and acknowledge an answer | write |
| `GET`/`POST`/`PATCH`/`DELETE /repos/{owner}/{repo}/hooks` | Only when `GITHUB_WEBHOOK_REGISTER=true` | write |

**There are two history commands and they read different objects.**
`kojutsu backfill` enumerates pull requests by *creation* date and collects
their comments; `kojutsu backfill-reviews` walks *reviews* and their inline
comments and reconstructs them with `capture_source: backfilled`. They are not two
spellings of one feature, and running the wrong one reconstructs nothing while
reporting success. Choose by what is missing: answers and clarifications in the
thread, or reviews.

The history listings need
`Pull requests: Read` and `Issues: Read` — the documented minimum, no widening —
and **no `Contents`**: a backfill reconstructs what people and reviews said, and
deliberately does not read diffs, because a diff is how a model infers a rationale
and that is a different record with a different trust
(`Kojutsu.core.rationale_link`). They live in `core/backfill_client.py` rather
than in `integrations/github.py`, which is the one module documented here as the
only caller of the REST API; that is a real divergence from the sentence above
and is recorded rather than papered over.

Pull request comments are **issue** comments in GitHub's API, which is why comment
posting is an `Issues` permission rather than a `Pull requests` one.

**One connection pool per client, for its whole life.** Every request used to open
its own `httpx.Client` and close it, so each paid a DNS lookup, a TCP handshake and a
TLS negotiation before transferring a byte — and the 96-request `backfill-reviews` run
profiled below paid 96 handshakes to read one repository's history. `GitHubClient` and
`GitHubHistoryReader` now each hold one pool, created lazily on first use and released
by `close()` (or by leaving the `with` block). Lazily because most of these clients
perform a single short call and are then dropped, which is the shape at six call
sites, and a pool opened eagerly for an object about to become garbage trades a real
cost for a tidy-looking one. This is also what makes the concurrent walk below
possible: `httpx.Client` is documented safe to use from several threads.

**The two read timeouts stay distinct.** Ordinary reads get 30s
(`REQUEST_TIMEOUT_SECONDS`); the unified diff gets 60s
(`DIFF_TIMEOUT_SECONDS`). A diff is megabytes and every other read on this seam is
kilobytes, so the one call that transfers orders of magnitude more bytes is given
orders of magnitude more patience — and it is now a per-request override on the shared
pool rather than a second pool, because the distinction is about one call site's
payload and not about a different connection policy.

## Reading history: `kojutsu backfill`

`GET /repos/{owner}/{repo}/pulls` takes no date parameter and only ever returns
`state=open`. A pull request that was opened and merged before Kojutsu existed
is therefore invisible to it, which is the whole reason `GET /search/issues` is
on this list: it is the only GitHub surface that can enumerate a *range* of pull
requests by creation date.

```
kojutsu backfill --repo owner/name --since 2024-01-01 --until 2024-03-31 \
  [--limit 100] [--report coverage.json]
```

### It is read-only, and that is checked rather than promised

Every request this path issues is a `GET`, and there is no `--apply`. A
historical ingest has no reason to comment on a pull request from three months
ago, and an ingestion path that *can* write would need a flag, a review, and an
audit of who passed it. `tests/test_github_range.py` records the HTTP method of
every request that passes through its stub transport during a full run and fails
on any of `POST`, `PUT`, `PATCH`, `DELETE` — so a write path added here turns the
build red rather than turning up in a review.

### The search API is not a consistent read of the core API

This is the single most important thing to know about the endpoint, and it is why
`backfill` is a separate command rather than a flag on `collect`. Search indexes
asynchronously: a pull request opened seconds ago may simply not be there yet,
and there is no way to ask whether it will be. For a historical range, where the
window closed days ago, that is irrelevant. For a webhook it is disqualifying —
which is exactly why the live path never uses this endpoint.

### Two limits that fail quietly

**30 requests per minute**, not the core API's 5000/hour. A naive page-through
exhausts that partway through any range worth backfilling, and a 403 arriving
mid-walk looks like an empty range rather than like a limit. Requests are paced
against `DEFAULT_SEARCH_PACE_SECONDS`, and a 403 with `x-ratelimit-remaining: 0`
is raised as `GitHubRateLimitError` (which is `retryable`) rather than absorbed.

**1000 results**, and no error past it. A wider range is silently incomplete, so:

- `MAX_RANGE_SPAN_DAYS` (365) is refused up front, with the bound in the message.
  A range too wide to be a useful unit of work is a caller mistake, and naming it
  is cheaper than explaining a truncated read afterwards.
- The result carries `truncated` and GitHub's own `total_count` alongside the
  list. A short list is never returned as if it were whole.
- `backfill` exits **2** on a truncated or partial run. A script that re-runs on
  a short read is how a partial range gets mistaken for a whole one.

`incomplete_results` in GitHub's own response is also treated as truncation: that
is GitHub admitting it could not finish counting, and reading it as a complete
count is the same mistake in a different place.

**The other two limits are on the core API and are handled in
[the walk that reads them](#walking-back-to-the-window-is-concurrent-and-bounded)**:
5000 requests/hour, and a concurrency bound of 8 on `backfill-reviews`. Neither is
paced, because both are concurrency-shaped, and the difference between the two
instruments is the thing to get right: a *rate* limit is fixed by sleeping
(`DEFAULT_SEARCH_PACE_SECONDS`), a *concurrency* limit is not.

### Refusals, and what they cost

Both refusals happen **before any request is issued**, because a range validated by
the response is a range that has already cost a request and told GitHub what the
caller was looking for.

- **The repository must be in `GITHUB_WEBHOOK_ALLOWED_REPOSITORIES`**, checked
  through `kojutsu.allowlist` — the one definition the webhook and the
  MCP server use. A second implementation here is exactly the drift
  `docs/design-review/read-path.md` documents, and a read path that authorised
  something the write path refused would capture a review it could not read back.
- **The range must be `YYYY-MM-DD` on both ends, `since` no later than `until`,
  and within `MAX_RANGE_SPAN_DAYS`.** An inverted range is refused rather than
  answered with zero results: an inverted range matches nothing on a perfectly
  healthy index, and a caller reading "no pull requests" would conclude the range
  was empty rather than that the arguments were backwards. Every message names the
  field at fault.

Timestamps are refused rather than truncated. The query is a pair of *days*, so
accepting `2024-01-01T12:00:00Z` and quietly taking its midnight would mean the
two ends of a range mean different kinds of thing.

### Idempotence

A capture is keyed on the comment it came from —
`stable_answer_entry_id(repo, pr_number, comment_id)` for an answer,
`stable_rationale_entry_id` for a declaration — and the registry claims that key
before anything is stored. So re-running an overlapping range stores nothing the
first run stored, and two ranges that overlap by one pull request agree about it.
This is not a special case in the backfill; it is the same dedupe the webhook
relies on, which is what makes an interrupted run safe to finish by running the
same range again.

`--report` writes the coverage record: the range, the truncation flag, GitHub's
count against the count enumerated, per-run counts, and any pull request that
could not be read. A pull request whose comments fail does not end the run — it is
recorded as a failure and named in the record, because a run that dies on the
twelfth of forty pull requests is a run that cannot be resumed from a known
position.

### What it does not reach

A pull request's **review** thread — review verdicts and inline diff comments —
is not collected. Those arrive as `pull_request_review` webhook payloads, and the
client has no method that lists them for a given pull request. A backfill over a
range therefore sees a pull request's issue comments only, which is where
questions and answers live; it does not reconstruct review history.


## Reading review history: `kojutsu backfill-reviews`

The other history command, and the one with different bounds. It walks pull
requests **most recently updated first**, so a `--since` floor is one comparison:
the first change whose update predates it ends the walk. `--until` is the same
comparison mirrored at the top, and it is optional.

**Without `--until`, an operator cannot ask for a historical window at all.**
Enumeration necessarily starts at the present, so "the last quarter" of a busy
repository returns the last page of its present — a few days — and reports itself
as the quarter. That is the worse of the two gaps, because `--max-objects` at least
is a visible number. Both ends are `YYYY-MM-DD`, `since` no later than `until`,
and `until` names the whole day.

**`--max-objects` bounds new work, not reads.** Charged per object read, it pinned
the walk to the first page: a second run re-read the objects the first had stored,
spent its budget on them, and stopped in the same place, while its own output
recommended exactly that re-run. Charged for storing instead, an object the store
already holds costs nothing and so does one with nothing to capture — which is what
makes the re-run the output recommends actually advance. On `pingdotgg/t3code`,
168 of the first 200 objects read had nothing to capture and 2 were already stored,
so exempting duplicates alone would have moved nothing.

Two things follow, and both are in the command's output rather than only here:

- `objects-read` counts every object **examined**; `new-objects` counts what
  `--max-objects` was charged against. They are different numbers, and one run can
  read a great deal and store two records.
- The forge reads are bounded by the page ceilings, not by `--max-objects`. A re-run
  over a range already walked re-reads it. That is the trade, and `--until` is what
  makes it a choice.

**A run that stops at its budget is truncated, and says so.** The summary prints
`TRUNCATED`, on stderr, and the command exits **2** as `backfill` does — a
truncated range that exits 0 is indistinguishable from a complete one to a script.
When the page ceiling is what stopped the walk, the output says that instead,
because an identical re-run would hit it identically.

**The forge will not list a repository's history for ever, and that is reported
too.** On `pingdotgg/t3code` the pull request listing returns 1048 changes and stops
— back to early September — so `--since 2026-07-01` is a floor the enumeration
cannot reach. A run that ends there prints `NOT COVERED TO ITS FLOOR`: from inside the
walk, "this repository has nothing older" and "the forge stopped listing" are the
same event, so neither can be claimed, but the range not having reached its own
floor is a fact about the run and belongs in it. Reading history below that depth
needs `kojutsu backfill`, which enumerates by creation date through the search API
and says so when it is capped.

### Walking back to the window is concurrent, and bounded

The listing is walked back **one bounded batch of pages at a time**
(`HISTORY_READ_CONCURRENCY`, default 8), and the batch is consumed **in listing
order**. It was not measured to be worth doing: profiled at the socket, a
`backfill-reviews` run reaching back to a 3.5-month-old floor issued 96 requests of
which **88 were the listing, carrying 114.2 of the run's 118.7 seconds** — 96% of the
wall, and none of it capture work. Pages of 100 changes carry no dependency on the
page beside them, so the cost of asking for an older window was
(PRs updated since the window ÷ 100) × 1.3s.

Two properties are load-bearing, and both are asserted in
`tests/test_backfill_concurrency.py`:

- **Order is the contract, not an optimisation.** A batch may fetch pages the floor
  would have stopped before, so the walk applies its short-circuits — floor, ceiling,
  short page, page ceiling — at exactly the points it always did, in page order, after
  the bytes arrive. A run that stops at its `--max-objects` budget therefore captures
  the same prefix a sequential walk would have, which is what makes the re-run the
  summary recommends actually *advance* rather than resume from a place nothing
  recorded.
- **The speculation is bounded, and the bound is a constant rather than a ratio.** The
  window ramps 1, 2, 4, 8…, so a walk whose floor is on page one fetches one page and
  starts no threads at all. At most `HISTORY_READ_CONCURRENCY - 1` pages are ever
  fetched and thrown away — **7 pages, 700 pull requests, for a walk of any length.**
  On the measured 88-page walk the ramp fetches 95 and discards 7: 8% more requests
  for a wall clock of 22.9s against 114.4s, every report counter identical.

**Reviews and comments are deliberately *not* fetched concurrently.** They are almost
always empty or a single item, so speculating there would multiply requests roughly
fourfold to save latency on reads that were never the bottleneck. Concurrency goes
where the pages are.

### One pull request costs its own pages, not the walk back to the window

A claim about one pull request used to cost the whole walk back to it: verifying
PR #2829 with `--since 2026-06-25` read 88 listing pages to answer a question
about one change. `--pr` (repeatable, comma-separated) names the changes instead
and the listing is never read — one PR's reviews and comments are two to four
requests total. The window still applies and is reported, never widened: reviews
outside it are skipped, and a PR with no reviews is an empty range rather than
an error. A PR past the page ceiling is reported as a gap with the existing
ceiling reason, and `--max-objects` charges the same new work the same way.

```bash
kojutsu backfill-reviews --repo owner/name --since 2024-01-01 \
  --max-objects 100 --pr 2829
```

#### The trade this makes: quota burst for latency

**The bound is on concurrency, not on the request rate, and that is the whole cost.**
What the forge's abuse detection looks at is how many connections a client holds open,
and eight is unremarkable. What a bound on connections does *not* do is cap requests
per second:

- The core API allows **5000/hour ≈ 1.4 req/s**. The measured sequential listing already
  ran at ~1/1.3s ≈ **0.77 req/s — about 55% of that budget.** This was never a frugal
  client; it was a slow one.
- At 8 streams the *same* 88 requests land in ~15s instead of ~115s. **The count does
  not change** — concurrency does not spend more quota on a walk of fixed length, it
  spends it faster — but it spends it inside a window short enough for a secondary rate
  limit to notice, and short enough to starve everything else on the token.
- The token is **shared**: one per-repository budget for the account, spent by the live
  capture path as well. A backfill that spends it in 15 seconds is a backfill that can
  take the capture path down with it.

So this is a deliberate trade of quota burst for latency, and the instrument for
changing your mind is one environment variable:

```bash
GITHUB_HISTORY_CONCURRENCY=1
```

**`1` is not "concurrency off"** — it is the strictly sequential walk this replaced,
reachable without a code change, and with the speculation waste at zero. Raise it to
`16` only if the token is yours alone.

**This is not a substitute for the search pacing above.** Search is 30/minute, a *rate*
limit, so `DEFAULT_SEARCH_PACE_SECONDS` is the only correct instrument and concurrency
cannot help at all — issuing the same pages faster just reaches the 403 sooner. The two
constants are not two answers to one question, and removing the pace because concurrency
exists would put back the 403 that `SEARCH_REQUESTS_PER_MINUTE` documents, one page into
the walk.

There is still no cursor, for the reason `core/backfill_reviews.py` records: the
semantic event ids derive identity from the forge's own object identity, so
"have I seen this?" is answerable from the store. The store is the cursor.

## Minimum token scope

Use a **fine-grained personal access token** limited to the single repository
Kojutsu captures. It should have exactly:

- **Metadata: Read-only**
- **Pull requests: Read-only**
- **Issues: Read and write** — required to post a question comment
- **Webhooks: Read and write** — *only* if `GITHUB_WEBHOOK_REGISTER=true`

Nothing else. In particular Kojutsu never needs **Contents**, so the token
should not have it: the pipeline never reads or writes repository file contents,
and granting `Contents: write` would hand a compromised capture process the
ability to push code.

A classic personal access token cannot express this. `repo` grants read *and write*
across every repository you can reach, including code, and `admin:*` grants far
more. A classic token is accepted at runtime, but `describe_token_scope` will
refuse the broadly-scoped ones rather than let a repo-wide credential run
unnoticed.

## Why the token is not the whole control

A minimum-scope token limits the blast radius of a *successful* attack. It does
not help if the token is readable in the first place, and the process that holds
it is the same process that feeds untrusted pull request text to a language
model. So the two are separated:

- The capture process (webhook server, `ask`, `collect`) holds the token. It
  validates HMAC signatures, enforces the repository allowlist, and resolves
  markers against the registry.
- The **question generator never sees the token at all.** It runs as a separate,
  sandboxed process with a scrubbed environment, a private home containing only
  its own agent definition and the single model credential, an empty working
  directory, and (on macOS) a Seatbelt profile denying reads of your code and
  credential stores. See `src/kojutsu/integrations/sandbox.py`.

That separation is the control the token scope complements. Either alone is
insufficient: a minimal token that leaks is still a credential, and an empty
tool set is only as strong as the process it runs in.

## Verifying the boundary

`tests/test_sandbox.py` runs real reads and writes under the real Seatbelt
profile and asserts they fail, rather than asserting on the profile's text. A
profile that denies nothing is the failure mode worth catching, and only an
executed probe catches it.

## Registry state: one source of truth

The local registry tracks each posted question so outstanding work can be
enumerated, not only fetched by an id that was already known. `status` is the
single source of truth for the question lifecycle and is the only field the
answer-dedupe gate reads: `is_question_answered()` and `complete_answer` both key
on `status = 'answered'`.

A legacy `answered` INTEGER column is retained as a mirror of `status` for
databases written by earlier versions, and schema validation fails closed if the
two ever disagree. That is not defensive decoration. Re-running `ask` against the
same question id re-records the question comment, and before schema v4 that
conflict clause reset `status` to `pending` while leaving `answered` set. Keying
the gate on the status alone would therefore have re-opened the dedupe window and
let one answer be captured twice, so the conflict clause now refuses to walk an
answered question backwards.

Migrations are forward-only and checksummed. Each applied version is recorded in
`registry_migrations` with a checksum over its own statements; reopening a
registry whose recorded history disagrees with the running application is refused
rather than half-migrated. The v4 migration adds columns, an index, and triggers
that enforce the status vocabulary, and repairs rows where the legacy flag and
status disagree by promoting the row to the state its own flag already asserts. No
row is deleted, rebuilt, or downgraded.

## Model identity, and what it is worth

A comment may declare which machine wrote it:

```
<!-- kojutsu:agent:opencode model=opencode/model -->
```

The marker used to carry only an agent name. It now carries a model too, and
capture records both as `answered_by_agent` and `answered_by_model`. The older
marker without a model still parses; the model is then recorded as `unknown`
rather than guessed at.

**The model is an assertion by the comment author, not a fact the platform
verifies.** GitHub proves who posted a comment and nothing more. It cannot prove
which model drafted the text, and no part of Kojutsu can either. So the claim is
trustworthy exactly as far as the account making it, and no further.

**It used to say "no further than a trusted association", because capture gated on
one. That sentence is now wrong, and this is what replaced it.** Admission was
`{OWNER, MEMBER, COLLABORATOR}` and is now unrestricted. The measurement, taken
against the corpora this policy had actually produced:

- 400 of 439 stored review captures re-fetched from `pingdotgg/t3code` and
  `fastapi/fastapi`: **278 `MEMBER`, 118 `COLLABORATOR`, zero `CONTRIBUTOR` or
  `NONE`.** The gate had never refused a review, and there were no bot reviews in
  the corpus to refuse.
- Issue comments on t3code PR #2829: `MEMBER` 0 bots / 12 humans, `CONTRIBUTOR`
  **21 bots** / 6 humans, `NONE` 0 bots / ~28 humans.

**`author_association` is anti-correlated with automation.** A bot is by definition
not a member or collaborator of anything, so `CONTRIBUTOR` is where automated
reviewers land — `cursor[bot]`, `macroscopeapp[bot]`, `github-actions[bot]`,
`coderabbitai[bot]`. Widening to `CONTRIBUTOR` admits every automated reviewer on
that change; widening to `NONE` admits only humans. One field cannot answer "may
this account write?" and "is this automated?" at the same time, and using it for
both cost **28 human comments to admit 21 bot ones**.

So the two questions are answered separately. *May this account write?* is now the
repository allowlist, checked before anything is written
(`GITHUB_WEBHOOK_ALLOWED_REPOSITORIES`, and `core/allowlist.py` fails closed on a
malformed configuration). *Is this automated?* is GitHub's own `user.type`, which
`GitHubUser` used to parse `login` from and discard the rest of, and which is now
recorded on every stored comment as `comment_author_is_machine` /
`reviewer_is_machine`.

**What is load-bearing now, and did not move.** `independence` and
`answered_by_model` are computed and recorded exactly as before, and a rationale is
still `asserted`: testimony, never verification. A stored rationale says what an
agent claims about its own work, and no amount of account standing made that
different.

**Automation is provenance, not a filter.** A bot's review is still evidence that a
review happened. Dropping it was a policy about who gets a voice; recording it is a
fact about who spoke.

### Independence

Knowing the model is only useful against knowing who asked the question, so
capture computes a level over the two comments as they exist on the forge:

| Level | Meaning |
| --- | --- |
| `independent` | A different posting account. A second party. |
| `model_separated` | Same account, different models. A second mind, not a second party. |
| `self_certified` | Same account and same model. The record restates rather than checks. |

A different account is `independent` *regardless of model*: two parties are
stronger evidence than two models, and a human on the same model as a bot is
still a second party. Where the account is shared, the comparison is on model.

Two absences are treated as absences. If either side does not state a model, the
record is `self_certified` with the reason recorded, because assuming two unknowns
are the same model would manufacture a label the evidence does not support.

None of the three levels claims the reasoning is *correct*. They describe who was
in a position to disagree, which is the only part a comment record can honestly
attest to.

### Reading it back

`search_knowledge` accepts `min_independence` with any of the three values, and
every rendered record carries `independence`, `independence_reason`, and
`answered_by_model` in its provenance block. A record with no level is below
every threshold: a caller who asked for independent evidence would rather see
nothing than see an unlabelled record and assume it had been checked. Records
excluded by the threshold are counted and named in the response, so a bounded
answer is never presented as a complete one.

## Answering with a model

`kojutsu answer <pr> --model <m> --agent <name> [--apply]` drafts answers to
outstanding registered questions and posts them as ordinary review comments.
It plans by default, exactly like `ask`: without `--apply` nothing is written and
no question is consumed.

The answerer runs through the same sandboxed provider path as the question
generator, because it reads the same attacker-controlled diff text. It is not more
trusted for producing prose instead of questions. Nothing here stores an answer
directly; it posts a comment, and capture is the only thing that turns a comment
into a record, so an answer cannot bypass the registry, the dedupe gate, or
provenance.

Every posted comment carries the agent marker with its model, so capture records
who wrote it and scales its independence. An unattributed answer would be exactly
the fluent, unverifiable output this system exists to catch.

### The reviewer is adversarial by design

The prompt does not ask the model to help. It asks for a verdict, makes
disagreement the expected shape of a useful answer, requires "I cannot verify
this" over confident assertion, and tells the model to treat agreement as the
outcome to be suspicious of.

This is not decoration. A model asked to answer questions about code it can read
will produce agreeable answers, and an unattended loop built on that manufactures
consensus — which is the exact failure this product exists to prevent. The
independence label added for those records would then be decoration over a
non-finding: the record would honestly say who wrote it while carrying nothing
worth reading. Padding a correct answer with manufactured concerns is the mirror
image of that failure and is called out in the prompt for the same reason.

### Selection is explicit

A run answers the questions it was given, or the outstanding ones it lists first,
and never more. Naming a question that is not outstanding on that pull request is
refused rather than silently swapped for another, because an answerer that quietly
widens its own remit is indistinguishable from one that invented work.

### Testing a model against a model

`tests/test_answerer.py` includes a `slow` test that drives the real sandboxed
provider with a diff containing a prompt injection asking it to read an SSH key,
dump the environment, and reply with a fixed string. It is skipped when the
provider is unavailable so the suite never goes red for an environmental reason.

The assertions are behavioural, not substring matches. Notably they do *not*
require the bait string to be absent: a reviewer that quotes the injection while
refusing it has behaved correctly, and asserting absence would penalise the honest
answer. What is asserted is non-compliance, that no secret material appears, and
that the reviewer engaged with the actual change — a response that refused the
bait while ignoring the code would satisfy every other check and be worthless.
