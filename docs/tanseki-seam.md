# Tanseki seam contract (as used by Kojutsu)

Kojutsu is a consumer of the **Tanseki** knowledge store over a versioned HTTP
API. This document records the canonical surface Kojutsu expects. The client
that implements it is `src/kojutsu/integrations/tanseki.py`; if the contract
changes, only that module (plus the mapping) should need edits.

## Collection

Kojutsu writes to `collection = "kojutsu"` (`TANSEKI_COLLECTION`).

## Transport

- Base URL: `{TANSEKI_URL}/v1`
- Auth: `X-API-Key: <TANSEKI_API_KEY>`
- **Document ids are path-derived and contain `/`** (e.g.
  `org/repo/pr-42/entry-1`), so document operations that address an id use
  **custom methods** — `POST /v1/documents:get|upsert|delete|traverse` — carrying
  the id in the request body, not as a path segment. Collection listing/count uses
  `GET /v1/documents` and does not expose document ids as path segments.
- Upserts send an `Idempotency-Key` (sha256 of `id|content`) so relay replays
  are safe server-side.

## Document mapping

`KnowledgeEntry` → Tanseki `Document` (see `src/kojutsu/core/tanseki_mapping.py`):

| Tanseki field | Value |
|---|---|
| `id` | `<repo>/pr-<n>/<entry_id>` — path-derived, extension stripped |
| `path` | `<id>.md` — adapter-relative, unique per collection |
| `content` | canonical Markdown **including** the frontmatter block |
| `frontmatter.title` | question text (truncated to 200 chars) |
| `frontmatter.author` | entry author (or `"unknown"`) |
| `frontmatter.tags` | list of tag strings |
| `frontmatter.updated_at` | `answered_at` (ISO-8601) |
| `frontmatter.repo` | `owner/name` |
| `frontmatter.pr` | PR number, **as an integer** |
| `frontmatter.jira` | Jira ticket key |
| `frontmatter.category` | question category |
| `frontmatter.record_kind` | `answer`, `review_verdict`, `inline_review_comment`, `pr_lifecycle` |
| `frontmatter.change_author_account` | the account that opened the pull request |
| `frontmatter.pr_opened_at` | the change's `created_at`, ISO-8601 |
| `frontmatter.pr_merged_at` | the change's `merged_at`, ISO-8601 |
| `frontmatter.pr_outcome` | `merged` or `closed_unmerged` |
| `frontmatter.review_id` | the forge's review id |
| `frontmatter.text_sanitisation` | what the character policy removed before storing; **absent when nothing was** |
| `frontmatter.github_author_association` | the association GitHub attributed to the posting account, recorded because the forge says it and not because anything acts on it |
| `frontmatter.comment_author` | the login that posted the comment |
| `frontmatter.comment_author_is_machine` | whether GitHub reports that account as an application; **absent on a review, which carries `reviewer_is_machine` instead** |
| `frontmatter.reviewer_is_machine` | the same question about a *review's* account; `false`, not absent, for a person |
| `frontmatter.pr_url`, `session_id`, `github_comment_id`, `answered_at` | extras |

### Admission is unrestricted, and the two automation flags are what replaced the filter

Every comment on a change in an allow-listed repository is a candidate for storage,
whatever the poster's `author_association`. That was not always so: the set admitted
was `{OWNER, MEMBER, COLLABORATOR}`, and it measured as the wrong filter — on
`pingdotgg/t3code` PR #2829 it discarded **28 human comments to admit 21 bot
ones**, because a bot is by definition not a member or collaborator of anything and
therefore lands on `CONTRIBUTOR`, the very value the set excluded. Over 400 stored
review captures re-fetched from `t3code` and `fastapi` there were 278 `MEMBER`, 118
`COLLABORATOR` and **zero** `CONTRIBUTOR` or `NONE`, so the gate had never refused a
review at all. `docs/github-seam.md` carries the full measurement and
`core/answer_collector.ADMIT_ALL_ASSOCIATIONS` carries the reason in the code.

**What this document must not let a reader conclude.** These records are no longer
filtered by standing, and nothing else about them got stricter:

- The bound on *who can cause a write* is the repository allowlist
  (`GITHUB_WEBHOOK_ALLOWED_REPOSITORIES`), not the author's relationship to the
  project. It is checked before anything is written and fails closed on a malformed
  configuration.
- `github_author_association` is still recorded on every record, because it is what a
  reader weighs now that it gates nothing. Dropping the filter did not drop the fact.
- `independence` and `answered_by_model` are computed exactly as before, so a reader
  can still tell how far a record is from checking itself.
- A rationale is still `asserted` and is still testimony. No account standing ever
  made a declared reason verification, and removing the gate does not.

**`comment_author_is_machine` and `reviewer_is_machine` are what replaced the
filter, and they are provenance rather than a judgement.** They resolve from
GitHub's own `user.type` where the payload carries it, falling back to the `[bot]`
login suffix where it does not. An absent `type` is the forge declining to say, not
a person, so the fallback is the weaker signal and the docstring says so. A bot's
review is still evidence that a review happened; the flag is what lets a reader weigh
it rather than what decides to keep it.

### Values keep the type the model holds

A number, a flag and a list reach the store as a number, a flag and a list. They
used to be stringified at the storage boundary, so `pr` arrived as `"42"` and
`reviewer_is_machine` as `"True"` — spellings, not values. The cost was concrete
rather than aesthetic: a document a person opens shows the number quoted, and
anything numeric (a range, a sort, a count) has to parse the value back before it
can be compared. `github_comment_id`, `review_id` and a question's `attempts` were
affected the same way.

**`fm=` filtering currently matches nothing for an extra, and that was measured.**
Against a real `tanseki-daemon` (main `9ac4130`) a typed value is *stored*
faithfully — `:get` returns `pr_int: 42` as a JSON number beside `pr_str: "42"` in
the same block — but no `fm=` equality matches, for typed values or strings alike:

| stored | filter | matched |
|---|---|---|
| `plain: "hello"` (string) | `fm=plain=hello` | 0 |
| `pr_str: "42"` (string) | `fm=pr_str=42` | 0 |
| `pr_int: 42` (int) | `fm=pr_int=42` | 0 |
| the same document | `q=<body token>` | **1** |

So this is **not** a cost of sending types, and it is not a typed-value limitation —
a plain string extra fails the same way. The document reaches the index, which is
what the free-text hit proves; its frontmatter terms simply do not match a filter.
`LuceneLookup`'s own suite passes and covers this exact case, so the `Lookup` is
correct in isolation and the defect is upstream of it, in the path from a
`documents:upsert` to an indexed document. Filed as Tanseki `P4CCWK8T`.

Two consequences worth stating plainly. Nothing is filterable yet, so preserving a
type buys no *queryable* behaviour at the moment — it is still worth preserving,
because it is what a document shows a person and what any future consumer will
compare against, and the string form is lossy in a way the typed form is not. And
`tags` is separately unfilterable by construction: it is a `CONTRACT_KEY`, the typed
`values` map must not contain it, so `fm_tags` is never indexed at all.

Timestamps stay ISO-8601 **strings**, and the sentinels stay strings: `pr_opened_at`
as a number would be a different kind of wrong, and `answered_by_model`,
`capture_source` and `structure` are read as text by a reader who has to tell "named
no model" from "not recorded here".

A value read back from a document written **before** this is still quoted, so a
consumer must accept both spellings — which is what `allow_string_numbers` on the
read path is for, and why it must outlive the change rather than being tidied up with
it.

Every change-describing key is **absent when the payload did not carry it**,
rather than defaulted. The builders skip `None` and `""`, and that is the only
mechanism here that expresses absence honestly — a placeholder string passes the
same filter and is then indistinguishable, to any reader, from a value somebody
actually stated. `answered_by_model` in particular is absent when no model was
named, rather than reading `unknown`. An **empty collection is not absence**: `[]`
says nothing matched and `None` says nobody looked, and collapsing the two makes an
outage look like a finding, so an extra with an empty list is stored as an empty
list.

## Sanitised text, and the absence that is the claim

Third-party text is stored in a form a reader can trust to render as what it says:
`core/text_hygiene.py` strips the C0/C1 controls and the invisible and
bidirectional formatting characters, and records what it took out under
`text_sanitisation` — one flat sentence naming each code point, its Unicode name
and a count. The characters go rather than being escaped visibly, because an
attacker-chosen character left in stored prose is the rendering attack the control
exists to stop, and the record is never refused for carrying one: a hostile comment
is evidence *of* a hostile comment, so it is stored with its prose and a note.

| Tanseki field | Value |
|---|---|
| `frontmatter.text_sanitisation` | `removed N characters before storing: U+202E RIGHT-TO-LEFT OVERRIDE (x2), …` |

**The key's absence is the claim.** A document with no `text_sanitisation` is
byte-for-byte the contributor's text, and that is a stronger statement than "it
looked fine" — it is the only reason a reader may treat the stored prose as a
quotation. ("Byte-for-byte" modulo canonical composition: `é` arrives as either one
code point or an `e` plus a combining acute, and it is stored as the first, because
those are the same characters to a reader and different bytes to everything
downstream.) So the key is omitted rather than written empty: `""` would be a claim
that sanitisation ran and found nothing, which is a different statement from a
record that never needed one, and `build_frontmatter` skips empty strings, so an
empty value would vanish anyway. The note cannot itself carry a refused character,
so a document cannot report a problem by reproducing it.

That claim is exact for every writer *except* the two described below, where a
narrower upstream pass or a comment that was never itself the stored text makes
silence mean something weaker. Neither exception is left implicit: both are stated
below, because a reader who meets the general claim first has to be told where it
stops.

Which writers carry the key: the answer, the review verdict, the inline review
comment, the lifecycle title and the check name — every field a reader sees
rendered, including a *name*, where an override would be a claim about who wrote
something that nobody made. The clarification carries it too, and the rationale
does as of this revision; see the two paragraphs below, because a clarification is
a **quotation** and that makes its case different from every other writer here.

**A clarification is a quotation, and it is still sanitised.** This is the one
record kind whose author is *any human who can comment on a pull request*, and the
one whose `ClarificationEntry.statement` is documented as the human text verbatim,
so the two considerations genuinely pull against each other and the argument is
written out in `core/clarification_collector.py`'s module docstring. Briefly: the
refused set holds no propositional content — no word, digit, or punctuation a human
meant — and the characters that *do* carry meaning (ZWJ, ZWNJ, the Arabic letter
mark, the invisible plus, the emoji tag block) are excluded from it by name and by
an import-time tripwire. So the cost is zero for honest content, while a byte-faithful
copy containing U+202E is not a *more* faithful quotation: it renders the tail of a
stored sentence reversed, and the reader of a Markdown document or an MCP-served
one has no reason to be looking for that.

**What a reader of a stored clarification is therefore entitled to** is the *wording*,
plus a `text_sanitisation` note saying what was taken out of it, plus a
re-fetchable `github_comment_id` to check the wording against. Not the bytes. A
document with no `text_sanitisation` is byte-for-byte the comment, which is a stronger
statement than "it looked fine" and is the only reason the stored wording may be
quoted as the author's own. The forge-issued comment id is the anchor, and the comment
on the forge is not edited by the filter.

**A rationale document carries it too, and its anchor now reaches the document.**
`build_rationale_frontmatter` copies named fields out of the model, and it now reads
`metadata` as well: for `text_sanitisation` and `github_comment_id`. Previously
neither reached the document, which meant a stored rationale — the most persuasive
unverified record the store could hold — was silent about what had been removed from
it, and had no anchor in it at all. Both were stated here as limits rather than
properties; they are now properties.

A rationale is nonetheless **not** byte-for-byte its comment even when the key is
absent. The marker extractors run `normalise_captured_text` on the way in, which
removes fourteen of the same characters silently before the wider policy runs, and
the capture server posts the reason **unsanitised** while storing a sanitised
rendering of it. So a rationale's silence means "this policy took nothing", not "the
comment was untouched" — and the comment it was read from is the thing to re-fetch.

| Tanseki field | Value |
|---|---|
| `frontmatter.text_sanitisation` | `removed N characters before storing: U+202E RIGHT-TO-LEFT OVERRIDE (x2), …` |
| `frontmatter.github_comment_id` | the forge comment the clarification was quoted from, or the declaration was read from; **absent when a declaration was never posted** |

**One declaration, two routes, one document.** A rationale reaches
`<repo>/pr-<n>/rationale/<entry_id>` from two places: directly from
`mcp_server/capture_server.py`, and later from `core/rationale_collector.py` reading
the comment that capture server posted. They are mutually exclusive per declaration —
`claim_rationale` refuses a row that is already `completed`, and the capture server
completes before it stores — so the direct route writes the surviving document in
almost every case, and the collector's route runs only where the direct one did not
finish. Both therefore apply `sanitise`, and both write the same `text_sanitisation`
note, so a reader need not know which route wrote the document. What differs is the
anchors, and it is a superset: the collector additionally records `comment_author`,
`github_author_association`, `comment_author_is_machine` and `delivery_id` in the
entry's metadata, and dates the declaration from the comment rather than from the
write. Of those, only `github_comment_id` reaches the document — the limitation it
has always had, and for the same reason: this builder reads the model, and the
comment-derived facts live in `metadata`.

The reason is posted **unsanitised** and stored sanitised, which looks like an
inconsistency and is not. The capture process is a courier: if it edited the text it
would be its author, and the collector's note — measured against the delivered comment
— would then find nothing to report and a fallback-route document would claim
byte-for-byte fidelity over prose that had been edited. Posting the raw string keeps
the refused characters in the comment, so the note either route writes names the same
code points with the same counts, and the forge keeps the thing a reader can re-fetch.

`record_kind` is authoritative and `tags` are a projection of it, computed in
`core.answer_collector._record_tags` so the two cannot drift. A record written
before `record_kind` existed has no kind, and that is reported as absence:
inferring one from its tags would record an interpretation as data.

`session_id` identifies a **change**, not a working session. It is a
deterministic `uuid5` over the pull request URL, so it is stable across runs and
one change is one id however many principals touched it. A duration computed from
it is a statement about the pull request.

## Projected decision requests

The question registry is projected into the store as its own record kind, in its
own namespace: `<repo>/pr-<n>/question/<question_id>`. A question is a request
for a reason, never a record of one, so it must not land where a reader would
take it for an answer.

`QuestionRecord` is the **allowlist**, and being a model is the point. It has no
`claim_token` field and no `last_error` field, so `core/question_projection.py`
cannot publish either — not because it filters them out, but because there is
nowhere for them to be read from. `claim_token` is the more important: it is a
`secrets.token_urlsafe(32)` capability and the `WHERE` guard on releasing a
claim, so anyone holding it can release a claim somebody else holds, and Tanseki is
readable over MCP by agents. A deny-list would make the safe behaviour depend on
remembering to update a blocklist, and its failure is silent — a document merely
lacks a key and nothing reports it.

| Tanseki field | Value |
|---|---|
| `id` | `<repo>/pr-<n>/question/<question_id>` |
| `frontmatter.tags` | `question`, `question_<status>` |
| `frontmatter.question_status` | `answered`, `failed`, or `superseded` |
| `frontmatter.status_as_of` | when that status was observed |
| `frontmatter.question_id`, `repo`, `pr`, `pr_url`, `category`, `jira`, `session_id`, `question_author`, `assignee`, `attempts`, `answer_comment_id`, `created_at`, `answered_at` | extras |

**Only terminal statuses are projected.** A `pending` or `claimed` question is
outstanding work: operational, transient, and rewritten on every claim and
release. Projecting it would put a live work queue into a knowledge store, where
it reads as a set of open decisions rather than as requests this process is still
waiting on, and it would put a rewrite-per-lease onto a delivery path built for
immutable records. `created_at` and `answered_at` are both stored, so the wait is
computable without persisting a transient state.

**No `capture_source` and no `independence`,** and the omission is the mechanism:
that is what excludes a question from an evidence-only query, since nobody stated
anything. Writing either field would be a way to make a request look like a
checked conclusion.

**The projection is eventually consistent,** so the status is written with
`status_as_of`. A status with no age attached is a claim about *now* made by a
document about the past, and a reader who cannot see the gap will compute a wait
that includes time the store never saw. The document id is derived from
`(repo, pr, question_id)` and never from the status, so re-projecting an unchanged
question upserts the same document rather than creating a second one.

`Runtime.project_questions()` reports the outbox depth the sweep left behind. A
question is re-derivable from the registry; a captured answer is the only copy of
a human's words. If the sweep is queueing work it is spending the valuable thing
to store the derivable one, and the right response is to reconsider the sweep
rather than tune it.

## Design records in the store

A design proposal and a design reconciliation are both stored as rationale
records, and the four metadata keys that tell them apart now reach the document:

| Tanseki field | Value |
|---|---|
| `frontmatter.design_role` | `proposal` or `reconciliation` |
| `frontmatter.design_plan_digest` | sha256 of the reconciled plan (reconciliations only) |
| `frontmatter.design_proposal_ids` | entry ids the reconciliation covered (reconciliations only) |
| `frontmatter.design_discarded` | per discarded proposal, its entry id, principal and reason |

A proposal carries none of the reconciliation's keys, so those keys are **absent
rather than empty** -- there was nothing to put in them, and the shared absence
rule drops them. `design_discarded: []` on a reconciliation is the opposite
case: the reconciler considered the proposals and discarded none, which is a
result worth keeping rather than a gap worth hiding.

**Stored and free-text findable, but not filterable.** A typed extra is stored
faithfully -- `:get` hands back the value with its type -- but `fm=` equality
filtering matches nothing at all for an extra, typed or not. That was measured
against a real daemon (`9ac4130`) rather than assumed, and it holds for string
extras identically, so it is not a typed-value limitation. Do not write a query
against these keys and expect it to narrow: the document is findable by free
text and readable by a person, and a filter is a promise this seam cannot keep.
The Tanseki-side gap has its own tickets.

**The local read still depends on the entry-id prefix.** Frontmatter lands in
the knowledge store, not in the registry: `rationale_captures` has no metadata
column, so `find_design_reconciliations` separates the two record kinds by the
id prefix and keeps doing so. The day the registry grows a metadata column, the
honest change is to read the column and delete the prefix logic -- not to keep
both and prefer the easier read.

## Check runs

A check run is a **machine report about a commit**, and it is the nearest thing
GitHub offers to an outcome signal. It is stored, in its own `record_kind` of
`check_run`, and never where a review would be.

The distinction is carried all the way through, because it is the whole risk in
putting a CI conclusion into a corpus whose provenance apparatus is built around
who said what:

- the conclusion is stored **verbatim** — `action_required` is not restated as
  "failed", because translating a tool's wording into kojutsu's own is
  editorialising, and the difference between the two is the difference between a
  fact about a check and a judgement about a person;
- the **title** says a check *concluded*; the body says what it *reported*;
- **no independence level.** Nobody was positioned to disagree about whether a
  check passed, so a level would let an evidence filter return it as a second
  opinion;
- the author is the **check**, not the change's author. Naming a tool as the
  author of a record is honest; pretending a person asserted it is not.

| Tanseki field | Value |
|---|---|
| `frontmatter.record_kind` | `check_run` |
| `frontmatter.tags` | `check`, `check_state_<conclusion>` |
| `frontmatter.check_id`, `check_name`, `check_status`, `check_conclusion` | verbatim from the forge |
| `frontmatter.head_sha` | the commit the check ran against |

Only `completed` runs are captured. `created`, `requested` and `rerequested` say
nothing yet, and a re-run arrives under `completed` with a new check run id — so
a re-run is a separate record from the run it replaced rather than an overwrite
of it.

**A branch check has no pull request**, and the anchor rules used to require one.
Rather than invent a `pr_number` or downgrade a genuine signed delivery to
`asserted`, `capture_anchor_gaps` accepts a `check_id` in its place. That *adds* a
checkable anchor rather than weakening the requirement: the check run's id is
forge-issued, globally unique, and names the exact run. Nothing else substitutes —
a record with neither is still missing what makes it checkable. The rule lives in
`models.capture_anchor_gaps` so the write path and the MCP read path cannot
disagree, which is the defect `read-path.md` records having already happened once.

**Not implemented: revert detection.** A revert is a commit whose message says
so, or a cross-reference GitHub infers, which means reading commit messages —
`Contents` scope that `github-seam.md` deliberately withholds. A heuristic over
commit messages is not a signal but a guess with a low threshold, and a corpus
that records guesses as facts is the failure this project exists to prevent. The
gap is recorded rather than papered over with a fragile detector.

## Census records

A census record is an **observation that captured nothing**: a delivery was
processed and no knowledge came out of it. It exists to make the corpus's
denominator visible, because from outside a system that only writes down what it
saw, "we looked and found nothing worth keeping" and "we never saw this" are the
same observation.

**It makes the bias visible. It does not repair it.** The bias from "kojutsu
only knows about changes that attracted attention" stays permanent for anything it
did not witness; only a backfill addresses that, and only by making the corpus
larger.

- **it is not knowledge.** No author said anything, so it has no `independence`
  and is never counted as a capture — the same treatment a rationale gets;
- **no reason, ever.** Kojutsu cannot know *why* nothing was captured: nobody
  commented, the reviewer was not authorised, the marker was malformed, the change
  was never reviewed. Each is a hypothesis about someone else's intent. An
  unauthorised reviewer is the tempting case — the system does know it declined
  the capture — but that is a fact about the system's configuration, not about the
  author, and the fields are absent rather than left empty;
- **`capture_source` is `webhook`.** A real signed delivery produced it; how the
  delivery arrived and what it yielded are different questions. This is also why
  the anchor rules needed no new rule: `delivery_id` is the anchor, and a record
  without one is refused rather than written;
- **it is about an *event*, not a change.** A change opened quietly and reviewed
  the next day with a capture has both. The document id is keyed on
  `(repo, pr, action)` rather than on the delivery, so a redelivery upserts one
  document and a count over census documents is a count over changes.

| Tanseki field | Value |
|---|---|
| `frontmatter.record_kind` | `census` |
| `frontmatter.tags` | `census` |
| `id` / `path` | `<repo>/pr-<n>/census/<action>` |
| `frontmatter.capture_source` | `webhook` |
| `frontmatter.delivery_id` | the delivery that was processed |
| `frontmatter.observed_at` | when the system observed the event |

A reader wanting "changes with no knowledge" must ask for changes with no
knowledge *record*; the census is one cheap input to that question, not the answer.

**Which events write one is in `webhook-integration.md`, and two rows deliberately
do not.** A `pull_request` lifecycle action that stored nothing did not see nothing —
a `None` there means the record for that event already exists, so an observation
would sit beside the capture for the same event. `issue_comment` is not covered at
all: a comment with no question marker is not something this system is configured to
capture from, so its absence is a configuration fact rather than a gap in
observation, and covering it is a decision that has not been made.

**Not backfilled.** A backfill has no delivery behind it, and a census record's
anchor *is* its delivery — "we looked and kept nothing" needs a signed delivery to
be evidence about. A backfill's own silence is reported in its run counts instead,
which is why `kojutsu backfill` does not write observations.

## Backfilled history

`capture_source: backfilled` marks a record reconstructed from history by an
authenticated read that happened **after** the event, by a process that was not
present for it.

The other three sources — `webhook`, `collect`, `asserted` — are all about *how*
the text was obtained. `backfilled` is the only one that also says *when*. That
makes the trust axis answer two questions at once, deliberately: a second axis was
considered and rejected for the schema cost, so the cost is paid here and the
mitigation is that the value's own name says `backfilled`. A reader filtering on
`capture_source` is filtering on both axes, and this document is where that is
stated rather than left to be discovered.

It is named `backfilled` and **not** `reconstructed` because
`RationaleSource.reconstructed` already exists and means something different: a
model *inferred* a reason from a diff. Two axes using one word for different
things, over records both apply to, is the confusion `rationale_link.py` exists to
prevent.

**The guarantee is weaker, and the record says so.** A backfilled record shows what
the forge says *now*, not what it said then: a comment edited since, a review body
rewritten, a review deleted outright are all indistinguishable from ones that were
not. Nothing in the anchor rule can fix that — it can only make the weakness
visible.

**The anchor is the read, not a delivery.** Nothing was delivered to a backfill, so
there is no delivery id to ask for, and asking would push a caller toward
inventing one. `capture_anchor_gaps` requires instead the repo, the change,
`captured_at` (which for this source is the read time — the only evidence there
is), and the id of whatever was read: `github_comment_id` or `review_id`. A
delivery id alone does **not** satisfy it. A backfilled record with neither read
id is reported as missing `capture_read_anchor`.

**PR lifecycle transitions are deliberately not reconstructed.** A lifecycle
record's read *is* the pull request, whose id is `pr_number` — already required of
every captured record, so accepting it as the read anchor would make the rule
vacuous and a record with nothing behind it would pass. Rather than weaken the
rule or invent a review id, `kojutsu backfill` skips lifecycle transitions and
reports them as gaps. Reviews and their inline comments, which do name a review or
a comment, are reconstructed.

`kojutsu backfill` stamps this source through a sink wrapper rather than a
parameter on the collectors, because the collectors were written for signed
deliveries and set `capture_source` from the delivery id they were handed — a
`delivery_id=None` alone would leave a review claiming a delivery that never
existed. There is exactly one writer still: the wrapper rewrites the provenance of
records the existing collectors produced, and rebuilds them through the model so
the anchor rule is enforced rather than bypassed.

## Record structure

`structure` is a **second axis**, deliberately not an extension of
`capture_source`. That one answers *where did this text come from*; this one
answers *was this structure established or inferred*.

| Value | Meaning | What a reader may conclude |
|---|---|---|
| `anchored` | A question was asked, an answer captured, and the record joins them on something checkable | Somebody asked and somebody answered |
| `inferred` | A model matched the two sides | Knowledge exists; the *conversation* did not |

The case where this matters is not exotic. A capture can have a real delivery id
and a real comment id, and still be a question/answer pairing **no one made** —
a model joined them. That record is completely checkable *as text* and unreadable
*as a conversation*, and no value of `capture_source` can express the difference.
Without this axis a reader holding an inferred pairing sees a row identical to one
where a human was asked and answered, with no way to prefer the honest one.

- **`inferred` is allowed in the store, because it is not allowed to be silent.**
  An inferred record is still knowledge, and refusing it would lose the record. It
  must also name the model that inferred it, in `structure_inferred_by`. An
  `inferred` label with no model named is an unattributed guess — it says only that
  *something* matched the two sides, which is the unreadable state the write path
  refuses to construct and the read path flags as an anomaly.
- **`anchored` is written only when it is not the default**, which is the opposite
  of how `capture_source` is treated and is deliberate. Adding the axis must not
  re-identify anything already stored: a document written before it existed is a
  real pairing by construction, so writing `structure: anchored` onto it would add
  a claim the writer never made and change the bytes of every stored record to say
  so. **Absence resolves to `anchored` on read**, so the value that is omitted is
  exactly the value a reader recovers.
- **Nothing detects an unlabelled inference.** No check can, and pretending
  otherwise would be the over-claim this document exists to prevent. What the
  default buys is that a record which *says* `inferred` is never served as a
  captured conversation. See `docs/design-review/record-structure.md`.

## Edge derivation

`repo`, `pr`, `jira` and `files` are the keys Tanseki's `EdgeDeriver` recognises
(`EdgeDeriver.kt` iterates exactly those four), so edges are derived
**server-side** from frontmatter and resolved `[[wikilinks]]`; Kojutsu does not
write edges. `files` resolves each path against a document that already exists, so
a `files` edge stays dangling until something writes documents for those paths —
the same position `repo`, `pr` and `jira` are in today. The frontmatter value is
useful without the edge; the edge is what a future producer would light up for
free.

`files` is stored as a list, bounded at `MAX_FRONTMATTER_FILES` and **saying so when
it truncates**, because a list quietly shortened reads as a complete description of
a change. A file list that could not be read is absent rather than empty: `None`
means the paths were unavailable, `[]` would mean the change touched no files, and
collapsing them would make a store outage look like a file-free change.

### The `files` workaround no longer works around anything, and deleting it would lose history

Tanseki's `WF6ENWNJ` (main `e419ddd`) replaced the hand-rolled frontmatter subset
with a real YAML parser, so `FrontmatterValue` models a sequence and the API seam
no longer flattens an array with `toString()`. **A `files` list sent today arrives
as a list.** The store half of the workaround is therefore obsolete, and it was the
only reason the store needed help.

The seam half is obsolete too: kojutsu's `extra` mapping no longer ends in
`frontmatter[key] = str(value)`, so folding `files` into it would no longer
reproduce the defect locally. The bypass is no longer a workaround.

**Removal is still not a cleanup, and the reason is invisible from this side.** Documents
captured before `WF6ENWNJ` hold the repr as *text* — `files: '["src/a.py"]'`. The
store used to recover those: `EdgeDeriver.parseRefs` stripped brackets and split on
commas, which is why the comma had become load-bearing. `WF6ENWNJ` deleted that
handling, and `Frontmatter.texts(key)` — `values[key]?.leaves()?.mapNotNull { it.asText() }` —
returns a `TextValue`'s whole string as one leaf. Each historical document therefore
now derives a **single `embeds` edge whose target is the literal repr string**, where
it used to derive the real paths.

So there are two honest end states, and the workaround moves only for one of them:
either the pre-`WF6ENWNJ` rows are re-captured or migrated so none survive, or the
bypass stays so those rows remain readable. Deleting it on the strength of "it works
now" is the one option that loses data, and the store can no longer repair it —
the parsing that made those rows recoverable is gone.

`head_sha` is an **anchor, not a verification**. It says which commit the capture
was taken against. It does not establish that the record is still true there, that
the change was reviewed, or that the two correspond to the same code.

Not every record carries these. The review and lifecycle paths read the head and
the changed files straight off the webhook payload; an *answer* does not, because
the commit that matters there is the one the question was asked against, and the
registry does not yet record it.

## Endpoints

| Method & path | Body / params | Returns |
|---|---|---|
| `GET /v1/health` | — | 2xx when healthy |
| `GET /v1/documents` | `?limit=1&collection=kojutsu-real` | `{total}`, used for the document count |
| `POST /v1/documents:get` | `{id, collection}` | Document JSON, or 404 |
| `POST /v1/documents:upsert` | `{collection, id, path, content, frontmatter, author, message}` + `Idempotency-Key` header | `{revision, created}`, or 409 on conflict |
| `POST /v1/documents:delete` | `{id, collection, message, author}` | `{revision}`, or 404 |
| `GET /v1/search` | `?q=&limit=&collection=` + repeatable `&tags=` and `&fm=key=value` | `{hits: [{id, score, snippet}], total}` |
| `POST /v1/documents:traverse` | `{id, rel, depth, collection}` | `{ids: [...]}`, or 404 |

## Guarantees Kojutsu relies on

- **Idempotency** by document id: re-writing the same entry upserts the same
  document, and the `Idempotency-Key` makes relay replays safe.
- **Durability**: Kojutsu also keeps a local outbox, so an Tanseki outage queues
  writes (drained by `uv run kojutsu relay` and on webhook startup).
- **Derived edges** are rebuilt from documents; clients never manage them.
- **Server-side filtering**: `search` pushes `tags` and frontmatter equality
  filters (`fm=repo=org/repo`, `fm=jira=ABC-1`) to the store rather than
  filtering client-side.
- **Count**: `GET /v1/documents?limit=1&collection=kojutsu-real` returns the
  collection's `total`; Kojutsu uses this bounded listing for status/count
  surfaces.
- **Listing**: the dev console's knowledge dashboard reads the same endpoint
  with a larger `limit` to enumerate document ids, then fetches each with
  `:get`. The seam documents only `{total}`, so the client reads the array
  defensively (`documents`/`items`/`results`/`hits`, entries as bare ids or
  objects) and raises rather than guessing silently.
- **Bounded search**: callers cap searches at 50 results. Tanseki has no document
  batch endpoint in the current contract, so Kojutsu fetches ranked document
  GETs with at most four concurrent requests and omits stale search hits.
- **`total` is required by `/documents` and optional in `/search`.** The endpoint
  table above is the documented seam, which lists `total` for both. The live Tanseki
  build disagrees for search: its paginated envelope is `{hits, limit, offset,
  hasMore}` and carries no `total` at all, so a client that requires the documented
  field refuses every search against it. The contract was resolved at the **client**,
  not by changing the documented seam, because the two endpoints are genuinely
  different: a document count is a number Kojutsu displays, while a search total
  is a number it does not need in order to page — `hasMore` and `offset` answer that.
  So `_total` (required, rejects a missing or malformed value) reads `/documents`,
  and `_optional_count` reads `/search`, where a missing field is legitimate and a
  *present* malformed one is still a broken response. Both raise `TansekiResponseError`
  rather than coercing, so a response that changes shape is loud instead of being
  read as zero.
- **404 semantics**: `:get` → `null`, `:delete`/`:traverse` → not-found.
