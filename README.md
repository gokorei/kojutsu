# Kojutsu

Capture developer knowledge during code review by asking context-aware questions based on PR diffs and Jira tickets. Captured decisions are stored in the **Tanseki** knowledge store; transient ingestion state lives in a local SQLite registry.

The project consists of two servers:
1. **Data Ingestion Server** - Handles capturing knowledge from PRs (via CLI and webhook)
2. **MCP Server** - Provides tools for LLMs to search and retrieve the captured knowledge

The MCP surface is two processes rather than one: `kojutsu-mcp` reads, and
`kojutsu-capture` records a stated rationale. They are separate so the read side
stays structurally read-only. See [MCP Capture Server](#mcp-capture-server).

## Quick Start

```bash
# 1. Copy .env.example to .env and configure
cp .env.example .env

# 2. Install the locked development environment
uv sync --locked --all-extras

# 3. Configure TANSEKI_URL to a running Tanseki knowledge store (separate service)

# Ask a question on a PR and post the marked questions for human answers
uv run kojutsu ask https://github.com/owner/repo/pull/123 --apply

# 5. Use the MCP Server to search knowledge (in another terminal)
uv run kojutsu-mcp
# Then use the MCP client to call tools like search_knowledge
```

> Full local setup — starting Tanseki and Kojutsu together, plus a one-command
> end-to-end smoke (`scripts/dev-e2e.sh`) — is in
> [`docs/tanseki-quickstart.md`](docs/tanseki-quickstart.md).
>
> Running a second corpus — a different Tanseki collection, a different set of
> repositories — is `kojutsu --instance <name>`. Instances live in the committed
> [`kojutsu.toml`](kojutsu.toml); see
> [`docs/configuration.md`](docs/configuration.md). That file holds no credentials
> and the loader refuses one.

## Docker

The webhook server has a production-oriented image definition. The example below
keeps both SQLite files on the mounted `/data` volume even when `.env` contains
the home-directory paths from `.env.example` for local, non-Docker use:

```bash
docker build -t kojutsu .
docker volume create kojutsu-data
docker run --rm --env-file .env \
  --add-host=host.docker.internal:host-gateway \
  -e TANSEKI_OUTBOX_PATH=/data/tanseki-outbox.db \
  -e KOJUTSU_REGISTRY_PATH=/data/registry.db \
  -e TANSEKI_URL=http://host.docker.internal:8088 \
  -e OLLAMA_URL=http://host.docker.internal:11434 \
  -p 8000:8000 \
  --mount type=volume,src=kojutsu-data,dst=/data kojutsu
```

The explicit `-e` options override `.env`, including its local SQLite paths.
Docker Desktop provides `host.docker.internal` on macOS. On Linux, the shown
`--add-host=host.docker.internal:host-gateway` option maps it to the host. If
Tanseki and Ollama run as containers on the same Docker network, use their service
DNS names, such as `http://tanseki:8088` and `http://ollama:11434`, instead of
`localhost` or `host.docker.internal`.

The container runs `kojutsu serve` as a non-root user and listens on port `8000`.
A publicly bound server requires `GITHUB_WEBHOOK_SECRET`; provide all service
configuration through environment variables or `--env-file`. Do not replace the
`/data` overrides with home-directory paths.

## Environment Variables

```bash
# Required for GitHub operations
GITHUB_TOKEN=github_pat_xxxxxxxxxxxxx   # fine-grained token limited to the single repository being captured

# Required for webhook processing
GITHUB_WEBHOOK_SECRET=your_webhook_secret
GITHUB_WEBHOOK_ALLOWED_REPOSITORIES=owner/repo,owner/other

# Tanseki knowledge store (required) — the consumer seam
TANSEKI_URL=http://localhost:8088
TANSEKI_API_KEY=
TANSEKI_COLLECTION=kojutsu-real
TANSEKI_OUTBOX_PATH=~/.kojutsu/tanseki-outbox.db

# Local SQLite registry for ingestion state (questions, dedupe, sessions)
KOJUTSU_REGISTRY_PATH=~/.kojutsu/registry.db

# Read log: one local JSON Lines event per MCP read (opt-in; off by default).
# Enabling it fixes a start date nothing can move earlier. A read log is a
# behavioural record, so it is bounded on both age and entries, and anything
# removed is reported on the server's stderr rather than silently. See
# docs/design-review/read-log.md.
READ_LOG_ENABLED=false
READ_LOG_PATH=~/.kojutsu/read-log.jsonl
READ_LOG_MAX_AGE_DAYS=7
READ_LOG_MAX_ENTRIES=10000

# LLM Configuration (pick one)
LLM_PROVIDER=openai        # openai, anthropic, or ollama
LLM_MODEL=gpt-4o
LLM_API_KEY=sk-...         # Required for openai/anthropic
LLM_EXTERNAL_ENABLED=false # Required before sending repository data to OpenAI/Anthropic
LLM_ALLOWED_REPOSITORIES=org/repo,org/other
LLM_TIMEOUT_SECONDS=30
LLM_RETRIES=1

# Or use local Ollama; no data leaves the machine through the LLM provider
OLLAMA_URL=http://localhost:11434
LLM_MODEL=ollama/llama3

# Optional: Jira integration
JIRA_URL=https://example.atlassian.net
JIRA_USERNAME=you@example.com
JIRA_API_TOKEN=your_api_token
```

Webhook processing requires both `GITHUB_WEBHOOK_SECRET` and a non-empty
`GITHUB_WEBHOOK_ALLOWED_REPOSITORIES`. Repository names are case-insensitive
`owner/repo` entries. An empty or missing allowlist denies every repository with
HTTP 403. The wildcard `*` permits every repository only when the configured
webhook URL is local; public webhook URLs still require explicit repositories.
Do not use the wildcard for a public deployment.

## Data Ingestion Server

This server handles capturing knowledge from PRs.

### Commands

| Command | Description |
|---------|-------------|
| `kojutsu ask <PR-URL>` | Generate questions for a PR and post as GitHub comments |
| `kojutsu design` | Run one design phase and stop for review before creating tickets |
| `kojutsu collect <PR-URL>` | Manually collect answers (fallback if webhook not configured) |
| `kojutsu search -n 10` | Search captured knowledge (via Tanseki) |
| `kojutsu serve` | Run webhook server for real-time capture |
| `kojutsu relay` | Drain queued writes to Tanseki from the outbox |
| `kojutsu outbox` | Show pending Tanseki writes |
| `kojutsu questions` | List questions still awaiting an answer |
| `kojutsu status` | Show Tanseki reachability, outbox backlog, and registry path |
| `kojutsu console` | Run the read-only local Tanseki verification console |
| `kojutsu webhook-register/unregister/status` | Manage GitHub webhooks |

### How It Works

External LLM processing is denied by default. Before enabling OpenAI or Anthropic,
review their data-processing terms, set `LLM_EXTERNAL_ENABLED=true`, and list every
`owner/repo` that may cross the provider boundary in `LLM_ALLOWED_REPOSITORIES`.
Kojutsu sends only a bounded diff and bounded Jira fields, redacts common
credentials and email addresses, and strictly limits generated output. Local
Ollama does not require external-provider opt-in.

#### 1. Ask Questions

```bash
uv run kojutsu ask https://github.com/org/repo/pull/123 \
  --plan-file ./question-plan.json
uv run kojutsu ask --apply --plan-file ./question-plan.json
```

The plan file preserves the exact generated question IDs and text, is owner-only, and
is never overwritten. Applying the same plan repeatedly is idempotent; concurrent
applies serialize on an owner-only lock. Existing comments are reconciled only when
both their Kojutsu marker and authenticated GitHub author provenance match.

This:
1. Fetches the PR diff and changed files
2. Extracts Jira ticket from branch name (e.g., `feature/PROJ-123-description`)
3. Calls LLM to generate relevant questions
4. Persists the exact plan and posts its questions as authenticated GitHub comments

#### 2. Design a Phase, Then Approve It

Reconcile several proposals into one plan, review it, then create the tickets:

```bash
uv run kojutsu design --repo org/repo --topic "postmortem for the sync stall" \
  --proposer alice --proposer bob --reconciled-by carol \
  --plan-file ./design-plan.json

uv run kojutsu design --repo org/repo --topic "postmortem for the sync stall" \
  --approve --approved-by dave
```

Two invocations of one command. The first captures each proposal, reconciles the panel
**once**, prints the reconciliation — goals, each decision with its rationale and the
alternatives it rejected, the drafts with their dependency edges, what was discarded —
and stops. Exit code **3** means gated, which is neither success nor failure. Nothing is
created until you approve.

The second invocation reads the recorded reconciliation rather than running the model
again, so the plan you approved is the plan that gets built. Re-running the reconciler
would produce a *different* plan, which would either be created under an approval you
gave for the first or be refused as "the plan changed" for something you never did.

Two recorded reconciliations for one topic is a refusal naming both, not a silent choice.
`--proposer`/`--reconciled-by` are refused on an `--approve` run and vice versa, so a
flag that cannot mean anything is never quietly ignored. `--repo` must be in the capture
allowlist, because this command writes records into the same corpus capture does.

#### 3. Collect Answers

When developers answer in PR comments, they include the hidden association marker:

```
<!-- kojutsu:answer:abc-123 -->

We chose this because the alternative had performance issues...
```

The marker, rather than comment order, identifies the question. Invalid, stale, and question-author answers are ignored.

#### 4. Auto-Capture (via Webhook)

For real-time capture, run the webhook server:

```bash
# Terminal 1: Start server (requires public URL for GitHub webhooks)
ngrok http 8000
uv run kojutsu serve --port 8000

# In GitHub: Settings > Webhooks
# Payload URL: https://your-ngrok-url.io/webhook/github
# Events: Issue comments, Pull requests
```

### 4. Search Knowledge

```bash
# Search all captured knowledge
uv run kojutsu search "authentication"

# Filter by repo or Jira ticket
uv run kojutsu search --repo owner/repo --jira PROJ-123
```

The webhook service exposes `/webhook/health` for liveness and
`/webhook/ready` for readiness; readiness returns 503 until the process owns its
outbox and Tanseki is configured and reachable. `/webhook/status` reports configuration
presence, Tanseki reachability,
outbox pending/retrying/dead-letter counts, and the registry path. Keep the status
endpoint on a trusted interface because it exposes operational paths and health.

### 5. Relay & status

Writes are captured durably in a local outbox first and sent to Tanseki; transient
failures use bounded backoff and permanent failures become dead-letter entries.
The webhook server also runs a background relay. To drain or inspect it manually:

```bash
uv run kojutsu relay     # send queued writes to Tanseki
uv run kojutsu outbox    # list pending writes (attempts, last error)
uv run kojutsu status    # Tanseki reachability, outbox backlog, registry path
```

### 6. Outstanding capture work

Every question Kojutsu posts is tracked in a local registry so the work
waiting on a human can be found, not just fetched by an id that was already
known. A question moves through a closed lifecycle:

| State | Meaning |
|-------|---------|
| `pending` | Posted, still waiting for an answer. Outstanding work. |
| `claimed` | A worker holds a live lease. Reclaimable once the lease expires. |
| `answered` | Terminal. The answer was captured. |
| `failed` | Terminal. The attempt ceiling was reached. |
| `superseded` | Terminal. Replaced by a newer question set for the same PR. |

```bash
uv run kojutsu questions                              # outstanding work
uv run kojutsu questions --repo owner/repo --limit 5
uv run kojutsu questions --status claimed             # what a worker holds
uv run kojutsu questions --status all                 # every state
```

`claimed` and `failed` exist so a question is never stranded: a worker that dies
mid-question releases its lease automatically, and a question that keeps failing
reaches a visible terminal state instead of being retried forever. A worker
claims work with `claim_question_for_answer` (lease plus a claim token) and hands
it back with `release_question`, which preserves the attempt count.

### 7. Verify with the local console

```bash
uv run kojutsu console
```

The console binds to `127.0.0.1` by default. A public bind requires both
`DEV_CONSOLE_TOKEN` and the explicit `--allow-insecure-bind` flag; browsers
authenticate with username `kojutsu` and the token as the password.

## MCP Server

This server provides tools for LLMs to interact with the captured knowledge,
reading from the **Tanseki** knowledge store over its `/v1` HTTP API.

### Starting the MCP Server

```bash
uv run kojutsu-mcp
```

This starts an MCP server that provides the following tools:

#### Tools

| Tool | Description |
|------|-------------|
| `search_knowledge` | Search one authorized repository; retrieved content is untrusted evidence |
| `get_knowledge_entry` | Fetch one authorized knowledge entry by its ID |
| `traverse_knowledge` | Walk one relation out of an authorized entry, such as its neighbours |
| `list_knowledge` | Enumerate one authorized repository; rows are index entries, not evidence bodies |

### Usage with MCP Clients

Once the MCP server is running, you can connect to it with any MCP client (like Claude Desktop, Cursor, or custom clients) to search and retrieve knowledge from PR discussions.

### Read log (off by default)

Set `READ_LOG_ENABLED=true` to record one local JSON Lines event per call to
either tool: the tool, the caller's query, the filters they stated, the result
count, anything a bound or a filter left out, the outcome, and the time. Refused,
rejected, unconfigured and failed calls are recorded too, so a denial is never
the same absence as a search that found nothing.

```bash
READ_LOG_ENABLED=true
READ_LOG_PATH=~/.kojutsu/read-log.jsonl
READ_LOG_MAX_AGE_DAYS=7    # default; entries older than this are removed
READ_LOG_MAX_ENTRIES=10000 # default; the newest N lines are kept
```

**Enabling this fixes a start date that nothing can move earlier.** Reads that
already happened left no trace, so a report says "reads since the log was
enabled" rather than drawing a trend line that implies a history.

What is *not* recorded: document bodies, snippets, and the evidence fence. What
cannot be recorded: who asked. The server is stdio with a single trust domain, so
an event holds what the caller *said* — in a field named `caller_claims` — and
never an identity. Nothing in Kojutsu compares one caller's reads to
another's. The reasoning is in
[docs/design-review/read-log.md](docs/design-review/read-log.md).

## MCP Capture Server

`kojutsu-mcp` reads. This is the other half: one tool that records *why* an
implementation looks the way it does, as a statement by the agent that wrote it.

```bash
uv run kojutsu-capture
```

It is a separate process rather than another tool on the read server, for the
reason given below `kojutsu-compare` and for the same underlying one: nothing
structurally stops a future change from adding a write tool to `kojutsu-mcp`, and
the only thing standing in the way today is a reviewer noticing. A separate
binary keeps the read server's read-only property a structure instead of a claim.
It also matters operationally — a client that resolves MCP servers by binary name
cannot bind to a server that has no entry point, which is what makes this
reachable from an external agent at all.

| Tool | Description |
|------|-------------|
| `record_decision_rationale` | Record a declared rationale; with `pr_number` it is also published as a marked comment on that pull request |

The declaration is a claim by its author about its own intent, never evidence
that the code is correct, and it never counts as an independent check on the
change. Capture is what turns a comment into a stored record, so the call posts a
comment and the ordinary pipeline stores it — there is no privileged write path
here that bypasses the registry, the dedupe gate, or provenance. Calling again
with a higher `revision` appends rather than replaces; calling again with the
same identity stores nothing and says so.

### Environment when the server is spawned as a subprocess

Most people run this from a shell, where `.env` in the working directory is read
and none of the below needs saying. It matters for an **agent host that starts
the server itself**: the framework's `MCPClient` passes a base allowlist (`HOME`,
`LANG`, `PATH`, `TMPDIR`, …) plus whatever the server config supplies, so a
variable you did not put in that config simply does not arrive. Verified by
running the wheel-installed binary against nothing but that allowlist:

| Variable | If it is missing |
|----------|------------------|
| `GITHUB_WEBHOOK_ALLOWED_REPOSITORIES` | Every call is refused `repository_not_authorized`, before anything else happens. This is checked first, so a missing allowlist looks like a permissions problem rather than a configuration one. |
| `TANSEKI_URL` | The call fails `capture_unavailable`; the store client refuses an empty URL. |
| `GITHUB_TOKEN` | Needed **only** when the call passes `pr_number`. Without a pull request there is no comment to post, and the call succeeds with no token at all; with one, the empty `Bearer` header is rejected before the request leaves. |
| `KOJUTSU_REGISTRY_PATH` | Defaults to `~/.kojutsu/registry.db`, which works. Pass it explicitly when `$HOME` is not durable or not shared with the process that later relays the outbox — a sandboxed agent's home directory is neither. |
| `TANSEKI_OUTBOX_PATH` | Same default and same reasoning; the relay that delivers the queued record has to open the same file. |
| `TANSEKI_API_KEY` | **Not** required by the call. The capture server enqueues; the relay authenticates, so a missing key fails later and somewhere else. |

That table is the reason to pass the values explicitly even where a default
exists. The allowlist and the URL have no usable default and the call cannot
succeed without them; the two SQLite paths have defaults that are only correct
when the writing process and the relaying process agree on what `$HOME` means,
which is exactly the assumption an agent sandbox breaks. A refusal naming
`repository_not_authorized` almost always means the allowlist never arrived.

## Rationale Comparison

One change produces two rationales: a `declared` one, stated by the agent that did
the work, and a `reconstructed` one, inferred from the diff by a reviewer. They are
stored as two separate records — see
[the rationale decision record](docs/design-review/rationale.md) — and
`kojutsu-compare` reports what their relationship is.

```bash
# Compare one change's stated and inferred rationales
uv run kojutsu-compare owner/repo#123

# A GitHub pull request URL works too
uv run kojutsu-compare https://github.com/owner/repo/pull/123
```

| Outcome | Meaning |
|---------|---------|
| `divergent` | Both exist and disagree, so the intent is not visible in the change. |
| `concurrent` | Both exist and agree, across separate principals. |
| `restatement` | Both exist, same principal and same model. Agreement carries **no** information. |
| `declared_only` | Only a stated rationale exists. An absence of evidence, not agreement. |
| `reconstructed_only` | Only an inferred rationale exists, and is labelled a guess about intent. |
| `neither` | No rationale of either kind exists for the change. |

The independence level comes from the same provenance scale `search_knowledge` uses
and is printed before either text, so a reader meets "these are the same account"
before meeting prose that reads like two agreeing opinions. Rationale documents the
command could not reach or could not use — a full page of results, a superseded
revision, a reason it declined to clip — are named in the output rather than
dropped, so a bounded comparison never reads as a complete one.

The report says whether two statements agree, disagree, or come from the same mind.
It does not say which of them is right, and it stores nothing: both inputs are
already in Tanseki, so keep them and recompute.

This is a separate console entry point rather than a tool on `kojutsu-mcp`
because `tests/test_mcp_server.py` pins the read server's tool table to exactly
`{search_knowledge, get_knowledge_entry, traverse_knowledge, list_knowledge}` and
asserts every one of them is read-only. A separate process keeps that boundary
structural rather than a property of a tool table. It is a read surface, so it is
governed by the same `GITHUB_WEBHOOK_ALLOWED_REPOSITORIES` allowlist as
`search_knowledge`, and a repository outside the allowlist is refused before
anything reaches the store.

## Setup Guide

### GitHub Personal Access Token

1. GitHub → Settings → Developer settings → Fine-grained personal access tokens
2. Limit the token to the single repository being captured.
3. Grant only the pull-request and issue-comment permissions required for capture.
4. Copy and add it to `.env`; do not reuse a production or unattended token.

### Webhook for Real-Time Capture

Only needed if you want automatic capture. Otherwise use `uv run kojutsu collect` manually.

**Local development:**
```bash
# Get public URL
ngrok http 8000

# Configure GitHub webhook
# Settings > Webhooks > Add webhook
# - Payload URL: https://xxx.ngrok.io/webhook/github
# - Content type: application/json
# - Events: Issue comments, Pull requests
```

### Tanseki knowledge store

Captured knowledge lives in **Tanseki**, a separate service (its own repo). Point
Kojutsu at it with `TANSEKI_URL`; there is no local knowledge database. Ingestion
state (posted questions, dedupe, sessions) and the durable outbox are local
SQLite files. The supported deployment is one Kojutsu process, one replica,
and one Uvicorn worker; keep these files on persistent storage and back them up.
`KOJUTSU_SQLITE_WORKERS` and `KOJUTSU_SQLITE_REPLICAS` are guardrails that
must remain `1`; startup rejects other values, and they do not coordinate
multiple processes. Do not run multiple workers, containers, or replicas against
the same local files. Multiple replicas require replacing the registry and outbox
with a shared transactional store before deployment.

Both local files are opened with `journal_mode=WAL` and `synchronous=FULL`, so an
acknowledged write has reached stable storage. Startup fails loudly rather than
running without them, because the durability claim above depends on it.

> **Do not copy these files with `cp`.** In WAL mode the data lives partly in a
> `-wal` sibling, so a copy of the main file opens without error and is missing
> the most recent writes. Back up the whole set, or use SQLite's `.backup`
> command. See [`docs/design-review/durability.md`](docs/design-review/durability.md).

### What is in those files, and what that means

Both databases are **plaintext SQLite**. There is no encryption at rest and no
external secret store: anyone who can read the file can read every question, every
answer, every author's GitHub login, and the queued knowledge waiting to be written.
Restrictive permissions are the only protection, so the honest summary is that this
is protection against other *accounts on the machine*, not against anyone who gets
the file itself.

What each file holds:

| File | Contains | Why it is not merely a cache |
|---|---|---|
| `KOJUTSU_REGISTRY_PATH` | Question text, answer text, author logins and associations, PR and Jira URLs, delivery and claim state | It is the record of what was asked and by whom, including the independence level a reader relies on |
| `TANSEKI_OUTBOX_PATH` | Full knowledge records queued for Tanseki, plus dead letters with their last error | A dead letter can hold an entire unsent record, including a captured answer and its author |

The file is set to `0600` and narrowed on every open, so an existing file that is too
permissive is corrected rather than trusted. The parent directory is created `0700`
when Kojutsu creates it — a directory you made yourself is left as you made it,
because silently chmod'ing a path an operator chose is its own surprise, and the file
inside it is unreadable regardless. A database path that is a symlink, is not a
regular file, or is owned by another user is refused outright. This is asserted for both files in
`tests/test_question_registry.py` and `tests/test_outbox.py`, including across a
close and reopen.

**Retention.** Completed delivery rows are pruned on a bounded schedule
(`READ_LOG_*`-style configuration aside, retention is a constructor argument with a
default), and dead letters are removed by `kojutsu outbox-cleanup` — a dead letter
is not a queue, and leaving one to accumulate turns a transient store failure into a
permanent one. `kojutsu outbox-dead-letters` lists them first. Neither database is
pruned automatically beyond that: the registry is the durable record of what was
asked, and deleting from it silently changes what the program believes it has already
done.

**Backups.** A backup is only useful if it was taken with `.backup` or by copying the
whole `-wal` set, and it is only safe if it lands somewhere with the same protections —
a backup written to a world-readable directory is a plaintext copy of every answer
and author. `docs/tanseki-quickstart.md` has a drill that verifies a restore rather than
assuming one.

The reasoning behind these choices, and behind the work still open, is recorded
in [`docs/design-review/`](docs/design-review/README.md).

## Development

```bash
# Run tests; the 82% floor in pyproject.toml is enforced here, as in CI
uv run pytest --cov=src/kojutsu --cov=mcp_server --cov-report=term-missing

# Run lint and type checks
uv run ruff check src tests mcp_server scripts app.py
uv run mypy

# Run the data ingestion server with auto-reload
uv run kojutsu serve

# Run the MCP server
uv run kojutsu-mcp
```

## Architecture

- **Data Ingestion Server**:
  - CLI: Typer-based commands (`ask`, `collect`, `search`, `serve`)
  - Storage: the **Tanseki** knowledge store (via its HTTP API); a local SQLite
    registry holds ingestion state (questions, dedupe, sessions)
  - LLM: Pluggable providers via litellm (OpenAI, Anthropic, Ollama)
  - GitHub: REST API for PR operations, webhooks for event handling

- **MCP Server**:
  - Protocol: Model Context Protocol (MCP) for LLM tool access
  - Storage: the same Tanseki knowledge store
  - Tools: search and retrieval for LLMs

## License

Kojutsu is licensed under the GNU Affero General Public License v3.0 only. See [`LICENSE`](LICENSE) for details. Contributions are welcome; see [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`SECURITY.md`](SECURITY.md).

Note that the AGPL is a network-copyleft license: if you run a modified
version as a network service (including the webhook server), you must offer
its users the corresponding source. See `LICENSE` §13.
