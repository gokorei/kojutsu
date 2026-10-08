# GitHub Webhook Integration

Kojutsu captures developer answers by listening to GitHub webhooks. Each answer must carry an explicit Kojutsu answer marker identifying the question it answers; chronological comment adjacency is not used.

## How It Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Kojutsu Knowledge Capture                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. `uv run kojutsu ask https://github.com/org/repo/pull/123`             │
│     → Fetches the PR diff and optional Jira ticket                           │
│     → LLM generates relevant questions                                      │
│     → Posts questions as GitHub comments                                    │
│                                                                              │
│  2. Developer reviews the PR and adds an answer marker to the answer       │
│                                                                              │
│  3. GitHub sends a signed webhook to Kojutsu                             │
│     → /webhook/github receives the issue_comment event                       │
│     → Resolves the marker to the registered question                         │
│     → Queues the Q&A pair in the local outbox and writes it to Tanseki         │
│                                                                          │
│  4. Review evidence arrives on its own events                               │
│     → pull_request_review: the verdict, plus every inline comment           │
│       submitted with it, anchored to file and line                          │
│     → pull_request `synchronize`: a new commit landed, so the questions     │
│       asked about the previous diff are marked superseded                   │
│     → Stored as review records, never as PR lifecycle entries               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Events the server accepts

| Event | Captured | Observed when nothing is |
|-------|----------|--------------------------|
| `issue_comment` | An answer comment carrying a question marker | not observed |
| `pull_request` (`opened`, `reopened`, `closed`) | A PR lifecycle transition | not observed — `None` here means the record for *this event* exists |
| `pull_request` (`synchronize`) | New commit: outstanding questions for that PR are superseded | yes |
| `pull_request_review` | The review verdict, and each inline comment with its file/line anchor | yes, when no verdict and no comment was capturable |
| `check_run` (`completed`) | The machine report about a commit | not observed |

The third column is the **census**: a record saying this delivery was processed and
nothing was captured from it. It exists because from outside a system that only
writes down what it saw, "we looked and found nothing worth keeping" and "we never
saw this change" are the same observation — see `tanseki-seam.md`.

Two rows say "not observed" deliberately, and the reason is the same in both:

- **A lifecycle action that stored nothing did not see nothing.** A `None` return
  there means the record for that very event is already stored, or the action was
  rejected, or the change is unnamed. Writing an observation from it would put a
  "nothing was captured" document beside the capture for the *same event*, and a
  count over census documents would then report changes that did produce
  knowledge.
- **`issue_comment` is not observed at all.** A comment with no question marker is
  not an event this system was configured to capture from, so its absence is a
  configuration fact rather than a gap in what was observed. Covering it is a
  separate decision and has not been made.

No observation is ever written for a delivery that was **ignored** (rejected action,
repository outside the allowlist) or **deduplicated** — neither was processed, and
both are common. A review that was processed and captured nothing *is* observed,
and the reason is not recorded, because a decision about what to keep is a fact
about this system's policy rather than about the reviewer's intent.

**Admission is unrestricted.** Every comment on a change in an allow-listed
repository is now a candidate for storage regardless of the poster's
`author_association`. That set used to be `{OWNER, MEMBER, COLLABORATOR}` and it
measured as the wrong filter — on t3code PR #2829 it discarded 28 human comments
to admit 21 bot ones, because a bot is by definition not a member or collaborator
of anything and so lands on `CONTRIBUTOR`. See `docs/github-seam.md` for the full
measurement. What still bounds a write is the repository allowlist above, and what
now tells a reader who spoke is `comment_author_is_machine` / `reviewer_is_machine`
on the stored record, resolved from GitHub's own `user.type`.

Narrowing is still possible, on the paths that take a policy
(`backfill-reviews --authorized-associations`, and `ClarificationPolicy`). The
webhook has no such setting, so from here a declined review is only ever declined
for having nothing to capture or for having no attributable account.

Every event goes through the same repository allowlist, signature check and
delivery dedupe. A repository outside
`GITHUB_WEBHOOK_ALLOWED_REPOSITORIES` is refused identically for review events as
for comment events, and a redelivered review collapses to one record.

A review verdict is stored as its own record kind, not as a lifecycle entry. A
lifecycle entry records what the write path did; a verdict is a person's judgement
about the code, and keeping them in separate records is what stops a later reader
quoting a tool's own output as a reviewer's decision. Every review record carries
the reviewing principal and an `independence` level, so a verdict from a different
account can be told apart from one that came from the same account and model.

Review text is untrusted input, exactly like a pull request diff, and is stored as
quoted evidence. A `changes_requested` verdict is the strongest signal this ledger
captures; a body that reads like instructions to a model is still just a body.


## Requirements

### 1. Run the webhook server

Start the webhook server with a public URL for GitHub webhooks:

```bash
# Option 1: Use a tunnel for local development
ngrok http 8000

# Option 2: Deploy the included container or another HTTP server
# Use the deployed URL in GitHub webhook settings
```

```bash
uv run kojutsu serve --host 0.0.0.0 --port 8000
```

A publicly bound server requires `GITHUB_WEBHOOK_SECRET`; the server fails closed when it is unset. An explicit public `--webhook-url` is required only when `GITHUB_WEBHOOK_REGISTER=true`; the bind address is not treated as a public callback URL.

### 2. Configure the GitHub webhook

In your GitHub repository **Settings > Webhooks**:

| Setting | Value |
|---------|-------|
| Payload URL | `https://your-server.com/webhook/github` |
| Content type | `application/json` |
| Events | Select "Issue comments", "Pull requests" **and "Pull request reviews"** |
| Secret | Set the same value as `GITHUB_WEBHOOK_SECRET` |

### 3. Configure Kojutsu

Copy `.env.example` to `.env` and set the required service configuration:

```bash
# Fine-grained token for the single repository being captured. Metadata (read),
# Pull requests (read), Issues (read+write). Not Contents. See docs/github-seam.md.
GITHUB_TOKEN=github_pat_xxxxxxxxxxxxx
GITHUB_WEBHOOK_SECRET=your_secret_here
GITHUB_WEBHOOK_ALLOWED_REPOSITORIES=owner/repo,owner/other

TANSEKI_URL=http://localhost:8088
TANSEKI_API_KEY=
TANSEKI_COLLECTION=kojutsu-real
TANSEKI_OUTBOX_PATH=~/.kojutsu/tanseki-outbox.db
KOJUTSU_REGISTRY_PATH=~/.kojutsu/registry.db

LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
LLM_API_KEY=sk-...
```

`GITHUB_WEBHOOK_ALLOWED_REPOSITORIES` is required for processing; the values
are case-insensitive, comma-separated `owner/repo` entries. An empty or missing
allowlist denies every repository with HTTP 403. `*` permits every repository
only for a local webhook URL and does not bypass authorization on a public URL.
Always list repositories explicitly for production deployments.

Tanseki stores captured knowledge. The local SQLite registry stores operational state for question mapping and deduplication; the local outbox makes writes durable when Tanseki is temporarily unavailable. The home-directory paths above are for non-Docker local runs; use the Docker overrides shown below for the container.

## SQLite deployment topology

The supported deployment is one Kojutsu process, one replica, and one Uvicorn worker. `KOJUTSU_SQLITE_WORKERS` and `KOJUTSU_SQLITE_REPLICAS` are guardrails and must both remain `1`; startup rejects other values. They do not coordinate multiple processes. Do not run multiple workers, containers, or replicas against the same local files. Keep both SQLite files on persistent, backed-up storage. Multiple replicas require replacing the registry and outbox with a shared transactional store before deployment.

## Health and status endpoints

- `GET /webhook/health` is a liveness check and returns 200 while the process is running.
- `GET /webhook/ready` is a readiness check and returns 503 until this process owns the outbox and Tanseki is configured and reachable.
- `GET /webhook/status` returns configuration presence, Tanseki reachability, outbox pending/retrying/dead-letter counts, and the local registry path. It is an operational endpoint; keep it on a trusted interface because it exposes paths and service health.

## How answer detection works

Question comments contain a hidden question marker. Answers use the corresponding hidden answer marker:

```
Comment #1: <!-- kojutsu:question:abc-123 -->  ← Kojutsu question
Comment #2: <!-- kojutsu:answer:abc-123 -->

             Because we needed to handle edge case X
```

An answer is accepted only when the marker resolves to a registered question and the author is not the question author. Unrelated or interleaved comments are ignored. Answers with missing, malformed, or stale markers are not captured.

## Development vs Production

### Local development

```bash
# Terminal 1: Start Kojutsu
uv run kojutsu serve --port 8000

# Terminal 2: Start a tunnel
ngrok http 8000
# Use the tunnel URL in GitHub webhook settings
```

### Production

Build and run the webhook server with Docker. Keep both SQLite files on `/data`
and use container-reachable URLs for services running outside the container:

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

The explicit `-e` options override the local home-directory SQLite paths in
`.env.example`. Docker Desktop provides `host.docker.internal` on macOS; Linux
Docker Engine uses the shown `host-gateway` mapping. When Tanseki and Ollama are
containers on the same Docker network, use service DNS such as
`http://tanseki:8088` and `http://ollama:11434` instead. Container-local `localhost`
does not refer to services on the host.

The image runs `kojutsu serve` as a non-root user and exposes port `8000`. It provides a liveness healthcheck. Set all required environment variables in the runtime environment; do not bake secrets into the image. The dedicated `/data` volume is the safe persistent location for the SQLite registry and outbox.

## Manual Collection (Fallback)

If you cannot use webhooks, manually collect answers:

```bash
uv run kojutsu collect https://github.com/org/repo/pull/123
```

This scans all pages of PR comments and stores Q&A pairs only when a comment contains a valid Kojutsu answer marker.

## Troubleshooting

### Webhook not reaching server

1. Check that the public URL or tunnel is active.
2. Verify firewall rules allow port 8000.
3. Test the health endpoint: `curl http://localhost:8000/webhook/health`.

### Answers not captured

1. Confirm webhook events are enabled in GitHub.
2. Check server logs for incoming events.
3. Verify the answer includes the matching `kojutsu:answer` marker and is not from the question author.
4. Check `uv run kojutsu status` for Tanseki reachability, retrying writes, and dead-letter entries.

### Signature verification fails

1. Ensure `GITHUB_WEBHOOK_SECRET` matches the GitHub webhook secret.
2. Verify the payload URL uses `/webhook/github` and the content type is `application/json`.
3. Do not run a publicly bound server without a webhook secret; Kojutsu rejects unconfigured authentication.
