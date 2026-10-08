# Tanseki × Kojutsu — local quickstart

Get a working, single-instance setup: an Tanseki knowledge store, Kojutsu
capturing into it, and an agent reading back out. For the wire contract see
[`tanseki-seam.md`](tanseki-seam.md).

## Prerequisites

- **JDK 21** and the **Tanseki** repo, cloned next to this one so it lands at
  `../tanseki` (the default `TANSEKI_DIR` for `scripts/dev-e2e.sh` and
  `start.sh --with-tanseki`):

  ```bash
  git clone https://github.com/gokorei/tanseki.git
  ```

  Gradle provisions the JDK itself via the foojay toolchain resolver, so a newer
  default `java` on `PATH` does not need to be overridden.

**Tanseki is the product; the distribution it builds is called `tanseki`.** The
binaries are `tanseki-daemon` and `tanseki-mcp`, under a distribution directory
named `tanseki-daemon`, and `installDist` produces them. This is not a rename you
can ignore: a path written as `install/tanseki-daemon/bin/tanseki-daemon` does not
exist, and a script that builds and then invokes it fails at launch rather than
at build time. Environment variables keep the `TANSEKI_` prefix (`TANSEKI_PATH`,
`TANSEKI_URL`, `TANSEKI_HTTP_PORT`).
- **`pijul`** on `PATH` (`pijul --version`).
- **Python 3.11** with `uv` available. Install the locked development environment with `uv sync --locked --all-extras`.

## 1. Run the Tanseki store

```bash
cd ../tanseki
./gradlew :service:installDist

TANSEKI_PATH=/tmp/tanseki-vault \
TANSEKI_HTTP_PORT=8099 \
  service/build/install/tanseki-daemon/bin/tanseki-daemon
```

- `TANSEKI_PATH` is the vault directory (created if missing; initialised as a Pijul repo on first write).
- Optional: set `TANSEKI_API_KEY` to require `X-API-Key` on `/v1`.
- Check it: `curl -s localhost:8099/v1/health` → `{"status":"ok"}`.

## 2. Point Kojutsu at it

In this repo's `.env`:

```bash
TANSEKI_URL=http://localhost:8099
TANSEKI_COLLECTION=kojutsu-real
# TANSEKI_API_KEY=...            # only if the daemon was started with one
GITHUB_TOKEN=ghp_...          # for ask/collect/serve
```

## 3. Capture decisions

```bash
# Post questions on a PR (developers answer in comments)
uv run kojutsu ask https://github.com/org/repo/pull/123 --apply

# Harvest answers (or run `uv run kojutsu serve` for real-time webhooks)
uv run kojutsu collect https://github.com/org/repo/pull/123
```

Writes are enqueued in a local outbox and sent to Tanseki; the webhook server also
relays in the background. Inspect and flush manually:

```bash
uv run kojutsu status    # Tanseki reachability, outbox backlog, registry path
uv run kojutsu outbox    # pending writes
uv run kojutsu relay     # drain queued writes
```

## 4. Read it back

```bash
uv run kojutsu search "authentication" --repo org/repo
```

Or run the MCP server and call its tools from any MCP client:

```bash
uv run kojutsu-mcp
#  - search_knowledge(text=..., repo=..., jira_ticket_key...)
#  - get_knowledge_entry(entry_id=...)
```


Generic agents can also use Tanseki's own MCP server directly
(`../tanseki/service/build/install/tanseki-daemon/bin/tanseki-mcp`).

### Dev console

For a quick visual check (health, document count, outbox backlog, search + open
a document):

```bash
./start.sh --with-tanseki         # build+start a local tanseki-daemon, then the console
./start.sh                     # console only; uses TANSEKI_URL (env or .env)
# or, with TANSEKI_URL already exported:
uv run kojutsu console
```

`./start.sh` loads `.env`, warns if Tanseki is unreachable, and forwards extra args
to `uv run kojutsu console` (e.g. `./start.sh --port 9000`). `--with-tanseki` builds the
daemon if needed, runs it on `TANSEKI_HTTP_PORT` (default 8088) with a vault at
`TANSEKI_PATH` (default `~/.kojutsu/tanseki-vault`), logs to `~/.kojutsu/tanseki-daemon.log`,
and stops it again when the console exits. Set `TANSEKI_DIR` if the Tanseki repo is not
at `../tanseki`.

It proxies Tanseki's `/v1` read path and shows the local outbox/registry, so it
verifies the integration itself. Read-only; not a production surface.

#### Knowledge dashboard

`http://127.0.0.1:8090/dashboard` charts the captured knowledge in the
collection: totals, captures per day, and breakdowns by category, repository,
author, and PR, over a filterable list of questions and answers. The page polls
the console's `GET /api/knowledge` (default every 10s, adjustable in-page), so it
fills in as captures land during a demo. `/api/knowledge?limit=N` is the plain
JSON for scripting; `limit` defaults to 500 and caps at 2000.

The page is served from `scripts/knowledge_dashboard.html`, so it is only
available from a checkout — it is not bundled into the wheel. It can also be
opened directly from disk, in which case point its "console base URL" field at
the running console, or drop a saved `knowledge-snapshot.json` onto it.

## 5. Verify the whole loop

```bash
scripts/dev-e2e.sh
```

This starts a throwaway `tanseki-daemon`, captures a decision through Kojutsu's
real sink + outbox, reads it back, and tears everything down. `TANSEKI_DIR`,
`TANSEKI_PORT`, and `PYTHON` override the defaults.

To run the live cross-seam test suite itself — the tests that assert the
guarantees in `tanseki-seam.md` which an in-process fake cannot prove — start a
daemon as above and point the suite at it:

```bash
TANSEKI_URL=http://127.0.0.1:8099 uv run pytest tests/integration/test_tanseki_daemon_e2e.py -v
```

Without `TANSEKI_URL` those tests skip at collection time, which is why the
default `uv run pytest` is green on a machine with no Tanseki build. CI runs the
same file in the `tanseki-e2e` job, which builds the distribution and starts its
own daemon.

## Controlled five-agent pilot

The controlled pilot uses this project as the knowledge seam without ingesting
real engineering knowledge. It runs five isolated MCP readers against the same
allowlisted repository, delegates the model turns to the local OpenCode CLI,
has each agent review the next agent's work, and validates the final Python
artifact syntactically without executing it.

```bash
export GITHUB_WEBHOOK_ALLOWED_REPOSITORIES=your-org/pilot-repo
export PILOT_REPOSITORY=your-org/pilot-repo
export TANSEKI_URL=http://127.0.0.1:8099
export TANSEKI_COLLECTION=kojutsu-pilot
export TANSEKI_API_KEY=dedicated-pilot-key
export PILOT_OPENCODE_MODEL=local/north_code:latest
export PILOT_OPENCODE_BIN=opencode
export LLM_EXTERNAL_ENABLED=false
export KOJUTSU_REGISTRY_PATH=~/.kojutsu/pilot/registry.db
export TANSEKI_OUTBOX_PATH=~/.kojutsu/pilot/outbox.db
uv run python scripts/pilot_demo.py
```

The model agents receive only bounded MCP evidence. The delegator starts the
OpenCode CLI with an inline agent whose permissions deny every tool, and passes
only `PATH`, `HOME`, and OpenCode configuration; Tanseki and GitHub credentials
are not passed to the model process. The generated program is printed for human
review and is never executed by the harness. Set `PILOT_OPENCODE_MODEL` to an
exact `provider/model` value available to `opencode models`.

Before using the pilot with real engineering knowledge, run the focused state
and safety checks and perform a SQLite backup/restore drill:

```bash
uv run pytest -q tests/test_webhook_capture.py tests/test_outbox.py tests/test_mcp_server.py tests/test_runtime.py
mkdir -p "$HOME/.kojutsu/pilot/backup"
sqlite3 "$KOJUTSU_REGISTRY_PATH" ".backup '$HOME/.kojutsu/pilot/backup/registry.db'"
sqlite3 "$TANSEKI_OUTBOX_PATH" ".backup '$HOME/.kojutsu/pilot/backup/outbox.db'"
sqlite3 "$HOME/.kojutsu/pilot/backup/registry.db" "PRAGMA integrity_check;"
sqlite3 "$HOME/.kojutsu/pilot/backup/outbox.db" "PRAGMA integrity_check;"
```

Stop the single Kojutsu writer before restoring the files, keep the files at
mode `0600`, and rerun the duplicate delivery, outage, dead-letter requeue,
lease recovery, and cross-repository MCP denial checks. A human runs
`kojutsu ask --apply`; autonomous agents do not run `ask`, and the pilot must
not use `--no-save-session`.

## Notes for this phase

- Single instance only: the outbox and the question registry are local files.
  Run one Kojutsu process, one Uvicorn worker, and one replica; use a shared
  transactional registry/outbox before scaling out.
- `repo`/`pr`/`jira` frontmatter is searchable now; graph edges to those are
  dangling references (no target documents), so `traverse` won't show PR/repo
  lineage yet.
