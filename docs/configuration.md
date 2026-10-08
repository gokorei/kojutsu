# Configuration, and running more than one instance

Kojutsu reads its settings from four layers. This document is about which one wins,
and about the one thing the configuration file is not allowed to contain.

## The layers

Highest priority first:

| # | Layer | Tracked? | Holds |
|---|-------|----------|-------|
| 1 | Constructor arguments | — | What the calling code passed |
| 2 | Environment variables | — | Credentials, and one-off overrides |
| 3 | `[instances.<name>]` in `kojutsu.toml` | **committed** | Per-instance settings |
| 4 | `[defaults]` in `kojutsu.toml` | **committed** | Settings shared by every instance |
| 5 | `.env` | gitignored | Credentials, and the local leftovers |

**Why the committed file outranks `.env`.** `.env` is untracked and belongs to one
machine; `kojutsu.toml` is committed and reviewed. If `.env` won, the file in the
diff would be a document with no effect on the machine that matters, and reviewing it
would be theatre. The layer that is per-developer and invisible in git history is the
one that yields.

The environment still beats both. Overriding one value for one run is a legitimate
thing to do, and editing a file to do it would be worse.

`.env` remains the last file layer, so an existing deployment keeps working unchanged.

## A worked example

No credential appears in this file, and the loader refuses one — see below.

```toml
# kojutsu.toml

[defaults]
# Shared by every instance. Where the store is, and who may be asked questions.
tanseki_url = "http://127.0.0.1:8099"
llm_provider = "opencode"
llm_model = "opencode/model"
llm_external_enabled = true

[instances.real]
tanseki_collection = "kojutsu-real"
github_webhook_allowed_repositories = ["acme/widgets"]
llm_allowed_repositories = ["acme/widgets"]

[instances.secondary]
tanseki_collection = "kojutsu-secondary"
github_webhook_allowed_repositories = ["acme/service", "acme/library"]
kojutsu_registry_path = "~/.kojutsu/registry-secondary.db"
tanseki_outbox_path = "~/.kojutsu/tanseki-outbox-secondary.db"

[instances.pilot]
tanseki_collection = "kojutsu-pilot"
```

An instance inherits everything from `[defaults]` and overrides only what differs, so
the two copies of a shared setting cannot drift apart.

### Using one

```bash
# By name, for one invocation. Every subcommand in the invocation reads it.
uv run kojutsu --instance secondary backfill --repo acme/library \
  --since 2026-08-01 --until 2026-09-30

# For a shell, a deployment, or a container.
export KOJUTSU_INSTANCE=secondary
uv run kojutsu console --port 8090
```

The environment variable is `KOJUTSU_INSTANCE`, and `KOJUTSU_CONFIG` names the
file if it is not `kojutsu.toml` in the working directory:

```bash
KOJUTSU_CONFIG=/etc/kojutsu/production.toml \
KOJUTSU_INSTANCE=secondary \
  uv run kojutsu console
```

Set `KOJUTSU_CONFIG` to the empty string to run with no configuration file at all,
which is what the test suite does.

### Which file a run used

`kojutsu status` reports the collection it is pointed at. For a programmatic
answer, `Settings().config_path` is the absolute path that was read and
`Settings().instance_name` is the selected instance.

## The file holds no credentials

Any credential-named key with a non-empty value in `kojutsu.toml` **is refused**,
at the top level and inside every instance:

```
ConfigError: /path/kojutsu.toml sets defaults.github_token, which is a
credential. This file is meant to be committed, reviewed, and pasted into a pull
request, so it must hold no secrets.
```

That file is committed, reviewed, and pasted into issues. Those three things are only
safe while it contains nothing sensitive, and a property that is only maintained by
good intentions stops being maintained the first time somebody is in a hurry. The
guard is at load time because that is the only moment it is worth anything.

An **empty** value is allowed — it is the shape `.env.example` uses, and refusing it
would make the file that points at a credential's proper home unusable in the place
the error message sends you.

Put credentials in `.env` or the environment. Both are gitignored or unset by default.

## Errors are refusals, not warnings

The loader would rather stop than guess:

- **An unknown instance name** is refused, listing the names that exist. Falling back
  to `[defaults]` would point capture at `kojutsu-real` and write a corpus nobody
  asked for, with nothing in the output to say so.
- **A misspelled `[defualts]`** is refused. Otherwise it loads as a file with no
  settings in it and the run is indistinguishable from a correct run of an empty
  configuration.
- **A nested table** is refused, whether it is a typo (`llm = {model = "x"}`) or a
  section that reads like inheritance (`[instances.a.b]`). Neither is implemented,
  and a silently-dropped value is worse than a refusal.
- **A malformed file** names itself and the parse error, rather than being swallowed
  into defaults.

## Repositories are named one at a time

`github_webhook_allowed_repositories` and `llm_allowed_repositories` take
`owner/name` entries. A wildcard is **not** authorisation — `acme/*` narrows to
nothing and is dropped rather than widened. An entry that silently failed to parse
would be an entry the operator believes is being collected.

Allowlists are comma-separated or a TOML array of strings; both are accepted.
