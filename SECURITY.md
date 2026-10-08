# Security Policy

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use
[GitHub private security advisories](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability)
against this repository instead.

Include what you can of: the affected version or commit, steps to
reproduce, and what you think an attacker gains. We will acknowledge
receipt, investigate, and coordinate a fix and disclosure timeline with
you.

## What to protect in a deployment

Kojutsu has no hosted service and no central credential store. A
deployment is a set of secrets you hold plus two plaintext SQLite files:

- **Secrets (environment only, never `kojutsu.toml`):** `GITHUB_TOKEN`,
  `GITHUB_WEBHOOK_SECRET`, `TANSEKI_API_KEY`, `LLM_API_KEY`,
  `JIRA_API_TOKEN`, `DEV_CONSOLE_TOKEN`. The loader refuses a
  credential-named key with a non-empty value in `kojutsu.toml`; keep it
  that way. If any of these leak, rotate the leaked credential at its
  provider and restart the process.
- **Local state is plaintext:** the registry (`KOJUTSU_REGISTRY_PATH`)
  and the Tanseki outbox (`TANSEKI_OUTBOX_PATH`) hold question and
  answer text, author logins, and queued records with `0600` file
  permissions as the only protection. Anyone who can read the file can
  read its contents. Back up with SQLite's `.backup` (not `cp` of the
  main file in WAL mode) and store backups with the same protections.
  See `README.md` ("What is in those files, and what that means") and
  `docs/design-review/durability.md`.
- **Webhook endpoint:** a publicly bound `kojutsu serve` requires
  `GITHUB_WEBHOOK_SECRET` and an explicit
  `GITHUB_WEBHOOK_ALLOWED_REPOSITORIES` list. The `*` wildcard is for
  local URLs only. Keep `/webhook/status` on a trusted interface; it
  exposes operational paths and health.
- **LLM boundary:** external providers (OpenAI, Anthropic) require both
  `LLM_EXTERNAL_ENABLED=true` and an explicit
  `LLM_ALLOWED_REPOSITORIES` list. Diffs sent across that boundary are
  bounded and redacted, but review the provider's data-processing terms
  before enabling. Local Ollama needs no opt-in.

## Scope

In scope: the webhook signature check, the repository allowlist,
credential handling and redaction, SQLite file permissions and symlink /
ownership refusal, and the MCP servers' authorization checks.

Out of scope: the Tanseki store, LLM providers, GitHub, Jira, and the
network between them. Vulnerabilities there belong to their maintainers,
unless Kojutsu misuses their API in a way worth fixing here.
