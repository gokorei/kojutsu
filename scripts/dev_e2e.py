#!/usr/bin/env python3
"""Dev smoke test: capture a decision into Tanseki and read it back.

Assumes a running Tanseki daemon and ``TANSEKI_URL`` set. Exercises the real capture
path (mapping + outbox + client) and the Tanseki read path (search/get); also
checks the Kojutsu MCP surface when the ``mcp`` SDK is importable.
"""

from __future__ import annotations

import sys

from kojutsu.config import Settings
from kojutsu.core.knowledge_sink import TansekiKnowledgeSink
from kojutsu.core.outbox import TansekiOutbox
from kojutsu.integrations.tanseki import TansekiClient
from kojutsu.models import KnowledgeEntry, QuestionCategory

DOC_ID = "demo/repo/pr-1/dev-e2e-1"


def main() -> int:
    settings = Settings()
    if not settings.tanseki_enabled:
        print("TANSEKI_URL is required", file=sys.stderr)
        return 2

    entry = KnowledgeEntry(
        entry_id="dev-e2e-1",
        question_text="Why tokens over sessions?",
        answer_text="Because sessions do not scale across services.",
        category=QuestionCategory.DESIGN_DECISION,
        author="dev",
        tags=["dev-e2e"],
        metadata={"repo": "demo/repo", "pr_number": 1, "jira_ticket_key": "DEMO-1"},
    )

    with (
        TansekiOutbox(settings.tanseki_outbox_path) as outbox,
        TansekiClient.from_settings(settings) as client,
    ):
        outcome = TansekiKnowledgeSink(client, outbox).store(entry)
        queued = outbox.pending_count()
    print("delivery:", outcome.status.value)
    if queued:
        print(f"WARN: {queued} write(s) still queued — run `kojutsu relay`")

    with TansekiClient.from_settings(settings) as client:
        hits = client.search("tokens", frontmatter={"repo": "demo/repo"}, limit=5)
        print("search hits:", [h.id for h in hits])
        doc = client.get_document(DOC_ID)
    print("get:", "ok" if doc else "MISSING")

    from mcp_server.server import search_knowledge

    mcp_result = search_knowledge(text="tokens", repo="demo/repo")
    mcp_ok = mcp_result.ok and DOC_ID in mcp_result
    print("mcp search ok:", mcp_ok, "code:", mcp_result.code)
    if not mcp_ok:
        print(f"MCP search failed: {mcp_result.error or 'document not found'}", file=sys.stderr)
        return 1

    return 0 if doc is not None and outcome.status.value == "delivered" else 1


if __name__ == "__main__":
    raise SystemExit(main())
