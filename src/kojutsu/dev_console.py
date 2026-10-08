"""Minimal read-only web console for verifying the Kojutsu <-> Tanseki loop.

Serves one HTML page plus three JSON endpoints that proxy the Tanseki ``/v1`` API
and surface local queue/registry health. Intended for local verification, not
production (see ``docs/tanseki-quickstart.md``).
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import sqlite3
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic

from kojutsu.config import Settings, get_settings
from kojutsu.core.answer_collector import RECORD_KINDS
from kojutsu.core.outbox import OutboxOwnershipError, TansekiOutbox
from kojutsu.core.tanseki_mapping import (
    CENSUS_NAMESPACE,
    CLARIFICATION_TAG,
    EVALUATION_TAG,
    QUESTION_NAMESPACE,
    RECORD_KIND_KEY,
)
from kojutsu.integrations.tanseki import (
    MAX_LIST_RESULTS,
    MAX_SEARCH_RESULTS,
    TansekiClient,
    TansekiDocument,
    TansekiError,
    TansekiHit,
)
from kojutsu.models import STRUCTURE_INFERRED_BY, CaptureSource, structure_of
from kojutsu.net import is_loopback_host


class TansekiReadClient(Protocol):
    """The read-only slice of the Tanseki client the console depends on."""

    def health(self) -> bool: ...

    def count(self, collection: str | None = None) -> int: ...

    def search(
        self,
        query: str,
        *,
        tags: list[str] | None = None,
        frontmatter: dict[str, str] | None = None,
        limit: int = 10,
        collection: str | None = None,
    ) -> list[TansekiHit]: ...

    def get_document(
        self, doc_id: str, collection: str | None = None
    ) -> TansekiDocument | None: ...

    def list_documents(
        self, *, limit: int = MAX_LIST_RESULTS, collection: str | None = None
    ) -> list[str]: ...

    def close(self) -> None: ...


ClientFactory = Callable[[Settings], TansekiReadClient]

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 512
MAX_FILTER_CHARS = 256
MAX_DOCUMENT_ID_CHARS = 512
MAX_DOCUMENT_BYTES = 1_048_576
MAX_SNIPPET_CHARS = 4_096

DEFAULT_KNOWLEDGE_LIMIT = MAX_LIST_RESULTS
MAX_DASHBOARD_ANSWER_CHARS = 2_000

# Kojutsu document ids are "<repo>/pr-<n>/<entry_id>"; the frontmatter is the
# richer source, so the id is only a fallback for repo/pr.
_DOC_ID_RE = re.compile(r"^(?P<repo>.+)/pr-(?P<pr>\d+)/(?P<entry>[^/]+)$")
#: Documents in their own namespace nest one level deeper than the flat
#: ``<repo>/pr-<n>/<entry>`` form, so the flat pattern cannot match them and a
#: census or a question would arrive with no repo and no pull request. Read
#: separately rather than by loosening the flat one, which matches real captures.
_NAMESPACE_DOC_ID_RE = re.compile(
    r"^(?P<repo>.+)/pr-(?P<pr>\d+)/(?P<namespace>[^/]+)/(?P<leaf>.+)$"
)
_ANSWER_SECTION_RE = re.compile(r"^##[ \t]+Answer[ \t]*$", re.MULTILINE)
# A rationale stores its stated reason under "## Reason" and the unverifiable
# claim about who produced it under "## Attribution". Both headings are matched so
# the dashboard can show the reason without the attribution trailing after it.
_RATIONALE_SECTION_RE = re.compile(r"^##[ \t]+Reason[ \t]*$", re.MULTILINE)
# A clarification stores the quoted statement under "## Statement", ahead of the
# same "## Attribution" block. Matched for the same reason as the rationale's two:
# the words come first, and the reader is meant to see them without the name that
# would otherwise lend them authority they have not earned.
_CLARIFICATION_SECTION_RE = re.compile(r"^##[ \t]+Statement[ \t]*$", re.MULTILINE)
_ATTRIBUTION_SECTION_RE = re.compile(r"^##[ \t]+Attribution[ \t]*$", re.MULTILINE)
# An evaluation leads with its numbers under "## Result", ahead of "## Subject" and
# the limits section, for the same reason the other two put the content first: a
# model id in the opening line reads as a certification rather than a measurement.
# The limits are a section of their own and are deliberately NOT folded into the
# result text, because a reader who stops after the figures is precisely the reader
# who must still have been told what they are.
_EVALUATION_RESULT_RE = re.compile(r"^##[ \t]+Result[ \t]*$", re.MULTILINE)
_EVALUATION_SUBJECT_RE = re.compile(r"^##[ \t]+Subject[ \t]*$", re.MULTILINE)
# Captures carry an invisible provenance marker (e.g. "<!-- kojutsu:agent:opencode -->")
# that rendered Markdown would hide; strip it so it does not surface as answer text.
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

# The dashboard ships as a repo file rather than an inlined string, so it stays
# editable during a demo. It is absent from an installed wheel.
DASHBOARD_PATH = Path(__file__).resolve().parents[2] / "scripts" / "knowledge_dashboard.html"


def _close_client(client: TansekiReadClient) -> None:
    with suppress(Exception):
        client.close()


def _open_client(make_client: ClientFactory, settings: Settings) -> TansekiReadClient | None:
    """Open a read client, or nothing when the store is not configured."""
    return make_client(settings) if settings.tanseki_enabled else None


def _status_payload(settings: Settings, make_client: ClientFactory) -> dict[str, Any]:
    """Health and counters for Tanseki and the local outbox."""
    counts = {"pending": 0, "retrying": 0, "dead_letter": 0}
    local_error: str | None = None
    try:
        with TansekiOutbox(settings.tanseki_outbox_path) as queue:
            counts = queue.status_counts()
    except (OutboxOwnershipError, OSError, sqlite3.Error):
        local_error = "Local queue status unavailable."
    payload: dict[str, Any] = {
        "tanseki_url": settings.tanseki_url or None,
        "tanseki_collection": settings.tanseki_collection,
        "tanseki_reachable": None,
        "document_count": None,
        "outbox_pending": counts["pending"] + counts["retrying"],
        "outbox_captured_locally": counts["pending"],
        "outbox_delivery_failed": counts["retrying"] + counts["dead_letter"],
        "outbox_dead_letter": counts["dead_letter"],
        "outbox_path": settings.tanseki_outbox_path,
        "registry_path": settings.kojutsu_registry_path,
    }
    if local_error is not None:
        payload["error"] = local_error
    client = _open_client(make_client, settings)
    if client is not None:
        try:
            payload["tanseki_reachable"] = client.health()
            if payload["tanseki_reachable"]:
                payload["document_count"] = client.count()
        except TansekiError:
            payload["tanseki_reachable"] = False
            payload["error"] = "Tanseki health check failed."
        finally:
            _close_client(client)
    return payload


def _search_response(
    make_client: ClientFactory,
    settings: Settings,
    q: str,
    repo: str | None,
    jira: str | None,
    limit: int,
) -> Any:
    """Proxy a lexical search to Tanseki."""
    client = _open_client(make_client, settings)
    if client is None:
        return JSONResponse({"error": "Tanseki is not configured."}, status_code=503)
    frontmatter = {k: v for k, v in {"repo": repo, "jira": jira}.items() if v}
    try:
        hits = client.search(q, frontmatter=frontmatter or None, limit=limit)
    except TansekiError:
        return JSONResponse({"error": "Tanseki search failed."}, status_code=502)
    finally:
        _close_client(client)
    return {
        "hits": [
            {
                "id": h.id,
                "score": h.score,
                "snippet": (h.snippet or "")[:MAX_SNIPPET_CHARS],
            }
            for h in hits
        ]
    }


def _document_response(make_client: ClientFactory, settings: Settings, doc_id: str) -> Any:
    """Proxy a document fetch to Tanseki, bounded on the way out."""
    client = _open_client(make_client, settings)
    if client is None:
        return JSONResponse({"error": "Tanseki is not configured."}, status_code=503)
    try:
        doc = client.get_document(doc_id)
    except TansekiError:
        return JSONResponse({"error": "Tanseki document lookup failed."}, status_code=502)
    finally:
        _close_client(client)
    if doc is None:
        return JSONResponse({"error": "Document not found."}, status_code=404)
    if len(doc.content) > MAX_DOCUMENT_BYTES:
        return JSONResponse(
            {"error": f"Document exceeds the {MAX_DOCUMENT_BYTES}-byte maximum size."},
            status_code=413,
        )
    document_payload = {
        "id": doc.id,
        "path": doc.path,
        "collection": doc.collection,
        "revision": doc.revision,
        "updated_at": doc.updated_at,
        "frontmatter": doc.frontmatter,
        "content": doc.content,
    }
    response_size = len(
        json.dumps(document_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    if response_size > MAX_DOCUMENT_BYTES:
        return JSONResponse(
            {"error": f"Document exceeds the {MAX_DOCUMENT_BYTES}-byte maximum size."},
            status_code=413,
        )
    return document_payload


def _knowledge_response(make_client: ClientFactory, settings: Settings, limit: int) -> Any:
    """Aggregate the collection into the knowledge dashboard's payload.

    One request replaces exporting a snapshot per refresh. The list endpoint
    is unbounded in Tanseki, so ``limit`` caps both the listing and the
    resulting response, and ``truncated`` reports when the cap was hit.
    """
    client = _open_client(make_client, settings)
    if client is None:
        return JSONResponse({"error": "Tanseki is not configured."}, status_code=503)
    try:
        total = client.count()
        captures = [
            capture
            for capture in (
                capture_of(client, doc_id) for doc_id in client.list_documents(limit=limit)
            )
            if capture is not None
        ]
    except TansekiError:
        # The response stays generic so nothing internal reaches the browser;
        # the reason (e.g. a limit above Tanseki's cap) is only logged.
        logger.warning("Knowledge listing failed", exc_info=True)
        return JSONResponse({"error": "Tanseki knowledge listing failed."}, status_code=502)
    finally:
        _close_client(client)
    captures.sort(key=lambda c: (c["answered_at"], c["id"]), reverse=True)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "tanseki_url": settings.tanseki_url or None,
        "collection": settings.tanseki_collection,
        "total": total,
        "captured": len(captures),
        "truncated": total > len(captures),
        "captures": captures,
    }


def answer_from_content(content: str) -> str:
    """Pull the answer body out of a Kojutsu document's canonical Markdown.

    HTML comments are dropped: they carry provenance, not prose.
    """
    match = _ANSWER_SECTION_RE.search(content)
    if match is None:
        return ""
    body = content[match.end() :]
    return _HTML_COMMENT_RE.sub("", body).strip()


def rationale_from_content(content: str) -> str:
    """Pull the stated reason out of a rationale document's canonical Markdown.

    The reason only, not the attribution block. The document is written reason-first
    precisely so the claim about who produced it cannot be read as a claim about
    whether the reason is true, and the dashboard must not undo that by leading with
    the agent name.
    """
    match = _RATIONALE_SECTION_RE.search(content)
    if match is None:
        return ""
    body = content[match.end() :]
    # Stop at the attribution heading if one follows, so the reader sees the reason
    # and the claim separately rather than as one block of prose.
    body = _ATTRIBUTION_SECTION_RE.split(body)[0]
    return _HTML_COMMENT_RE.sub("", body).strip()


def clarification_from_content(content: str) -> str:
    """Pull the quoted statement out of a clarification document's Markdown.

    The statement only, never the attribution block, for the same reason the
    rationale reader stops before its own: a document that leads with a name lends
    the claim more authority than it has earned, and the dashboard must not undo
    the document's ordering to get there.
    """
    match = _CLARIFICATION_SECTION_RE.search(content)
    if match is None:
        return ""
    body = content[match.end() :]
    body = _ATTRIBUTION_SECTION_RE.split(body)[0]
    return _HTML_COMMENT_RE.sub("", body).strip()


def evaluation_result_from_content(content: str) -> str:
    """Pull the figures out of an evaluation document's Markdown, and nothing else.

    Stops before "## Subject" for the reason the two readers above do, and stops
    before the limits section for a second reason: a dashboard row is a summary, and
    a summary of a measurement that has had its own scope cut out of it is the exact
    misreading the scope paragraph was written to prevent. The row carries
    ``evaluation_scope`` beside the figures for the same reason — a reader who wants
    what the numbers can stand for should not have to open the document to find out
    that it says so.
    """
    match = _EVALUATION_RESULT_RE.search(content)
    if match is None:
        return ""
    body = content[match.end() :]
    body = _EVALUATION_SUBJECT_RE.split(body)[0]
    return _HTML_COMMENT_RE.sub("", body).strip()


_CENSUS_OBSERVED_RE = re.compile(r"^##[ \t]+What this does not say[ \t]*$", re.MULTILINE)
_QUESTION_REQUEST_RE = re.compile(r"^##[ \t]+Request[ \t]*$", re.MULTILINE)


def census_from_content(content: str) -> str:
    """The statement an observation makes, lifted out of its document.

    The body says the delivery was processed and nothing came of it, and then says
    what that does not mean. Both halves matter to a reader skimming a row, so the
    statement is returned whole rather than summarised into a caption.
    """
    match = re.search(r"^# Observed:.*$", content, re.MULTILINE)
    if match is None:
        return ""
    body = content[match.end() :]
    stop = _CENSUS_OBSERVED_RE.search(body)
    if stop is not None:
        body = body[: stop.start()]
    return _HTML_COMMENT_RE.sub("", body).strip()


def question_from_content(content: str) -> str:
    """The text of a decision request, so a question is never shown as an empty answer."""
    match = _QUESTION_REQUEST_RE.search(content)
    if match is None:
        return ""
    body = content[match.end() :]
    body = _ATTRIBUTION_SECTION_RE.split(body)[0]
    return _HTML_COMMENT_RE.sub("", body).strip()


def record_kind_of(frontmatter: dict[str, Any], tag_list: list[str], doc_id: str) -> str:
    """What kind of record this is, or that the document does not say.

    Counted rather than guessed. The first five are resolved by namespace or tag
    because a rationale, a projected question, a census record, an evaluation and a
    clarification do not all name themselves with a ``record_kind`` -- they are
    separate models dispatched separately at ``knowledge_sink.to_payload`` -- so the
    key is not there to read.

    A key this build cannot resolve reports ``unrecognised``, which is a *different*
    failure from an absent one and is kept distinct so a badge can say which. A
    record kind added to a writer with no reader here used to be an unhandled
    exception that took down the endpoint and hid every other record behind it --
    the cheapest failure hides the most.

    An absent key is resolved on evidence that is in the document rather than read
    into it: a record sitting in the flat answer namespace *and* stating a
    retrospective question category is a capture by construction, because only
    something that answers a question can have a category at all. Anything else
    with no key reports ``undeclared``, because defaulting that to ``capture`` would
    record the reader's interpretation as though the document had stated it.
    """
    namespace = _NAMESPACE_DOC_ID_RE.match(doc_id)
    named = namespace.group("namespace") if namespace else None
    if "rationale" in tag_list or named == "rationale":
        return "rationale"
    if "question" in tag_list or named == QUESTION_NAMESPACE:
        return "question"
    if "census" in tag_list or named == CENSUS_NAMESPACE:
        return "census"
    if EVALUATION_TAG in tag_list:
        return "evaluation"
    if CLARIFICATION_TAG in tag_list:
        return "clarification"
    raw = str(frontmatter.get(RECORD_KIND_KEY) or "").strip()
    if raw:
        return raw if raw in RECORD_KINDS else "unrecognised"
    if _DOC_ID_RE.match(doc_id) is not None and str(frontmatter.get("category") or "").strip():
        return "capture"
    return "undeclared"


def is_backfilled(frontmatter: dict[str, Any]) -> bool:
    """True for records reconstructed from history after the event.

    Surfaced on its own because ``backfilled`` is the one source that makes a weaker
    guarantee: it shows what the forge says now, not what it said then. A reader has
    to be able to tell which records they are holding rather than discovering it from
    a source string in a payload nobody opens.
    """
    return str(frontmatter.get("capture_source") or "").strip().casefold() == (
        CaptureSource.BACKFILLED.value
    )


@dataclass
class _CaptureContext:
    """Everything ``capture_of`` derives before it knows the record kind.

    Computed once, shared by every kind branch, so a record whose kind is
    declared but whose renderer has not caught up is reported rather than
    silently folded into the capture count.
    """

    document: TansekiDocument
    frontmatter: dict[str, Any]
    repo: str
    pr: str
    tag_list: list[str]
    resolved_kind: str
    answer: str
    agent: str
    structure: str
    structure_stated: bool
    structure_inferred_by: str


def _capture_context(document: TansekiDocument) -> _CaptureContext:
    """Derive the shared context; the kind dispatch happens in ``capture_of``."""
    frontmatter = document.frontmatter or {}
    id_match = _DOC_ID_RE.match(document.id) or _NAMESPACE_DOC_ID_RE.match(document.id)
    tags = frontmatter.get("tags")
    tag_list = [str(tag) for tag in tags] if isinstance(tags, list) else []
    repo = str(frontmatter.get("repo") or (id_match.group("repo") if id_match else ""))
    pr = str(frontmatter.get("pr") or (id_match.group("pr") if id_match else ""))
    # Resolved once, used by every branch, so a record whose kind is declared but
    # whose renderer has not caught up is reported rather than silently folded
    # into the capture count.
    resolved_kind = record_kind_of(frontmatter, tag_list, document.id)
    answer = answer_from_content(document.content)
    # ``author`` is whoever produced the answer (often the agent), while
    # ``comment_author`` is the human the answer was attributed to on GitHub.
    agent = str(frontmatter.get("answered_by_agent") or "")
    # Always present rather than absent. A dashboard row whose structure is
    # missing and a row whose structure is anchored have to be the same field for
    # the filter and the badge to work, and a reader asking "was this pairing
    # established or guessed?" must never be left to infer the answer from a
    # missing key. A value this code cannot read is reported as ``unknown``,
    # which is the truth about it -- resolving it to ``anchored`` would be a
    # guess made by the reader about a claim the store made.
    resolved_structure = structure_of(frontmatter.get("structure"))
    structure = resolved_structure.value if resolved_structure is not None else "unknown"
    # Whether the store actually stated it, as distinct from whether it reads as
    # anchored. Anchored is written only when it is not the default, so a document
    # from before the axis existed has no key at all and resolves to anchored by
    # truth — the capture path could not have inferred anything. That inference is
    # sound, but it is still the reader supplying it, and a reader asking "did
    # anyone state this, or am I being told?" deserves both answers. ``stated`` is
    # False for a rationale too: a stated reason has no pairing to state.
    structure_stated = bool(str(frontmatter.get("structure") or "").strip())
    # The model that inferred the pairing, carried beside the structure rather than
    # left for a reader to correlate. An ``inferred`` label with no model named is
    # an unattributed guess -- it says only that *something* matched the two sides,
    # which is the same unreadable state the write path refuses to construct and the
    # read path flags as an anomaly. The dashboard badges the row from this, and a
    # badge that has to be completed by a second lookup is a badge most readers
    # never complete. Always present, for the same reason ``structure`` is: absent
    # and empty have to be the same field for the badge to be decidable.
    structure_inferred_by = str(frontmatter.get(STRUCTURE_INFERRED_BY) or "")
    return _CaptureContext(
        document=document,
        frontmatter=frontmatter,
        repo=repo,
        pr=pr,
        tag_list=tag_list,
        resolved_kind=resolved_kind,
        answer=answer,
        agent=agent,
        structure=structure,
        structure_stated=structure_stated,
        structure_inferred_by=structure_inferred_by,
    )


def _rationale_row(ctx: _CaptureContext) -> dict[str, Any]:
    rationale = rationale_from_content(ctx.document.content)
    return {
        "id": ctx.document.id,
        "repo": ctx.repo or "unknown",
        "pr": int(ctx.pr) if ctx.pr.isdigit() else 0,
        "jira": str(ctx.frontmatter.get("jira") or ""),
        "pr_url": str(ctx.frontmatter.get("pr_url") or ""),
        "author": str(ctx.frontmatter.get("declared_by") or ctx.frontmatter.get("author") or ""),
        "answered_by_agent": str(ctx.frontmatter.get("declared_by") or ""),
        "comment_author": "",
        "capture_source": str(ctx.frontmatter.get("capture_source") or "asserted"),
        # A rationale writes no structure key, because a stated reason has no
        # question and answer to have inferred, so this resolves to anchored
        # for both kinds. That is not hiding a reconstruction: a reconstructed
        # reason is already labelled by ``rationale_source``, which this same
        # payload carries as its category.
        "structure": ctx.structure,
        "structure_stated": ctx.structure_stated,
        "structure_inferred_by": ctx.structure_inferred_by,
        "agent_authored": True,
        "kind": "rationale",
        "category": str(ctx.frontmatter.get("rationale_source") or "declared"),
        "tags": ctx.tag_list,
        "title": str(ctx.frontmatter.get("title") or ""),
        "answer": rationale[:MAX_DASHBOARD_ANSWER_CHARS],
        "answer_truncated": len(rationale) > MAX_DASHBOARD_ANSWER_CHARS,
        "rationale_model": str(ctx.frontmatter.get("declared_by_model") or ""),
        "rationale_revision": ctx.frontmatter.get("rationale_revision"),
        "answered_at": str(ctx.frontmatter.get("declared_at") or ctx.document.updated_at or ""),
    }


def _evaluation_row(ctx: _CaptureContext) -> dict[str, Any]:
    # A clarification is the other kind of record that carries no category: nobody
    # asked for it, so there is no question and every QuestionCategory value is
    # retrospective. Flattened as a capture it renders as an empty "uncategorized"
    # row — a conclusion with nothing behind it, which is the same failure this
    # model already suffered once, for a rationale. It is recognised here so the
    # dashboard can show it as a quotation with its own label.
    #
    # Unlike a rationale it *is* evidence: it was captured from a real comment, so
    # ``capture_source`` is reported as the channel it came through and the
    # dashboard must not badge it as unverified. And unlike an answer it has no
    # question to filter by, which is why ``category`` is reported empty rather
    # than defaulted: there is no axis for it to sit on.
    # A measurement is the one record kind that is a number about something rather
    # than a thing that happened, and it is the one most likely to be misread if it
    # arrives here flattened as a capture: the document id carries ``evaluation/``,
    # the content opens with figures, and every field a row is built from is either
    # absent or a word from the harness's vocabulary rather than a question category.
    # The result is the empty "uncategorized" row this function has now had to
    # rescue three times, and it is the worst of the three: an evaluation filed
    # beside eleven real records reads as a twelfth review rather than as a report
    # about a model.
    #
    # ``category`` is the harness's measurement name, not a QuestionCategory, because
    # every value in that vocabulary is retrospective and the nearest fit would be a
    # false claim about the repository the measurement was pointed at. The row says
    # which model was measured and that the target was a model, so a reader cannot
    # take the figures for a property of the code.
    result = evaluation_result_from_content(ctx.document.content)
    return {
        "id": ctx.document.id,
        "repo": ctx.repo or "unknown",
        "pr": int(ctx.pr) if ctx.pr.isdigit() else 0,
        "jira": "",
        "pr_url": str(ctx.frontmatter.get("pr_url") or ""),
        # The harness produced this, not a human and not a commenter, so
        # ``author`` names the process and ``comment_author`` stays empty rather
        # than borrowing a forge account the measurement has nothing to do with.
        "author": str(ctx.frontmatter.get("author") or "kojutsu"),
        "answered_by_agent": str(ctx.frontmatter.get("evaluated_model") or ""),
        "comment_author": "",
        "capture_source": str(ctx.frontmatter.get("capture_source") or "asserted"),
        "agent_authored": True,
        "kind": "evaluation",
        "category": str(ctx.frontmatter.get("evaluated_measurement") or ""),
        "tags": ctx.tag_list,
        "title": str(ctx.frontmatter.get("title") or ""),
        "answer": result[:MAX_DASHBOARD_ANSWER_CHARS],
        "answer_truncated": len(result) > MAX_DASHBOARD_ANSWER_CHARS,
        "evaluated_model": str(ctx.frontmatter.get("evaluated_model") or ""),
        "evaluation_target": str(ctx.frontmatter.get("evaluation_target") or ""),
        "evaluation_scope": str(ctx.frontmatter.get("evaluation_scope") or ""),
        # A measurement has no pairing to establish or infer, so it has no
        # structure to state. Reported as ``unknown`` rather than ``anchored``
        # so it cannot be counted by a filter asking which pairings were real.
        "structure": "unknown",
        "structure_stated": False,
        "structure_inferred_by": "",
        "answered_at": str(ctx.frontmatter.get("measured_at") or ctx.document.updated_at or ""),
    }


def _clarification_row(ctx: _CaptureContext) -> dict[str, Any]:
    statement = clarification_from_content(ctx.document.content)
    return {
        "id": ctx.document.id,
        "repo": ctx.repo or "unknown",
        "pr": int(ctx.pr) if ctx.pr.isdigit() else 0,
        "jira": str(ctx.frontmatter.get("jira") or ""),
        "pr_url": str(ctx.frontmatter.get("pr_url") or ""),
        # The forge account that posted the comment. A declared agent is
        # reported separately, and never in this field: nothing in the platform
        # verifies that a model wrote it, only that an account did.
        "author": str(ctx.frontmatter.get("author") or "unknown"),
        "answered_by_agent": str(ctx.frontmatter.get("clarified_by_agent") or ""),
        "comment_author": str(ctx.frontmatter.get("author") or ""),
        "capture_source": str(ctx.frontmatter.get("capture_source") or ""),
        "agent_authored": bool(ctx.frontmatter.get("clarified_by_agent")),
        "kind": "clarification",
        "category": "",
        "tags": ctx.tag_list,
        "title": str(ctx.frontmatter.get("title") or ""),
        "answer": statement[:MAX_DASHBOARD_ANSWER_CHARS],
        "answer_truncated": len(statement) > MAX_DASHBOARD_ANSWER_CHARS,
        "clarification_model": str(ctx.frontmatter.get("clarified_by_model") or ""),
        "clarification_comment_id": ctx.frontmatter.get("github_comment_id"),
        "clarification_association": str(ctx.frontmatter.get("github_author_association") or ""),
        # A clarification carries the same axis as an answer. Which of these
        # records were quoted from a comment is settled by the comment id; what
        # is not settled by that alone is whether anyone decided it was
        # answering something, which is what `structure` is for.
        "structure": ctx.structure,
        "structure_stated": ctx.structure_stated,
        "structure_inferred_by": ctx.structure_inferred_by,
        "answered_at": str(ctx.frontmatter.get("declared_at") or ctx.document.updated_at or ""),
    }


def _census_row(ctx: _CaptureContext) -> dict[str, Any]:
    # An observation is not a capture and has no reviewer, so it must not appear
    # in a capture listing with an empty author -- which is exactly how a
    # rationale used to render. What it does carry is the delivery a reader can
    # re-fetch, and that is its anchor: "we processed this and kept nothing" is
    # unverifiable without it.
    observation = census_from_content(ctx.document.content)
    return {
        "id": ctx.document.id,
        "repo": ctx.repo or "unknown",
        "pr": int(ctx.pr) if ctx.pr.isdigit() else 0,
        "jira": "",
        "pr_url": str(ctx.frontmatter.get("pr_url") or ""),
        # The login that opened the change. That is a fact about the change, not
        # a claim that this person asserted an observation -- which is why
        # nothing here names an answerer or a model.
        "author": str(ctx.frontmatter.get("change_author_account") or ""),
        "answered_by_agent": "",
        "comment_author": "",
        "capture_source": str(ctx.frontmatter.get("capture_source") or ""),
        "backfilled": False,
        "structure": "unknown",
        "structure_stated": False,
        "structure_inferred_by": "",
        "agent_authored": False,
        "kind": "census",
        "declared_kind": ctx.resolved_kind,
        # No category: every QuestionCategory value is retrospective and nobody
        # asked anything here, so one would be a false claim about the change.
        "category": "",
        "tags": ctx.tag_list,
        "title": str(ctx.frontmatter.get("title") or ""),
        "answer": observation[:MAX_DASHBOARD_ANSWER_CHARS],
        "answer_truncated": len(observation) > MAX_DASHBOARD_ANSWER_CHARS,
        "observation_delivery_id": str(ctx.frontmatter.get("delivery_id") or ""),
        "observation_action": str(ctx.frontmatter.get("action") or ""),
        "answered_at": str(ctx.frontmatter.get("observed_at") or ctx.document.updated_at or ""),
    }


def _question_row(ctx: _CaptureContext) -> dict[str, Any]:
    # A request for a reason, not a record of one. Rendering it as a capture that
    # found nothing to say is the specific misreading the projection exists to
    # prevent, so it carries its own kind and its own text.
    request = question_from_content(ctx.document.content)
    return {
        "id": ctx.document.id,
        "repo": ctx.repo or "unknown",
        "pr": int(ctx.pr) if ctx.pr.isdigit() else 0,
        "jira": str(ctx.frontmatter.get("jira") or ""),
        "pr_url": str(ctx.frontmatter.get("pr_url") or ""),
        "author": str(
            ctx.frontmatter.get("question_author") or ctx.frontmatter.get("author") or ""
        ),
        "answered_by_agent": "",
        "comment_author": "",
        # No capture source and no independence: a question carries neither, and
        # that absence is what excludes it from an evidence-only query.
        "capture_source": "",
        "backfilled": False,
        "structure": "unknown",
        "structure_stated": False,
        "structure_inferred_by": "",
        "agent_authored": False,
        "kind": "question",
        "declared_kind": ctx.resolved_kind,
        "category": str(ctx.frontmatter.get("category") or ""),
        "tags": ctx.tag_list,
        "title": str(ctx.frontmatter.get("title") or ""),
        "answer": request[:MAX_DASHBOARD_ANSWER_CHARS],
        "answer_truncated": len(request) > MAX_DASHBOARD_ANSWER_CHARS,
        "question_id": str(ctx.frontmatter.get("question_id") or ""),
        "question_status": str(ctx.frontmatter.get("question_status") or ""),
        "answered_at": str(ctx.frontmatter.get("created_at") or ctx.document.updated_at or ""),
    }


def _answer_row(ctx: _CaptureContext) -> dict[str, Any]:
    return {
        "id": ctx.document.id,
        "repo": ctx.repo or "unknown",
        "pr": int(ctx.pr) if ctx.pr.isdigit() else 0,
        "jira": str(ctx.frontmatter.get("jira") or ""),
        "pr_url": str(ctx.frontmatter.get("pr_url") or ""),
        "author": str(ctx.frontmatter.get("author") or "unknown"),
        "answered_by_agent": ctx.agent,
        "comment_author": str(ctx.frontmatter.get("comment_author") or ""),
        "capture_source": str(ctx.frontmatter.get("capture_source") or ""),
        "backfilled": is_backfilled(ctx.frontmatter),
        "structure": ctx.structure,
        "structure_stated": ctx.structure_stated,
        "structure_inferred_by": ctx.structure_inferred_by,
        "agent_authored": "agent_authored" in ctx.tag_list,
        # Classified rather than assumed. Every other kind has its own branch above,
        # so reaching here with ``undeclared`` or ``unrecognised`` means the document
        # said something this build cannot resolve -- reported as itself rather than
        # counted as a capture, because a count that looks complete while absorbing
        # an unknown kind is the failure this classification exists to prevent.
        "kind": ctx.resolved_kind,
        "declared_kind": str(ctx.frontmatter.get(RECORD_KIND_KEY) or ""),
        "category": str(ctx.frontmatter.get("category") or "uncategorized"),
        "tags": ctx.tag_list,
        "title": str(ctx.frontmatter.get("title") or ""),
        "answer": ctx.answer[:MAX_DASHBOARD_ANSWER_CHARS],
        "answer_truncated": len(ctx.answer) > MAX_DASHBOARD_ANSWER_CHARS,
        "answered_at": str(
            ctx.frontmatter.get("answered_at")
            or ctx.frontmatter.get("updated_at")
            or ctx.document.updated_at
            or ""
        ),
    }


def capture_of(client: TansekiReadClient, doc_id: str) -> dict[str, Any] | None:
    """Flatten a Kojutsu document into a dashboard capture record."""
    document = client.get_document(doc_id)
    if document is None:
        return None
    ctx = _capture_context(document)
    # A rationale is not a capture. It carries no category, because every capture
    # category is retrospective and this is a statement made before anyone asked.
    # Flattened as a capture it renders as an empty "uncategorized" row with no
    # text, which is the one thing it must never look like: a conclusion with
    # nothing behind it. Recognised here so it can be shown as a claim, with its
    # reason, and never counted as evidence.
    if "rationale" in ctx.tag_list:
        return _rationale_row(ctx)
    if EVALUATION_TAG in ctx.tag_list:
        return _evaluation_row(ctx)
    if CLARIFICATION_TAG in ctx.tag_list:
        return _clarification_row(ctx)
    if ctx.resolved_kind == "census":
        return _census_row(ctx)
    if ctx.resolved_kind == "question":
        return _question_row(ctx)
    return _answer_row(ctx)


def create_console_app(
    settings: Settings | None = None,
    client_factory: ClientFactory | None = None,
    *,
    bind_host: str = "127.0.0.1",
    allow_insecure_bind: bool = False,
    access_token: str = "",
) -> FastAPI:
    """Build a local-only console, with explicit opt-in and auth for public binds."""
    if not is_loopback_host(bind_host):
        if not allow_insecure_bind:
            raise ValueError("Non-loopback console binds require --allow-insecure-bind.")
        if not access_token.strip():
            raise ValueError("Non-loopback console binds require DEV_CONSOLE_TOKEN.")
    settings = settings or get_settings()
    make_client: ClientFactory = client_factory or TansekiClient.from_settings
    basic_auth = HTTPBasic(auto_error=False)

    async def _require_auth(request: Request) -> None:
        if not access_token:
            return
        credentials = await basic_auth(request)
        valid = (
            credentials is not None
            and secrets.compare_digest(credentials.username, "kojutsu")
            and secrets.compare_digest(credentials.password, access_token)
        )
        if not valid:
            raise HTTPException(
                status_code=401,
                detail="Authentication required.",
                headers={"WWW-Authenticate": "Basic"},
            )

    app = FastAPI(
        title="Kojutsu Dev Console",
        version="0.1.0",
        dependencies=[Depends(_require_auth)],
    )

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        """The console page."""
        return HTMLResponse(_PAGE)

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        """Health and counters for Tanseki and the local outbox."""
        return _status_payload(settings, make_client)

    @app.get("/api/search")
    def search(
        q: str = Query("", max_length=MAX_QUERY_CHARS, description="Free-text query"),
        repo: str | None = Query(
            None, max_length=MAX_FILTER_CHARS, description="Filter by frontmatter repo"
        ),
        jira: str | None = Query(
            None, max_length=MAX_FILTER_CHARS, description="Filter by frontmatter jira key"
        ),
        limit: int = Query(20, ge=1, le=MAX_SEARCH_RESULTS),
    ) -> Any:
        """Proxy a lexical search to Tanseki."""
        return _search_response(make_client, settings, q, repo, jira, limit)

    @app.get("/api/document")
    def document(
        id: str = Query(..., max_length=MAX_DOCUMENT_ID_CHARS, description="Document id"),
    ) -> Any:
        """Proxy a document fetch to Tanseki."""
        return _document_response(make_client, settings, id)

    @app.get("/api/knowledge")
    def knowledge(
        limit: int = Query(
            DEFAULT_KNOWLEDGE_LIMIT, ge=1, le=MAX_LIST_RESULTS, description="Documents to include"
        ),
    ) -> Any:
        """Aggregate the collection into the knowledge dashboard's payload.

        One request replaces exporting a snapshot per refresh. The list endpoint
        is unbounded in Tanseki, so ``limit`` caps both the listing and the
        resulting response, and ``truncated`` reports when the cap was hit.
        """
        return _knowledge_response(make_client, settings, limit)

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        """The knowledge dashboard, which polls ``/api/knowledge``."""
        try:
            return HTMLResponse(DASHBOARD_PATH.read_text(encoding="utf-8"))
        except OSError:
            return HTMLResponse(
                "<!doctype html><title>Kojutsu dashboard</title>"
                "<p>scripts/knowledge_dashboard.html was not found. Run the console "
                "from a Kojutsu checkout.</p>",
                status_code=503,
            )

    return app


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<link rel="icon" href="data:," />
<title>Kojutsu dev console</title>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --fg:#e6e8ee; --muted:#8b93a7;
          --accent:#6ea8fe; --ok:#3fb950; --bad:#f85149; --line:#262b36; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; }
  header { padding:16px 20px; border-bottom:1px solid var(--line); }
  h1 { margin:0 0 8px; font-size:15px; font-weight:600; }
  #status { color:var(--muted); font-size:13px; }
  #status b { color:var(--fg); font-weight:600; }
  .ok { color:var(--ok); } .bad { color:var(--bad); }
  main { display:grid; grid-template-columns:1fr 1fr; gap:16px; padding:16px 20px; }
  @media (max-width:820px) { main { grid-template-columns:1fr; } }
  section { background:var(--panel); border:1px solid var(--line);
            border-radius:8px; padding:14px; min-width:0; }
  h2 { margin:0 0 12px; font-size:12px; text-transform:uppercase;
       letter-spacing:.08em; color:var(--muted); }
  form { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px; }
  input { background:#0c0e13; border:1px solid var(--line); color:var(--fg);
          border-radius:6px; padding:6px 8px; font:inherit; }
  input[name=q] { flex:1 1 160px; }
  input[name=repo], input[name=jira] { flex:0 1 130px; }
  input[name=limit] { width:64px; }
  button { background:var(--accent); color:#0b0d12; border:0; border-radius:6px;
           padding:6px 12px; font:inherit; font-weight:600; cursor:pointer; }
  button:hover { filter:brightness(1.1); }
  ul { list-style:none; margin:0; padding:0; }
  li { padding:8px; border-bottom:1px solid var(--line); cursor:pointer;
       border-radius:6px; border:1px solid transparent; }
  li:hover { border-color:var(--accent); }
  li .id { color:var(--accent); }
  li .score { color:var(--muted); }
  li .snip { color:var(--muted); font-size:12px; }
  pre { white-space:pre-wrap; word-break:break-word; background:#0c0e13;
        border:1px solid var(--line); border-radius:6px; padding:10px;
        max-height:340px; overflow:auto; }
  .fm { color:var(--muted); font-size:12px; margin:8px 0; }
  .empty { color:var(--muted); }
</style>
</head>
<body>
<header>
  <h1>Kojutsu dev console</h1>
  <div id="status">loading…</div>
</header>
<main>
  <section>
    <h2>Search</h2>
    <form id="search-form">
      <input name="q" placeholder="query (e.g. authentication)" autofocus />
      <input name="repo" placeholder="repo: owner/name" />
      <input name="jira" placeholder="jira: ABC-123" />
      <input name="limit" type="number" min="1" max="50" value="20" />
      <button type="submit">Search</button>
    </form>
    <ul id="hits"><li class="empty">No search yet.</li></ul>
  </section>
  <section>
    <h2>Document</h2>
    <div id="doc" class="empty">Select a hit to inspect it.</div>
  </section>
</main>
<script>
const $ = (s) => document.querySelector(s);
const escapeHtml = (v) => String(v ?? '').replace(/[&<>"']/g,
  (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

async function refreshStatus() {
  try {
    const s = await (await fetch('/api/status')).json();
    const dot = s.tanseki_reachable
      ? '<span class="ok">&#9679;</span>' : '<span class="bad">&#9679;</span>';
    $('#status').innerHTML =
      `${dot} tanseki <b>${escapeHtml(s.tanseki_url || '(unset)')}</b>` +
      ` &middot; collection <b>${escapeHtml(s.tanseki_collection)}</b>` +
      ` &middot; reachable <b>${escapeHtml(s.tanseki_reachable)}</b>` +
      ` &middot; docs <b>${escapeHtml(s.document_count ?? '?')}</b>` +
      ` &middot; outbox <b>${escapeHtml(s.outbox_pending)}</b>` +
      ` &middot; registry <b>${escapeHtml(s.registry_path)}</b>` +
      (s.error ? ` &middot; <span class="bad">${escapeHtml(s.error)}</span>` : '');
  } catch (e) {
    $('#status').innerHTML = `<span class="bad">status error: ${escapeHtml(e)}</span>`;
  }
}

async function runSearch(ev) {
  ev.preventDefault();
  const params = new URLSearchParams();
  for (const [k, v] of new FormData(ev.target).entries()) {
    if (String(v).trim()) params.set(k, v);
  }
  const list = $('#hits');
  list.innerHTML = '<li class="empty">searching…</li>';
  try {
    const data = await (await fetch('/api/search?' + params)).json();
    if (data.error) { list.innerHTML = `<li class="bad">${escapeHtml(data.error)}</li>`; return; }
    if (!data.hits.length) { list.innerHTML = '<li class="empty">No matches.</li>'; return; }
    list.innerHTML = data.hits.map((h) =>
      `<li data-id="${escapeHtml(h.id)}">` +
      `<span class="id">${escapeHtml(h.id)}</span> ` +
      `<span class="score">${Number(h.score).toFixed(2)}</span><br>` +
      `<span class="snip">${escapeHtml(h.snippet || '')}</span></li>`
    ).join('');
    list.querySelectorAll('li[data-id]').forEach((li) => {
      li.onclick = () => openDoc(li.dataset.id);
    });
  } catch (e) { list.innerHTML = `<li class="bad">${escapeHtml(e)}</li>`; }
}

async function openDoc(id) {
  const box = $('#doc');
  box.className = '';
  box.textContent = 'loading…';
  try {
    const r = await fetch('/api/document?id=' + encodeURIComponent(id));
    const d = await r.json();
    if (!r.ok) { box.innerHTML = `<span class="bad">${escapeHtml(d.error || r.status)}</span>`; return; }
    box.innerHTML =
      `<div class="fm"><span class="id">${escapeHtml(d.id)}</span> &middot; ` +
      `${escapeHtml(d.path)} &middot; ${escapeHtml(d.collection)} &middot; ` +
      `rev ${escapeHtml(d.revision || '?')} &middot; ${escapeHtml(d.updated_at || '')}</div>` +
      `<div class="fm">${escapeHtml(JSON.stringify(d.frontmatter || {}))}</div>` +
      `<pre>${escapeHtml(d.content)}</pre>`;
  } catch (e) { box.innerHTML = `<span class="bad">${escapeHtml(e)}</span>`; }
}

$('#search-form').addEventListener('submit', runSearch);
refreshStatus();
setInterval(refreshStatus, 10000);
</script>
</body>
</html>
"""
