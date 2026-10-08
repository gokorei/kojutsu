"""Tests for the clarification record: a human statement that answered no question.

The record exists because ``KnowledgeEntry`` cannot hold this content, and every
test below is one of the ways it would go wrong if it could.

Two sections matter more than the rest and are the ones a later change is most
likely to break quietly:

**Identity is derived from the comment, not the text.** The golden test pins the
exact digest. The entry id is the outbox key and the Tanseki document id is derived
from it, so a moved derivation does not merely change a value — it orphans every
document already in the store. See ``docs/design-review/identity-and-limits.md``.

**Trust runs opposite to a rationale's.** A rationale is permanently ``ASSERTED``
because no provider delivery is behind a stated reason. A clarification is a
quotation from a real comment, so it is ``COLLECT`` or ``WEBHOOK`` and is held to
the same anchors as any other capture. Both halves are tested here, because
conflating them in either direction is the mistake this record was made to avoid:
a clarification stored as an assertion is real evidence thrown away, and a
rationale stored as a capture is an agent's own account of its work served as
review evidence.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from kojutsu.core.knowledge_sink import (
    KnowledgeDeliveryStatus,
    StorableRecord,
    TansekiKnowledgeSink,
    to_payload,
)
from kojutsu.core.outbox import TansekiOutbox
from kojutsu.core.question_registry import (
    CLARIFICATION_IDENTITY_VERSION,
    stable_answer_entry_id,
    stable_clarification_entry_id,
    stable_rationale_entry_id,
)
from kojutsu.core.tanseki_mapping import (
    build_clarification_content,
    build_clarification_frontmatter,
    clarification_document_id,
    document_id,
    to_clarification_upsert_payload,
)
from kojutsu.dev_console import capture_of, clarification_from_content
from kojutsu.models import (
    UNKNOWN_MODEL,
    CaptureSource,
    ClarificationEntry,
    KnowledgeEntry,
    QuestionCategory,
    RationaleEntry,
    capture_anchor_gaps,
)

MODEL = "opencode/model"
DECLARED_AT = datetime(2026, 1, 1, tzinfo=UTC)
CAPTURED_AT = datetime(2026, 1, 2, tzinfo=UTC)
DELIVERY_ID = "0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0"
STATEMENT = "This is deliberate; the narrow window is the accepted cost for v0.1."

#: The golden digest. Changing ANY input to the derivation, or the derivation
#: itself, must fail here rather than quietly re-identifying stored records.
#:
#: Provenance: SHA-256 over the length-prefixed, domain-labelled preimage
#: ``("kojutsu.clarification.v1", "org/repo", "42", "201")``.
#: Recompute deliberately, and bump ``CLARIFICATION_IDENTITY_VERSION`` with it.
GOLDEN_ENTRY_ID = (
    "clarification-v1-e9ee0278ea7a0cd11f0bcd7f9003a79b321ef1afed6d3da760235ee18086e9ce"
)

DASHBOARD = Path(__file__).resolve().parents[1] / "scripts" / "knowledge_dashboard.html"
_SCRIPT_RE = re.compile(r"<script[^>]*>(.*?)</script>", re.DOTALL)


def make_clarification(**overrides: object) -> ClarificationEntry:
    fields: dict[str, object] = {
        "entry_id": stable_clarification_entry_id(
            repo="org/repo", pr_number=42, github_comment_id=201
        ),
        "repo": "org/repo",
        "pr_number": 42,
        "statement": STATEMENT,
        "author": "davy",
        "author_association": "OWNER",
        "github_comment_id": 201,
        "declared_at": DECLARED_AT,
        "capture_source": CaptureSource.COLLECT,
        "captured_at": CAPTURED_AT,
    }
    fields.update(overrides)
    return ClarificationEntry(**fields)  # type: ignore[arg-type]


# --- identity ----------------------------------------------------------------


def test_the_identity_derivation_is_pinned_to_a_golden_value() -> None:
    """A refactor that moves the digest must fail here, not in the store.

    Everything else in this section is a convention. This is the only thing
    standing between an innocuous-looking edit and re-identifying every stored
    clarification, which orphans the Tanseki documents whose ids derive from it.
    """
    assert (
        stable_clarification_entry_id(repo="org/repo", pr_number=42, github_comment_id=201)
        == GOLDEN_ENTRY_ID
    )
    assert CLARIFICATION_IDENTITY_VERSION == 1


def test_the_same_comment_derives_the_same_id() -> None:
    """Determinism is the point: a re-delivery must collapse to one record."""
    first = stable_clarification_entry_id(repo="org/repo", pr_number=42, github_comment_id=201)
    second = stable_clarification_entry_id(repo="org/repo", pr_number=42, github_comment_id=201)
    assert first == second


def test_reworded_text_is_the_same_record_and_a_new_comment_is_not() -> None:
    """The two halves of "identity is the comment, not the text".

    A digest over the text would make every rephrasing an unrelated record and
    orphan the earlier one. It would also collapse two people who independently
    said the same thing into a single row, and that agreement is the finding.
    """
    reworded = make_clarification(
        statement="Deliberate. The narrow window is what v0.1 costs.",
    )
    different_comment = make_clarification(
        entry_id=stable_clarification_entry_id(
            repo="org/repo", pr_number=42, github_comment_id=202
        ),
        github_comment_id=202,
    )

    assert reworded.entry_id == make_clarification().entry_id
    assert different_comment.entry_id != make_clarification().entry_id
    assert different_comment.statement == make_clarification().statement


def test_a_delimiter_in_a_component_cannot_forge_another_record() -> None:
    """Length-prefixing, not delimiter-joining, for the same reason as the rationale.

    Repository names are attacker-influenced and may contain any character.
    """
    ambiguous_left = stable_clarification_entry_id(
        repo="org/repo", pr_number=4, github_comment_id=2
    )
    ambiguous_right = stable_clarification_entry_id(
        repo="org/rep", pr_number=42, github_comment_id=1
    )
    assert ambiguous_left != ambiguous_right, (
        "two different preimages produced one id, so a dedupe check would pass by "
        "accident and a real record could be dropped as a duplicate"
    )


def test_the_identity_is_separated_from_every_other_namespace() -> None:
    """The domain label keeps a clarification from colliding with any other record."""
    clarification_id = stable_clarification_entry_id(
        repo="org/repo", pr_number=42, github_comment_id=201
    )

    assert clarification_id != stable_answer_entry_id("org/repo", 42, 201), (
        "an answer and a clarification can quote the same comment, and are still "
        "two different records"
    )
    assert clarification_id != stable_rationale_entry_id(
        repo="org/repo", pr_number=42, branch="feat/x", declared_by="opencode", revision=1
    )
    assert clarification_id.startswith("clarification-v1-")


def test_the_repository_is_matched_case_insensitively() -> None:
    """GitHub defines repository names as case-insensitive, so the digest must be too."""
    assert stable_clarification_entry_id(
        repo="org/repo", pr_number=42, github_comment_id=201
    ) == stable_clarification_entry_id(repo="ORG/Repo", pr_number=42, github_comment_id=201)


@pytest.mark.parametrize("comment_id", [0, -1, True])
def test_a_clarification_with_no_comment_is_refused(comment_id: object) -> None:
    """Without a comment id there is nothing to re-fetch, so it is not a quotation."""
    with pytest.raises(ValueError, match="comment"):
        stable_clarification_entry_id(
            repo="org/repo",
            pr_number=42,
            github_comment_id=comment_id,  # type: ignore[arg-type]
        )


# --- trust -------------------------------------------------------------------


def test_a_clarification_cannot_be_asserted() -> None:
    """A record that quotes nobody is a typed claim, which is what a rationale is.

    The mirror of ``RationaleEntry``'s refusal of ``webhook``: a record's trust
    axis must be the one its content can support, in both directions.
    """
    with pytest.raises(ValidationError, match="RationaleEntry"):
        make_clarification(capture_source=CaptureSource.ASSERTED)


def test_the_capture_source_must_be_stated() -> None:
    """No default. The question "how was this captured?" has no safe default answer."""
    with pytest.raises(ValidationError):
        ClarificationEntry(
            entry_id="clarification-v1-x",
            repo="org/repo",
            pr_number=42,
            statement=STATEMENT,
            author="davy",
            author_association="OWNER",
            github_comment_id=201,
        )


@pytest.mark.parametrize("source", [CaptureSource.COLLECT, CaptureSource.WEBHOOK])
def test_a_captured_clarification_is_evidence(source: CaptureSource) -> None:
    """Both real capture channels are accepted, and both are captured records.

    This is the opposite of a rationale, which can never be either of these.
    """
    clarification = make_clarification(
        capture_source=source,
        capture_delivery_id=DELIVERY_ID if source is CaptureSource.WEBHOOK else None,
    )

    assert clarification.capture_source is source
    assert source is not CaptureSource.ASSERTED


def test_a_clarification_passes_capture_anchor_gaps_unchanged() -> None:
    """The one definition of "captured means checkable", reused rather than restated.

    A rule that exists in two places drifts, and the read path is the one that
    decides what a reader may conclude. The anchor here is the comment id, so this
    record satisfies the same check a ``collect`` answer does.
    """
    clarification = make_clarification()

    assert (
        capture_anchor_gaps(
            capture_source=clarification.capture_source,
            repo=clarification.repo,
            pr_number=clarification.pr_number,
            captured_at=clarification.captured_at,
            delivery_id=clarification.capture_delivery_id,
            comment_id=clarification.github_comment_id,
        )
        == []
    )
    assert capture_anchor_gaps(
        capture_source=CaptureSource.WEBHOOK,
        repo=clarification.repo,
        pr_number=clarification.pr_number,
        captured_at=clarification.captured_at,
        delivery_id=None,
        comment_id=clarification.github_comment_id,
    ) == ["capture_delivery_id"]


@pytest.mark.parametrize(
    ("overrides", "missing"),
    [
        ({"repo": "  "}, "metadata.repo"),
        ({"github_comment_id": 0}, "metadata.github_comment_id"),
    ],
)
def test_an_unanchored_clarification_is_refused(overrides: dict[str, object], missing: str) -> None:
    with pytest.raises(ValidationError, match=missing.split(".")[-1]):
        make_clarification(**overrides)


def test_a_webhook_clarification_without_a_delivery_id_is_refused() -> None:
    with pytest.raises(ValidationError, match="capture_delivery_id"):
        make_clarification(capture_source=CaptureSource.WEBHOOK)


def test_an_empty_statement_is_not_a_quotation() -> None:
    """The mirror of "no empty question_text": an empty body quotes nothing."""
    with pytest.raises(ValidationError, match="statement"):
        make_clarification(statement="   \n")


def test_a_blank_association_is_refused_and_a_messy_one_is_normalised() -> None:
    """ "The owner said this" and "somebody said this" are different statements."""
    with pytest.raises(ValidationError, match="association"):
        make_clarification(author_association="   ")

    assert make_clarification(author_association=" owner ").author_association == "OWNER"


def test_a_clarification_has_no_question_field_at_all() -> None:
    """Absent, not empty. An empty string cannot be told from a lost value."""
    fields = set(ClarificationEntry.model_fields)

    assert "question_text" not in fields
    assert "category" not in fields
    assert "question_category" not in fields
    assert not any("question" in name for name in fields), (
        "a clarification answers no question; a field that names one would be an "
        "invitation to fill it in, and an empty value there is indistinguishable "
        "from a lost one"
    )
    with pytest.raises(ValidationError):
        ClarificationEntry(
            entry_id="clarification-v1-x",
            repo="org/repo",
            pr_number=42,
            statement=STATEMENT,
            author="davy",
            author_association="OWNER",
            github_comment_id=201,
            capture_source=CaptureSource.COLLECT,
            captured_at=CAPTURED_AT,
            question_text="Why is the window so narrow?",
        )


def test_an_agent_authored_clarification_records_its_marker_and_model() -> None:
    """A bot's statement is worth nothing to a reader who cannot see a bot wrote it.

    The same terms as an answer: the agent and model are read from the comment's
    own marker, and both are self-declarations the platform never verifies.
    """
    clarification = make_clarification(
        author="kojutsu-bot",
        authored_by_agent="opencode",
        authored_by_model=MODEL,
    )
    frontmatter = build_clarification_frontmatter(clarification)

    assert clarification.is_agent_authored is True
    assert frontmatter["author"] == "kojutsu-bot"
    assert frontmatter["clarified_by_agent"] == "opencode"
    assert frontmatter["clarified_by_model"] == MODEL
    assert "agent_authored" in frontmatter["tags"]


def test_a_comment_with_no_marker_records_no_agent_and_reports_the_model_unknown() -> None:
    """Absence is reported as absence, never guessed at in either direction."""
    clarification = make_clarification()

    assert clarification.is_agent_authored is False
    assert "clarified_by_agent" not in build_clarification_frontmatter(clarification)
    assert build_clarification_frontmatter(clarification)["clarified_by_model"] == UNKNOWN_MODEL


# --- storage -----------------------------------------------------------------


def make_entry() -> KnowledgeEntry:
    return KnowledgeEntry(
        entry_id="answer-1",
        question_text="Why?",
        answer_text="Because.",
        category=QuestionCategory.DESIGN_DECISION,
        author="dev",
        metadata={"repo": "org/repo", "pr_number": 1},
    )


def test_the_document_id_keeps_a_clarification_out_of_the_answer_namespace() -> None:
    """Its own path segment, mirroring how ``rationale/`` was done."""
    clarification = make_clarification()
    document = clarification_document_id(clarification)

    assert document == f"org/repo/pr-42/clarification/{clarification.entry_id}"
    assert document.split("/")[:3] == ["org", "repo", "pr-42"]
    assert document.split("/")[3] == "clarification"
    # A reader browsing the store must be able to tell what they are looking at
    # from the id alone, so this must not be shaped like an answer.
    assert "/rationale/" not in document
    assert document_id(make_entry()) == "org/repo/pr-1/answer-1"


def test_the_frontmatter_carries_the_anchor_and_no_category() -> None:
    clarification = make_clarification()
    frontmatter = build_clarification_frontmatter(clarification)

    assert frontmatter["capture_source"] == "collect"
    assert frontmatter["repo"] == "org/repo"
    assert frontmatter["pr"] == 42
    assert frontmatter["github_comment_id"] == 201
    assert frontmatter["github_author_association"] == "OWNER"
    assert frontmatter["declared_at"] == DECLARED_AT.isoformat()
    assert frontmatter["captured_at"] == CAPTURED_AT.isoformat()
    assert "clarification" in frontmatter["tags"]
    assert "category" not in frontmatter
    assert not any(key.startswith("category") for key in frontmatter), (
        "every QuestionCategory value is retrospective, so any of them on this "
        "document would be a claim about a decision nobody was asked to make"
    )


def test_the_statement_comes_first_and_the_attribution_after() -> None:
    """Leading with a name lends a claim more authority than it has earned."""
    content = build_clarification_content(make_clarification())
    body = content.split("---", 2)[-1]

    assert body.index("## Statement") < body.index("## Attribution")
    assert body.index(STATEMENT) < body.index("davy")
    # The quoted words are readable on their own, which is the only way to check
    # them against the comment they were taken from.
    assert clarification_from_content(content) == STATEMENT
    assert "Author association: OWNER" not in clarification_from_content(content)
    assert "Comment: 201" in body


def test_the_upsert_payload_carries_the_frontmatter_and_the_body() -> None:
    clarification = make_clarification()
    payload = to_clarification_upsert_payload(clarification)

    assert payload["id"] == clarification_document_id(clarification)
    assert payload["path"] == f"{payload['id']}.md"
    assert payload["frontmatter"]["github_comment_id"] == 201
    assert STATEMENT in payload["content"]
    assert payload["author"] == "davy"


class FakeWriter:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.payloads: list[dict] = []

    def upsert_document(self, payload: dict) -> dict:
        if self.fail:
            from kojutsu.integrations.tanseki import TansekiError

            raise TansekiError("down")
        self.payloads.append(payload)
        return {}


def test_a_clarification_travels_the_same_outbox_as_every_other_record(
    tmp_path: Path,
) -> None:
    """One delivery path. A second queue is a second thing that can lose a record.

    The relay that retries and dead-letters a failure has to see this record; a
    kind of knowledge that reached the store by a private route would be invisible
    to it.
    """
    writer = FakeWriter()
    with TansekiOutbox(tmp_path / "outbox.db") as outbox:
        outcome = TansekiKnowledgeSink(writer, outbox).store(make_clarification())
        assert outcome.status is KnowledgeDeliveryStatus.DELIVERED
        assert outbox.pending_count() == 0

    assert writer.payloads[0]["id"] == clarification_document_id(make_clarification())


def test_a_failed_clarification_delivery_is_left_for_the_relay(tmp_path: Path) -> None:
    """The failure half of the same claim: it is queued, not dropped."""
    with TansekiOutbox(tmp_path / "outbox.db") as outbox:
        outcome = TansekiKnowledgeSink(FakeWriter(fail=True), outbox).store(make_clarification())
        assert outcome.status is KnowledgeDeliveryStatus.QUEUED
        assert outbox.pending_count() == 1
        assert outbox.pending()[0].payload["id"] == clarification_document_id(make_clarification())


def test_dispatch_is_on_type_so_no_record_kind_is_rendered_as_another() -> None:
    """Each kind has its own document. The rationale cannot be a question either."""
    clarification = make_clarification()
    rationale = RationaleEntry(
        entry_id="rationale-v1-x",
        repo="org/repo",
        pr_number=1,
        declared_by="opencode",
        rationale_text="Because X.",
    )
    assert isinstance(clarification, StorableRecord)
    assert isinstance(rationale, StorableRecord)
    assert isinstance(make_entry(), StorableRecord)

    clarification_payload = to_payload(clarification)
    rationale_payload = to_payload(rationale)
    entry_payload = to_payload(make_entry())

    assert "/clarification/" in clarification_payload["id"]
    assert "/rationale/" in rationale_payload["id"]
    assert clarification_payload["id"] != rationale_payload["id"] != entry_payload["id"]
    # A clarification rendered by the entry renderer would grow a question and an
    # answer it never had, which is the failure this record was made to avoid.
    assert "## Question" not in clarification_payload["content"]
    assert "## Answer" not in clarification_payload["content"]


# --- read path ---------------------------------------------------------------


def _doc(doc_id: str, content: str, frontmatter: dict[str, Any]) -> Any:
    return SimpleNamespace(
        id=doc_id,
        content=content,
        frontmatter=frontmatter,
        updated_at="2026-03-04T12:00:00Z",
        revision=1,
    )


class _OneDocumentClient:
    def __init__(self, document: Any) -> None:
        self._document = document

    def get_document(self, doc_id: str) -> Any:
        return self._document if doc_id == self._document.id else None


def test_a_clarification_is_served_as_its_own_kind_not_as_a_capture() -> None:
    """Flattened as a capture it renders as an empty "uncategorized" row.

    The failure this model already suffered once, for a rationale: a record with
    no question text, read on a path that expects one, produces a conclusion with
    nothing behind it.
    """
    clarification = make_clarification(authored_by_agent="opencode", authored_by_model=MODEL)
    document = _doc(
        clarification_document_id(clarification),
        build_clarification_content(clarification),
        build_clarification_frontmatter(clarification),
    )
    record = capture_of(_OneDocumentClient(document), document.id)

    assert record is not None
    assert record["kind"] == "clarification"
    assert record["category"] == ""
    assert record["capture_source"] == "collect"
    # The id carries an extra path segment, so the repo and PR come from the
    # frontmatter rather than the document id — and a row without them is a row
    # a reader cannot place in the world.
    assert record["repo"] == "org/repo"
    assert record["pr"] == 42
    assert record["answer"].strip(), "a row with no text is the failure being guarded"
    assert record["answer"] == STATEMENT
    assert record["comment_author"] == "davy"
    assert record["answered_by_agent"] == "opencode"
    assert record["clarification_model"] == MODEL
    assert record["clarification_comment_id"] == 201
    assert record["clarification_association"] == "OWNER"
    assert record["agent_authored"] is True


def test_a_captured_clarification_is_not_reported_as_unverified() -> None:
    """The mirror of the rationale's exemption, for the opposite reason.

    A rationale is exempt from the "unverified" badge because it never claimed to
    be a capture. A clarification is exempt because it is one: its anchor is the
    comment id, and a comment can be re-fetched by anyone.
    """
    document = _doc(
        clarification_document_id(make_clarification()),
        build_clarification_content(make_clarification()),
        build_clarification_frontmatter(make_clarification()),
    )
    record = capture_of(_OneDocumentClient(document), document.id)

    assert record is not None
    assert record["capture_source"] in {"collect", "webhook"}


def test_the_mcp_read_surface_shows_the_agent_that_wrote_a_clarification() -> None:
    """Visible where an answer's is, or the record is unreadable to an agent.

    Rendered through the same fenced evidence envelope, so this asserts the keys
    are surfaced in the provenance block rather than that the content parses.
    """
    from mcp_server.server import _render_tanseki_document

    clarification = make_clarification(
        capture_source=CaptureSource.WEBHOOK,
        capture_delivery_id="0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0",
        authored_by_agent="opencode",
        authored_by_model=MODEL,
    )
    rendered = _render_tanseki_document(
        SimpleNamespace(
            id=clarification_document_id(clarification),
            path=f"{clarification_document_id(clarification)}.md",
            collection="kojutsu",
            content=build_clarification_content(clarification),
            frontmatter=build_clarification_frontmatter(clarification),
        )
    )
    payload = json.loads(rendered.splitlines()[1])

    assert payload["provenance"]["clarified_by_agent"] == "opencode"
    assert payload["provenance"]["clarified_by_model"] == MODEL
    assert payload["provenance"]["capture_source"] == "webhook"
    assert "provenance_anomalies" not in payload["provenance"], (
        "a fully anchored capture must not be reported as anomalous, or the record "
        "is served as evidence and labelled a forgery at the same time"
    )


# --- dashboard ---------------------------------------------------------------

_HARNESS = """
function __makeNode(tag) {
  return {
    tag: tag, className: '', textContent: '', innerHTML: '', value: '',
    style: {}, dataset: {}, childNodes: [],
    addEventListener() {},
    classList: { add() {}, remove() {} },
    append(...kids) { for (const k of kids) if (k != null) this.childNodes.push(k); },
    replaceChildren(...kids) { this.childNodes = []; this.append(...kids); },
    get options() {
      const found = this.childNodes.filter((n) => n && n.tag === 'option');
      return found.length ? found : [{ value: '' }];
    },
  };
}
const __nodes = new Map();
function __nodeFor(selector) {
  if (!__nodes.has(selector)) {
    const node = __makeNode('div');
    if (selector === '#filters') {
      for (const name of __CONTROLS__) node[name] = __makeNode(name);
      node.addEventListener = () => {};
    } else if (selector === '#source') {
      node.base = __makeNode('input');
      node.every = __makeNode('select');
      node.addEventListener = () => {};
    }
    __nodes.set(selector, node);
  }
  return __nodes.get(selector);
}
globalThis.document = {
  querySelector: __nodeFor,
  createElement: (tag) => __makeNode(tag),
  createTextNode: (text) => ({ tag: '#text', textContent: String(text), childNodes: [] }),
};
globalThis.location = { protocol: 'http:', origin: 'http://console.test' };
globalThis.setInterval = () => 0;
globalThis.clearInterval = () => {};
// A fetch that never settles: the harness drives render() itself, and an
// unresolved promise cannot race the assertions.
globalThis.fetch = () => new Promise(() => {});
"""


_FILTERS_FORM_RE = re.compile(r'<form id="filters">(.*?)</form>', re.DOTALL)
_CONTROL_NAME_RE = re.compile(r'<(?:input|select|textarea)[^>]*\bname="([^"]+)"')


def _filter_control_names() -> list[str]:
    """The named controls the filters form actually has, read out of the dashboard.

    Derived rather than listed. A hand-maintained list here is a second copy of the
    form, and the two drift the moment a filter is added: the dashboard then reads
    ``form.<name>.value`` for a control this stub never created, and the page throws
    a TypeError while every API call keeps returning 200. That is the same blank-page
    failure the ``node --check`` guard exists for, reached a different way, and it
    was reached by adding one filter.
    """
    form = _FILTERS_FORM_RE.search(DASHBOARD.read_text(encoding="utf-8"))
    assert form is not None, "the dashboard lost its filters form"
    return _CONTROL_NAME_RE.findall(form.group(1))


def _text_of(node: Any) -> str:
    if isinstance(node, str):
        return node
    if not hasattr(node, "childNodes"):
        return str(getattr(node, "textContent", "") or "")
    parts = [str(node.textContent or "")]
    for child in node.childNodes:
        parts.append(_text_of(child))
    return "".join(parts)


def _run_dashboard(
    captures: list[dict[str, Any]],
    tmp_path: Path,
    *,
    category: str = "",
) -> dict[str, Any]:
    """Render ``captures`` through the dashboard's own code and read the DOM back.

    The dashboard is JavaScript, so the only honest way to test that it counts a
    clarification under its own heading is to run it. No Python test can see a
    wrong count here, and a wrong count is a headline number.

    ``category`` is then applied as the dashboard's own category filter, and the
    rows it leaves behind are reported as ``filtered_rows`` — a record kind with no
    question must not be reachable through a question axis.
    """
    if shutil.which("node") is None:
        pytest.skip("node is required to render the dashboard")
    script = "\n".join(_SCRIPT_RE.findall(DASHBOARD.read_text(encoding="utf-8")))
    harness = f"""
const __items = {json.dumps(captures)};
render({{ captures: __items, meta: {{}} }});
function __textOf(node) {{
  if (node == null) return '';
  if (typeof node === 'string') return node;
  if (!Array.isArray(node.childNodes)) return String(node.textContent || '');
  // The dashboard writes rendered Markdown through innerHTML, which a real DOM
  // would parse into child nodes. The stub does not, so it is read back as the
  // string it is or every row would look empty.
  let out = String(node.textContent || '') + (node.innerHTML ? String(node.innerHTML) : '');
  for (const child of node.childNodes) out += __textOf(child);
  return out;
}}
const kpis = {{}};
for (const card of $('#kpis').childNodes) {{
  kpis[__textOf(card.childNodes[1])] = __textOf(card.childNodes[0]);
}}
const __categories = $('#by-category').childNodes.map(__textOf);
const rows = $('#captures').childNodes.map(__textOf);
const note = $('#source-note').textContent;
$('#filters').category.value = {json.dumps(category)};
renderCaptures();
process.stdout.write(JSON.stringify({{
  kpis,
  categories: __categories,
  rows,
  filtered_rows: $('#captures').childNodes.map(__textOf),
  note,
}}));
"""
    script_file = tmp_path / "dashboard_harness.js"
    script_file.write_text(
        _HARNESS.replace("__CONTROLS__", json.dumps(_filter_control_names()))
        + "\n"
        + script
        + "\n"
        + harness,
        encoding="utf-8",
    )
    result = subprocess.run(  # noqa: S603 - node checks a file this test just wrote
        ["node", str(script_file)],  # noqa: S607 - node via PATH is the documented check
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"the dashboard harness failed:\n{result.stderr.strip()}"
    return json.loads(result.stdout)


def test_the_dashboard_counts_a_clarification_under_its_own_heading(
    tmp_path: Path,
) -> None:
    clarification = make_clarification()
    rendered = _run_dashboard(
        [
            {
                "id": clarification_document_id(clarification),
                "kind": "clarification",
                "category": "",
                "repo": "org/repo",
                "pr": 42,
                "author": "davy",
                "comment_author": "davy",
                "capture_source": "collect",
                "agent_authored": False,
                "title": f"Clarification {clarification.entry_id}",
                "answer": STATEMENT,
                "answered_at": DECLARED_AT.isoformat(),
            },
            {
                "id": "org/repo/pr-42/answer-1",
                "kind": "capture",
                "category": "trade_off",
                "repo": "org/repo",
                "pr": 42,
                "author": "opencode",
                "comment_author": "davy",
                "capture_source": "webhook",
                "agent_authored": True,
                "title": "Why is the window so narrow?",
                "answer": "Because of the index rebuild.",
                "answered_at": DECLARED_AT.isoformat(),
            },
        ],
        tmp_path,
    )

    assert rendered["kpis"]["clarifications"] == "1"
    assert rendered["kpis"]["rationales"] == "0"
    assert rendered["kpis"]["categories"] == "1", (
        "a clarification has no question, so counting it would put an empty "
        "'uncategorized' row on the question-category axis"
    )
    # It is a capture, so it stays inside the capture total rather than being
    # hidden from it.
    assert rendered["kpis"]["captures"] == "2"
    assert not any("uncategorized" in bar for bar in rendered["categories"])
    assert any("trade_off" in bar for bar in rendered["categories"])
    assert any("clarification" in row and STATEMENT in row for row in rendered["rows"])
    assert "clarifications" in rendered["note"]


def test_a_clarification_is_not_reachable_through_the_category_filter(tmp_path: Path) -> None:
    """The filter is a question axis, and this record is not on it.

    A clarification carries no category, so with the filter on it must simply not
    match — not be folded into "uncategorized", which would put a record with no
    question in the bucket that means "the question text was lost".
    """
    clarification = make_clarification()
    rendered = _run_dashboard(
        [
            {
                "id": clarification_document_id(clarification),
                "kind": "clarification",
                "category": "",
                "repo": "org/repo",
                "pr": 42,
                "author": "davy",
                "comment_author": "davy",
                "capture_source": "collect",
                "agent_authored": False,
                "title": f"Clarification {clarification.entry_id}",
                "answer": STATEMENT,
                "answered_at": DECLARED_AT.isoformat(),
            },
            {
                "id": "org/repo/pr-42/answer-1",
                "kind": "capture",
                "category": "trade_off",
                "repo": "org/repo",
                "pr": 42,
                "author": "opencode",
                "comment_author": "davy",
                "capture_source": "webhook",
                "agent_authored": True,
                "title": "Why is the window so narrow?",
                "answer": "Because of the index rebuild.",
                "answered_at": DECLARED_AT.isoformat(),
            },
        ],
        tmp_path,
        category="trade_off",
    )

    assert any("index rebuild" in row for row in rendered["filtered_rows"])
    assert not any(STATEMENT in row for row in rendered["filtered_rows"])


def test_the_dashboard_never_badges_a_clarification_unverified(tmp_path: Path) -> None:
    """A ``collect`` clarification is real evidence, and the badge would libel it."""
    rendered = _run_dashboard(
        [
            {
                "id": "org/repo/pr-42/clarification/clarification-v1-a",
                "kind": "clarification",
                "category": "",
                "repo": "org/repo",
                "pr": 42,
                "author": "davy",
                "comment_author": "davy",
                "capture_source": "collect",
                "agent_authored": False,
                "title": "Clarification a",
                "answer": STATEMENT,
                "answered_at": DECLARED_AT.isoformat(),
            }
        ],
        tmp_path,
    )

    row = next(row for row in rendered["rows"] if STATEMENT in row)
    assert "unverified" not in row
    assert "clarification · collect" in row
    # The note may say the record was collected rather than webhook-delivered —
    # that is true, and the comment id is why — but it must not call it unverified.
    assert "collected from the provider API" in rendered["note"]
    assert "unverified" not in rendered["note"]
    assert "asserted" not in rendered["note"]


def test_the_kind_tag_is_stored_once_however_many_callers_name_it() -> None:
    """A tag stored twice is a record counted twice.

    Found by backfilling a real repository and reading the stored document back: the
    collector names the kind in its own tags and the frontmatter builder prepends it,
    so the document carried ``["clarification", "clarification",
    "clarification_collect"]``. The builder already guarded ``agent_authored`` against
    exactly this and did not guard the kind tag it owns, which is the usual way a rule
    that lives in two places drifts -- and nothing in the suite noticed, because the
    suite only ever built a clarification whose tags did not collide.
    """
    clarification = make_clarification(tags=["clarification", "clarification_collect"])

    frontmatter = build_clarification_frontmatter(clarification)

    tags = frontmatter["tags"]
    assert tags == ["clarification", "clarification_collect"]
    assert len(tags) == len(set(tags)), f"duplicate tags stored: {tags}"


def test_agent_authored_is_still_added_when_the_collector_omits_it() -> None:
    """The dedupe above must not become a way to drop a tag the caller did not send."""
    clarification = make_clarification(authored_by_agent="opencode", tags=["clarification"])

    frontmatter = build_clarification_frontmatter(clarification)

    assert "agent_authored" in frontmatter["tags"]
