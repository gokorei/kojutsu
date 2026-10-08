"""Tests for the ``kojutsu-compare`` surface.

The documents these tests read are built by the real writer —
``to_rationale_upsert_payload`` — rather than hand-assembled, so a change to the
stored layout or the frontmatter contract breaks this file instead of leaving it
green against a shape the store no longer holds.

What each test pins is a property the surface is *not* allowed to lose, not a
particular output string: the caveat leads, an absence is never an agreement, a
restatement says it teaches nothing, and a bounded comparison names what it did
not reach.
"""

from __future__ import annotations

import asyncio
import importlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from kojutsu import compare as compare_module
from kojutsu.core.rationale_link import ComparisonOutcome
from kojutsu.core.tanseki_mapping import to_rationale_upsert_payload
from kojutsu.integrations.tanseki import TansekiDocument, TansekiHit
from kojutsu.models import RationaleEntry, RationaleSource

runner = CliRunner()

REPO = "org/repo"
PR = 7
CHANGE = f"{REPO}#{PR}"

MODEL_A = "opencode/model"
MODEL_B = "anthropic/claude-opus-5"


# --- documents, built by the real writer --------------------------------------


def rationale_document(
    *,
    entry_id: str,
    source: RationaleSource = RationaleSource.DECLARED,
    declared_by: str = "bot",
    model: str | None = MODEL_A,
    text: str = "Chose exponential backoff for idempotency.",
    revision: int = 1,
    repo: str = REPO,
    pr_number: int | None = PR,
) -> TansekiDocument:
    """Build a stored rationale exactly as the write path would store it."""
    entry = RationaleEntry(
        entry_id=entry_id,
        repo=repo,
        pr_number=pr_number,
        branch="feat/backoff",
        declared_by=declared_by,
        declared_model=model,
        rationale_text=text,
        source=source,
        revision=revision,
        revises=None if revision == 1 else f"{entry_id}-v{revision - 1}",
        declared_at=datetime(2026, 1, 2, 3, 4, tzinfo=UTC),
    )
    payload = to_rationale_upsert_payload(entry)
    return TansekiDocument(
        id=payload["id"],
        path=payload["path"],
        collection="kojutsu-real",
        content=payload["content"],
        frontmatter=payload["frontmatter"],
    )


# --- a faked Tanseki, at the seam the command builds ------------------------------


class FakeTanseki:
    """The read slice of the Tanseki client, plus two ways to misbehave on purpose."""

    def __init__(
        self,
        documents: dict[str, TansekiDocument],
        *,
        vanish: tuple[str, ...] = (),
        ignore_filters: bool = False,
    ) -> None:
        self.documents = documents
        self.vanish = set(vanish)
        self.ignore_filters = ignore_filters
        self.closed = False
        self.calls: list[tuple[str, object]] = []

    def search(
        self,
        query: str,
        *,
        tags: list[str] | None = None,
        frontmatter: dict[str, str] | None = None,
        limit: int = 10,
        collection: str | None = None,
    ) -> list[TansekiHit]:
        self.calls.append(("search", dict(frontmatter or {})))
        if self.ignore_filters:
            return [TansekiHit(id=key, score=1.0) for key in self.documents][:limit]
        filters = frontmatter or {}
        matches = [
            TansekiHit(id=key, score=1.0)
            for key, document in self.documents.items()
            if all(str(document.frontmatter.get(k)) == str(v) for k, v in filters.items())
            and all(tag in document.frontmatter.get("tags", []) for tag in tags or [])
        ]
        return matches[:limit]

    def get_documents(
        self, doc_ids: list[str], collection: str | None = None, **kwargs: Any
    ) -> list[TansekiDocument | None]:
        return [None if doc_id in self.vanish else self.documents.get(doc_id) for doc_id in doc_ids]

    def close(self) -> None:
        self.closed = True


def install_tanseki(
    monkeypatch: pytest.MonkeyPatch,
    documents: dict[str, TansekiDocument],
    *,
    vanish: tuple[str, ...] = (),
    ignore_filters: bool = False,
    tanseki_url: str = "https://tanseki.test",
    allowed: str = REPO,
) -> FakeTanseki:
    """Point the command at a faked store, as ``mcp_server.server.configure`` does."""
    fake = FakeTanseki(documents, vanish=vanish, ignore_filters=ignore_filters)
    monkeypatch.setenv("TANSEKI_URL", tanseki_url)
    monkeypatch.setenv("TANSEKI_COLLECTION", "kojutsu-real")
    monkeypatch.setenv("GITHUB_WEBHOOK_ALLOWED_REPOSITORIES", allowed)
    monkeypatch.setattr(compare_module, "TansekiClient", _stub_client_class(fake))
    return fake


def _stub_client_class(fake: FakeTanseki) -> type:
    class StubClient:
        def __init__(self, settings: Any) -> None:
            self.settings = settings

        @classmethod
        def from_settings(cls, settings: Any) -> StubClient:
            return cls(settings)

        def search(self, *args: Any, **kwargs: Any) -> list[TansekiHit]:
            return fake.search(*args, **kwargs)

        def get_documents(self, *args: Any, **kwargs: Any) -> list[TansekiDocument | None]:
            return fake.get_documents(*args, **kwargs)

        def close(self) -> None:
            fake.close()

    return StubClient


def run(*args: str) -> Any:
    return runner.invoke(compare_module.app, list(args))


# --- reachable, and says what it found ----------------------------------------


def test_a_change_with_both_rationales_is_reachable_and_reports_divergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The finding the whole programme exists for, end to end through the command."""
    declared = rationale_document(entry_id="rl-declared", text="The retry exists for idempotency.")
    reconstructed = rationale_document(
        entry_id="rl-reconstructed",
        source=RationaleSource.RECONSTRUCTED,
        declared_by="reviewer",
        model=MODEL_B,
        text="The retry exists for rate limiting.",
    )
    install_tanseki(monkeypatch, {declared.id: declared, reconstructed.id: reconstructed})

    result = run(CHANGE)

    assert result.exit_code == 0, result.output
    assert f"Outcome: {ComparisonOutcome.DIVERGENT.value}" in result.output
    assert "Independence: independent" in result.output
    # Both records are named so the reader can go and look at them.
    assert declared.id in result.output
    assert reconstructed.id in result.output
    assert "idempotency" in result.output
    assert "rate limiting" in result.output


def test_the_query_asks_the_store_for_this_change_s_rationale_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Filters are pushed to the store, and it is asked for the rationale tag."""
    document = rationale_document(entry_id="rl-declared")
    fake = install_tanseki(monkeypatch, {document.id: document})

    assert run(CHANGE).exit_code == 0

    assert fake.calls == [("search", {"repo": REPO, "pr": str(PR)})]
    assert fake.closed is True


def test_a_change_with_neither_rationale_reports_the_absence_as_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither side existing is an outcome, not an empty answer."""
    install_tanseki(monkeypatch, {})

    result = run(CHANGE)

    assert result.exit_code == 0, result.output
    assert f"Outcome: {ComparisonOutcome.NEITHER.value}" in result.output
    assert "No rationale of either kind exists" in result.output
    assert "Stated by" not in result.output
    assert "Inferred by" not in result.output


# --- the caveat leads, and an absence is never an agreement -------------------


def test_the_rendering_states_independence_and_the_note_before_either_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ordering ``render_comparison`` already guarantees, preserved on the way out.

    Presenting the two texts first is how a restatement ends up reading as a second
    opinion, so this asserts the order on the command's own output rather than on
    the function that produces it.
    """
    declared = rationale_document(entry_id="rl-declared", text="For idempotency.")
    reconstructed = rationale_document(
        entry_id="rl-reconstructed",
        source=RationaleSource.RECONSTRUCTED,
        text="For rate limiting.",
    )
    install_tanseki(monkeypatch, {declared.id: declared, reconstructed.id: reconstructed})

    output = run(CHANGE).output

    assert output.index("Independence") < output.index("Stated by")
    assert output.index("Independence") < output.index("Inferred by")
    assert output.index("Outcome") < output.index("Stated by")


def test_a_one_sided_change_names_the_missing_side_and_does_not_claim_agreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One side missing is a finding about the store, never a second opinion."""
    declared = rationale_document(entry_id="rl-declared")
    install_tanseki(monkeypatch, {declared.id: declared})

    output = run(CHANGE).output

    assert f"Outcome: {ComparisonOutcome.DECLARED_ONLY.value}" in output
    assert "No reviewer rationalised it" in output
    assert "Inferred by" not in output


def test_a_reconstruction_without_a_declaration_is_labelled_as_a_guess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reconstructed = rationale_document(
        entry_id="rl-reconstructed",
        source=RationaleSource.RECONSTRUCTED,
        declared_by="reviewer",
        model=MODEL_B,
    )
    install_tanseki(monkeypatch, {reconstructed.id: reconstructed})

    output = run(CHANGE).output

    assert f"Outcome: {ComparisonOutcome.RECONSTRUCTED_ONLY.value}" in output
    assert "Nothing stated what the author intended" in output
    assert "Stated by" not in output


def test_agreement_from_one_principal_on_one_model_is_reported_as_a_restatement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The outcome the module was written to make impossible to present as agreement."""
    declared = rationale_document(entry_id="rl-declared", text="Backoff, for idempotency.")
    reconstructed = rationale_document(
        entry_id="rl-reconstructed",
        source=RationaleSource.RECONSTRUCTED,
        text="Backoff, for idempotency.",
    )
    install_tanseki(monkeypatch, {declared.id: declared, reconstructed.id: reconstructed})

    output = run(CHANGE).output

    assert f"Outcome: {ComparisonOutcome.RESTATEMENT.value}" in output
    assert "NO information" in output
    assert "second opinion" not in output


# --- the bound is visible, and nothing is reported over unread text -----------


def test_a_bounded_comparison_names_what_it_did_not_reach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full page of hits means a bounded answer, and the bound is reported."""
    documents = {}
    for index in range(4):
        document = rationale_document(
            entry_id=f"rl-{index}",
            text=f"Reason number {index}.",
        )
        documents[document.id] = document
    install_tanseki(monkeypatch, documents)

    result = run(CHANGE, "--limit", "2")

    assert result.exit_code == 0, result.output
    assert "NOT COMPARED" in result.output
    assert "beyond the first 2" in result.output


def test_a_superseded_revision_is_named_rather_than_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the latest revision is compared, and the earlier one stays visible."""
    first = rationale_document(entry_id="rl-v1", text="Backoff, for idempotency.")
    second = rationale_document(
        entry_id="rl-v2",
        text="Dropped the jitter.",
        revision=2,
    )
    reconstructed = rationale_document(
        entry_id="rl-reconstructed",
        source=RationaleSource.RECONSTRUCTED,
        declared_by="reviewer",
        model=MODEL_B,
        text="Jitter is there for latency.",
    )
    install_tanseki(
        monkeypatch,
        {first.id: first, second.id: second, reconstructed.id: reconstructed},
    )

    output = run(CHANGE).output

    assert "revision 2" in output
    assert "only the latest revision on each side is compared" in output
    assert first.id in output
    assert "Outcome: divergent" in output


def test_a_rationale_whose_source_is_unknown_is_named_not_assigned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An honest ``unknown`` stays unknown; neither side is invented for it."""
    document = rationale_document(entry_id="rl-unknown", source=RationaleSource.UNKNOWN)
    install_tanseki(monkeypatch, {document.id: document})

    output = run(CHANGE).output

    assert f"Outcome: {ComparisonOutcome.NEITHER.value}" in output
    assert "states no known rationale_source" in output
    assert "Stated by" not in output


def test_an_unreadable_rationale_is_named_instead_of_compared_as_empty_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two empty reasons would compare as agreement, which is the failure to avoid."""
    hollow = rationale_document(entry_id="rl-hollow")
    document = TansekiDocument(
        id=hollow.id,
        path=hollow.path,
        collection=hollow.collection,
        content=hollow.content.replace("## Reason\n", "## Something Else\n"),
        frontmatter=hollow.frontmatter,
    )
    install_tanseki(monkeypatch, {document.id: document})

    output = run(CHANGE).output

    assert "has no readable Reason section" in output
    assert f"Outcome: {ComparisonOutcome.NEITHER.value}" in output


def test_an_oversized_reason_is_refused_rather_than_clipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clipping before the comparison could turn a divergence into an agreement."""
    document = rationale_document(
        entry_id="rl-huge", text="x" * (compare_module.MAX_COMPARABLE_TEXT_CHARS + 1)
    )
    install_tanseki(monkeypatch, {document.id: document})

    output = run(CHANGE).output

    assert "characters" in output
    assert "NOT COMPARED" in output
    assert "x" * 100 not in output


def test_an_unstated_model_is_not_compared_as_a_model_named_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The writer's ``unknown`` placeholder must not manufacture a separation.

    Two rationales from one account, one of which stated no model, land on
    ``SELF_CERTIFIED``. Handing the placeholder string to ``compute_independence``
    instead would read it as a second model and claim a second mind the record
    does not support.
    """
    declared = rationale_document(entry_id="rl-declared", model=None)
    reconstructed = rationale_document(
        entry_id="rl-reconstructed",
        source=RationaleSource.RECONSTRUCTED,
        model=None,
        text="Something else entirely.",
    )
    install_tanseki(monkeypatch, {declared.id: declared, reconstructed.id: reconstructed})

    output = run(CHANGE).output

    assert "Independence: self_certified" in output
    assert "model not stated" in output
    assert f"Outcome: {ComparisonOutcome.RESTATEMENT.value}" in output


def test_a_document_the_store_returned_from_outside_this_change_is_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defence in depth: the returned id prefix is re-checked, not trusted."""
    elsewhere = rationale_document(
        entry_id="rl-other-pr", text="A different change entirely.", pr_number=8
    )
    install_tanseki(monkeypatch, {elsewhere.id: elsewhere}, ignore_filters=True)

    output = run(CHANGE).output

    assert "not stored under this change's rationale/" in output
    assert f"Outcome: {ComparisonOutcome.NEITHER.value}" in output


def test_a_rationale_that_vanished_mid_read_is_named_as_a_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A store-side race is never read as a rationale that was never stored."""
    document = rationale_document(entry_id="rl-declared")
    install_tanseki(monkeypatch, {document.id: document}, vanish=(document.id,))

    output = run(CHANGE).output

    assert "vanished between the search and the fetch" in output
    assert f"Outcome: {ComparisonOutcome.NEITHER.value}" in output


# --- the scope, and the surfaces it must not have reached ----------------------


def test_a_repository_outside_the_allowlist_is_refused_before_the_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = rationale_document(entry_id="rl-declared", repo="other/repo")
    fake = install_tanseki(monkeypatch, {document.id: document}, allowed=REPO)

    result = run("other/repo#7")

    assert result.exit_code == 2
    assert "not authorized" in result.output
    assert fake.calls == []


def test_a_malformed_allowlist_denies_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared predicate fails closed, so a typo denies rather than crashing."""
    monkeypatch.setenv("GITHUB_WEBHOOK_ALLOWED_REPOSITORIES", "not-a-repository")
    monkeypatch.setenv("TANSEKI_URL", "https://tanseki.test")

    result = run(CHANGE)

    assert result.exit_code == 2
    assert "not authorized" in result.output


def test_a_change_that_is_not_a_change_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TANSEKI_URL", "https://tanseki.test")
    monkeypatch.setenv("GITHUB_WEBHOOK_ALLOWED_REPOSITORIES", REPO)

    result = run("org/repo-7")

    assert result.exit_code == 2
    assert "owner/repo#123" in result.output


def test_an_unconfigured_store_says_so_rather_than_reporting_no_rationales(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unconfigured store is not an empty change."""
    monkeypatch.setenv("TANSEKI_URL", "")
    monkeypatch.setenv("GITHUB_WEBHOOK_ALLOWED_REPOSITORIES", REPO)

    result = run(CHANGE)

    assert result.exit_code == 1
    assert "TANSEKI_URL" in result.output
    assert "neither" not in result.output


def test_the_read_mcp_server_tool_table_is_untouched() -> None:
    """The comparison is a separate process, so the read table cannot have grown.

    Pinned from this side as well as from ``tests/test_mcp_server.py``: that test
    says the read server is read-only, and this one says the read server is also
    still exactly two tools.
    """
    server = importlib.reload(importlib.import_module("mcp_server.server"))
    tools = {tool.name for tool in asyncio.run(server.server.list_tools())}

    assert tools == {
        "search_knowledge",
        "get_knowledge_entry",
        "traverse_knowledge",
        "list_knowledge",
    }

    # And the read server's module does not reach the comparison at all, so the
    # boundary is structural rather than a matter of what the table happens to say.
    source = Path(server.__file__ or "").read_text(encoding="utf-8")
    for forbidden in ("compare_rationales", "rationale_link", "kojutsu.compare"):
        assert forbidden not in source, (
            f"the read server references {forbidden!r}. The comparison belongs on its "
            "own console entry point, so adding it here would widen a tool table that "
            "was frozen on purpose."
        )
