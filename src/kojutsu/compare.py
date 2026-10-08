"""Compare one change's two rationales: the stated one and the inferred one.

:mod:`kojutsu.core.rationale_link` does the comparison and does it carefully —
it classifies the relationship between a declared rationale and a reconstructed
one, computes independence through the one provenance scale the codebase has,
renders the caveat before either text, and carries ``not_reached`` for a bounded
comparison. All of it was reachable only from its own test file, which is the
worse of the two states a finished module can be in, because an unreachable
function looks finished.

This module is the surface that runs it, and deliberately nothing more. It reads
the stored rationales for one change through the existing Tanseki read path, hands
them to :func:`~kojutsu.core.rationale_link.compare_rationales` unchanged, and
prints :func:`~kojutsu.core.rationale_link.render_comparison` unchanged. The
outcome vocabulary, the independence scale, the classification rules and the render
order are the ones that module defines. None of them is restated, re-derived,
reordered or extended here, because two vocabularies for one comparison is exactly
what that module exists to prevent — and the failure would be quiet, because both
would be right about most pairs.

**Why a separate command rather than a tool on the read server.**
``tests/test_mcp_server.py`` pins the read server's tool table to exactly
``{search_knowledge, get_knowledge_entry}`` and asserts its module reaches no write
path. The comparison is a derived reading of two stored records, so a tool there
would be a legitimate read, and adding it would widen a table that was frozen on
purpose. A separate console entry point keeps the read server's boundary a
property of *which process an agent is connected to* — the same argument
``docs/design-review/open-questions.md`` makes for putting the capture surface on
its own server. ``kojutsu`` and ``kojutsu-mcp`` are already separate entry
points; this is the third, and it reads only.

**What this deliberately does not do.**

- **It does not say which rationale is right.** The comparison says whether two
  statements about a change agree, disagree, or come from the same mind. It does
  not say which is correct, whether the disagreement mattered, or what anyone
  should do about it, and no output here is arranged to invite "the reviewer
  disagreed with the implementer, therefore…". Whether a divergence mattered is a
  question about one specific change, and answering it is a person's job.
- **It does not store the comparison.** Both inputs are already stored. A stored
  derived conclusion invites being read as something a person concluded rather than
  as an arithmetic result over two claims, and it goes stale the moment either
  rationale is revised. Keep the two rationales and recompute.
- **It does not compare across changes, count divergences, or feed a gate.** A
  divergence rate that starts influencing what gets built stops being a
  measurement, one level above the failure ``rationale_link`` already guards
  against.
- **It never reports agreement over text it did not read in full.** Where a stored
  rationale cannot be summarised honestly, the document is named in
  ``not_reached`` and the comparison becomes one-sided, rather than comparing a
  truncated text that could turn a divergence into an agreement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import typer

from kojutsu import allowlist
from kojutsu.config import get_settings
from kojutsu.core.rationale_link import (
    RationaleComparison,
    RationaleSummary,
    compare_rationales,
    render_comparison,
)
from kojutsu.dev_console import rationale_from_content
from kojutsu.integrations.github import parse_pr_identifier
from kojutsu.integrations.tanseki import (
    MAX_SEARCH_RESULTS,
    TansekiClient,
    TansekiDocument,
    TansekiError,
    bound_search_limit,
)
from kojutsu.models import UNKNOWN_MODEL, RationaleSource

#: Every stored rationale carries this tag (``build_rationale_frontmatter``), so it
#: is what keeps this query inside the ``rationale/`` namespace rather than
#: answering with every capture on the change.
RATIONALE_TAG = "rationale"

#: Documents per comparison. A change with more rationales than this is a bounded
#: comparison, and a bounded comparison says so by name.
DEFAULT_RATIONALE_LIMIT = 10

#: Ceiling on one rationale's stated reason, well above the ``MAX_RATIONALE_CHARS``
#: the capture path already enforces at write time. A document past it is not
#: truncated: clipping the text before the comparison would compare a prefix and
#: could report agreement where the reasons differ further in, which is the one
#: thing this surface must never do. It is named in ``not_reached`` instead.
MAX_COMPARABLE_TEXT_CHARS = 20_000

app = typer.Typer(
    help="Compare a change's stated rationale with the one inferred from its diff.",
)


@dataclass(frozen=True)
class RationaleComparisonReport:
    """What the command ran, so a reader can go and look at the records.

    This is the addressing of a comparison, not a second one. It carries the
    comparison ``rationale_link`` produced and the document ids it was made from,
    and holds no outcome of its own: the only classification in this module is the
    one that module defines.
    """

    repo: str
    pr_number: int
    comparison: RationaleComparison
    #: Store addresses of the documents placed on either side of the comparison.
    compared: tuple[str, ...] = ()


def _scalar(frontmatter: dict[str, Any], key: str) -> str | None:
    """Read one frontmatter value as text, or ``None`` when it states nothing.

    Absent is reported as absent. Nothing here defaults a missing value, because a
    guessed principal or a guessed model is the manufactured provenance this whole
    axis exists to keep out of the store.
    """
    value = frontmatter.get(key)
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _revision(value: object) -> int | None:
    """Read ``rationale_revision``, which the writer stores stringified."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _model(frontmatter: dict[str, Any]) -> str | None:
    """Read the declared model, reading a legacy placeholder back as an absence.

    ``build_rationale_frontmatter`` used to write ``UNKNOWN_MODEL`` into
    ``declared_by_model`` whenever the author stated no model, which
    :data:`kojutsu.models.UNKNOWN_MODEL`'s own docstring calls out as exactly
    the encoding it was written to stop: a principal nobody named came to look
    like one who declared a model called ``"unknown"``, indistinguishable to a
    reader of that key from a principal who really named that model. The writer no
    longer does this, so new documents leave the key absent.

    The tolerance stays for one reason: documents already in the store carry the
    placeholder, and dropping it would make the current corpus unreadable rather
    than honest. Handing that string to
    :func:`~kojutsu.models.compute_independence` would compare it as a model
    and manufacture a separation the record does not support, so it is read back
    as the absence it stands for — which lands on the conservative
    ``SELF_CERTIFIED`` side, the direction ``rationale_link`` wants.

    Two absences must therefore be told apart by anyone who cares: a key that is
    *absent* is a document written after the fix, and a key reading ``"unknown"``
    is a legacy one. Both mean no model was stated, and neither means a model was.
    """
    model = _scalar(frontmatter, "declared_by_model")
    if model is None or model.casefold() == UNKNOWN_MODEL:
        return None
    return model


def _source(frontmatter: dict[str, Any]) -> tuple[RationaleSource | None, str]:
    """Read which side of the comparison a document belongs on, or why it has none."""
    raw = _scalar(frontmatter, "rationale_source")
    if raw is None:
        return None, "it states no rationale_source"
    try:
        source = RationaleSource(raw.casefold())
    except ValueError:
        return None, "its rationale_source is not a value the comparison knows"
    if source is RationaleSource.UNKNOWN:
        # An explicit ``unknown`` is an honest absence and stays one. Placing it on
        # either side would be the reader guessing at provenance the writer declined
        # to claim, and dropping it silently would report a one-sided change as
        # though it were a one-sided *store*.
        return None, "it states no known rationale_source"
    return source, ""


def _summarise(document: TansekiDocument) -> tuple[RationaleSummary | None, str]:
    """Return the comparison's view of one stored rationale, or why there is none.

    Every refusal here names the document rather than dropping it, so a rationale
    the comparison could not use is visible as a rationale the comparison did not
    reach. That is the same rule the bounded case follows, applied to a record
    that turned out to be unusable: a side reported as missing must be a side the
    reader can see was looked for.
    """
    frontmatter = document.frontmatter
    if not isinstance(frontmatter, dict):
        return None, "its frontmatter is unreadable"
    source, reason = _source(frontmatter)
    if source is None:
        return None, reason
    declared_by = _scalar(frontmatter, "declared_by")
    if declared_by is None:
        return None, "it names no declaring principal"
    revision = _revision(frontmatter.get("rationale_revision"))
    if revision is None:
        return None, "it states no usable rationale_revision"
    # The reason is read out of the body rather than the frontmatter because that
    # is where the writer put it: ``_rationale_body`` leads with ``## Reason`` and
    # keeps the unverifiable claim about who produced it under ``## Attribution``.
    # Reusing the console's reader keeps one definition of how a stored reason is
    # found, so the dashboard and this cannot disagree about what a document says.
    text = rationale_from_content(document.content)
    if not text:
        return None, "it has no readable Reason section"
    if len(text) > MAX_COMPARABLE_TEXT_CHARS:
        return None, (
            f"its stated reason is over the {MAX_COMPARABLE_TEXT_CHARS} characters "
            "this command reads in full"
        )
    return (
        RationaleSummary(
            entry_id=document.id.rsplit("/", 1)[-1],
            source=source,
            declared_by=declared_by,
            model=_model(frontmatter),
            revision=revision,
            text=text,
        ),
        "",
    )


def compare_change(
    client: TansekiClient,
    *,
    repo: str,
    pr_number: int,
    limit: int = DEFAULT_RATIONALE_LIMIT,
) -> RationaleComparisonReport:
    """Read one change's rationales from Tanseki and compare the two sides.

    The query is the one the storage layout already answers: documents tagged
    ``rationale`` whose frontmatter names this repository and pull request, which
    ``rationale_document_id`` writes as ``<repo>/pr-<n>/rationale/<entry_id>``. A
    declared and a reconstructed rationale for a change therefore arrive from the
    same query, and a change with neither arrives as an empty result rather than as
    a request that had to know in advance which kind to look for.

    The returned id prefix is re-checked here. Tanseki filters on the frontmatter
    columns, so a document outside this change's ``rationale/`` path means the
    store returned more than was asked for; it is named and dropped rather than
    compared as though it were about this change.
    """
    effective_limit = bound_search_limit(limit)
    prefix = f"{repo}/pr-{pr_number}/rationale/"
    hits = client.search(
        "",
        tags=[RATIONALE_TAG],
        frontmatter={"repo": repo, "pr": str(pr_number)},
        limit=effective_limit,
    )
    documents = client.get_documents([hit.id for hit in hits])

    not_reached: list[str] = []
    # ``search`` returns a prefix of the rank order and says nothing about what it
    # dropped, so a full page is indistinguishable from a truncated one. Reported
    # as possibly truncated rather than silently treated as the whole change: the
    # bound may exclude nothing, and claiming a complete comparison when it might
    # not be one is the failure this whole path exists to avoid.
    budget_saturated = len(hits) >= effective_limit
    by_source: dict[RationaleSource, list[tuple[str, RationaleSummary]]] = {}
    for hit, document in zip(hits, documents, strict=True):
        if document is None:
            # A store-side race between the search and the fetch, reported as its
            # own thing so it is never read as a rationale that was never stored.
            not_reached.append(f"{hit.id} (it vanished between the search and the fetch)")
            continue
        if not document.id.startswith(prefix):
            not_reached.append(f"{document.id} (it is not stored under this change's rationale/)")
            continue
        summary, reason = _summarise(document)
        if summary is None:
            not_reached.append(f"{document.id} ({reason})")
            continue
        by_source.setdefault(summary.source, []).append((document.id, summary))

    compared: list[str] = []
    chosen: dict[RationaleSource, RationaleSummary] = {}
    for source in (RationaleSource.DECLARED, RationaleSource.RECONSTRUCTED):
        candidates = by_source.get(source, [])
        if not candidates:
            continue
        # The latest revision on each side, because an early intent is usually the
        # one a later revision contradicts. The document id breaks a tie so two
        # runs over the same store pick the same pair. Every other candidate is
        # named rather than dropped: a reader told two rationales agree should
        # still be able to see the third that was not consulted.
        best_id, best = max(candidates, key=lambda item: (item[1].revision, item[0]))
        chosen[source] = best
        compared.append(best_id)
        for document_id, summary in candidates:
            if document_id != best_id:
                not_reached.append(
                    f"{document_id} (revision {summary.revision}; only the latest "
                    f"revision on each side is compared)"
                )
    if budget_saturated:
        not_reached.append(
            f"rationale documents for {repo} pr-{pr_number} beyond the first "
            f"{effective_limit} (the store returned a full page, so more may exist)"
        )

    return RationaleComparisonReport(
        repo=repo,
        pr_number=pr_number,
        comparison=compare_rationales(
            chosen.get(RationaleSource.DECLARED),
            chosen.get(RationaleSource.RECONSTRUCTED),
            not_reached=tuple(not_reached),
        ),
        compared=tuple(compared),
    )


@app.command()
def compare(
    change: str = typer.Argument(
        ...,
        help="The change, as owner/repo#123 or a GitHub pull request URL",
    ),
    limit: int = typer.Option(
        DEFAULT_RATIONALE_LIMIT,
        "--limit",
        min=1,
        max=MAX_SEARCH_RESULTS,
        help="Rationale documents to read; a full page is reported as a bounded comparison",
    ),
) -> None:
    """Report what one change's two rationales have to say about each other.

    The report says whether the stated reason and the inferred one agree, disagree,
    or come from the same mind. It does not say which of them is right, and it
    stores nothing.
    """
    parsed = parse_pr_identifier(change)
    if parsed is None:
        typer.echo(
            "Provide a change as owner/repo#123 or a GitHub pull request URL.",
            err=True,
        )
        raise typer.Exit(2)
    repo, pr_number = parsed

    # The one shared allowlist predicate, called before anything reaches the store.
    # A read surface is governed exactly as ``search_knowledge`` is, and the write
    # path that captured these records authorises against the same definition, so a
    # change that can be written can be compared.
    if not allowlist.is_valid_repository(repo):
        typer.echo("Repository must be a valid owner/name value.", err=True)
        raise typer.Exit(2)
    settings = get_settings()
    if not allowlist.repository_allowed(repo, settings):
        typer.echo("Repository is not authorized.", err=True)
        raise typer.Exit(2)
    if not settings.tanseki_enabled:
        typer.echo("Tanseki is not configured; set TANSEKI_URL.", err=True)
        raise typer.Exit(1)

    client = TansekiClient.from_settings(settings)
    try:
        report = compare_change(client, repo=repo, pr_number=pr_number, limit=limit)
    except TansekiError as exc:
        typer.echo(f"Tanseki request failed: {exc}", err=True)
        raise typer.Exit(1) from None
    finally:
        client.close()

    # Addressing only. Naming the change and the two documents ahead of the report
    # tells a reader which two records this is about, and adds nothing about whether
    # they agree — which stays the first thing the rendered comparison says.
    typer.echo(f"Rationale comparison for {repo} pr-{pr_number}")
    for document_id in report.compared:
        typer.echo(f"Compared: {document_id}")
    typer.echo("")
    typer.echo(render_comparison(report.comparison))


def main() -> None:
    """Entry point for the ``kojutsu-compare`` console script."""
    app()


if __name__ == "__main__":
    main()
