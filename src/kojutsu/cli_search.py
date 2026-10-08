"""CLI commands: search knowledge and list registry work."""

from typing import Any, cast

import typer

from kojutsu.cli_shared import _require_tanseki, _settings
from kojutsu.core.question_registry import QUESTION_STATUSES, build_registry
from kojutsu.integrations.tanseki import TansekiClient, TansekiError, bound_search_limit


def search_cmd(
    text: str | None = typer.Argument(None, help="Free-text search over captured decisions"),
    repo: str | None = typer.Option(None, "--repo", help="Filter by repository owner/name"),
    jira: str | None = typer.Option(None, "--jira", help="Filter by Jira ticket key"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
) -> None:
    """Query the Tanseki knowledge store."""
    settings = _settings()
    _require_tanseki(settings)

    filters: dict[str, str] = {}
    if repo:
        filters["repo"] = repo
    if jira:
        filters["jira"] = jira

    try:
        limit = bound_search_limit(limit)
        client = TansekiClient.from_settings(settings)
        try:
            documents: list = []
            search_documents = cast(Any, getattr(client, "search_documents", None))
            if search_documents is not None:
                documents = list(
                    search_documents(
                        text or "",
                        frontmatter=filters or None,
                        limit=limit,
                    )
                )
            else:
                for hit in client.search(text or "", frontmatter=filters or None, limit=limit):
                    document = client.get_document(hit.id)
                    if document is not None:
                        documents.append(document)
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    except TansekiError:
        typer.echo(
            "Unable to search Tanseki. Verify TANSEKI_URL, TANSEKI_API_KEY, and service health.",
            err=True,
        )
        raise typer.Exit(1) from None

    for doc in documents:
        fm = doc.frontmatter
        typer.echo(f"[{fm.get('repo', '?')} #{fm.get('pr', '?')}] {fm.get('category', '?')}")
        typer.echo(f"  {doc.content[:200]}")
        typer.echo()


def questions_cmd(
    status: str | None = typer.Option(
        "pending", "--status", help="Lifecycle state to show, or 'all' for every state"
    ),
    repo: str | None = typer.Option(None, "--repo", help="Filter by repository owner/name"),
    pr_number: int | None = typer.Option(None, "--pr", help="Filter by PR number"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max questions to show"),
) -> None:
    """List outstanding capture work in the local question registry.

    Defaults to the questions still awaiting an answer, which is the set a worker
    consumes. The per-state counts above the listing always cover every state, so
    the queue is visible even when the listing is filtered or bounded.
    """
    settings = _settings()
    if limit < 0:
        typer.echo("--limit must be non-negative.", err=True)
        raise typer.Exit(1)
    if status is not None and status != "all" and status not in QUESTION_STATUSES:
        typer.echo(
            f"Unknown status: {status}. Choose from: all, {', '.join(QUESTION_STATUSES)}.", err=True
        )
        raise typer.Exit(1)

    selected = None if status == "all" else status
    try:
        with build_registry(settings) as registry:
            counts = {name: registry.count_questions(status=name) for name in QUESTION_STATUSES}
            outstanding = registry.count_questions(status="pending")
            rows = registry.list_questions(
                status=selected, repo=repo, pr_number=pr_number, limit=limit
            )
    except (OSError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None

    typer.echo(f"registry: {settings.kojutsu_registry_path}")
    typer.echo(f"outstanding: {outstanding} (pending)")
    typer.echo("  ".join(f"{name}={count}" for name, count in counts.items()))
    if not rows:
        typer.echo("no questions match")
        return
    for row in rows:
        typer.echo(
            f"  {row['question_id']} state={row['status']} repo={row['repo'] or '-'} "
            f"pr={row['pr_number'] if row['pr_number'] is not None else '-'} "
            f"attempts={row['attempts']} assignee={row['assignee'] or '-'} "
            f"error={row['last_error'] or '-'}"
        )


def register(app: typer.Typer) -> None:
    """Attach these commands to a Typer app."""
    app.command("search")(search_cmd)
    app.command("questions")(questions_cmd)
