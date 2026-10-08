"""CLI commands: inspect and drain the local Tanseki outbox."""

import typer

from kojutsu.cli_shared import (
    _age,
    _echo_outbox_quota,
    _open_outbox,
    _require_tanseki,
    _settings,
)
from kojutsu.core.outbox import OutboxOwnershipError
from kojutsu.runtime import build_runtime


def relay(
    limit: int = typer.Option(100, "--limit", "-n", help="Max queued writes to drain"),
) -> None:
    """Drain queued Tanseki writes from the outbox."""
    settings = _settings()
    _require_tanseki(settings)

    with build_runtime(settings) as runtime:
        result = runtime.relay(limit=limit)
    typer.echo(
        f"relayed: sent={result.sent} failed={result.failed} dead_lettered={result.dead_lettered}"
    )
    if result.failed:
        raise typer.Exit(1)


def outbox() -> None:
    """Show captured and delivery-failed Tanseki writes in the outbox."""
    settings = _settings()
    with _open_outbox(settings) as queue:
        pending = queue.pending(limit=20)
        total = queue.pending_count()
        counts = queue.status_counts()
        usage = queue.quota_usage()
    typer.echo(f"pending: {total} ({queue.path})")
    typer.echo(f"captured-locally: {counts['pending']}")
    typer.echo(
        f"delivery-failed: {counts['retrying'] + counts['dead_letter']} "
        f"retrying: {counts['retrying']} dead_letter: {counts['dead_letter']}"
    )
    _echo_outbox_quota(usage)
    for item in pending:
        typer.echo(
            f"  {item.entry_id} state={item.state} attempts={item.attempts} "
            f"age={_age(item.created_at)} "
            f"error={item.last_error or '-'}"
        )


def outbox_dead_letters(
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum dead letters to show"),
) -> None:
    """List dead-lettered Tanseki writes."""
    settings = _settings()
    with _open_outbox(settings) as queue:
        items = queue.dead_letters(limit=limit)
        total = queue.status_counts()["dead_letter"]
    typer.echo(f"dead-letters: {total} ({queue.path})")
    for item in items:
        typer.echo(
            f"  {item.entry_id} state={item.state} attempts={item.attempts} "
            f"failed_at={item.dead_lettered_at or '-'} error={item.last_error or '-'}"
        )


def outbox_requeue(entry_id: str = typer.Argument(..., help="Dead-letter entry ID")) -> None:
    """Move one dead-lettered write back to the active outbox."""
    settings = _settings()
    with _open_outbox(settings) as queue:
        requeued = queue.requeue_dead_letter(entry_id)
    if not requeued:
        typer.echo(f"Dead-letter entry not found: {entry_id}", err=True)
        raise typer.Exit(1)
    typer.echo(f"requeued: {entry_id}")


def outbox_retry(entry_id: str = typer.Argument(..., help="Dead-letter entry ID")) -> None:
    """Requeue and immediately deliver one dead-lettered write."""
    settings = _settings()
    _require_tanseki(settings)
    try:
        with build_runtime(settings) as runtime:
            result = runtime.retry(entry_id)
    except OutboxOwnershipError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    if result.sent == 0 and result.failed == 0:
        typer.echo(f"Dead-letter entry not found: {entry_id}", err=True)
        raise typer.Exit(1)
    typer.echo(
        f"retried: {entry_id} sent={result.sent} failed={result.failed} "
        f"dead_lettered={result.dead_lettered}"
    )
    if result.failed:
        raise typer.Exit(1)


def outbox_cleanup(
    older_than_days: int = typer.Option(
        30, "--older-than-days", help="Delete dead letters older than this many days"
    ),
) -> None:
    """Delete expired dead letters from the local outbox."""
    if older_than_days < 0:
        typer.echo("--older-than-days must be non-negative.", err=True)
        raise typer.Exit(1)
    settings = _settings()
    with _open_outbox(settings) as queue:
        removed = queue.cleanup_dead_letters(retention_days=older_than_days)
    typer.echo(f"removed: {removed}")


def register(app: typer.Typer) -> None:
    """Attach these commands to a Typer app."""
    app.command()(relay)
    app.command()(outbox)
    app.command("outbox-dead-letters")(outbox_dead_letters)
    app.command("outbox-requeue")(outbox_requeue)
    app.command("outbox-retry")(outbox_retry)
    app.command("outbox-cleanup")(outbox_cleanup)
