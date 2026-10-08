"""Shared helpers for the kojutsu CLI command groups."""

from datetime import UTC, datetime

import typer

from kojutsu.config import Settings, get_settings
from kojutsu.core.outbox import OutboxOwnershipError, OutboxQuotaUsage, TansekiOutbox


def _settings() -> Settings:
    return get_settings()


def _require_github_token(settings: Settings) -> None:
    """Refuse a forge-reading command without a token, naming the variable."""
    if not settings.github_token:
        typer.echo("Set GITHUB_TOKEN.", err=True)
        raise typer.Exit(1)


def _require_tanseki(settings: Settings) -> None:
    """Refuse a store-reading command without a configured store."""
    if not settings.tanseki_enabled:
        typer.echo("Set TANSEKI_URL.", err=True)
        raise typer.Exit(1)


def _open_outbox(settings: Settings) -> TansekiOutbox:
    try:
        return TansekiOutbox(settings.tanseki_outbox_path)
    except OutboxOwnershipError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None


def _age(stamp: str, *, now: datetime | None = None) -> str:
    """How long ago an ISO timestamp was, compact enough to sit in a table.

    ``-`` when the stamp is absent or unreadable, because a legacy row with an
    empty ``created_at`` is a fact about the row and blanking the column would
    hide it. A stamp in the future reads ``0s`` rather than a negative age:
    clock skew is not an age, and printing one invites the reader to treat a
    broken clock as a real duration.
    """
    if not stamp:
        return "-"
    try:
        moment = datetime.fromisoformat(stamp)
    except ValueError:
        return "-"
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    seconds = int(((now or datetime.now(UTC)) - moment).total_seconds())
    if seconds <= 0:
        return "0s"
    for size, unit in ((86400, "d"), (3600, "h"), (60, "m")):
        if seconds >= size:
            return f"{seconds // size}{unit}"
    return f"{seconds}s"


def _echo_outbox_quota(usage: OutboxQuotaUsage) -> None:
    """Report the growth bound, and say plainly when the next entry is refused.

    The usage line prints whether or not the quota is full, so an operator can
    see a spool approaching its ceiling before it refuses anything. A quota that
    only speaks at 100% gives no warning at all, which is the same failure the
    bound exists to prevent one step earlier.

    It does not exit non-zero. ``status`` already exits 1 when Tanseki is
    unreachable, and a second non-zero path would make the exit code ambiguous
    about which thing went wrong; the refusal itself is reported as its own
    outcome where it happens, in ``OutboxQuotaError``.
    """
    typer.echo(
        f"outbox_quota: {usage.entries}/{usage.max_entries} entries "
        f"{usage.bytes}/{usage.max_bytes} bytes"
        + (" FULL: the next entry will be refused" if usage.full else "")
    )
