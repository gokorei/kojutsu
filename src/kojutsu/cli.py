"""Kojutsu CLI: ask, collect, search."""

from typing import Annotated

import typer

from kojutsu import (
    cli_ask,
    cli_backfill,
    cli_design,
    cli_outbox,
    cli_search,
    cli_webhook,
    cli_worker,
)
from kojutsu.cli_shared import _echo_outbox_quota, _open_outbox, _settings
from kojutsu.config import set_selected_instance
from kojutsu.integrations.tanseki import TansekiClient, TansekiError
from kojutsu.logging_config import configure_logging
from kojutsu.net import is_loopback_host
from kojutsu.webhook.urls import is_local_webhook_url

app = typer.Typer(help="Capture developer knowledge during code review.")
cli_webhook.register(app)
cli_worker.register(app)
cli_outbox.register(app)
cli_ask.register(app)
cli_search.register(app)
cli_backfill.register(app)
cli_design.register(app)


@app.callback()
def _select_instance(
    instance: Annotated[
        str | None,
        typer.Option(
            "--instance",
            "-i",
            help=(
                "Named [instances.<name>] table from kojutsu.toml. One flag "
                "covers every subcommand in the invocation, so a server and a "
                "backfill against the same instance cannot drift apart."
            ),
        ),
    ] = None,
) -> None:
    """Select the configuration instance every command in this run reads.

    Recorded via :func:`kojutsu.config.set_selected_instance` rather than
    ``os.environ`` mutation, so the selection is per-invocation state the
    ``Settings`` file layer already reads -- not process-wide ambient state
    that leaks into other threads. ``Settings`` still honours
    ``KOJUTSU_INSTANCE`` as a fallback, so exported environments keep working.

    An unknown name is refused by ``Settings`` on first load rather than here, so
    the error names the instances that do exist. Validating in this callback as well
    would mean reading and parsing the file twice, and the second parse could
    disagree with the first.
    """
    if instance is None:
        return
    set_selected_instance(instance)


@app.command()
def status() -> None:
    """Show Tanseki, outbox, and registry status."""
    settings = _settings()
    with _open_outbox(settings) as queue:
        counts = queue.status_counts()
        usage = queue.quota_usage()
    pending = counts["pending"] + counts["retrying"]

    typer.echo(f"tanseki_url: {settings.tanseki_url or '(unset)'}")
    typer.echo(f"tanseki_collection: {settings.tanseki_collection}")
    if settings.tanseki_enabled:
        try:
            with TansekiClient.from_settings(settings) as client:
                reachable = client.health()
        except TansekiError:
            typer.echo("tanseki_reachable: unavailable")
            raise typer.Exit(1) from None
        typer.echo(f"tanseki_reachable: {reachable}")
    else:
        typer.echo("tanseki_reachable: n/a (TANSEKI_URL unset)")
    typer.echo(f"outbox_pending: {pending}")
    typer.echo(f"outbox_captured_locally: {counts['pending']}")
    typer.echo(f"outbox_retrying: {counts['retrying']}")
    typer.echo(f"outbox_delivery_failed: {counts['retrying'] + counts['dead_letter']}")
    typer.echo(f"outbox_dead_letter: {counts['dead_letter']}")
    _echo_outbox_quota(usage)
    typer.echo(f"registry_path: {settings.kojutsu_registry_path}")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),  # noqa: S104 - operator-chosen bind address
    port: int = typer.Option(8000, "--port"),
    webhook_url: str = typer.Option(None, "--webhook-url", help="Custom webhook URL"),
    cors_origins: str = typer.Option(None, "--cors-origins", help="Comma-separated CORS origins"),
    reload_server: bool = typer.Option(
        False, "--reload/--no-reload", help="Enable development auto-reload"
    ),
) -> None:
    """Run the webhook server for GitHub events."""
    import uvicorn

    from kojutsu.webhook import create_webhook_app

    configure_logging()
    settings = _settings()
    public_bind = not is_loopback_host(host)
    if public_bind and not settings.github_webhook_secret:
        typer.echo(
            "Set GITHUB_WEBHOOK_SECRET before binding the webhook server publicly.", err=True
        )
        raise typer.Exit(1)
    if public_bind and settings.github_webhook_register:
        if not webhook_url:
            typer.echo(
                "Set --webhook-url to a public webhook URL for automatic registration.",
                err=True,
            )
            raise typer.Exit(1)
        if is_local_webhook_url(webhook_url):
            typer.echo(
                "A public webhook URL is required for automatic registration on a public bind.",
                err=True,
            )
            raise typer.Exit(1)

    cors_list = cors_origins.split(",") if cors_origins else None
    try:
        app = create_webhook_app(
            webhook_url=webhook_url,
            cors_origins=cors_list,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload_server,
    )


@app.command()
def console(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8090, "--port"),
    allow_insecure_bind: bool = typer.Option(
        False,
        "--allow-insecure-bind",
        help="Explicitly allow a non-loopback bind; still requires DEV_CONSOLE_TOKEN.",
    ),
) -> None:
    """Run the minimal read-only dev console (verify Tanseki capture/search)."""
    import uvicorn

    from kojutsu.dev_console import create_console_app

    configure_logging()
    settings = _settings()
    try:
        app = create_console_app(
            bind_host=host,
            allow_insecure_bind=allow_insecure_bind,
            access_token=settings.dev_console_token,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    app()
