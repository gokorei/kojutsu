"""CLI commands for GitHub webhook management.

Commands are plain functions; :func:`register` attaches them to a Typer app.
This avoids importing ``kojutsu.cli`` here (which would create a circular
import and break ``python -m kojutsu.cli``).
"""

import httpx
import typer

from kojutsu.cli_shared import _require_github_token
from kojutsu.config import get_settings
from kojutsu.integrations.webhook_client import (
    GitHubWebhookManager,
    redact_webhook_url,
    validate_webhook_url,
)


def webhook_register(
    repos: list[str] = typer.Argument(..., help="List of repositories as owner/repo"),
    webhook_url: str = typer.Option(
        "http://localhost:8000/webhook/github",
        help="Webhook endpoint URL",
    ),
    dry_run: bool = typer.Option(False, help="Show what would be registered without executing"),
) -> None:
    """Register webhooks for GitHub repositories."""
    settings = get_settings()
    _require_github_token(settings)
    try:
        webhook_url = validate_webhook_url(webhook_url)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    if not dry_run and not settings.github_webhook_secret:
        typer.echo("Set GITHUB_WEBHOOK_SECRET before registering webhooks.", err=True)
        raise typer.Exit(1)

    manager = GitHubWebhookManager(settings.github_token, settings.github_webhook_secret)

    if dry_run:
        typer.echo("DRY RUN: Would register webhooks for:")
        for repo in repos:
            typer.echo(f"  - {repo}")
        typer.echo(f"Target URL: {webhook_url}")
        if settings.github_webhook_secret:
            typer.echo(f"Secret: {'set' if settings.github_webhook_secret else 'not set'}")
        typer.echo("\nNo changes made.")
        return

    typer.echo(f"Registering webhooks for {len(repos)} repository(s)...")
    results = manager.register_all_repos(repos, webhook_url)

    success_count = 0
    for repo, result in results.items():
        if result:
            success_count += 1
            typer.echo(f"✓ {repo}: Webhook registered (ID: {result['id']})")
        else:
            typer.echo(f"✗ {repo}: Registration failed")

    typer.echo(f"\nSuccessfully registered {success_count}/{len(repos)} webhooks.")


def webhook_unregister(
    repos: list[str] = typer.Argument(..., help="List of repositories as owner/repo"),
    webhook_url: str = typer.Option(
        "http://localhost:8000/webhook/github",
        help="Webhook endpoint URL",
    ),
    dry_run: bool = typer.Option(False, help="Show what would be unregistered without executing"),
) -> None:
    """Unregister webhooks for GitHub repositories."""
    settings = get_settings()
    _require_github_token(settings)
    try:
        webhook_url = validate_webhook_url(webhook_url)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None

    manager = GitHubWebhookManager(settings.github_token, settings.github_webhook_secret)

    if dry_run:
        typer.echo("DRY RUN: Would unregister webhooks for:")
        for repo in repos:
            typer.echo(f"  - {repo}")
        typer.echo(f"Target URL: {webhook_url}")
        typer.echo("\nNo changes made.")
        return

    typer.echo(f"Unregistering webhooks for {len(repos)} repository(s)...")
    results = manager.unregister_all_repos(repos, webhook_url=webhook_url)

    success_count = 0
    for repo, success in results.items():
        if success:
            success_count += 1
            typer.echo(f"✓ {repo}: Webhook unregistered")
        else:
            typer.echo(f"✗ {repo}: Unregistration failed or not found")

    typer.echo(f"\nSuccessfully unregistered {success_count}/{len(repos)} webhooks.")


def webhook_status(
    repos: list[str] = typer.Argument(None, help="List of repositories as owner/repo"),
    webhook_url: str = typer.Option(
        "http://localhost:8000/webhook/github",
        help="Webhook endpoint URL",
    ),
) -> None:
    """Check webhook registration status for repositories."""
    settings = get_settings()
    _require_github_token(settings)
    try:
        webhook_url = validate_webhook_url(webhook_url)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None

    manager = GitHubWebhookManager(settings.github_token, settings.github_webhook_secret)

    if repos:
        all_repos = repos
    else:
        # List all webhooks for the authenticated user
        try:
            with httpx.Client(timeout=30.0) as client:
                r = client.get(
                    "https://api.github.com/user",
                    headers={"Authorization": f"Bearer {settings.github_token}"},
                )
                r.raise_for_status()
                user = r.json()
                username = user.get("login", "")
                all_repos = []
                # This is a simplified approach - in reality you'd need to list orgs and repos
                typer.echo(f"Listing webhooks for user: {username}")
                # For now, just show example
                typer.echo("Repository status check requires specific repo list.")
                typer.echo("Use --help to see available options.")
                return
        except Exception as e:
            typer.echo(f"Error checking status: {e}", err=True)
            raise typer.Exit(1) from e

    typer.echo(f"Checking webhook status for {len(all_repos)} repository(s)...")
    success_count = 0
    total_count = 0

    for repo in all_repos:
        owner, repo_name = repo.split("/", 1)
        try:
            webhooks = manager.client.list_webhooks(owner, repo_name)
            matching_hooks = [
                hook for hook in webhooks if hook.get("config", {}).get("url") == webhook_url
            ]
            total_count += 1
            if matching_hooks:
                success_count += 1
                typer.echo(f"✓ {repo}: {len(matching_hooks)} webhook(s) found")
                for hook in matching_hooks:
                    hook_id = hook.get("id", "unknown")
                    hook_url = hook.get("config", {}).get("url", "")
                    typer.echo(f"  - ID: {hook_id}, URL: {redact_webhook_url(hook_url)}")
            else:
                typer.echo(f"✗ {repo}: No webhooks found")
        except Exception as e:
            typer.echo(f"✗ {repo}: Error checking status ({e})")

    typer.echo(f"\nStatus: {success_count}/{total_count} repositories have webhooks configured.")


def register(app: typer.Typer) -> None:
    """Attach the webhook-management commands to a Typer app."""
    app.command()(webhook_register)
    app.command()(webhook_unregister)
    app.command()(webhook_status)
