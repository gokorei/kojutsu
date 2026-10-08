"""FastAPI lifecycle management for GitHub webhooks."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from kojutsu import relay_worker as relay_worker_module
from kojutsu import runtime as runtime_module
from kojutsu.config import Settings, get_settings, webhook_registration_repos
from kojutsu.core.outbox import OutboxOwnershipError
from kojutsu.integrations.webhook_client import GitHubWebhookManager, validate_webhook_url
from kojutsu.webhook import server as server_module
from kojutsu.webhook.urls import default_webhook_url, is_local_webhook_url

logger = logging.getLogger(__name__)


class WebhookLifecycle:
    """Manages webhook registration during FastAPI startup/shutdown."""

    def __init__(self, app: FastAPI, settings: Settings | None = None) -> None:
        self.app = app
        self.settings = settings or get_settings()
        self.manager: GitHubWebhookManager | None = None
        self.webhook_url: str
        self.registered_repos: list[str] = []

    def configure(self, webhook_url: str | None = None) -> None:
        """Configure webhook lifecycle with optional custom URL."""
        self.webhook_url = validate_webhook_url(webhook_url or self._get_default_webhook_url())
        logger.info("Webhook lifecycle configured with URL: %s", self.webhook_url)

    def _get_default_webhook_url(self) -> str:
        """Get default webhook URL based on environment."""
        return default_webhook_url()

    async def startup(self) -> None:
        """Startup event handler: register webhooks if configured."""
        runtime_module.validate_sqlite_topology(self.settings)
        if not is_local_webhook_url(self.webhook_url) and not self.settings.github_webhook_secret:
            raise ValueError("GITHUB_WEBHOOK_SECRET is required for a public webhook URL")
        if not self.settings.github_token:
            logger.info("No GITHUB_TOKEN set - skipping webhook registration")
            return

        if not self.settings.github_webhook_secret:
            logger.warning("No GITHUB_WEBHOOK_SECRET set - local webhooks cannot authenticate")
            return

        self.manager = GitHubWebhookManager(
            self.settings.github_token,
            self.settings.github_webhook_secret,
        )

        # Check if webhook registration is enabled
        if not getattr(self.settings, "github_webhook_register", False):
            logger.info("GitHub webhook registration disabled in config")
            return

        # Get repositories to register from config: the allowlist of record,
        # with the legacy ``github_webhook_repos`` as fallback (see
        # ``kojutsu.config.webhook_registration_repos`` for the role mapping).
        repos = webhook_registration_repos(self.settings)
        if not repos:
            logger.info("No repositories configured for webhook registration")
            return

        logger.info(f"Registering webhooks for {len(repos)} repositories...")
        results = self.manager.register_all_repos(repos, self.webhook_url)

        success_count = sum(1 for r in results.values() if r is not None)
        self.registered_repos = self.manager.get_created_repos()

        logger.info(f"Successfully registered {success_count}/{len(repos)} webhooks")
        if success_count < len(repos):
            logger.warning(f"Failed to register {len(repos) - success_count} webhooks")

    async def shutdown(self) -> None:
        """Shutdown event handler: cleanup registered webhooks."""
        if self.manager is None:
            return

        if self.settings.github_webhook_cleanup:
            logger.info(f"Cleaning up {len(self.registered_repos)} registered webhooks...")
            results = self.manager.unregister_all_repos(
                self.registered_repos, webhook_url=self.webhook_url
            )
            success_count = sum(results.values())
            logger.info(
                f"Successfully cleaned up {success_count}/{len(self.registered_repos)} webhooks"
            )
        else:
            logger.info("GitHub webhook cleanup disabled - leaving webhooks registered")


def create_webhook_app(
    webhook_url: str | None = None,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Create FastAPI app with webhook lifecycle management."""
    main_app = FastAPI(title="Kojutsu Webhook Server")

    # Configure CORS if needed
    if cors_origins:
        main_app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Include webhook routes
    main_app.include_router(server_module.router)

    # Setup lifecycle management
    lifecycle = WebhookLifecycle(main_app)
    lifecycle.configure(webhook_url)

    # Register lifecycle events
    main_app.router.on_startup.append(lifecycle.startup)
    main_app.router.on_shutdown.append(lifecycle.shutdown)

    # Store lifecycle in app state for access if needed
    main_app.state.webhook_lifecycle = lifecycle

    async def _relay_startup() -> None:
        """Drain once, then keep a background relay running."""
        import asyncio

        interval = relay_worker_module.resolve_relay_interval()
        try:
            await relay_worker_module.relay_once()
        except OutboxOwnershipError:
            raise
        except Exception as exc:
            logger.info("Outbox relay (initial) skipped: %s", exc)
        main_app.state.relay_task = asyncio.create_task(relay_worker_module.relay_loop(interval))

    async def _relay_shutdown() -> None:
        """Stop the relay and release the runtime's connections."""
        import asyncio
        import contextlib

        task = getattr(main_app.state, "relay_task", None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        relay_drained = await relay_worker_module.wait_for_relay_shutdown()
        webhook_drained = await server_module.wait_for_webhook_processing()
        if not relay_drained or not webhook_drained:
            logger.warning(
                "Runtime shutdown deferred: relay_drained=%s webhook_drained=%s",
                relay_drained,
                webhook_drained,
            )
            return
        try:
            runtime_module.reset_runtime()
        except Exception as exc:
            logger.info("Runtime shutdown skipped: %s", exc)

    main_app.router.on_startup.append(_relay_startup)
    main_app.router.on_shutdown.append(_relay_shutdown)

    return main_app
