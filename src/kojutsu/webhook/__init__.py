"""FastAPI webhook server for GitHub events."""

from kojutsu.webhook.lifecycle import create_webhook_app

#: The default application: the same factory ``asgi`` uses, so importing the
#: app and serving it cannot disagree about routes, middleware, or lifespan.
#: Built here rather than in ``server`` so ``server`` never imports
#: ``lifecycle`` while ``lifecycle`` needs the server's router -- one direction,
#: no cycle, no function-level import.
app = create_webhook_app()

webhook_app = app

__all__ = ["app", "create_webhook_app", "webhook_app"]
