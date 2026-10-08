"""Webhook URL helpers: defaults and loopback checks."""

import os
from urllib.parse import urlparse

from kojutsu.net import is_loopback_host


def default_webhook_url() -> str:
    """Get default webhook URL based on environment."""
    host = os.getenv("WEBHOOK_HOST", "localhost")
    port = os.getenv("PORT", "8000")
    return f"http://{host}:{port}/webhook/github"


def is_local_webhook_url(url: str) -> bool:
    """True when a webhook URL targets this machine."""
    host = urlparse(url).hostname
    if host is None:
        return False
    return is_loopback_host(host)
