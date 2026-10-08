"""
GitHub webhook client for registration, cleanup, and lifecycle management.

Handles webhook registration/unregistration with GitHub API and integrates
with FastAPI lifecycle for automatic management.
"""

import logging
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, Field

from kojutsu.net import is_loopback_host

logger = logging.getLogger(__name__)


class WebhookURLError(ValueError):
    pass


class WebhookLookupError(RuntimeError):
    """Listing webhooks failed, so absence cannot be established.

    ``None`` from :meth:`find_webhook_by_url` means verified-absent (the list
    succeeded and no hook matched). Anything else -- 429/5xx/auth/network --
    raises this instead, so callers never read a transient failure as absent
    and create a duplicate webhook.
    """


def validate_webhook_url(webhook_url: str) -> str:
    if not isinstance(webhook_url, str) or not webhook_url.strip():
        raise WebhookURLError("Webhook URL must not be empty")
    normalized_url = webhook_url.strip()
    parsed = urlsplit(normalized_url)
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or "?" in normalized_url
        or "#" in normalized_url
    ):
        raise WebhookURLError(
            "Webhook URL must be an HTTP(S) URL without credentials, query, or fragment"
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise WebhookURLError("Webhook URL has an invalid port") from exc
    if port is not None and not 0 < port <= 65535:
        raise WebhookURLError("Webhook URL has an invalid port")
    if parsed.scheme.casefold() == "http" and not is_loopback_host(parsed.hostname or ""):
        raise WebhookURLError("Webhook URL must use HTTPS unless it targets loopback")
    return normalized_url


def redact_webhook_url(webhook_url: str) -> str:
    try:
        parsed = urlsplit(webhook_url)
        if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
            return "[redacted webhook URL]"
        hostname = parsed.hostname
        if hostname is None:
            return "[redacted webhook URL]"
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        port = parsed.port
        netloc = f"{hostname}:{port}" if port is not None else hostname
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    except ValueError:
        return "[redacted webhook URL]"


class WebhookConfig(BaseModel):
    """Webhook configuration for GitHub."""

    owner: str
    repo: str
    url: str = Field(default_factory=lambda: "http://localhost:8000/webhook/github")
    content_type: str = "json"
    secret: str | None = None
    #: Subscribed by default to the events that produce capture. ``pull_request_review``
    #: covers both a submitted review and the inline comments delivered with it, so
    #: subscribing to it once gives both kinds of review evidence. Omitting it is the
    #: reason review verdicts and inline comments are silently not captured: the
    #: webhook simply never receives them.
    events: list[str] = [
        "issue_comment",
        "pull_request",
        "pull_request_review",
        "pull_request_review_comment",
    ]


class GitHubWebhookClient:
    """Client for managing GitHub webhooks via REST API."""

    def __init__(self, token: str, base_url: str = "https://api.github.com") -> None:
        self._token = token
        self._base_url = base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        self._created_hooks: set[tuple[str, str, str]] = set()

    def _url(self, path: str) -> str:
        """Build full API URL."""
        path = path.lstrip("/")
        return f"{self._base_url}/{path}"

    def _make_request(
        self,
        method: str,
        path: str,
        json_data: dict[str, Any] | None = None,
        raise_for_status: bool = True,
    ) -> httpx.Response:
        """Make authenticated API request."""
        url = self._url(path)
        with httpx.Client(timeout=30.0) as client:
            response = client.request(
                method,
                url,
                headers=self._headers,
                json=json_data,
            )
            if raise_for_status:
                response.raise_for_status()
            return response

    def list_webhooks(self, owner: str, repo: str) -> list[dict[str, Any]]:
        """List all webhooks for a repository."""
        path = f"repos/{owner}/{repo}/hooks"
        response = self._make_request("GET", path)
        return response.json()

    def get_webhook(self, owner: str, repo: str, webhook_id: str) -> dict[str, Any]:
        """Get details for a specific webhook."""
        path = f"repos/{owner}/{repo}/hooks/{webhook_id}"
        response = self._make_request("GET", path)
        return response.json()

    def create_webhook(
        self,
        owner: str,
        repo: str,
        config: WebhookConfig,
    ) -> dict[str, Any]:
        """Create a new webhook for a repository."""
        webhook_url = validate_webhook_url(config.url)
        webhook_data = {
            "name": "web",
            "active": True,
            "config": {
                "url": webhook_url,
                "content_type": config.content_type,
                "secret": config.secret,
                "insecure_ssl": "0",
            },
            "events": config.events,
        }

        path = f"repos/{owner}/{repo}/hooks"
        response = self._make_request("POST", path, json_data=webhook_data)
        return response.json()

    def update_webhook(
        self,
        owner: str,
        repo: str,
        webhook_id: str,
        config: WebhookConfig,
    ) -> dict[str, Any]:
        """Update an existing webhook to the desired configuration."""
        webhook_url = validate_webhook_url(config.url)
        webhook_data = {
            "active": True,
            "config": {
                "url": webhook_url,
                "content_type": config.content_type,
                "secret": config.secret,
                "insecure_ssl": "0",
            },
            "events": config.events,
        }
        path = f"repos/{owner}/{repo}/hooks/{webhook_id}"
        response = self._make_request("PATCH", path, json_data=webhook_data)
        return response.json()

    def delete_webhook(self, owner: str, repo: str, webhook_id: str) -> None:
        """Delete a webhook by ID."""
        path = f"repos/{owner}/{repo}/hooks/{webhook_id}"
        self._make_request("DELETE", path)

    def find_webhook_by_url(
        self,
        owner: str,
        repo: str,
        target_url: str,
    ) -> dict[str, Any] | None:
        """Find webhook by target URL; None means verified-absent."""
        target_url = validate_webhook_url(target_url)
        try:
            webhooks = self.list_webhooks(owner, repo)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 404:
                return None
            logger.warning("Failed to list webhooks for %s/%s: HTTP %s", owner, repo, status)
            raise WebhookLookupError(f"could not list webhooks for {owner}/{repo}") from exc
        except httpx.HTTPError as exc:
            logger.warning("Failed to list webhooks for %s/%s: %s", owner, repo, exc)
            raise WebhookLookupError(f"could not list webhooks for {owner}/{repo}") from exc
        for hook in webhooks:
            hook_config = hook.get("config", {})
            if hook_config.get("url") == target_url:
                return hook
        return None

    def is_created_hook(self, owner: str, repo: str, hook: dict[str, Any]) -> bool:
        hook_id = hook.get("id")
        return hook_id is not None and (owner, repo, str(hook_id)) in self._created_hooks

    def register_webhook(
        self,
        owner: str,
        repo: str,
        webhook_url: str,
        secret: str | None = None,
    ) -> dict[str, Any]:
        """Register webhook, reusing existing if found."""
        webhook_url = validate_webhook_url(webhook_url)
        config = WebhookConfig(
            owner=owner,
            repo=repo,
            url=webhook_url,
            secret=secret,
        )
        existing = self.find_webhook_by_url(owner, repo, webhook_url)
        if existing:
            existing_config = existing.get("config") or {}
            existing_events = set(existing.get("events") or [])
            desired_events = set(config.events)
            needs_update = (
                not existing.get("active", False)
                or existing_config.get("content_type") != config.content_type
                or existing_config.get("secret") != config.secret
                or existing_config.get("insecure_ssl") not in (None, "0", 0)
                or existing_events != desired_events
            )
            if needs_update:
                logger.info(f"Updating existing webhook ID {existing['id']}")
                return self.update_webhook(owner, repo, str(existing["id"]), config)
            logger.info(f"Reusing existing webhook ID {existing['id']}")
            return existing

        created = self.create_webhook(owner, repo, config)
        if created.get("id") is not None:
            self._created_hooks.add((owner, repo, str(created["id"])))
        return created

    def unregister_webhook(self, owner: str, repo: str, webhook_url: str) -> bool:
        """Unregister webhook by URL."""
        existing = self.find_webhook_by_url(owner, repo, webhook_url)
        if existing:
            webhook_id = existing["id"]
            self.delete_webhook(owner, repo, webhook_id)
            logger.info(f"Deleted webhook ID {webhook_id}")
            return True
        return False


class GitHubWebhookManager:
    """High-level manager for webhook registration with error handling."""

    def __init__(self, token: str, webhook_secret: str | None = None) -> None:
        self.client = GitHubWebhookClient(token)
        self.webhook_secret = webhook_secret
        self.registered_hooks: dict[str, dict[str, Any]] = {}
        self.created_hooks: dict[str, dict[str, Any]] = {}

    def register_for_repo(
        self,
        owner: str,
        repo: str,
        webhook_url: str,
    ) -> dict[str, Any] | None:
        """Register webhook for a specific repository."""
        webhook_url = validate_webhook_url(webhook_url)
        try:
            hook = self.client.register_webhook(
                owner,
                repo,
                webhook_url,
                self.webhook_secret,
            )
            key = f"{owner}/{repo}"
            self.registered_hooks[key] = hook
            if self.client.is_created_hook(owner, repo, hook):
                self.created_hooks[key] = hook
            logger.info(f"Registered webhook for {owner}/{repo}: {hook['id']}")
        except WebhookLookupError:
            # Lookup failure must not read as absent (which would duplicate on retry).
            logger.error(
                "Failed to register webhook for %s/%s: webhook lookup failed",
                owner,
                repo,
                exc_info=True,
            )
            raise
        except Exception as e:
            logger.error(f"Failed to register webhook for {owner}/{repo}: {e}", exc_info=True)
            return None
        else:
            return hook

    def unregister_for_repo(
        self,
        owner: str,
        repo: str,
        webhook_url: str = "http://localhost:8000/webhook/github",
    ) -> bool:
        """Unregister webhook for a specific repository."""
        try:
            key = f"{owner}/{repo}"
            if key in self.registered_hooks:
                hook = self.registered_hooks[key]
                webhook_id = hook["id"]
                self.client.delete_webhook(owner, repo, webhook_id)
                del self.registered_hooks[key]
                self.created_hooks.pop(key, None)
                logger.info(f"Unregistered webhook for {owner}/{repo}")
                return True
            else:
                return self.client.unregister_webhook(owner, repo, webhook_url)
        except WebhookLookupError:
            logger.error(
                "Failed to unregister webhook for %s/%s: webhook lookup failed",
                owner,
                repo,
                exc_info=True,
            )
            raise
        except Exception as e:
            logger.error(f"Failed to unregister webhook for {owner}/{repo}: {e}", exc_info=True)
            return False

    def register_all_repos(
        self,
        repos: list[str],
        webhook_url: str,
    ) -> dict[str, dict[str, Any] | None]:
        """Register webhooks for multiple repositories."""
        results = {}
        for repo in repos:
            owner, repo_name = repo.split("/", 1)
            results[repo] = self.register_for_repo(owner, repo_name, webhook_url)
        return results

    def unregister_all_repos(
        self,
        repos: list[str],
        webhook_url: str = "http://localhost:8000/webhook/github",
    ) -> dict[str, bool]:
        """Unregister webhooks for multiple repositories."""
        results = {}
        for repo in repos:
            owner, repo_name = repo.split("/", 1)
            results[repo] = self.unregister_for_repo(owner, repo_name, webhook_url)
        return results

    def get_registered_repos(self) -> list[str]:
        """Get list of registered repositories."""
        return list(self.registered_hooks.keys())

    def get_created_repos(self) -> list[str]:
        return list(self.created_hooks.keys())

    def cleanup_all(self) -> int:
        """Cleanup all registered webhooks."""
        deleted_count = 0
        for repo in list(self.created_hooks):
            owner, repo_name = repo.split("/", 1)
            if self.unregister_for_repo(owner, repo_name):
                deleted_count += 1
        return deleted_count
