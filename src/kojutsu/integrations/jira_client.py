"""Jira REST API client for ticket details."""

from __future__ import annotations

import ipaddress
import logging
import random
import time
from collections.abc import Callable
from typing import Any, TypedDict
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

DEFAULT_JIRA_TIMEOUT_SECONDS = 5.0
MAX_JIRA_TIMEOUT_SECONDS = 5.0

#: How many times one ``get_issue`` is attempted before the failure is raised.
#: Three attempts, not unbounded retries: Jira context enriches a capture but is
#: never worth stalling it, and a store that is down for three attempts is down,
#: not blipping.
JIRA_MAX_ATTEMPTS = 3

#: Ceiling for one wait between attempts. Small on purpose: the per-request
#: timeout above already bounds each attempt, so the backoff only spaces them.
JIRA_MAX_BACKOFF_SECONDS = 5.0


class JiraIntegrationError(RuntimeError):
    """Base error for Jira integration failures."""


class JiraConfigurationError(JiraIntegrationError):
    """The configured Jira endpoint is unsafe or invalid."""


class JiraAuthenticationError(JiraIntegrationError):
    """Jira rejected the configured credentials."""


class JiraUnavailableError(JiraIntegrationError):
    """Jira could not be reached."""


class JiraPayloadError(JiraIntegrationError):
    """Jira returned an invalid issue response."""


class JiraIssueFields(TypedDict):
    """The bounded issue fields LLM context is built from, and nothing else.

    All strings: the shape is fixed (unlike the raw issue payload, whose
    ``fields`` carry per-instance custom fields), so callers read attributes
    the checker can see instead of string keys it cannot.
    """

    summary: str
    description: str
    issue_type: str
    acceptance_criteria: str
    key: str


def _validate_jira_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    parsed = urlparse(normalized)
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise JiraConfigurationError("Jira URL must be a valid HTTP(S) URL without credentials.")
    try:
        port = parsed.port
    except ValueError as exc:
        raise JiraConfigurationError("Jira URL has an invalid port.") from exc
    if port is not None and not 0 < port <= 65535:
        raise JiraConfigurationError("Jira URL has an invalid port.")
    if parsed.scheme.casefold() == "http":
        host = parsed.hostname.removeprefix("[").removesuffix("]").casefold()
        loopback = host == "localhost"
        if not loopback:
            try:
                loopback = ipaddress.ip_address(host).is_loopback
            except ValueError:
                loopback = False
        if not loopback:
            raise JiraConfigurationError("Jira URL must use HTTPS unless it targets loopback.")
    return normalized


class JiraClient:
    """Client for Jira REST API: issue details for knowledge context.

    **One connection pool for the client's whole life, like ``GitHubClient``.**
    The pool is opened lazily on first use and held until :meth:`close`, so a
    retry is the same read the call already decided to make -- not a fresh TLS
    negotiation on top of the backoff sleep.
    """

    def __init__(
        self,
        base_url: str,
        username: str,
        api_token: str,
        *,
        timeout_seconds: float = DEFAULT_JIRA_TIMEOUT_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Jira timeout must be greater than zero.")
        if timeout_seconds > MAX_JIRA_TIMEOUT_SECONDS:
            logger.warning(
                "Jira timeout %.1fs exceeds the %.1fs maximum and is clamped; "
                "a Jira read that blocks longer would stall the capture path "
                "waiting on it.",
                timeout_seconds,
                MAX_JIRA_TIMEOUT_SECONDS,
            )
        self._base = _validate_jira_base_url(base_url)
        self._auth = (username, api_token)
        self._timeout_seconds = min(timeout_seconds, MAX_JIRA_TIMEOUT_SECONDS)
        self._headers = {"Accept": "application/json", "Content-Type": "application/json"}
        self._sleep = sleep
        self._client: httpx.Client | None = None
        import threading as _threading

        self._client_lock = _threading.Lock()

    @classmethod
    def from_settings(cls, settings: Any) -> JiraClient:
        """Build a client from application settings."""
        return cls(
            settings.jira_url,
            settings.jira_username,
            settings.jira_api_token,
        )

    def http(self) -> httpx.Client:
        """The one pool this client reads through, built on first use.

        Left open until :meth:`close`. A client held for a run hands its
        sockets back when the run ends; a single-call client can simply be
        dropped.
        """
        client = self._client
        if client is not None:
            return client
        with self._client_lock:
            if self._client is None:
                self._client = httpx.Client(timeout=self._timeout_seconds)
            return self._client

    def close(self) -> None:
        """Close the connection pool. Safe to call twice, and safe unopened."""
        client = self._client
        if client is not None:
            client.close()

    def __enter__(self) -> JiraClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _url(self, path: str) -> str:
        return f"{self._base}/rest/api/3/{path.lstrip('/')}"

    @staticmethod
    def _retry_after_seconds(response: httpx.Response) -> float | None:
        """The wait Jira asked for, or ``None`` when it did not say."""
        header = (response.headers.get("retry-after") or "").strip()
        if not header:
            return None
        try:
            return max(0.0, float(header))
        except ValueError:
            return None

    def _backoff(self, response: httpx.Response | None, attempt: int) -> float:
        """Wait before the next attempt: the forge's ask first, else doubling.

        Jittered so concurrent captures do not wake in lockstep, bounded so a
        large ``Retry-After`` cannot park a capture behind a Jira outage.
        """
        requested = self._retry_after_seconds(response) if response is not None else None
        if requested is not None:
            return min(requested, JIRA_MAX_BACKOFF_SECONDS)
        return min(
            float(2 ** max(attempt - 1, 0)) * random.uniform(0.5, 1.0),  # noqa: S311 - backoff jitter, not cryptographic
            JIRA_MAX_BACKOFF_SECONDS,
        )

    def get_issue(self, issue_key: str) -> dict[str, Any] | None:
        """Fetch an issue, retrying transient failures, or raise a safe domain error.

        Retried is a dropped connection, a 429, or a 5xx -- the failures that say
        nothing about the request. A 404 still returns ``None``, a 401/403 still
        raises authentication, and any other 4xx still raises immediately: those
        are answers, and retrying an answer is how a misconfigured credential
        becomes a slow misconfigured credential. When the bounded attempts are
        exhausted the last failure raises as the same error a single attempt
        would have raised, so callers see no new error type.

        The pool is opened once around the loop, not once per attempt: a retry
        is the same read the call already decided to make, and re-handshaking
        for it would spend the backoff sleeping and then pay a fresh TLS
        negotiation on top.
        """
        attempt = 0
        client = self.http()
        while True:
            attempt += 1
            try:
                response = client.get(
                    self._url(f"issue/{issue_key}"),
                    auth=self._auth,
                    headers=self._headers,
                )
            except httpx.TransportError as exc:
                if attempt >= JIRA_MAX_ATTEMPTS:
                    raise JiraUnavailableError(
                        "Jira is unavailable; retry or verify JIRA_URL."
                    ) from exc
                self._sleep(self._backoff(None, attempt))
                continue
            if response.status_code == 404:
                return None
            if response.status_code in {401, 403}:
                raise JiraAuthenticationError(
                    "Jira authentication failed; verify Jira credentials."
                )
            if response.status_code == 429 or response.status_code >= 500:
                if attempt >= JIRA_MAX_ATTEMPTS:
                    raise JiraUnavailableError("Jira is unavailable; retry later.")
                self._sleep(self._backoff(response, attempt))
                continue
            if response.status_code >= 400:
                raise JiraIntegrationError(
                    f"Jira rejected issue {issue_key} ({response.status_code})."
                )
            try:
                data = response.json()
            except ValueError as exc:
                raise JiraPayloadError("Jira returned invalid JSON.") from exc
            if not isinstance(data, dict) or not isinstance(data.get("fields"), dict):
                raise JiraPayloadError("Jira returned an invalid issue response.")
            return data

    def get_issue_fields(self, issue_key: str) -> JiraIssueFields | None:
        """Return bounded Jira fields suitable for LLM context, or None when absent."""
        data = self.get_issue(issue_key)
        if not data:
            return None
        fields = data["fields"]
        raw_summary = fields.get("summary")
        if raw_summary is None:
            summary = ""
        elif isinstance(raw_summary, str):
            summary = raw_summary
        else:
            # A non-string summary is a shaped object this reader does not know,
            # and passing it through as-is would put a dict where every caller
            # reads a string. Coerced rather than refused: the record keeps its
            # field, labelled as what it was.
            summary = str(raw_summary)
        description_obj = fields.get("description")
        if isinstance(description_obj, dict):
            if description_obj.get("type") == "doc" and "content" in description_obj:
                description = _extract_doc_text(description_obj)
            else:
                description = str(description_obj)
        else:
            description = str(description_obj) if description_obj else ""

        issue_type = ""
        issue_type_obj = fields.get("issuetype")
        if isinstance(issue_type_obj, dict):
            issue_type = issue_type_obj.get("name", "")

        acceptance_criteria = ""
        for name in ("acceptance criteria", "Acceptance Criteria", "customfield_10014"):
            value = fields.get(name)
            if value is not None:
                if isinstance(value, dict) and value.get("type") == "doc" and "content" in value:
                    acceptance_criteria = _extract_doc_text(value)
                else:
                    acceptance_criteria = str(value)
                break

        return {
            "summary": summary,
            "description": description,
            "issue_type": issue_type,
            "acceptance_criteria": acceptance_criteria,
            "key": issue_key,
        }


def _extract_doc_text(node: dict) -> str:
    """Recursively extract plain text from Atlassian document structure."""
    if node.get("type") == "text":
        return node.get("text", "")
    content = node.get("content") or []
    return " ".join(_extract_doc_text(c) for c in content if isinstance(c, dict))
