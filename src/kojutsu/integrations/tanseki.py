"""Client for the Tanseki knowledge store — the consumer seam.

Tanseki is a separate service; Kojutsu only reaches it over its versioned HTTP
API (``/v1``). This module is the *only* place Kojutsu talks to the store, so
the seam contract lives here. The request/response shapes mirror Tanseki's domain
model (``Document``/``Edge``/``Revision``) and its OpenAPI spec. If the seam
evolves, only this module changes.

Contract (canonical Tanseki ``/v1``):

- ``GET  /v1/health``                        -> 200
- ``GET  /v1/documents``  {limit, collection} -> {total, documents: [...]}; 400 above 500
- ``POST /v1/documents:get``   {id, collection} -> Document | 404
- ``POST /v1/documents:upsert`` {id, collection, content, frontmatter, ...} -> {revision, created}
- ``POST /v1/documents:delete`` {id, collection, message, author} -> {revision} | 404
- ``GET  /v1/search?q=&limit=&collection=&tags=`` -> {hits: [{id, score, snippet}], limit, offset, hasMore}
- ``POST /v1/documents:traverse`` {id, rel, depth, collection} -> {ids: [...]}

Ids are path-derived and contain ``/`` (e.g. ``org/repo/pr-1/abc``), so documents
are addressed in the body via custom methods, not a path segment.
"""

from __future__ import annotations

import hashlib
import ipaddress
import math
import random
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any, Protocol, TypedDict, runtime_checkable
from urllib.parse import urlparse

import httpx

from kojutsu.config import Settings
from kojutsu.identity import IDEMPOTENCY_IDENTITY_DOMAIN, identity_preimage

DEFAULT_COLLECTION = "kojutsu-real"
MAX_SEARCH_RESULTS = 50
MAX_SEARCH_FETCH_WORKERS = 4
# Tanseki rejects ``GET /v1/documents`` with a 400 above 500; keep callers inside it.
MAX_LIST_RESULTS = 500
_RETRY_TRANSPORT_STATUS = {408, 425, 429, 500, 502, 503, 504}
_RETRY_AUTH_STATUS = {401, 403}
_PERMANENT_PAYLOAD_STATUS = {400, 404, 405, 413, 415, 422}

#: Longest this client will wait between attempts, whichever form the wait took.
#:
#: It bounds the header Tanseki sends *and* the schedule this client computes, so a
#: single knob decides how long a rate-limited caller can be parked. It is not
#: the number of attempts; that is ``max_retries``, and the two are separate
#: because a client that retries quickly for a long time and one that retries
#: slowly for a short time fail differently.
MAX_RETRY_DELAY_SECONDS = 300.0

#: First computed delay, doubled per attempt, before jitter.
RETRY_BASE_DELAY_SECONDS = 0.2

#: Fraction of the computed window that jitter randomises.
#:
#: Half. Full jitter can return almost zero, which is how a client being
#: rate-limited ends up hammering instead of backing off; no jitter at all is how
#: every client that failed at the same instant retries at the same instant,
#: which is how one store's blip becomes every client's simultaneous retry. Half
#: keeps a real floor under the delay and still spreads the herd.
RETRY_JITTER_RATIO = 0.5

#: Hard ceiling on retries, whatever ``max_retries`` is configured to.
#:
#: Attempts bounded only by configuration are not bounded: a typo or a copied
#: config turns a transient 503 into a retry loop that outlives the caller and
#: keeps the store answering requests from a process that has already given up.
MAX_RETRIES_CEILING = 8


def _retry_after_seconds(response: httpx.Response, *, now: float) -> float | None:
    """The wait Tanseki asked for, in seconds, from either form of ``Retry-After``.

    RFC 9110 allows a delay in seconds or an HTTP-date. Only the seconds form was
    read before, so a store — or a proxy rewriting the header on the way in —
    that sent the date form was treated as having said nothing at all, and this
    client retried on its own schedule rather than the one it was given. A date
    in the past means "now", which is zero rather than a reason to fall back to
    backoff: the window Tanseki named has already passed.

    ``now`` is the caller's clock rather than a fresh reading, so the date form
    is testable without a test that has to agree with the wall clock.
    """
    raw = (response.headers.get("Retry-After") or "").strip()
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        pass
    else:
        return min(MAX_RETRY_DELAY_SECONDS, max(0.0, seconds))
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    try:
        delta = when.timestamp() - now
    except (OverflowError, OSError, ValueError):
        return None
    return min(MAX_RETRY_DELAY_SECONDS, max(0.0, delta))


def _retry_delay_seconds(*, attempt: int, retry_after: float | None, jitter: float) -> float:
    """Seconds to wait before the next attempt, always in ``[0, ceiling]``.

    A ``Retry-After`` is used as given, and never jittered: a store that says
    "wait thirty seconds" means at least thirty seconds, and a client that
    shaved a few off it has not hardened its retry, it has ignored the only
    instruction the store gave. Jitter is for the schedule this client invented,
    where nothing else is coordinating the herd.

    ``jitter`` is a fraction in ``[0, 1)``. The result is clamped on both sides
    anyway, so a caller that passes something outside that range produces a valid
    delay rather than a negative sleep or a wait past the ceiling.
    """
    if retry_after is not None:
        return max(0.0, min(retry_after, MAX_RETRY_DELAY_SECONDS))
    window = min(MAX_RETRY_DELAY_SECONDS, RETRY_BASE_DELAY_SECONDS * (2**attempt))
    spread = window * RETRY_JITTER_RATIO
    return max(0.0, min(window, window * (1 - RETRY_JITTER_RATIO) + jitter * spread))


def _raise_operation_error(
    operation: str, doc_id: str, response: httpx.Response, *, now: float
) -> None:
    status = response.status_code
    if status in _RETRY_AUTH_STATUS:
        raise TansekiAuthenticationError(
            f"{operation} authentication failed ({status})",
            retry_after=_retry_after_seconds(response, now=now),
        )
    if status in _RETRY_TRANSPORT_STATUS:
        raise TansekiUnavailableError(
            f"{operation} failed ({status})",
            retry_after=_retry_after_seconds(response, now=now),
        )
    if status == 409:
        raise TansekiConflictError(f"conflict writing {doc_id}")
    if status in _PERMANENT_PAYLOAD_STATUS:
        raise TansekiPermanentError(f"{operation}({doc_id}) failed: {status}")
    raise TansekiResponseError(f"{operation} returned unexpected status {status}")


def _response_json(response: httpx.Response, operation: str) -> Any:
    try:
        return response.json()
    except (TypeError, ValueError) as exc:
        raise TansekiResponseError(f"{operation} returned invalid JSON") from exc


def bound_search_limit(limit: int) -> int:
    """Validate a positive result limit and cap it to the shared maximum."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("Search limit must be an integer of at least 1.")
    return min(limit, MAX_SEARCH_RESULTS)


def bound_list_limit(limit: int) -> int:
    """Validate a listing limit and hold it inside Tanseki's listing cap.

    A distinct bound from :func:`bound_search_limit`, and deliberately not the
    same number. The search cap is 50 because a search is a ranked answer; the
    listing cap is 500 because Tanseki answers 400 above it. Reusing the search
    number here would quietly enumerate a fifth of the collection and present it
    as the listing, which is indistinguishable from a collection that holds a
    fifth as many documents -- the defect ``docs/design-review/read-path.md`` is
    about, one layer down. Clamping rather than refusing keeps the two bounds
    distinguishable: a caller above the cap gets the cap, which is why a surface
    that must report a shortened answer still owes the reader a note.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("Listing limit must be an integer of at least 1.")
    return min(limit, MAX_LIST_RESULTS)


def idempotency_key(
    *,
    document_id: str | None,
    collection: str | None,
    content: str | None,
) -> str:
    """Derive the ``Idempotency-Key`` for one upsert, from named fields.

    A function of named fields rather than of the request body, for two reasons that
    are both about the encoding rather than about convenience.

    **A field can contain the separator.** This was a ``|``-joined string over two
    attacker-influenced fields, so ``{"id": "a|b", "content": "c"}`` and
    ``{"id": "a", "content": "b|c"}`` derived the same key. A collision here is
    worse than one on a record id: the store reads a repeated key as *this is a
    replay of something already sent*, so a genuine write is answered as a duplicate
    and silently does not happen. Nothing stops a document id or a body containing
    a pipe.

    **Absent is not empty.** ``body.get("id", "")`` could not tell a missing field
    from an empty one, so a body without an id and a body with ``"id": ""`` derived
    the same key. ``None`` is carried through as JSON ``null`` rather than being
    flattened, which keeps the two apart.

    ``collection`` is in the key because the document id is not collection-scoped —
    it is path-derived from repo and entry id alone — so the same id and content in
    two collections are the same bytes without it. Whether the store scopes its
    idempotency records per collection decides whether that was ever a live
    collision; including the field is correct either way, and excluding a field
    because a downstream might not need it is how the pipe collision happened.

    This is a per-request key and nothing is keyed on it, so changing it re-sends
    in-flight outbox rows rather than re-identifying stored records. That is a
    retry, which is what the outbox already does when a row comes back unacknowledged.
    """
    preimage = identity_preimage(
        IDEMPOTENCY_IDENTITY_DOMAIN,
        (document_id, collection, content),
    )
    return hashlib.sha256(preimage).hexdigest()


def _idempotency_key(body: Mapping[str, Any]) -> str:
    """Read the fields the key is derived from, by name, out of a request body.

    Separate from :func:`idempotency_key` so the derivation takes named fields and
    this is the only place that knows a request body exists. A missing key stays
    ``None`` on the way through — it is not defaulted to ``""``, which is the
    conflation the derivation exists to refuse.
    """
    return idempotency_key(
        document_id=body.get("id"),
        collection=body.get("collection"),
        content=body.get("content"),
    )


@runtime_checkable
class TansekiWriter(Protocol):
    """The subset of the Tanseki client the outbox relay depends on.

    Declared as a protocol so the relay can be tested with lightweight fakes.
    """

    def upsert_document(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class TansekiError(RuntimeError):
    """Base error for Tanseki interactions."""


class TansekiResponseError(TansekiError):
    """The response body did not match the Tanseki contract."""


class TansekiPermanentError(TansekiError):
    """The request is deterministically invalid and cannot be retried unchanged."""


class TansekiConfigurationError(TansekiPermanentError):
    """The configured Tanseki endpoint is unsafe or invalid."""


class TansekiAuthenticationError(TansekiError):
    """Tanseki rejected credentials; an operator must repair authentication before retrying."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after

    @property
    def operator_action_required(self) -> bool:
        return True


class TansekiUnavailableError(TansekiError):
    """The store could not be reached (network, timeout, or 5xx after retries)."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class TansekiNotFoundError(TansekiError):
    """The requested document does not exist."""


class TansekiConflictError(TansekiError):
    """A revision / compare-and-swap conflict requires a fresh delivery attempt."""


def _required_string(data: dict[str, Any], key: str, operation: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TansekiResponseError(f"{operation} returned an invalid {key}")
    return value


def _optional_string(data: dict[str, Any], key: str, operation: str) -> str | None:
    value = data.get(key)
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise TansekiResponseError(f"{operation} returned an invalid {key}")
    return value


def _total(data: dict[str, Any], operation: str) -> int:
    value = data.get("total")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TansekiResponseError(f"{operation} returned an invalid total")
    return value


def _optional_count(data: dict[str, Any], key: str, operation: str) -> int | None:
    """Validate an optional non-negative count field, rejecting a malformed one.

    Tanseki's paginated search envelope is ``{hits, limit, offset, hasMore}`` and
    carries no ``total``, so a missing field is legitimate here. A field that is
    present but not a non-negative integer is still a broken response.
    """
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TansekiResponseError(f"{operation} returned an invalid {key}")
    return value


def _optional_bool(data: dict[str, Any], key: str, operation: str) -> bool | None:
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if not isinstance(value, bool):
        raise TansekiResponseError(f"{operation} returned an invalid {key}")
    return value


@dataclass(frozen=True)
class TansekiUpsertResult(TypedDict):
    """What a document upsert reports: the revision stored and whether it is new.

    A fixed shape (unlike the mapper-produced payload going in, whose
    frontmatter keys vary per record kind), so callers read fields the
    checker can see.
    """

    revision: str
    created: bool


@dataclass(frozen=True)
class TansekiDocument:
    """A document as returned by Tanseki."""

    id: str
    path: str
    collection: str
    content: str
    content_hash: str | None = None
    revision: str | None = None
    updated_at: str | None = None
    frontmatter: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(
        cls,
        data: dict[str, Any],
        *,
        collection: str,
        expected_id: str | None = None,
    ) -> TansekiDocument:
        if not isinstance(data, dict):
            raise TansekiResponseError("get_document returned an invalid response")
        document_id = _required_string(data, "id", "get_document")
        path = _required_string(data, "path", "get_document")
        content = data.get("content")
        if not isinstance(content, str):
            raise TansekiResponseError("get_document returned an invalid content")
        resolved_collection = data.get("collection")
        if not isinstance(resolved_collection, str) or not resolved_collection.strip():
            raise TansekiResponseError("get_document returned an invalid collection")
        if expected_id is not None and document_id != expected_id:
            raise TansekiResponseError("get_document returned an unexpected document identity")
        if resolved_collection != collection:
            raise TansekiResponseError("get_document returned an unexpected collection")
        frontmatter = data.get("frontmatter", {})
        if not isinstance(frontmatter, dict):
            raise TansekiResponseError("get_document returned an invalid frontmatter")
        content_hash = data.get("contentHash", data.get("content_hash"))
        if content_hash is not None and (not isinstance(content_hash, str) or not content_hash):
            raise TansekiResponseError("get_document returned an invalid content hash")
        updated_at = data.get("updatedAt", data.get("updated_at"))
        if updated_at is not None and (not isinstance(updated_at, str) or not updated_at):
            raise TansekiResponseError("get_document returned an invalid updated at")
        return cls(
            id=document_id,
            path=path,
            collection=resolved_collection,
            content=content,
            content_hash=content_hash,
            revision=_optional_string(data, "revision", "get_document"),
            updated_at=updated_at,
            frontmatter=frontmatter,
        )


@dataclass(frozen=True)
class TansekiHit:
    """A ranked search hit from Tanseki."""

    id: str
    score: float
    snippet: str | None = None

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> TansekiHit:
        if not isinstance(data, dict):
            raise TansekiResponseError("search returned an invalid hit")
        hit_id = _required_string(data, "id", "search")
        score = data.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TansekiResponseError("search returned an invalid score")
        if not math.isfinite(float(score)):
            raise TansekiResponseError("search returned an invalid score")
        snippet = data.get("snippet")
        if snippet is not None and not isinstance(snippet, str):
            raise TansekiResponseError("search returned an invalid snippet")
        return cls(id=hit_id, score=float(score), snippet=snippet)


def _validate_tanseki_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    parsed = urlparse(normalized)
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise TansekiConfigurationError(
            "Tanseki URL must be a valid HTTP(S) URL without credentials."
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise TansekiConfigurationError("Tanseki URL has an invalid port.") from exc
    if port is not None and not 0 < port <= 65535:
        raise TansekiConfigurationError("Tanseki URL has an invalid port.")
    if parsed.scheme.casefold() == "http":
        host = parsed.hostname.removeprefix("[").removesuffix("]").casefold()
        loopback = host == "localhost"
        if not loopback:
            try:
                loopback = ipaddress.ip_address(host).is_loopback
            except ValueError:
                loopback = False
        if not loopback:
            raise TansekiConfigurationError(
                "Tanseki URL must use HTTPS unless it targets loopback."
            )
    return normalized


class TansekiClient:
    """Thin, retrying HTTP client over the Tanseki ``/v1`` API."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "",
        collection: str = DEFAULT_COLLECTION,
        timeout: float = 10.0,
        max_retries: int = 2,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        normalized_base_url = _validate_tanseki_base_url(base_url)
        self.collection = collection
        # Clamped rather than trusted, because this is the only bound on how long
        # a caller can be made to wait and the configuration is not code.
        self.max_retries = max(0, min(max_retries, MAX_RETRIES_CEILING))
        #: Injected so a test can assert on the retry schedule without spending
        #: it. The arithmetic is the behaviour under test; a test that actually
        #: sleeps out a ``Retry-After`` is a test nobody runs and everybody
        #: disables.
        self._sleep = sleep
        #: Wall clock, used only to turn an HTTP-date ``Retry-After`` into a
        #: delay. Separate from ``sleep`` because the two answer different
        #: questions: one is "how long do I wait", the other "what time is it".
        self._clock = clock
        #: Source of the jitter fraction. Injected for the same reason as
        #: ``sleep``: asserting that two callers do not retry in lockstep means
        #: choosing the values, not hoping the RNG cooperates.
        self._jitter = jitter
        headers = {"Accept": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key
        self._headers = headers
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=normalized_base_url + "/v1",
            headers=headers,
            timeout=timeout,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> TansekiClient:
        """Build a client from application settings."""
        return cls(
            settings.tanseki_url,
            api_key=settings.tanseki_api_key,
            collection=settings.tanseki_collection,
            timeout=settings.tanseki_timeout_seconds,
        )

    def __enter__(self) -> TansekiClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Issue a request with bounded retries on transient failures.

        Bounded twice over, and both bounds are load-bearing. ``max_retries``
        caps how many times the client asks at all, so a rate-limited store is
        not asked forever; :data:`MAX_RETRY_DELAY_SECONDS` caps how long it waits
        between asks, so the retries it does make cannot add up to an outage of
        its own. A client that retries a 429 without a ceiling is not being
        patient, it is the load.

        The wait before each retry is ``Retry-After`` when Tanseki sent one — in
        either RFC 9110 form, seconds or HTTP-date — and otherwise an
        exponentially growing window with jitter. ``retry_after`` is re-read per
        attempt rather than carried forward, because a delay belongs to the
        response that asked for it: applying one response's ``Retry-After`` to a
        later transport error waits out a window the store never opened.
        """
        headers = {**self._headers, **(kwargs.pop("headers", None) or {})}
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            retry_after: float | None = None
            try:
                response = self._client.request(method, url, headers=headers, **kwargs)
            except httpx.TransportError as exc:
                last_error = exc
            else:
                status = response.status_code
                if status in _RETRY_AUTH_STATUS:
                    retry_after = _retry_after_seconds(response, now=self._clock())
                    last_error = TansekiAuthenticationError(
                        f"{method} {url} -> {status}", retry_after=retry_after
                    )
                elif status in _RETRY_TRANSPORT_STATUS:
                    retry_after = _retry_after_seconds(response, now=self._clock())
                    last_error = TansekiUnavailableError(
                        f"{method} {url} -> {status}", retry_after=retry_after
                    )
                else:
                    return response
            if attempt < self.max_retries:
                self._sleep(
                    _retry_delay_seconds(
                        attempt=attempt, retry_after=retry_after, jitter=self._jitter()
                    )
                )
        if isinstance(last_error, (TansekiAuthenticationError, TansekiUnavailableError)):
            raise last_error
        raise TansekiUnavailableError(f"{method} {url} failed: {type(last_error).__name__}")

    # -- operations ---------------------------------------------------------

    def health(self) -> bool:
        """Return True if the store answers its health endpoint."""
        try:
            response = self._request("GET", "/health")
        except (TansekiAuthenticationError, TansekiUnavailableError):
            return False
        return 200 <= response.status_code < 300

    def count(self, collection: str | None = None) -> int:
        """Number of documents in the collection."""
        response = self._request(
            "GET",
            "/documents",
            params={"limit": 1, "collection": collection or self.collection},
        )
        if response.status_code >= 400:
            _raise_operation_error("count", self.collection, response, now=self._clock())
        data = _response_json(response, "count")
        if not isinstance(data, dict):
            raise TansekiResponseError("count returned an invalid response")
        return _total(data, "count")

    def get_document(self, doc_id: str, collection: str | None = None) -> TansekiDocument | None:
        """Fetch a document, or None when it does not exist."""
        response = self._request(
            "POST",
            "/documents:get",
            json={"id": doc_id, "collection": collection or self.collection},
        )
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            _raise_operation_error("get_document", doc_id, response, now=self._clock())
        data = _response_json(response, "get_document")
        if not isinstance(data, dict):
            raise TansekiResponseError("get_document returned an invalid response")
        return TansekiDocument.from_json(
            data,
            collection=collection or self.collection,
            expected_id=doc_id,
        )

    def get_documents(
        self,
        doc_ids: list[str],
        collection: str | None = None,
        *,
        max_workers: int = MAX_SEARCH_FETCH_WORKERS,
    ) -> list[TansekiDocument | None]:
        """Fetch unique documents with bounded concurrency while preserving rank order."""
        unique_ids = list(dict.fromkeys(doc_ids))
        if not unique_ids:
            return []
        workers = max(1, min(max_workers, MAX_SEARCH_FETCH_WORKERS))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(
                executor.map(
                    lambda doc_id: self.get_document(doc_id, collection=collection),
                    unique_ids,
                )
            )

    def list_documents(
        self,
        *,
        limit: int = MAX_LIST_RESULTS,
        collection: str | None = None,
    ) -> list[str]:
        """Return the collection's document ids from the bounded listing endpoint.

        ``GET /v1/documents`` is otherwise only used for counting, and the seam
        documents just ``{total}``, so the envelope is read defensively instead
        of assumed: the array is accepted under any of the usual keys, and its
        entries may be bare ids or objects carrying an ``id``.

        The requested limit is held inside :data:`MAX_LIST_RESULTS` here rather
        than left to each caller, because the cap is the store's and not ours: an
        over-cap request is answered 400, and a surface that forgot the cap would
        report a caller mistake for what is really its own unstated precondition.
        """
        limit = bound_list_limit(limit)
        response = self._request(
            "GET",
            "/documents",
            params={"limit": limit, "collection": collection or self.collection},
        )
        if response.status_code >= 400:
            _raise_operation_error("list_documents", self.collection, response, now=self._clock())
        data = _response_json(response, "list_documents")
        if not isinstance(data, dict):
            raise TansekiResponseError("list_documents returned an invalid response")
        items = next(
            (
                data[key]
                for key in ("documents", "items", "results", "hits")
                if isinstance(data.get(key), list)
            ),
            None,
        )
        if items is None:
            raise TansekiResponseError("list_documents returned no document array")
        ids: list[str] = []
        for item in items:
            if isinstance(item, str):
                ids.append(item)
            elif isinstance(item, dict) and isinstance(item.get("id"), str):
                ids.append(item["id"])
        return ids

    def upsert_document(self, payload: dict[str, Any]) -> TansekiUpsertResult:
        """Create or update a document. ``payload`` is a mapper-produced dict."""
        body = {**payload, "collection": self.collection}
        doc_id = body.get("id")
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise TansekiError("upsert_document requires a payload with a non-blank string 'id'")
        response = self._request(
            "POST",
            "/documents:upsert",
            json=body,
            headers={"Idempotency-Key": _idempotency_key(body)},
        )
        if response.status_code == 409:
            raise TansekiConflictError(f"conflict writing {doc_id}")
        if response.status_code >= 400:
            _raise_operation_error("upsert_document", doc_id, response, now=self._clock())
        if not response.content:
            raise TansekiResponseError("upsert_document returned an empty response")
        data = _response_json(response, "upsert_document")
        if not isinstance(data, dict):
            raise TansekiResponseError("upsert_document returned an invalid response")
        revision = _required_string(data, "revision", "upsert_document")
        created = data.get("created")
        if not isinstance(created, bool):
            raise TansekiResponseError("upsert_document returned an invalid created flag")
        return TansekiUpsertResult(revision=revision, created=created)

    def delete_document(
        self,
        doc_id: str,
        *,
        message: str,
        author: str,
        collection: str | None = None,
    ) -> bool:
        """Delete a document. Returns False when it did not exist."""
        response = self._request(
            "POST",
            "/documents:delete",
            json={
                "id": doc_id,
                "collection": collection or self.collection,
                "message": message,
                "author": author,
            },
        )
        if response.status_code == 404:
            return False
        if response.status_code >= 400:
            _raise_operation_error("delete_document", doc_id, response, now=self._clock())
        data = _response_json(response, "delete_document")
        if not isinstance(data, dict):
            raise TansekiResponseError("delete_document returned an invalid response")
        _required_string(data, "revision", "delete_document")
        return True

    def search(
        self,
        query: str,
        *,
        tags: list[str] | None = None,
        frontmatter: dict[str, str] | None = None,
        limit: int = 10,
        collection: str | None = None,
    ) -> list[TansekiHit]:
        """Lexical search across the collection, with tag/frontmatter filters."""
        limit = bound_search_limit(limit)
        params: list[tuple[str, Any]] = [
            ("q", query),
            ("limit", limit),
            ("collection", collection or self.collection),
        ]
        for tag in tags or []:
            params.append(("tags", tag))
        for key, value in (frontmatter or {}).items():
            params.append(("fm", f"{key}={value}"))
        response = self._request("GET", "/search", params=params)
        if response.status_code >= 400:
            _raise_operation_error("search", "query", response, now=self._clock())
        data = _response_json(response, "search")
        if not isinstance(data, dict) or not isinstance(data.get("hits"), list):
            raise TansekiResponseError("search returned an invalid response")
        # Tanseki's search response is a pagination envelope (hits/limit/offset/hasMore)
        # and deliberately carries no ``total``; only ``DocumentListResponse`` and
        # ``HistoryResponse`` do. Every present field is still validated, and a
        # ``total`` sent by some other server is honoured rather than demanded.
        raw_hits = data["hits"]
        _optional_count(data, "offset", "search")
        _optional_count(data, "limit", "search")
        _optional_bool(data, "hasMore", "search")
        total = _optional_count(data, "total", "search")
        if total is not None and total < len(raw_hits):
            raise TansekiResponseError("search returned fewer results than its total")
        hits = [TansekiHit.from_json(hit) for hit in raw_hits]
        return hits[:limit]

    def search_documents(
        self,
        query: str,
        *,
        tags: list[str] | None = None,
        frontmatter: dict[str, str] | None = None,
        limit: int = 10,
        collection: str | None = None,
    ) -> list[TansekiDocument]:
        """Return bounded ranked documents, omitting IDs that disappeared after search."""
        limit = bound_search_limit(limit)
        hits = self.search(
            query,
            tags=tags,
            frontmatter=frontmatter,
            limit=limit,
            collection=collection,
        )
        documents = self.get_documents(
            [hit.id for hit in hits],
            collection=collection,
        )
        return [document for document in documents if document is not None][:limit]

    def traverse(
        self,
        doc_id: str,
        rel: str,
        *,
        depth: int = 3,
        collection: str | None = None,
    ) -> list[str]:
        """Traverse the graph from a document, returning related document ids."""
        response = self._request(
            "POST",
            "/documents:traverse",
            json={
                "id": doc_id,
                "rel": rel,
                "depth": depth,
                "collection": collection or self.collection,
            },
        )
        if response.status_code == 404:
            return []
        if response.status_code >= 400:
            _raise_operation_error("traverse", doc_id, response, now=self._clock())
        data = _response_json(response, "traverse")
        if not isinstance(data, dict) or not isinstance(data.get("ids"), list):
            raise TansekiResponseError("traverse returned an invalid response")
        ids = data["ids"]
        if any(not isinstance(doc_id, str) or not doc_id.strip() for doc_id in ids):
            raise TansekiResponseError("traverse returned an invalid document id")
        return ids
