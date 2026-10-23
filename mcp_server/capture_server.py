#!/usr/bin/env python3
"""MCP server for *writing* declared decision rationale.

This is a separate server from ``mcp_server/server.py`` on purpose, and the
separation is the design rather than an organisational preference.

``kojutsu-knowledge`` is read-only, and ``docs/design-review/open-questions.md``
names exactly why that matters: nothing structurally prevents a future change from
adding a write tool to it, and the only thing standing in the way today is a
reviewer noticing. Keeping the write surface in its own module means the read
server's read-only property stays a *structure* rather than a claim, and the test
that asserts it stays writable.

**What this tool does is post a comment *and* write the record, and that is two
routes to one document rather than one route and a shadow of it.** A rationale
reaches ``<repo>/pr-<n>/rationale/<entry_id>`` from exactly two places: from here,
immediately, out of the text the agent passed to this tool; and from
``core/rationale_collector.py``, later, out of the comment this tool posted. Both go
through the same claim and the same outbox, and both derive the same entry id from
the same anchor, so one declaration cannot become two records.

**The two are mutually exclusive per declaration, and which one wins is not obvious.**
:meth:`~kojutsu.core.question_registry.QuestionRegistry.claim_rationale` refuses a
declaration whose row is already ``completed``, and this module completes the claim
*before* it stores. So for any declaration that runs to the end, the webhook that
reaches the collector finds the claim closed and writes nothing — the collector's
route runs only where this one did not finish: the post failed and released the
claim, or ``complete_rationale`` was refused. **This direct route is therefore the
one that writes the surviving document, almost always.**

That is why the character policy is applied here as well as in the collector, and the
reason is not redundancy: :func:`kojutsu.core.text_hygiene.sanitise` on the
collector side is unreachable for the common case, because the claim gate has already
closed by the time the comment comes back. An agent declaring a reason has read
attacker-controlled issue and pull request text and is now writing prose about what
it found there, so this is the record where "the author is a machine, so its bytes are
safe" is least true.

**The two routes store the same wording, for a reason that is not "both sanitise".**
The collector applies its own narrower normalisation inside the extractor before
:func:`~kojutsu.core.text_hygiene.sanitise` runs, so a naive expectation is that one
route stores ``sanitise(x)`` and the other ``sanitise(normalise(x))`` and they differ.
They do not, and the reason is that the refused set is a *superset* of the extractor's:
composing the narrower pass with the wider one removes the same set as the wider one
alone. That is a property of two enumerated sets rather than of this function, which
is why the equality is asserted in the tests instead of asserted here and trusted.

**The comment is posted unsanitised, and that is deliberate.** This process is a
courier: it posts what the agent declared and lets capture do the reading. Sanitising
first would make kojutsu the author of the wording, and it would leave the collector's
note measuring a comment with nothing left in it -- so on the fallback route, where the
collector is the writer, the document would claim byte-for-byte fidelity over prose that
had been edited. Posting the raw string keeps the refused characters in the comment, so
the note either route writes names the same ones with the same counts, and the forge
keeps the closest thing to the original that anybody can re-fetch.

What genuinely differs between the routes is the anchors, and it is a superset rather
than a contradiction: the collector additionally records ``comment_author``,
``github_author_association`` and ``delivery_id``, and dates the declaration from the
comment rather than from the write. A document written here names the comment it was
posted as and nothing else, which is the most this process can honestly claim.

**The agent cannot hold the token that makes this work.** The sandbox strips
``GITHUB_TOKEN`` from the model process and provisions the agent with no tools, so
an agent that could post its own declaration would collapse the separation
``docs/github-seam.md`` describes. Kojutsu posts; the agent's contribution is
the text.

**What this cannot claim.** A stdio server with a single trust domain has no
caller identity, so this module cannot know who invoked it. The honest statement
is narrow: this build exposes exactly one tool, it can only post a comment on an
allow-listed repository using the capture process's token, and the declaration it
carries is a self-assertion by the account that posted it. That is not the same
claim as "no caller can induce a write".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import BaseModel

from kojutsu import allowlist
from kojutsu.config import Settings
from kojutsu.core.knowledge_sink import TansekiKnowledgeSink
from kojutsu.core.outbox import TansekiOutbox
from kojutsu.core.question_registry import (
    build_registry,
    stable_rationale_entry_id,
)
from kojutsu.core.text_hygiene import (
    SANITISATION_KEY,
    describe_removals,
    sanitise,
)
from kojutsu.integrations.github import (
    GitHubClient,
    rationale_comment_body_as_agent,
)
from kojutsu.integrations.llm import MAX_RATIONALE_DECLARATION_CHARS
from kojutsu.integrations.tanseki import TansekiClient
from kojutsu.models import (
    RationaleChannel,
    RationaleEntry,
    RationaleSource,
)
from kojutsu.repo_name import split_repo
from kojutsu.text_limits import (
    MAX_AGENT_ID_CHARS,
    MAX_BRANCH_CHARS,
    MAX_MODEL_ID_CHARS,
    MAX_REPOSITORY_CHARS,
)

#: Bounded at the same edge that receives the input. An MCP tool on a stdio
#: transport is reachable by anything that can speak the protocol to it, so
#: unbounded input here is the same denial-of-service primitive
#: ``open-questions.md`` discusses for the webhook. The honest answer is the same
#: too: bound it where it arrives.
MAX_RATIONALE_TEXT_CHARS = MAX_RATIONALE_DECLARATION_CHARS * 8
MAX_REASON_CHARS = 4_000

#: A write is not idempotent in the HTTP sense -- it creates a comment -- but
#: declaring it idempotent is accurate about the *effect*: the semantic
#: declaration key means a repeat call finds the existing record and posts
#: nothing. It is deliberately not marked read-only or non-destructive.
_WRITE_ANNOTATIONS = ToolAnnotations.model_validate(
    {
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)


class ToolResult(BaseModel):
    """Machine-readable MCP tool result.

    Mirrors the read server's result shape so a host handles one format, while
    staying a separate type: the two servers have different trust assumptions and
    sharing the class would invite a caller to treat a write refusal as a read
    failure.
    """

    ok: bool
    code: str
    retryable: bool
    error: str | None = None
    result: str | None = None

    def __contains__(self, value: object) -> bool:
        return str(value) in (self.result or "")


def _success(result: str) -> ToolResult:
    return ToolResult(ok=True, code="ok", retryable=False, result=result)


def _error(code: str, error: str, *, retryable: bool = False) -> ToolResult:
    return ToolResult(ok=False, code=code, retryable=retryable, error=error)


server = MCPServer(
    "kojutsu-capture",
    version="0.1.0",
    instructions=(
        "Record why an implementation looks the way it does, as a statement by the "
        "agent that wrote it. This posts a comment on the forge; capture then stores "
        "it. A recorded rationale is a claim about intent, never evidence that the "
        "code is correct, and it never counts as an independent check on the change."
    ),
)


@dataclass
class _ServerOverrides:
    """Owns this server's injectable construction: settings, client, registry, sink.

    The capture server builds its collaborators per call from whatever is
    configured, so tests and embedded hosts inject through :func:`configure`
    rather than by assigning module globals. The one instance below is the
    only module state; every read and write goes through it.
    """

    settings: Settings | None = None
    client_factory: Any = None
    registry_factory: Any = None
    sink_factory: Any = None

    def current_settings(self) -> Settings:
        return self.settings or Settings()


_overrides = _ServerOverrides()

logger = logging.getLogger(__name__)


def configure(
    *,
    settings: Settings | None = None,
    client_factory: Any = None,
    registry_factory: Any = None,
    sink_factory: Any = None,
) -> None:
    """Inject settings, client, registry, and sink construction for tests or embedded use."""
    if client_factory is not None:
        _overrides.client_factory = client_factory
    if registry_factory is not None:
        _overrides.registry_factory = registry_factory
    if sink_factory is not None:
        _overrides.sink_factory = sink_factory
    if settings is not None:
        _overrides.settings = settings


def _current_settings() -> Settings:
    return _overrides.current_settings()


def _github_client(settings: Settings) -> GitHubClient:
    if _overrides.client_factory is not None:
        return _overrides.client_factory(settings)
    return GitHubClient(settings.github_token)


def _registry(settings: Settings) -> Any:
    if _overrides.registry_factory is not None:
        return _overrides.registry_factory(settings)
    return build_registry(settings)


def _bounded_text(
    value: str | None, *, field: str, max_chars: int
) -> tuple[str | None, ToolResult | None]:
    """Normalise one bounded text field, or explain which bound it broke.

    A length violation is ``invalid_input`` and never retryable: the caller sent
    too much and sending it again will not help. A distinction that matters here
    because the alternative codes in this module are retryable and mean "the store
    or the forge is unwell".
    """
    if value is None:
        return None, _error("invalid_input", f"{field} is required.")
    stripped = value.strip()
    if not stripped:
        return None, _error("invalid_input", f"{field} must not be empty.")
    if len(stripped) > max_chars:
        return None, _error("invalid_input", f"{field} must be at most {max_chars} characters.")
    return stripped, None


def _sink(settings: Settings) -> Any:
    """The store's own sink, so a direct declaration gets the same delivery.

    Built from the runtime's outbox rather than a second queue. That is the whole
    reason the direct route is not a second privileged write path: it claims
    through the same registry table, derives the same identity, and is relayed by
    the same durable outbox with the same retry and dead-letter semantics. A
    rationale that reached the store by a different mechanism would be a thing
    that could be lost differently from everything else, and that is the failure
    mode the outbox exists to prevent.
    """
    if _overrides.sink_factory is not None:
        return _overrides.sink_factory(settings)
    outbox = TansekiOutbox(settings.tanseki_outbox_path)
    client = TansekiClient.from_settings(settings)
    return TansekiKnowledgeSink(client, outbox)


class _ValidatedDeclaration(BaseModel):
    """Bounds-checked tool inputs, before policy and identity."""

    repo: str
    reason: str
    agent_id: str
    branch: str
    model: str | None
    revision: int
    pr_number: int | None


def _validate_declaration_inputs(
    repo: str,
    reason: str,
    agent_id: str,
    branch: str,
    pr_number: int | None,
    model: str | None,
    revision: int,
) -> tuple[_ValidatedDeclaration | None, ToolResult | None]:
    normalized_repo, error = _bounded_text(repo, field="repo", max_chars=MAX_REPOSITORY_CHARS)
    if error is not None:
        return None, error
    normalized_reason, error = _bounded_text(
        reason, field="reason", max_chars=MAX_RATIONALE_TEXT_CHARS
    )
    if error is not None:
        return None, error
    normalized_agent, error = _bounded_text(
        agent_id, field="agent_id", max_chars=MAX_AGENT_ID_CHARS
    )
    if error is not None:
        return None, error
    normalized_branch, error = _bounded_text(
        branch or "", field="branch", max_chars=MAX_BRANCH_CHARS
    )
    if error is not None:
        return None, error
    # Narrowed for the checker: a None error means every field above is
    # present, which is the helper's contract.
    normalized_repo = cast(str, normalized_repo)
    normalized_reason = cast(str, normalized_reason)
    normalized_agent = cast(str, normalized_agent)
    normalized_branch = cast(str, normalized_branch)
    normalized_model: str | None = None
    if model is not None:
        normalized_model, error = _bounded_text(model, field="model", max_chars=MAX_MODEL_ID_CHARS)
        if error is not None:
            return None, error
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        return None, _error("invalid_input", "revision must be an integer of at least 1.")
    if pr_number is not None and (
        isinstance(pr_number, bool) or not isinstance(pr_number, int) or pr_number < 1
    ):
        return None, _error("invalid_input", "pr_number must be a positive integer or omitted.")
    if pr_number is None and not normalized_branch:
        return None, _error(
            "invalid_input",
            "a declaration must name a change: pass pr_number, or branch when there is "
            "no pull request yet. A rationale attached to nothing is one nobody will "
            "find again.",
        )
    # Format before policy. A malformed repository is a caller mistake whatever the
    # allowlist says, and answering it with "not authorized" would make the two
    # indistinguishable to a caller probing the boundary.
    try:
        split_repo(normalized_repo)
    except ValueError:
        return None, _error("invalid_input", "repo must be 'owner/name'.")
    return (
        _ValidatedDeclaration(
            repo=normalized_repo,
            reason=normalized_reason,
            agent_id=normalized_agent,
            branch=normalized_branch,
            model=normalized_model,
            revision=revision,
            pr_number=pr_number,
        ),
        None,
    )


class _AuthorisedDeclaration(BaseModel):
    """A declaration that passed policy: sanitised text plus store settings."""

    model_config = {"arbitrary_types_allowed": True}

    validated: _ValidatedDeclaration
    rationale_text: str
    sanitisation_note: str | None
    settings: Settings


def _authorise_declaration(
    validated: _ValidatedDeclaration,
) -> tuple[_AuthorisedDeclaration | None, ToolResult | None]:
    # Policy before anything is written. Ordering is the whole argument:
    # ``reason`` is in no identity derivation, so rewriting the prose cannot
    # re-identify a stored record.
    reason_field = sanitise(validated.reason)
    rationale_text = reason_field.text
    sanitisation_note = describe_removals(validated.reason)
    if not rationale_text.strip():
        return None, _error(
            "invalid_input",
            "the reason is empty once the display-deceptive characters are removed, so "
            "there is nothing to record. Nothing was posted or stored. What was found: "
            f"{sanitisation_note}.",
        )
    settings = _current_settings()
    try:
        allowed = allowlist.configured_repositories(settings)
    except allowlist.AllowlistError:
        return None, _error(
            "allowlist_invalid",
            "The configured repository allowlist could not be parsed; nothing was posted.",
        )
    if validated.repo.casefold() not in allowed:
        return None, _error("repository_not_authorized", "Repository is not authorized.")
    return (
        _AuthorisedDeclaration(
            validated=validated,
            rationale_text=rationale_text,
            sanitisation_note=sanitisation_note,
            settings=settings,
        ),
        None,
    )


def _rationale_ids(declaration: _ValidatedDeclaration) -> tuple[str, str | None]:
    entry_id = stable_rationale_entry_id(
        repo=declaration.repo,
        pr_number=declaration.pr_number,
        branch=declaration.branch,
        declared_by=declaration.agent_id,
        revision=declaration.revision,
    )
    revises = None
    if declaration.revision > 1:
        revises = stable_rationale_entry_id(
            repo=declaration.repo,
            pr_number=declaration.pr_number,
            branch=declaration.branch,
            declared_by=declaration.agent_id,
            revision=declaration.revision - 1,
        )
    return entry_id, revises


def _post_rationale_comment(
    authorised: _AuthorisedDeclaration, owner: str, name: str
) -> int | None:
    """Post the raw declaration as a forge comment, or None without a PR.

    The reason **as the agent wrote it**, not the sanitised copy: this process
    is a courier, and a courier that edits the text makes itself its author.
    """
    declaration = authorised.validated
    if declaration.pr_number is None:
        return None
    client = _github_client(authorised.settings)
    return client.post_issue_comment(
        owner,
        name,
        declaration.pr_number,
        rationale_comment_body_as_agent(
            declaration.reason,
            declaration.agent_id,
            declaration.model,
            declaration.revision,
            declaration.branch,
        ),
    ).id


def _build_rationale_entry(
    authorised: _AuthorisedDeclaration,
    entry_id: str,
    revises: str | None,
    comment_id: int | None,
) -> RationaleEntry:
    declaration = authorised.validated
    return RationaleEntry(
        entry_id=entry_id,
        repo=declaration.repo,
        pr_number=declaration.pr_number,
        branch=declaration.branch,
        declared_by=declaration.agent_id,
        declared_model=declaration.model,
        rationale_text=authorised.rationale_text,
        source=RationaleSource.DECLARED,
        channel=(
            RationaleChannel.FORGE_COMMENT
            if declaration.pr_number is not None
            else RationaleChannel.CAPTURE_SERVER
        ),
        revision=declaration.revision,
        revises=revises,
        metadata={
            **({"github_comment_id": comment_id} if comment_id is not None else {}),
            **(
                {SANITISATION_KEY: authorised.sanitisation_note}
                if authorised.sanitisation_note
                else {}
            ),
        },
    )


def _store_rationale_entry(registry: Any, authorised: _AuthorisedDeclaration) -> ToolResult:
    """Claim, publish, complete and queue one declaration."""
    declaration = authorised.validated
    entry_id, revises = _rationale_ids(declaration)
    token = registry.claim_rationale(
        entry_id=entry_id,
        repo=declaration.repo,
        pr_number=declaration.pr_number,
        branch=declaration.branch,
        declared_by=declaration.agent_id,
        declared_model=declaration.model,
        source="declared",
        revision=declaration.revision,
        revises=revises,
        rationale_text=authorised.rationale_text,
    )
    if token is None:
        # Already captured, whichever route it took. Posting again would put
        # the same declaration on the forge twice.
        return _success(
            f"This declaration is already recorded as {entry_id}. Nothing was stored. "
            "Call again with a higher revision to append a later rationale."
        )
    owner, name = split_repo(declaration.repo)
    comment_id: int | None = None
    if declaration.pr_number is not None:
        try:
            comment_id = _post_rationale_comment(authorised, owner, name)
        except Exception:
            # Release rather than hold: the claim is what makes a repeat call a
            # no-op, and a claim left held by a failed post would silently
            # refuse the retry that would have succeeded.
            registry.release_rationale(entry_id, token, "post failed")
            raise
    entry = _build_rationale_entry(authorised, entry_id, revises, comment_id)
    if not registry.complete_rationale(entry_id, token):
        registry.release_rationale(entry_id, token, "completion refused")
        return _error(
            "claim_lost",
            "The rationale was written but its claim could not be finalised; it may "
            "be captured again. This is safe: capture deduplicates on the "
            "declaration key.",
        )
    delivery = _sink(authorised.settings).store(entry)
    delivery_note = (
        f" Delivery {delivery.status.value}."
        if delivery is not None
        else " Delivery is queued; delivery is uncertain."
    )
    if comment_id is not None:
        return _success(
            f"Posted rationale revision {declaration.revision} as comment {comment_id} on "
            f"{declaration.repo}#{declaration.pr_number} and recorded it as {entry_id}."
            f"{delivery_note} It is a claim about intent, not evidence that the code is correct."
        )
    return _success(
        f"Recorded rationale revision {declaration.revision} as {entry_id} without publishing it "
        f"anywhere: there is no pull request to post it on.{delivery_note} It is a "
        "claim about intent, not evidence that the code is correct."
    )


@server.tool(
    name="record_decision_rationale",
    description=(
        "Record why an implementation looks the way it does, as a statement by the "
        "agent that wrote it. With a pr_number it is also published as a marked "
        "comment, so a human reading the change can see the reason; without one it is "
        "recorded directly, for an agent with nowhere to publish. Either way the "
        "record is a claim about intent, not evidence, and calling again with a "
        "higher revision appends rather than replaces."
    ),
    annotations=_WRITE_ANNOTATIONS,
)
def record_decision_rationale(
    repo: str,
    reason: str,
    agent_id: str,
    branch: str = "",
    pr_number: int | None = None,
    model: str | None = None,
    revision: int = 1,
) -> ToolResult:
    """Record a declared rationale, publishing it only when there is a change to publish to."""
    validated, error = _validate_declaration_inputs(
        repo, reason, agent_id, branch, pr_number, model, revision
    )
    if error is not None or validated is None:
        return error or _error("invalid_input", "Invalid declaration.")
    authorised, error = _authorise_declaration(validated)
    if error is not None or authorised is None:
        return error or _error("invalid_input", "Not authorized.")
    registry = None
    try:
        registry = _registry(authorised.settings)
        return _store_rationale_entry(registry, authorised)
    except Exception as exc:  # reported to the caller, logged here
        # Logged rather than returned, and deliberately not summarised into the
        # response. A GitHub error body can echo the URL, and the URL is the one
        # place a token might appear, so the cause never crosses into a tool
        # result. An operator reads it from the server's own stderr instead.
        logger.exception("Recording a declared rationale failed: %s", type(exc).__name__)
        return _error(
            "capture_unavailable",
            "Unable to record the rationale. Verify the store and service health, then "
            "retry; nothing was stored.",
            retryable=True,
        )
    finally:
        if registry is not None:
            registry.close()


def main() -> None:
    """Entry point: run the capture MCP server over stdio."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
