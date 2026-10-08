"""Opencode provider implementation for LLM completion.

Kojutsu's other providers reach a raw completion endpoint. Opencode is an agent
runtime, so this adapter has a stricter job: the prompt contains untrusted pull
request text, and it must not be able to reach anything worth stealing or changing.

The controls live in :mod:`kojutsu.integrations.sandbox` and are layered, because
the important one is not the tool list:

- the child gets a **scrubbed environment**, so no repository or provider token is
  inherited. Seatbelt does not filter ``environ``, so this cannot be delegated to
  the OS sandbox and is enforced here.
- the child gets a **private home** with only its own agent definition and the one
  model credential it needs, so the operator's opencode configuration, MCP servers,
  and session data are unreachable.
- on macOS the child runs under a **Seatbelt profile** that denies reads and writes
  outside the scratch tree.
- the agent is provisioned with **no tools**, as defence in depth.

Opencode is treated as an external provider: :func:`validate_llm_privacy` still
requires an explicit opt-in and an allow-listed repository, because a hosted model
endpoint leaves the host just as a vendor API does.
"""

from __future__ import annotations

import asyncio
import functools
import json
import shutil
import subprocess

from . import sandbox
from .llm import (
    MAX_LLM_RESPONSE_CHARS,
    LLMConfigurationError,
    LLMProviderError,
    LLMResponseError,
)

_OPENCODE_BIN = "opencode"
AGENT_NAME = "kojutsu-complete"
_AGENT_NOT_FOUND_MARKER = "agent not found"

#: Characters of model output accepted per requested token, estimated. The
#: opencode CLI cannot pre-limit generation the way a token API can, so the
#: caller's ``max_tokens`` bound is enforced on the way out instead: output past
#: ``max_tokens * _CHARS_PER_TOKEN`` is refused rather than truncated, because a
#: truncated answer reads as a complete one. Four characters per token is the
#: standard English estimate; denser scripts pack more meaning per token and so
#: pass more easily, which errs toward accepting rather than refusing.
_CHARS_PER_TOKEN = 4


def _require_binary() -> None:
    if shutil.which(_OPENCODE_BIN) is None:
        raise LLMProviderError(
            f"LLM provider opencode requires the {_OPENCODE_BIN!r} executable on PATH."
        )


def _extract_text(stream: str) -> str:
    """Collect only ``text`` parts from opencode's JSON event stream."""
    chunks: list[str] = []
    for line in stream.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            event = json.loads(stripped)
        except ValueError:
            continue
        if not isinstance(event, dict) or event.get("type") != "text":
            continue
        part = event.get("part")
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            chunks.append(part["text"])
    return "\n".join(chunks).strip()


def completion(
    prompt: str,
    model: str,
    max_tokens: int = 768,
    *,
    timeout_seconds: float = 30.0,
    max_retries: int = 1,
    base_url: str = "",
    api_key: str = "",
    system: str | None = None,
) -> str:
    """Generate a completion through a sandboxed, tool-less opencode agent.

    ``base_url`` and ``api_key`` are accepted for interface parity with the other
    providers and deliberately unused: opencode carries its own credentials, so
    Kojutsu neither needs nor stores a key for it.

    The agent's system prompt is provisioned per (provider, model) and is fixed, so a
    per-task instruction is carried in the user prompt under explicit ``<task>`` and
    ``<input>`` delimiters instead. The agent's own untrusted-source rule stays in the
    system prompt where it is hardest to lose, and the delimiters make the boundary
    between instruction and data explicit rather than positional.

    ``max_tokens`` cannot pre-limit generation -- the opencode CLI takes no token
    budget flag -- so it is enforced as an output acceptance bound instead of
    being silently dropped: output past ``max_tokens * _CHARS_PER_TOKEN``
    characters is refused. The absolute ceiling is ``MAX_LLM_RESPONSE_CHARS``,
    shared with every other provider, so no provider smuggles more context
    downstream than the rest.
    """
    del base_url, api_key
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise LLMConfigurationError(
            f"LLM provider opencode requires a positive integer max_tokens, got {max_tokens!r}."
        )
    _require_binary()

    if system is not None:
        prompt = f"<task>\n{system.strip()}\n</task>\n\n<input>\n{prompt}\n</input>"

    try:
        box = sandbox.provision(sandbox.root_for_model(model), model=model)
    except sandbox.SandboxError as exc:
        raise LLMProviderError(f"LLM provider opencode could not be isolated: {exc}") from exc

    command = [
        _OPENCODE_BIN,
        "run",
        "--agent",
        AGENT_NAME,
        "--model",
        model,
        "--format",
        "json",
        prompt,
    ]
    # The first call into a newly created private home can fail while opencode
    # initialises its own state, so the configured retry budget is spent here too.
    attempts = max(1, max_retries + 1)
    result: subprocess.CompletedProcess[str] | None = None
    for attempt in range(attempts):
        try:
            result = sandbox.run(box, command, timeout_seconds=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            raise LLMProviderError(
                f"LLM request to opencode timed out after {timeout_seconds:g} seconds; "
                f"retry or increase LLM_TIMEOUT_SECONDS."
            ) from exc
        if result.returncode == 0:
            break
        if attempt + 1 < attempts:
            continue

    if result is None:
        raise LLMProviderError(
            "LLM provider opencode finished without a result; retry the request."
        )
    if result.returncode != 0:
        if _AGENT_NOT_FOUND_MARKER in (result.stderr or "").lower():
            raise LLMProviderError(
                f"LLM provider opencode could not find the {AGENT_NAME!r} agent."
            )
        raise LLMProviderError(
            "LLM provider opencode failed; verify the model id with "
            "`opencode models opencode-go`, then retry."
        )
    text = _extract_text(result.stdout)
    if not text:
        raise LLMResponseError("LLM provider returned an empty or malformed response.")
    accepted_chars = min(MAX_LLM_RESPONSE_CHARS, max_tokens * _CHARS_PER_TOKEN)
    if len(text) > accepted_chars:
        raise LLMResponseError(
            f"LLM provider response exceeded the {accepted_chars}-character limit "
            f"(max_tokens={max_tokens})."
        )
    return text


def get_model_id(provider: str, model: str) -> str:
    """Get the full model ID for the opencode provider."""
    return model if "/" in model else f"{provider}/{model}"


async def completion_async(
    prompt: str,
    model: str,
    max_tokens: int = 768,
    *,
    timeout_seconds: float = 30.0,
    max_retries: int = 1,
    base_url: str = "",
    api_key: str = "",
) -> str:
    """Await :func:`completion` without blocking the event loop.

    The whole call is blocking -- sandbox provisioning and a subprocess that runs
    for the duration of the model call -- so all of it is dispatched to a worker
    thread rather than making the call half-async, which would leave the blocking
    half still blocking. This is the same ``asyncio.to_thread`` dispatch the relay
    worker uses for the outbox.

    Two principals may await this concurrently. Their sandboxes are separate homes
    keyed by ``(provider, model)``, so neither can observe the other's credential or
    agent definition; only provisioning the *same* model contends, and that is
    serialised by the per-root lock.
    """
    return await asyncio.to_thread(
        functools.partial(
            completion,
            prompt,
            model,
            max_tokens,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            base_url=base_url,
            api_key=api_key,
        )
    )
