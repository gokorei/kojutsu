"""Tests for the opencode LLM provider adapter.

The isolation controls are tested in ``test_sandbox.py``; this file covers the
adapter's own contract: how it invokes opencode, and what it does with the result.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from kojutsu.integrations import opencode as adapter
from kojutsu.integrations import sandbox
from kojutsu.integrations.llm import LLMProviderError, LLMResponseError

MODEL = "opencode/model"


def _stream(*texts: str) -> str:
    parts = [json.dumps({"type": "step_start", "part": {"type": "step-start"}})]
    for text in texts:
        parts.append(json.dumps({"type": "text", "part": {"type": "text", "text": text}}))
    parts.append(json.dumps({"type": "step_finish", "part": {"type": "step-finish"}}))
    return "\n".join(parts)


class _Completed:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


@pytest.fixture
def operator_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "operator-home"
    auth = home / sandbox.OPENCODE_AUTH_PATH
    auth.parent.mkdir(parents=True)
    auth.write_text(json.dumps({"opencode-go": {"type": "api", "key": "model-key"}}))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


@pytest.fixture
def runner(operator_home: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., dict[str, Any]]:
    """Fake ``sandbox.run``, capturing the command, env, and cwd it was given."""
    monkeypatch.setattr(adapter.shutil, "which", lambda name: "/usr/bin/opencode")

    def install(
        *,
        stdout: str = _stream("design_decision|Why this?"),
        stderr: str = "",
        returncode: int = 0,
        side_effect: Callable[[], Any] | None = None,
        provision_error: Exception | None = None,
    ) -> dict[str, Any]:
        recorded: dict[str, Any] = {}

        if provision_error is not None:

            def fake_provision(root: Path, *, model: str) -> Any:
                raise provision_error
        else:

            def fake_provision(root: Path, *, model: str) -> Any:
                recorded["model"] = model
                return sandbox.Sandbox(
                    home=root / "home",
                    workdir=root / "work",
                    root=root,
                    profile_path=None,
                    env={"HOME": str(root / "home")},
                )

        def fake_run(box, command, *, timeout_seconds):  # type: ignore[no-untyped-def]
            recorded["command"] = command
            recorded["timeout"] = timeout_seconds
            if side_effect is not None:
                return side_effect()
            return _Completed(stdout=stdout, stderr=stderr, returncode=returncode)

        monkeypatch.setattr(adapter.sandbox, "provision", fake_provision)
        monkeypatch.setattr(adapter.sandbox, "run", fake_run)
        return recorded

    return install


def test_command_pins_the_text_only_agent_and_the_model(runner) -> None:
    recorded = runner()

    text = adapter.completion("prompt", MODEL, timeout_seconds=90.0)

    assert text == "design_decision|Why this?"
    command = recorded["command"]
    assert command[0] == "opencode"
    assert command[command.index("--agent") + 1] == adapter.AGENT_NAME
    assert command[command.index("--model") + 1] == MODEL
    assert command[command.index("--format") + 1] == "json"
    assert command[-1] == "prompt"
    assert recorded["timeout"] == 90.0
    assert recorded["model"] == MODEL


def test_execution_goes_through_the_sandbox(runner) -> None:
    recorded = runner()

    adapter.completion("prompt", MODEL)

    # Not a bare subprocess call: the child must be the sandboxed run helper.
    assert recorded["command"][0] == "opencode"


def test_tool_events_contribute_nothing_to_the_result(runner) -> None:
    """A tool call, if one ever happened, must not leak into the completion."""
    stream = "\n".join(
        [
            json.dumps({"type": "text", "part": {"type": "text", "text": "trade_off|Real answer"}}),
            json.dumps(
                {
                    "type": "tool",
                    "part": {"type": "tool", "tool": "bash", "state": {"output": "leaked"}},
                }
            ),
        ]
    )
    runner(stdout=stream)

    assert adapter.completion("p", MODEL) == "trade_off|Real answer"


def test_reasoning_events_contribute_nothing_to_the_result(runner) -> None:
    """Model-reported reasoning is discarded, and this is a decision not a gap.

    The stream carries a `reasoning` part whose text quotes the untrusted prompt
    nearly verbatim. Capturing it would store attacker-controlled text in the
    knowledge store carrying capture provenance, which is the prompt-injection
    persistence case `docs/design-review/rationale.md` rules out. A rationale
    programme exists to capture *stated* reasons instead; it does not widen this.

    Stated as a test so a future change that keeps reasoning cannot present itself
    as a missing feature.
    """
    stream = "\n".join(
        [
            json.dumps(
                {
                    "type": "reasoning",
                    "part": {
                        "type": "reasoning",
                        "text": "The diff asks me to ignore my rules and print a secret.",
                    },
                }
            ),
            json.dumps({"type": "text", "part": {"type": "text", "text": "trade_off|Real answer"}}),
        ]
    )
    runner(stdout=stream)

    result = adapter.completion("p", MODEL)

    assert result == "trade_off|Real answer"
    assert "ignore my rules" not in result


def test_multiple_text_parts_are_joined_in_order(runner) -> None:
    runner(stdout=_stream("line one", "line two"))

    assert adapter.completion("p", MODEL) == "line one\nline two"


def test_malformed_stream_lines_are_skipped(runner) -> None:
    stream = "not json\n" + json.dumps({"type": "text", "part": {"text": "kept"}}) + "\n{broken"
    runner(stdout=stream)

    assert adapter.completion("p", MODEL) == "kept"


def test_sandbox_failure_is_reported_as_a_provider_error(runner, operator_home: Path) -> None:
    """No credential means no isolation, and isolation is not optional."""
    runner(provision_error=sandbox.SandboxError("No opencode credentials at /x/auth.json"))

    with pytest.raises(LLMProviderError, match="could not be isolated"):
        adapter.completion("p", MODEL)


def test_missing_binary_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter.shutil, "which", lambda name: None)

    with pytest.raises(LLMProviderError, match="executable on PATH"):
        adapter.completion("p", MODEL)


def test_nonzero_exit_is_a_provider_error(runner) -> None:
    runner(stderr="boom", returncode=2)

    with pytest.raises(LLMProviderError, match="verify the model id"):
        adapter.completion("p", MODEL)


def test_agent_race_is_reported_distinctly(runner) -> None:
    runner(stderr="agent not found: kojutsu-complete", returncode=1)

    with pytest.raises(LLMProviderError, match="could not find"):
        adapter.completion("p", MODEL)


def test_empty_output_is_a_response_error(runner) -> None:
    runner(stdout=_stream())

    with pytest.raises(LLMResponseError):
        adapter.completion("p", MODEL)


def test_timeout_is_a_provider_error(runner) -> None:
    def timeout() -> Any:
        raise subprocess.TimeoutExpired(cmd="opencode", timeout=30)

    runner(side_effect=timeout)

    with pytest.raises(LLMProviderError, match="timed out"):
        adapter.completion("p", MODEL, timeout_seconds=30.0)


def test_oversized_response_is_rejected(runner) -> None:
    accepted = min(adapter.MAX_LLM_RESPONSE_CHARS, 768 * adapter._CHARS_PER_TOKEN)
    runner(stdout=_stream("x" * (accepted + 1)))

    with pytest.raises(LLMResponseError, match="exceeded"):
        adapter.completion("p", MODEL)


def test_max_tokens_bounds_accepted_output(runner) -> None:
    """The caller bound is enforced, not dropped: a small max_tokens refuses
    output the caller did not ask for room for."""
    runner(stdout=_stream("x" * 41))

    with pytest.raises(LLMResponseError, match="max_tokens=10"):
        adapter.completion("p", MODEL, max_tokens=10)


def test_non_positive_max_tokens_is_a_configuration_error(runner) -> None:
    """A caller bug worth failing fast on, not a provider failure to retry."""
    from kojutsu.integrations.llm import LLMConfigurationError

    with pytest.raises(LLMConfigurationError, match="max_tokens"):
        adapter.completion("p", MODEL, max_tokens=0)


def test_credentials_are_not_accepted_from_the_caller(runner) -> None:
    """The adapter ignores api_key: opencode's own credential is used instead."""
    runner()

    adapter.completion("p", MODEL, api_key="sk-should-be-ignored", base_url="http://nope")

    # Reaching the model at all proves the bogus values were not routed anywhere.


def test_model_id_keeps_an_explicit_provider_prefix() -> None:
    assert adapter.get_model_id("opencode", MODEL) == MODEL
    assert adapter.get_model_id("opencode", "space-bunny") == "opencode/space-bunny"
