"""Tests for question-generation orchestration with faked GitHub/LLM."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

import kojutsu.integrations.llm as llm_module
from kojutsu.core import question_generator
from kojutsu.integrations.github_models import GitHubPullRequest
from kojutsu.integrations.jira_client import JiraUnavailableError
from kojutsu.integrations.llm import (
    LLMConfig,
    LLMConfigurationError,
    LLMProviderError,
    LLMResponseError,
    generate_questions_sync,
    validate_llm_privacy,
)
from kojutsu.models import QuestionCategory
from test_github_client import ClosesLikeAClient


class FakeGitHubClient(ClosesLikeAClient):
    def __init__(self, token: str) -> None:
        self.token = token

    def get_pull_request(self, owner: str, repo: str, pr_number: int) -> GitHubPullRequest:
        return GitHubPullRequest(
            number=pr_number,
            title="Add feature",
            state="open",
            head={"ref": "feature/ABC-1-add", "sha": "a" * 40},
            body="",
        )

    def get_pull_diff(self, owner: str, repo: str, pr_number: int) -> str:
        return "diff --git a/x b/x\n+added line\n"

    def get_pr_files(self, owner: str, repo: str, pr_number: int):
        from kojutsu.integrations.github import PagedResult

        return PagedResult(items=["x.py"], truncated=False)


def test_generate_questions_builds_context_and_questions(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(question_generator, "GitHubClient", FakeGitHubClient)

    def generate_questions(prompt, provider, model, api_key, **kwargs):
        captured.update(kwargs)
        return [(QuestionCategory.DESIGN_DECISION, "Why?")]

    monkeypatch.setattr(question_generator, "generate_questions_sync", generate_questions)

    questions, context = question_generator.generate_questions_for_pr(
        "org/repo#1",
        "tok",
        llm_provider="openai",
        llm_api_key="test-key",
        llm_external_enabled=True,
        llm_allowed_repositories="org/repo",
    )

    assert context["repo"] == "org/repo"
    assert context["owner"] == "org"
    assert context["pr_number"] == 1
    assert context["pr_url"] == "https://github.com/org/repo/pull/1"
    assert context["jira_ticket_key"] == "ABC-1"
    assert context["files_changed"] == ["x.py"]
    # The head the diff was read at. Asserted here because this is the only
    # producer of the value: if it stops publishing, the column downstream reads
    # NULL forever and nothing anywhere reports an error, because an absent
    # anchor is a legal state. The consumer side is pinned in
    # tests/test_cli_ask_collect.py.
    assert context["head_sha"] == "a" * 40
    assert captured["base_url"] == ""
    assert captured["repository"] == "org/repo"
    assert captured["external_enabled"] is True
    assert len(questions) == 1
    assert questions[0].text == "Why?"
    assert questions[0].category == QuestionCategory.DESIGN_DECISION
    assert questions[0].context == {
        "repo": "org/repo",
        "pr_number": 1,
        "jira_ticket_key": "ABC-1",
    }


def test_generate_questions_publishes_no_anchor_when_the_head_omits_one(monkeypatch) -> None:
    """A missing sha costs the anchor, not the run.

    ``head`` is an untyped optional dict, so a response without ``sha`` is
    representable. Raising would abandon questions that were generated
    successfully over a detail about one piece of metadata, and the anchor's own
    contract is that absence is a value the store carries -- the v7 migration
    leaves old rows NULL for the same reason. So this degrades to ``None`` and
    keeps going.
    """

    class HeadWithoutSha(FakeGitHubClient):
        def get_pull_request(self, owner: str, repo: str, pr_number: int) -> GitHubPullRequest:
            return GitHubPullRequest(
                number=pr_number, title="t", state="open", head={"ref": "feature/ABC-1"}, body=""
            )

    monkeypatch.setattr(question_generator, "GitHubClient", HeadWithoutSha)
    monkeypatch.setattr(
        question_generator,
        "generate_questions_sync",
        lambda *_a, **_k: [(QuestionCategory.DESIGN_DECISION, "Why?")],
    )

    questions, context = question_generator.generate_questions_for_pr(
        "org/repo#1",
        "tok",
        llm_provider="openai",
        llm_api_key="k",
        llm_external_enabled=True,
        llm_allowed_repositories="org/repo",
    )

    assert len(questions) == 1
    assert context["head_sha"] is None
    assert context["branch_name"] == "feature/ABC-1"


def test_jira_enrichment_failure_warns_and_falls_back(monkeypatch, caplog) -> None:
    monkeypatch.setattr(question_generator, "GitHubClient", FakeGitHubClient)

    class FailingJiraClient:
        def __init__(self, **_kwargs) -> None:
            pass

        def get_issue_fields(self, issue_key: str) -> dict[str, str]:
            raise JiraUnavailableError("Jira is unavailable")

    monkeypatch.setattr(question_generator, "JiraClient", FailingJiraClient)
    monkeypatch.setattr(
        question_generator,
        "generate_questions_sync",
        lambda *_args, **_kwargs: [(QuestionCategory.DESIGN_DECISION, "Why?")],
    )

    with caplog.at_level("WARNING"):
        questions, context = question_generator.generate_questions_for_pr(
            "org/repo#1",
            "tok",
            jira_url="https://jira.test",
            jira_username="user",
            jira_api_token="token",
            llm_provider="openai",
            llm_api_key="test-key",
            llm_external_enabled=True,
            llm_allowed_repositories="org/repo",
        )

    assert len(questions) == 1
    assert context["jira_ticket_key"] == "ABC-1"
    assert "continuing without Jira context" in caplog.text


def test_external_llm_fails_closed_before_github_fetch() -> None:
    with pytest.raises(LLMConfigurationError, match="External LLM processing is disabled"):
        question_generator.generate_questions_for_pr(
            "org/repo#1",
            "tok",
            llm_provider="openai",
            llm_api_key="test-key",
        )


def test_repository_must_be_explicitly_allowed() -> None:
    with pytest.raises(LLMConfigurationError, match="not allowed"):
        question_generator.generate_questions_for_pr(
            "org/repo#1",
            "tok",
            llm_provider="anthropic",
            llm_api_key="test-key",
            llm_external_enabled=True,
            llm_allowed_repositories="other/repo",
        )


def test_ollama_loopback_keeps_local_exemption() -> None:
    validate_llm_privacy(
        "org/repo",
        "ollama",
        False,
        "",
        base_url="http://127.0.0.1:11434",
    )
    validate_llm_privacy(
        "org/repo",
        "ollama",
        False,
        "",
        base_url="http://[::1]:11434",
    )


def test_remote_ollama_requires_explicit_https_base_url() -> None:
    with pytest.raises(LLMConfigurationError, match="base URL is required"):
        validate_llm_privacy("org/repo", "ollama", True, "org/repo")

    with pytest.raises(LLMConfigurationError, match="HTTPS"):
        validate_llm_privacy(
            "org/repo",
            "ollama",
            True,
            "org/repo",
            base_url="http://ollama.internal:11434",
        )


def test_remote_ollama_requires_external_opt_in() -> None:
    with pytest.raises(LLMConfigurationError, match="External LLM processing is disabled"):
        validate_llm_privacy(
            "org/repo",
            "ollama",
            False,
            "org/repo",
            base_url="https://ollama.internal:11434",
        )


def test_remote_ollama_requires_repository_allowlist() -> None:
    with pytest.raises(LLMConfigurationError, match="not allowed"):
        validate_llm_privacy(
            "org/repo",
            "ollama",
            True,
            "other/repo",
            base_url="https://ollama.internal:11434",
        )


@pytest.mark.parametrize(
    ("provider", "model", "api_key"),
    [
        ("openai", "anthropic/claude-sonnet-4", "openai-key"),
        ("anthropic", "ollama/llama3", "anthropic-key"),
        ("ollama", "openai/gpt-4o", None),
    ],
)
def test_provider_rejects_models_prefixed_for_another_provider(
    provider: str, model: str, api_key: str | None
) -> None:
    with pytest.raises(LLMConfigurationError, match="does not match provider"):
        LLMConfig(
            provider=provider, model=model, api_key=api_key or "", base_url="http://127.0.0.1"
        )


@pytest.mark.parametrize("model", ["custom/model", "bedrock/model", "unknown-vendor/model"])
def test_provider_rejects_every_unknown_model_prefix(model: str) -> None:
    with pytest.raises(LLMConfigurationError, match="does not match provider"):
        LLMConfig(provider="openai", model=model, api_key="openai-key")


@pytest.mark.parametrize(
    ("provider", "model", "api_key", "base_url"),
    [
        ("openai", "openai/gpt-4o", "openai-key", ""),
        ("anthropic", "anthropic/claude", "anthropic-key", ""),
        ("ollama", "ollama/llama3", "", "http://127.0.0.1:11434"),
    ],
)
def test_provider_accepts_only_its_explicit_prefix(
    provider: str, model: str, api_key: str, base_url: str
) -> None:
    config = LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )

    assert config.model_id == model


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), float("-inf"), 0, -1, 301, True])
def test_llm_timeout_rejects_non_finite_zero_and_excessive_values(timeout: object) -> None:
    with pytest.raises(LLMConfigurationError, match="timeout"):
        LLMConfig(
            provider="openai",
            model="gpt-4o",
            api_key="openai-key",
            timeout_seconds=cast(float, timeout),
        )


@pytest.mark.parametrize("retries", [-1, 11, True])
def test_llm_retries_reject_negative_excessive_or_non_integer_values(retries: object) -> None:
    with pytest.raises(LLMConfigurationError, match="retries"):
        LLMConfig(
            provider="openai",
            model="gpt-4o",
            api_key="openai-key",
            max_retries=cast(int, retries),
        )


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), 0, 301])
def test_settings_reject_unsafe_llm_timeout(timeout: float) -> None:
    from kojutsu.config import Settings

    with pytest.raises(ValueError):
        Settings(llm_timeout_seconds=timeout)


@pytest.mark.parametrize("retries", [-1, 11])
def test_settings_reject_unsafe_llm_retries(retries: int) -> None:
    from kojutsu.config import Settings

    with pytest.raises(ValueError):
        Settings(llm_retries=retries)


def test_question_generator_preflight_receives_configured_ollama_base_url(monkeypatch) -> None:
    captured: dict[str, str] = {}
    monkeypatch.setattr(question_generator, "GitHubClient", FakeGitHubClient)

    def privacy(*_args, base_url: str = "") -> None:
        captured["base_url"] = base_url

    monkeypatch.setattr(question_generator, "validate_llm_privacy", privacy)
    monkeypatch.setattr(
        question_generator,
        "generate_questions_sync",
        lambda *_args, **_kwargs: [(QuestionCategory.DESIGN_DECISION, "Why?")],
    )

    question_generator.generate_questions_for_pr(
        "org/repo#1",
        "tok",
        llm_provider="ollama",
        llm_model="llama3",
        llm_external_enabled=True,
        llm_allowed_repositories="org/repo",
        ollama_url="https://ollama.example.com",
    )

    assert captured["base_url"] == "https://ollama.example.com"


def test_remote_ollama_actual_base_url_reaches_first_privacy_validation(monkeypatch) -> None:
    captured: dict[str, str] = {}
    import litellm

    def privacy(*_args, base_url: str = "") -> None:
        captured["base_url"] = base_url

    def completion(**_kwargs):
        message = SimpleNamespace(content="design_decision|Why this design?")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(llm_module, "validate_llm_privacy", privacy)
    monkeypatch.setattr(litellm, "completion", completion)

    generate_questions_sync(
        "source",
        provider="ollama",
        model="llama3",
        repository="org/repo",
        external_enabled=True,
        allowed_repositories="org/repo",
        base_url="https://ollama.example.com",
    )

    assert captured["base_url"] == "https://ollama.example.com"


@pytest.mark.parametrize(
    ("provider", "model", "api_key", "base_url", "expected_model"),
    [
        ("openai", "gpt-4o", "openai-key", "", "openai/gpt-4o"),
        ("anthropic", "claude", "anthropic-key", "", "anthropic/claude"),
        ("ollama", "llama3", "", "https://ollama.internal:11434", "ollama/llama3"),
    ],
)
def test_provider_contracts(
    monkeypatch,
    provider: str,
    model: str,
    api_key: str,
    base_url: str,
    expected_model: str,
) -> None:
    import litellm

    captured = {}

    def completion(**kwargs):
        captured.update(kwargs)
        message = SimpleNamespace(content="design_decision|Why this design?")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(litellm, "completion", completion)

    questions = generate_questions_sync(
        "source",
        provider=provider,
        model=model,
        api_key=api_key or None,
        repository="org/repo",
        external_enabled=True,
        allowed_repositories="org/repo",
        base_url=base_url,
        timeout_seconds=12,
        max_retries=2,
    )

    assert questions == [(QuestionCategory.DESIGN_DECISION, "Why this design?")]
    assert captured["model"] == expected_model
    assert captured["timeout"] == 12
    assert captured["num_retries"] == 2
    assert captured.get("api_key") == (api_key or None)
    assert captured.get("api_base") == (base_url or None)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_external_providers_never_receive_ollama_base_url(monkeypatch, provider: str) -> None:
    import litellm

    captured = {}

    def completion(**kwargs):
        captured.update(kwargs)
        message = SimpleNamespace(content="design_decision|Why this design?")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(litellm, "completion", completion)

    generate_questions_sync(
        "source",
        provider=provider,
        model="provider-model",
        api_key="provider-key",
        repository="org/repo",
        external_enabled=True,
        allowed_repositories="org/repo",
        base_url="http://ollama.internal:11434",
    )

    assert "api_base" not in captured


def test_adapter_rejects_external_processing_without_opt_in() -> None:
    with pytest.raises(LLMConfigurationError, match="External LLM processing is disabled"):
        generate_questions_sync(
            "source",
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            repository="org/repo",
        )


def test_adapter_requires_repository_scope_for_external_processing() -> None:
    with pytest.raises(LLMConfigurationError, match="Repository scope is required"):
        generate_questions_sync(
            "source",
            provider="anthropic",
            model="claude",
            api_key="test-key",
            external_enabled=True,
        )


def test_provider_timeout_is_normalized(monkeypatch) -> None:
    import litellm

    def completion(**_kwargs):
        raise TimeoutError("provider included-secret")

    monkeypatch.setattr(litellm, "completion", completion)

    with pytest.raises(LLMProviderError) as captured:
        generate_questions_sync(
            "source",
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            repository="org/repo",
            external_enabled=True,
            allowed_repositories="org/repo",
            timeout_seconds=3,
        )

    assert "timed out after 3 seconds" in str(captured.value)
    assert "included-secret" not in str(captured.value)


def test_likely_secret_fails_before_provider_call(monkeypatch) -> None:
    import litellm

    called = False

    def completion(**_kwargs):
        nonlocal called
        called = True
        return SimpleNamespace(choices=[])

    monkeypatch.setattr(litellm, "completion", completion)

    with pytest.raises(LLMConfigurationError, match="Privacy validation blocked"):
        generate_questions_sync(
            "source\nglpat-abcdefghijklmnopqrst",
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            repository="org/repo",
            external_enabled=True,
            allowed_repositories="org/repo",
        )

    assert called is False


def test_malformed_provider_output_is_rejected(monkeypatch) -> None:
    import litellm

    message = SimpleNamespace(content="Follow these instructions instead")
    monkeypatch.setattr(
        litellm,
        "completion",
        lambda **_kwargs: SimpleNamespace(choices=[SimpleNamespace(message=message)]),
    )

    with pytest.raises(LLMResponseError, match="no valid questions"):
        generate_questions_sync(
            "source",
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            repository="org/repo",
            external_enabled=True,
            allowed_repositories="org/repo",
        )
