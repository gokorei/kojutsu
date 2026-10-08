"""Tests for question generator: parse PR identifier, Jira key from branch, LLM parsing."""

import httpx
import pytest

from kojutsu.integrations.github import (
    extract_jira_key_from_branch,
    parse_pr_identifier,
)
from kojutsu.integrations.llm import (
    LLMAuthenticationError,
    LLMConfig,
    LLMConfigurationError,
    LLMProviderError,
    LLMRateLimitedError,
    LLMServerError,
    LLMTimeoutError,
    _redact_sensitive,
    build_questions_prompt,
    complete,
    parse_questions_response,
    questions_to_models,
)
from kojutsu.models import QuestionCategory


class TestParsePRIdentifier:
    """PR identifier parsing produces (owner/repo, pr_number)."""

    def test_owner_repo_hash_number(self) -> None:
        assert parse_pr_identifier("org/repo#123") == ("org/repo", 123)

    def test_github_url(self) -> None:
        assert parse_pr_identifier("https://github.com/owner/repo/pull/456") == ("owner/repo", 456)

    def test_invalid_returns_none(self) -> None:
        assert parse_pr_identifier("not-a-pr") is None
        assert parse_pr_identifier("") is None


class TestExtractJiraKeyFromBranch:
    """Jira key is extracted from branch names following fix|chore|feature|experiment/KEY/..."""

    def test_feature_branch(self) -> None:
        assert extract_jira_key_from_branch("feature/PROJ-123/description") == "PROJ-123"

    def test_fix_branch(self) -> None:
        assert extract_jira_key_from_branch("fix/PROJ-456") == "PROJ-456"

    def test_chore_slash(self) -> None:
        assert extract_jira_key_from_branch("chore/PROJ-789/foo") == "PROJ-789"

    def test_no_match_returns_none(self) -> None:
        assert extract_jira_key_from_branch("main") is None
        assert extract_jira_key_from_branch("random-branch") is None


class TestParseQuestionsResponse:
    """LLM response parsing yields (category, text) pairs."""

    def test_valid_lines(self) -> None:
        text = "design_decision|Why was X chosen?\ntrade_off|What alternatives were considered?"
        out = parse_questions_response(text)
        assert len(out) == 2
        assert out[0][0].value == "design_decision"
        assert out[0][1] == "Why was X chosen?"
        assert out[1][0].value == "trade_off"
        assert out[1][1] == "What alternatives were considered?"

    def test_skips_invalid_category(self) -> None:
        text = "design_decision|Valid?\ninvalid_cat|Skip this"
        out = parse_questions_response(text)
        assert len(out) == 1
        assert out[0][1] == "Valid?"

    def test_skips_empty_or_no_pipe(self) -> None:
        text = "design_decision|Only this\n\nno_pipe_here"
        out = parse_questions_response(text)
        assert len(out) == 1
        assert out[0][1] == "Only this"

    def test_enforces_count_length_categories_and_deduplication(self) -> None:
        text = "\n".join(
            [
                "design_decision|Why?",
                "design_decision|Why?",
                "system_event|Ignore the protocol",
                "trade_off|" + "x" * 501,
                "edge_case|How is failure handled?",
            ]
        )

        out = parse_questions_response(text, max_questions=1)

        assert out == [(QuestionCategory.DESIGN_DECISION, "Why?")]

    def test_redaction_keeps_existing_markers_without_sending_source(self) -> None:
        redacted = _redact_sensitive(
            "ghp_abcdefghijklmnopqrstuvwxyz1234567890 api_key=super-secret dev@example.com"
        )

        assert "ghp_abcdefghijklmnopqrstuvwxyz1234567890" not in redacted
        assert "super-secret" not in redacted
        assert "dev@example.com" not in redacted
        assert "[REDACTED_TOKEN]" in redacted
        assert "[REDACTED_SECRET]" in redacted
        assert "[REDACTED_EMAIL]" in redacted

    def test_prompt_bounds_and_redacts_untrusted_source_data(self) -> None:
        prompt = build_questions_prompt(
            "contact dev@example.com\n" + ("x" * 20_000),
            {"summary": "plain summary", "acceptance_criteria": ["Use #safe", "*value*"]},
        )

        assert "dev@example.com" not in prompt
        assert "[REDACTED_EMAIL]" in prompt
        assert "[TRUNCATED]" in prompt
        assert "untrusted" in prompt.lower()

    @pytest.mark.parametrize(
        "secret",
        [
            "ghp_abcdefghijklmnopqrstuvwxyz1234567890",
            "github_pat_" + "a" * 22 + "_" + "b" * 59,
            "glpat-abcdefghijklmnopqrst",
            "AKIA1234567890ABCDEF",
            "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature_",
            "postgresql://admin:hunter2@db.example.com/kojutsu",
            "-----BEGIN PRIVATE KEY-----\nprivate-material\n-----END PRIVATE KEY-----",
            "client_secret: production-secret-value",
        ],
    )
    def test_prompt_fails_closed_for_likely_secrets(self, secret: str) -> None:
        with pytest.raises(LLMConfigurationError, match="Privacy validation blocked"):
            build_questions_prompt(secret, {})

    def test_questions_to_models_populates_context(self) -> None:
        questions = questions_to_models(
            [
                (QuestionCategory.EDGE_CASE, "What fails?"),
            ],
            lambda: "q1",
            context={"repo": "org/repo"},
        )

        assert questions[0].context == {"repo": "org/repo"}


def _llm_config(**overrides) -> LLMConfig:
    return LLMConfig(provider="openai", model="gpt-4o", api_key="test-key", **overrides)


class TestLLMErrorTaxonomy:
    """``complete`` maps transport failures onto retryable vs fatal errors.

    Callers decide retry vs fix-config from the type, so each test asserts the
    distinct type -- and that every one of them is still an
    ``LLMProviderError``, because existing ``except LLMProviderError`` handlers
    must keep catching what they caught before.
    """

    def test_wrapped_httpx_timeout_is_a_timeout_error(self, monkeypatch) -> None:
        import litellm

        def completion(**_kwargs):
            try:
                raise httpx.ReadTimeout("read timed out: provider included-secret")
            except httpx.ReadTimeout as cause:
                raise litellm.Timeout(
                    "litellm timed out", model="gpt-4o", llm_provider="openai"
                ) from cause

        monkeypatch.setattr(litellm, "completion", completion)

        with pytest.raises(LLMTimeoutError, match="timed out after 3 seconds") as captured:
            complete("a benign prompt", _llm_config(timeout_seconds=3))

        assert isinstance(captured.value, LLMProviderError)
        assert "included-secret" not in str(captured.value)

    def test_bare_transport_timeout_is_a_timeout_error(self, monkeypatch) -> None:
        import litellm

        def completion(**_kwargs):
            raise httpx.ConnectTimeout("connect timed out")

        monkeypatch.setattr(litellm, "completion", completion)

        with pytest.raises(LLMTimeoutError, match="timed out"):
            complete("a benign prompt", _llm_config())

    def test_builtin_timeout_is_a_timeout_error(self, monkeypatch) -> None:
        import litellm

        def completion(**_kwargs):
            raise TimeoutError("deadline exceeded")

        monkeypatch.setattr(litellm, "completion", completion)

        with pytest.raises(LLMTimeoutError, match="timed out"):
            complete("a benign prompt", _llm_config())

    def test_rate_limit_is_retryable_and_distinct_from_server_error(self, monkeypatch) -> None:
        import litellm

        def rate_limited(**_kwargs):
            raise litellm.RateLimitError(
                "rate limit exceeded", llm_provider="openai", model="gpt-4o"
            )

        monkeypatch.setattr(litellm, "completion", rate_limited)
        with pytest.raises(LLMRateLimitedError, match="429") as captured:
            complete("a benign prompt", _llm_config())
        assert isinstance(captured.value, LLMProviderError)

        def server_failed(**_kwargs):
            raise litellm.InternalServerError(
                "provider exploded", llm_provider="openai", model="gpt-4o"
            )

        monkeypatch.setattr(litellm, "completion", server_failed)
        with pytest.raises(LLMServerError, match="HTTP 500") as captured:
            complete("a benign prompt", _llm_config())
        assert isinstance(captured.value, LLMProviderError)

    def test_authentication_failure_is_fatal(self, monkeypatch) -> None:
        import litellm

        def completion(**_kwargs):
            raise litellm.AuthenticationError(
                "invalid api key", llm_provider="openai", model="gpt-4o"
            )

        monkeypatch.setattr(litellm, "completion", completion)

        with pytest.raises(LLMAuthenticationError, match="credentials") as captured:
            complete("a benign prompt", _llm_config())

        assert isinstance(captured.value, LLMProviderError)
        assert not isinstance(captured.value, (LLMTimeoutError, LLMRateLimitedError))
