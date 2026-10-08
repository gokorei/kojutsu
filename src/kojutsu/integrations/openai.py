"""OpenAI provider implementation for LLM completion."""

import litellm

from .llm import LLMConfig, complete


def set_api_key(api_key: str) -> None:
    """Set the OpenAI API key for litellm."""
    litellm.openai_key = api_key


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
    """Generate a completion using normalized OpenAI configuration."""
    config = LLMConfig(
        provider="openai",
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
    return complete(prompt, config, max_tokens=max_tokens, system=system)


def get_model_id(provider: str, model: str) -> str:
    """Get the full model ID for OpenAI provider."""
    return model if "/" in model else f"openai/{model}"
