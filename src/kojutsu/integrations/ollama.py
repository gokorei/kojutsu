"""Ollama provider implementation for LLM completion."""

import litellm

from .llm import LLMConfig, complete


def set_url(url: str) -> None:
    """Set the Ollama URL for litellm."""
    vars(litellm)["ollama_url"] = url


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
    """Generate a completion using the configured Ollama endpoint."""
    config = LLMConfig(
        provider="ollama",
        model=model,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
    return complete(prompt, config, max_tokens=max_tokens, system=system)


def get_model_id(provider: str, model: str) -> str:
    """Get the full model ID for Ollama provider."""
    return model if "/" in model else f"ollama/{model}"
