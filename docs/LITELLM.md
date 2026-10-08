# Litellm Integration Guide

This document explains how the Kojutsu project integrates with litellm for LLM-powered question generation.

## Overview

Litellm is used as a unified interface to multiple LLM providers, allowing the application to generate context-aware questions about code changes and Jira ticket context.

## Supported Providers

### Currently Supported
- **OpenAI**: Uses models like `gpt-4o`
- **Anthropic**: Uses models like `claude-3-5-sonnet`
- **Ollama**: Local LLM server (default URL: `http://localhost:11434`)

### Configuration

```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
LLM_API_KEY=your-api-key-here
LLM_EXTERNAL_ENABLED=true
LLM_ALLOWED_REPOSITORIES=org/repo
LLM_TIMEOUT_SECONDS=30
LLM_RETRIES=1
OLLAMA_URL=http://localhost:11434
```

OpenAI and Anthropic fail closed unless external processing is enabled and the exact
repository is allowlisted. Ollama is local and does not require that opt-in.
Review the selected provider's data-processing terms before enabling external
processing. Kojutsu redacts common credentials and email addresses, bounds
diff and Jira fields, and never sends Jira credentials to the LLM provider.

## Docker networking

The `localhost` Ollama URL above is for non-Docker local runs. When Kojutsu
runs in Docker and Ollama runs on the host, use
`OLLAMA_URL=http://host.docker.internal:11434` instead. Docker Desktop provides
that hostname on macOS. On Linux, add
`--add-host=host.docker.internal:host-gateway` to the Kojutsu container. If
both run on the same Docker network, use the Ollama service DNS name, such as
`OLLAMA_URL=http://ollama:11434`, and do not use `localhost` from the container.

## Integration Architecture

`LLMConfig` in `src/kojutsu/integrations/llm.py` normalizes provider, model,
credential, base URL, timeout, and retry settings. The provider modules are thin
adapters over the same bounded LiteLLM completion path:

- **OpenAI**: `src/kojutsu/integrations/openai.py`
- **Anthropic**: `src/kojutsu/integrations/anthropic.py`
- **Ollama**: `src/kojutsu/integrations/ollama.py`

Model names without a provider prefix are normalized as `gpt-4o` →
`openai/gpt-4o`, `claude` → `anthropic/claude`, and `llama3` → `ollama/llama3`.
The configured `OLLAMA_URL` is passed as the Ollama API base.

## Prompt and Output Controls

The system role treats PR and Jira content as untrusted source data. Context is
redacted and bounded before it reaches the user role. Responses are accepted
only as `CATEGORY|question` lines with allowed categories, at most six unique
questions, and at most 500 characters per question.

## Error Handling

- Invalid privacy, credential, model, timeout, or retry settings raise
  `LLMConfigurationError`.
- Provider failures are normalized to actionable `LLMProviderError` messages
  without exposing raw exception details.
- Empty, oversized, or protocol-invalid output raises `LLMResponseError`.


## Testing Considerations

- Tests isolate external environment variables
- Uses monkeypatch for environment isolation
- Unsets LLM-related environment variables by default

## Adding New Providers

To add support for a new LLM provider:

1. Create a new provider file in `src/kojutsu/integrations/`
2. Add the provider to the orchestrator in `llm.py`
3. Update configuration validation if needed
4. Test with the new provider

## Dependencies

```toml
litellm>=1.81.13
```

## Best Practices

- Use environment variables for sensitive API keys
- Test with different providers to ensure compatibility
- Monitor token usage and costs
- Handle API rate limits appropriately
- Validate model responses before processing