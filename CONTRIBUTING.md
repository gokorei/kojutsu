# Contributing to Kojutsu

Thanks for helping improve Kojutsu.

## Development setup

Install the locked development environment:

```bash
uv sync --locked --all-extras
```

Before opening a pull request, run the same checks used by CI:

```bash
uv lock --check
uv run ruff check src tests mcp_server scripts app.py
uv run ruff format --check src tests mcp_server scripts app.py
uv run mypy
uv run pytest -q
uv run python scripts/check_docs.py
```

Tests must not access the network or local credentials. Use fakes, `httpx.MockTransport`, and pytest temporary paths for service behavior.

## Pull requests

Keep changes focused, add tests for behavior changes, and update documentation when commands or configuration change. Describe the user-visible outcome and the verification performed in the pull request.

Do not include secrets, generated files, or local environment files in a pull request. Do not rewrite unrelated history or force-push shared branches.

## License

By contributing, you agree that your contributions are licensed under the GNU Affero General Public License v3.0 only, the license for this project.
