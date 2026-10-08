"""Pytest fixtures and configuration."""

import socket
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import httpx
import pytest

from kojutsu import runtime as runtime_module
from kojutsu.core.outbox import TansekiOutbox
from kojutsu.core.question_registry import SqliteQuestionRegistry

BEHAVIOR_ENV_VARS = (
    "KOJUTSU_CONFIG",
    "KOJUTSU_ENV_FILE",
    "KOJUTSU_INSTANCE",
    "KOJUTSU_REGISTRY_PATH",
    "KOJUTSU_RELAY_INTERVAL_SECONDS",
    "KOJUTSU_SQLITE_REPLICAS",
    "KOJUTSU_SQLITE_WORKERS",
    "DEV_CONSOLE_TOKEN",
    "DESIGN_PLAN_APPROVAL_LEDGER_PATH",
    "DESIGN_TICKET_SINK_PATH",
    "GITHUB_TOKEN",
    "GITHUB_WEBHOOK_ALLOWED_REPOSITORIES",
    "GITHUB_WEBHOOK_CLEANUP",
    "GITHUB_WEBHOOK_REGISTER",
    "GITHUB_WEBHOOK_REPOS",
    "GITHUB_WEBHOOK_SECRET",
    "JIRA_API_TOKEN",
    "JIRA_URL",
    "JIRA_USERNAME",
    "LITELLM_LOCAL_MODEL_COST_MAP",
    "LLM_ALLOWED_REPOSITORIES",
    "LLM_API_KEY",
    "LLM_EXTERNAL_ENABLED",
    "LLM_MODEL",
    "LLM_PROVIDER",
    "LLM_RETRIES",
    "LLM_TIMEOUT_SECONDS",
    "OLLAMA_URL",
    "TANSEKI_API_KEY",
    "TANSEKI_COLLECTION",
    "TANSEKI_OUTBOX_PATH",
    "TANSEKI_TIMEOUT_SECONDS",
    "TANSEKI_URL",
    "PORT",
    "READ_LOG_ENABLED",
    "READ_LOG_MAX_AGE_DAYS",
    "READ_LOG_MAX_ENTRIES",
    "READ_LOG_PATH",
    "WEBHOOK_HOST",
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("filterwarnings", "error::ResourceWarning")
    config.addinivalue_line("filterwarnings", "error::pytest.PytestUnraisableExceptionWarning")


def _close_resource(resource: object) -> Exception | None:
    close = getattr(resource, "close", None)
    if not callable(close):
        return None
    try:
        close()
    except Exception as exc:
        return exc
    return None


def _close_test_resources(resources: list[object]) -> None:
    runtime = runtime_module._runtime
    runtime_module._runtime = None
    first_error: Exception | None = None
    closed_ids: set[int] = set()

    if runtime is not None:
        runtime_closes_resources = isinstance(runtime, runtime_module.Runtime)
        error = _close_resource(runtime)
        if error is not None and first_error is None:
            first_error = error
        for name in ("outbox", "registry", "client"):
            resource = getattr(runtime, name, None)
            if resource is None:
                continue
            if runtime_closes_resources:
                closed_ids.add(id(resource))
                continue
            error = _close_resource(resource)
            if error is not None and first_error is None:
                first_error = error

    for resource in reversed(resources):
        if id(resource) in closed_ids:
            continue
        closed_ids.add(id(resource))
        error = _close_resource(resource)
        if error is not None and first_error is None:
            first_error = error

    if first_error is not None:
        raise first_error


@pytest.fixture(autouse=True)
def env_isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Generator[None, None, None]:
    """Isolate tests from local configuration, credentials, state, and network."""
    for key in BEHAVIOR_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "true")
    monkeypatch.setenv("KOJUTSU_ENV_FILE", "")
    # Opts out of config-file discovery entirely. `kojutsu.toml` is meant to be
    # committed, so it exists in the repository root, and the suite runs with that
    # root as its cwd -- so without this every test would read whatever the
    # developer last committed.
    monkeypatch.setenv("KOJUTSU_CONFIG", "")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("TANSEKI_OUTBOX_PATH", str(tmp_path / "tanseki-outbox.db"))
    monkeypatch.setenv("KOJUTSU_REGISTRY_PATH", str(tmp_path / "registry.db"))

    def deny_socket_connect(_socket: socket.socket, address: object) -> None:
        raise RuntimeError(f"Unexpected network access in tests: {address!r}")

    def deny_create_connection(address: object, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError(f"Unexpected network access in tests: {address!r}")

    monkeypatch.setattr(socket.socket, "connect", deny_socket_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_socket_connect)
    monkeypatch.setattr(socket, "create_connection", deny_create_connection)
    yield


@pytest.fixture(autouse=True)
def sqlite_resources(
    env_isolate: None, monkeypatch: pytest.MonkeyPatch
) -> Generator[None, None, None]:
    resources: list[object] = []
    registry_init = SqliteQuestionRegistry.__init__
    outbox_init = TansekiOutbox.__init__

    def tracked_registry_init(instance: SqliteQuestionRegistry, *args: Any, **kwargs: Any) -> None:
        registry_init(instance, *args, **kwargs)
        resources.append(instance)

    def tracked_outbox_init(instance: TansekiOutbox, *args: Any, **kwargs: Any) -> None:
        outbox_init(instance, *args, **kwargs)
        resources.append(instance)

    monkeypatch.setattr(SqliteQuestionRegistry, "__init__", tracked_registry_init)
    monkeypatch.setattr(TansekiOutbox, "__init__", tracked_outbox_init)
    yield
    _close_test_resources(resources)


#: Shared factories, so per-file hand-rolled fakes converge here instead of
#: drifting apart. Prefer these over local ``SqliteQuestionRegistry(...)`` /
#: ``TansekiOutbox(...)`` constructions and over bespoke ``MockTransport``
#: handlers: one definition, many callers, is the point. ``httpx.MockTransport``
#: is used directly (no respx dependency) because the suite only needs
#: in-process request routing, not a separate assertion library -- adding respx
#: would trade one fake for a heavier one without covering more branches.


@pytest.fixture
def registry_factory(tmp_path: Path) -> Callable[..., SqliteQuestionRegistry]:
    """Build isolated registries: ``registry_factory("r.db")``."""

    def _make(name: str = "registry.db") -> SqliteQuestionRegistry:
        return SqliteQuestionRegistry(tmp_path / name)

    return _make


@pytest.fixture
def registry(registry_factory: Callable[..., SqliteQuestionRegistry]) -> SqliteQuestionRegistry:
    """One isolated registry per test (auto-closed by ``sqlite_resources``)."""
    return registry_factory()


@pytest.fixture
def outbox_factory(tmp_path: Path) -> Callable[..., TansekiOutbox]:
    """Build isolated outboxes: ``outbox_factory("o.db")``."""

    def _make(name: str = "outbox.db") -> TansekiOutbox:
        return TansekiOutbox(tmp_path / name)

    return _make


@pytest.fixture
def outbox(outbox_factory: Callable[..., TansekiOutbox]) -> TansekiOutbox:
    """One isolated outbox per test (auto-closed by ``sqlite_resources``)."""
    return outbox_factory()


def make_mock_transport(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.MockTransport:
    """Shared ``httpx.MockTransport`` factory: route requests via ``handler``."""
    return httpx.MockTransport(handler)


@pytest.fixture
def mock_transport_factory() -> Callable[
    [Callable[[httpx.Request], httpx.Response]], httpx.MockTransport
]:
    """Fixture form of :func:`make_mock_transport` for tests that prefer injection."""
    return make_mock_transport


def make_mock_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.Client:
    """One ``httpx.Client`` over a mock transport (close when done)."""
    return httpx.Client(transport=make_mock_transport(handler))
