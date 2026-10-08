"""Tests for the runtime composition."""

from __future__ import annotations

import asyncio
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

import pytest

from kojutsu import relay_worker
from kojutsu import runtime as runtime_module
from kojutsu.config import Settings
from kojutsu.core.outbox import RelayResult
from kojutsu.runtime import Runtime, build_runtime, get_runtime, reset_runtime


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        tanseki_url="https://tanseki.test",
        tanseki_outbox_path=str(tmp_path / "outbox.db"),
        kojutsu_registry_path=str(tmp_path / "registry.db"),
    )


def test_settings_ignore_local_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("TANSEKI_URL=http://from-dotenv.test\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KOJUTSU_ENV_FILE", "")

    assert Settings().tanseki_url == ""


def test_build_runtime_requires_tanseki(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_runtime(
            Settings(
                tanseki_url="",
                tanseki_outbox_path=str(tmp_path / "o.db"),
                kojutsu_registry_path=str(tmp_path / "r.db"),
            )
        )


def test_build_runtime_rejects_multi_process_sqlite_topology(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.kojutsu_sqlite_replicas = 2
    with pytest.raises(ValueError, match="one process"):
        build_runtime(settings)


def test_build_runtime_wires_dependencies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with build_runtime(make_settings(tmp_path)) as runtime:
        monkeypatch.setattr(runtime.client, "health", lambda: False)
        assert runtime.outbox.pending_count() == 0
        assert runtime.registry is not None
        status = runtime.status()
        assert status["tanseki_url"] == "https://tanseki.test"
        assert status["outbox_pending"] == 0
        assert status["outbox_captured_locally"] == 0
        assert status["outbox_delivery_failed"] == 0


def test_get_runtime_caches_and_reset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_module, "_runtime", None)
    monkeypatch.setenv("TANSEKI_URL", "https://tanseki.test")
    monkeypatch.setenv("TANSEKI_OUTBOX_PATH", str(tmp_path / "o.db"))
    monkeypatch.setenv("KOJUTSU_REGISTRY_PATH", str(tmp_path / "r.db"))

    first = get_runtime()
    assert get_runtime() is first

    reset_runtime()
    assert runtime_module._runtime is None


def test_runtime_close_attempts_every_resource() -> None:
    closed: list[str] = []

    class Resource:
        def __init__(self, name: str, *, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        def close(self) -> None:
            closed.append(self.name)
            if self.fail:
                raise RuntimeError(f"close {self.name}")

    runtime = Runtime(
        settings=Settings(),
        client=cast(Any, Resource("client", fail=True)),
        outbox=cast(Any, Resource("outbox", fail=True)),
        sink=cast(Any, object()),
        registry=cast(Any, Resource("registry")),
    )

    with pytest.raises(RuntimeError, match="close outbox"):
        runtime.close()

    assert closed == ["outbox", "registry", "client"]
    runtime.close()
    assert closed == ["outbox", "registry", "client"]


def test_reset_runtime_clears_cache_when_close_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingRuntime:
        def close(self) -> None:
            raise RuntimeError("close failed")

    monkeypatch.setattr(runtime_module, "_runtime", cast(Any, FailingRuntime()))

    with pytest.raises(RuntimeError, match="close failed"):
        reset_runtime()

    assert runtime_module._runtime is None


def test_reset_runtime_restores_cached_runtime_after_close_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TimeoutRuntime:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise TimeoutError("still active")

    runtime = TimeoutRuntime()
    monkeypatch.setattr(runtime_module, "_runtime", cast(Any, runtime))

    with pytest.raises(TimeoutError, match="still active"):
        reset_runtime()
    assert runtime_module._runtime is runtime

    reset_runtime()
    assert runtime_module._runtime is None


def test_build_runtime_closes_created_resources_on_composition_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    closed: list[str] = []

    class Client:
        def close(self) -> None:
            closed.append("client")

    class Outbox:
        def __init__(self, _path: str) -> None:
            pass

        def close(self) -> None:
            closed.append("outbox")

    def fail_registry(_path: str) -> object:
        raise RuntimeError("registry failed")

    monkeypatch.setattr(runtime_module.TansekiClient, "from_settings", lambda _settings: Client())
    monkeypatch.setattr(runtime_module, "TansekiOutbox", Outbox)
    monkeypatch.setattr(runtime_module, "SqliteQuestionRegistry", fail_registry)

    with pytest.raises(RuntimeError, match="registry failed"):
        build_runtime(make_settings(tmp_path))

    assert closed == ["outbox", "client"]


def test_runtime_maintenance_preserves_dead_letters_unless_explicitly_requested() -> None:
    registry_cleanup_calls: list[int] = []
    dead_letter_cleanup_calls: list[int] = []

    class Registry:
        def cleanup_deliveries(self) -> int:
            registry_cleanup_calls.append(1)
            return 2

    class Outbox:
        def cleanup_dead_letters(self) -> int:
            dead_letter_cleanup_calls.append(1)
            return 1

        def status_counts(self) -> dict[str, int]:
            return {"pending": 0, "retrying": 0, "dead_letter": 3}

    runtime = Runtime(
        settings=Settings(),
        client=cast(Any, object()),
        outbox=cast(Any, Outbox()),
        sink=cast(Any, object()),
        registry=cast(Any, Registry()),
    )

    default_result = runtime.maintenance()
    explicit_result = runtime.maintenance(cleanup_dead_letters=True)

    assert default_result.registry_deliveries_deleted == 2
    assert default_result.outbox_dead_letters_deleted == 0
    assert default_result.outbox_dead_letters_remaining == 3
    assert explicit_result.outbox_dead_letters_deleted == 1
    assert registry_cleanup_calls == [1, 1]
    assert dead_letter_cleanup_calls == [1]


def test_runtime_close_waits_for_in_flight_relay(monkeypatch: pytest.MonkeyPatch) -> None:
    started = threading.Event()
    release = threading.Event()
    close_started = threading.Event()

    class Resource:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    client = Resource()
    outbox = Resource()

    def slow_relay(_outbox: object, _client: object, *, limit: int = 100) -> RelayResult:
        started.set()
        assert release.wait(timeout=2)
        return RelayResult(sent=0, failed=0)

    monkeypatch.setattr(runtime_module, "relay", slow_relay)
    runtime = Runtime(
        settings=Settings(),
        client=cast(Any, client),
        outbox=cast(Any, outbox),
        sink=cast(Any, object()),
        registry=cast(Any, object()),
    )
    delivery_errors: list[BaseException] = []
    close_errors: list[BaseException] = []

    def deliver() -> None:
        try:
            runtime.relay()
        except BaseException as exc:
            delivery_errors.append(exc)

    def close_runtime() -> None:
        close_started.set()
        try:
            runtime.close(timeout=1)
        except BaseException as exc:
            close_errors.append(exc)

    delivery_thread = threading.Thread(target=deliver)
    delivery_thread.start()
    assert started.wait(timeout=2)
    close_thread = threading.Thread(target=close_runtime)
    close_thread.start()
    assert close_started.wait(timeout=2)
    close_thread.join(timeout=0.05)
    assert close_thread.is_alive()
    assert not client.closed
    assert not outbox.closed
    release.set()
    delivery_thread.join(timeout=2)
    close_thread.join(timeout=2)
    assert not delivery_thread.is_alive()
    assert not close_thread.is_alive()
    assert delivery_errors == []
    assert close_errors == []
    assert client.closed
    assert outbox.closed


def test_runtime_close_timeout_leaves_resources_open(monkeypatch: pytest.MonkeyPatch) -> None:
    started = threading.Event()
    release = threading.Event()

    class Resource:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    client = Resource()
    outbox = Resource()

    def slow_relay(_outbox: object, _client: object, *, limit: int = 100) -> RelayResult:
        started.set()
        assert release.wait(timeout=2)
        return RelayResult(sent=0, failed=0)

    monkeypatch.setattr(runtime_module, "relay", slow_relay)
    runtime = Runtime(
        settings=Settings(),
        client=cast(Any, client),
        outbox=cast(Any, outbox),
        sink=cast(Any, object()),
        registry=cast(Any, object()),
    )
    delivery_thread = threading.Thread(target=runtime.relay)
    delivery_thread.start()
    assert started.wait(timeout=2)
    with pytest.raises(TimeoutError):
        runtime.close(timeout=0)
    assert runtime._closing is False
    assert not client.closed
    assert not outbox.closed
    release.set()
    delivery_thread.join(timeout=2)
    assert runtime.maintenance().total_deleted == 0
    runtime.close()
    assert client.closed
    assert outbox.closed


@pytest.mark.asyncio
async def test_wait_for_relay_shutdown_after_cancelled_relay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    class FakeRuntime:
        def relay(self) -> RelayResult:
            started.set()
            assert release.wait(timeout=2)
            return RelayResult(sent=0, failed=0)

    monkeypatch.setattr(runtime_module, "get_runtime", lambda: FakeRuntime())
    task = asyncio.create_task(relay_worker.relay_once())
    while not started.is_set():  # noqa: ASYNC110 - test spin on a cross-thread flag set ~instantly
        await asyncio.sleep(0)
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    assert not await relay_worker.wait_for_relay_shutdown(timeout_seconds=0)
    release.set()
    assert await relay_worker.wait_for_relay_shutdown(timeout_seconds=1)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "not-a-number"])
def test_relay_interval_rejects_invalid_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("KOJUTSU_RELAY_INTERVAL_SECONDS", value)
    with pytest.raises(ValueError, match="finite positive"):
        relay_worker.resolve_relay_interval()


def test_relay_interval_accepts_positive_finite_value() -> None:
    assert relay_worker.resolve_relay_interval(0.25) == 0.25
