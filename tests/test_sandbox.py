"""Tests for process isolation around the opencode question generator.

The prompt contains untrusted pull request text, so what matters is not whether the
model behaves but what it can reach if it does not. These tests pin the reachability
boundary itself. The macOS sandbox test executes real binaries under the real
profile rather than asserting on the profile's text, because a profile that looks
right and denies nothing is the failure mode worth catching.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import stat
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from kojutsu.integrations import sandbox


@pytest.fixture
def auth_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake opencode credential store with two providers in it."""
    home = tmp_path / "operator-home"
    auth = home / sandbox.OPENCODE_AUTH_PATH
    auth.parent.mkdir(parents=True)
    auth.write_text(
        json.dumps(
            {
                "opencode-go": {"type": "api", "key": "model-key"},
                "other-provider": {"type": "api", "key": "unrelated-secret"},
            }
        )
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return auth


# --- environment ------------------------------------------------------------


def test_child_environment_drops_every_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """The allowlist is the layer Seatbelt cannot provide: it does not filter environ.

    A repository token present in the capture process must not be inherited by a
    process that reads attacker-influenced text.
    """
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
    monkeypatch.setenv("TANSEKI_API_KEY", "tanseki-secret")
    monkeypatch.setenv("JIRA_API_TOKEN", "jira-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    monkeypatch.setenv("PATH", "/usr/bin")

    env = sandbox._minimal_env(home=Path("/scratch/home"), root=Path("/scratch"))

    assert "GITHUB_TOKEN" not in env
    assert "TANSEKI_API_KEY" not in env
    assert "JIRA_API_TOKEN" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert env["PATH"] == "/usr/bin"
    assert env["HOME"] == "/scratch/home"


def test_child_environment_ignores_unlisted_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allowlist, not denylist: a credential nobody thought of is still dropped."""
    monkeypatch.setenv("SOME_FUTURE_CREDENTIAL", "leak")

    env = sandbox._minimal_env(home=Path("/h"), root=Path("/r"))

    assert "SOME_FUTURE_CREDENTIAL" not in env


# --- private home -----------------------------------------------------------


def test_provisioned_home_holds_only_the_agent_and_one_credential(
    auth_store: Path, tmp_path: Path
) -> None:
    box = sandbox.provision(tmp_path / "box", fresh=True, model="opencode-go/m1")

    agent = box.home / sandbox.OPENCODE_AGENT_DIR / sandbox.AGENT_FILENAME
    assert agent.is_file()
    assert "tools: {}" in agent.read_text()
    assert "model: opencode-go/m1" in agent.read_text()

    stored = json.loads((box.home / sandbox.OPENCODE_AUTH_PATH).read_text())
    assert set(stored) == {"opencode-go"}
    assert "unrelated-secret" not in json.dumps(stored)


def test_provisioned_credentials_are_not_world_readable(auth_store: Path, tmp_path: Path) -> None:
    box = sandbox.provision(tmp_path / "box", fresh=True, model="opencode-go/m1")

    auth = box.home / sandbox.OPENCODE_AUTH_PATH
    assert not auth.stat().st_mode & (stat.S_IRGRP | stat.S_IROTH)


def test_workdir_starts_empty(auth_store: Path, tmp_path: Path) -> None:
    """No repository may be the child's default workspace."""
    box = sandbox.provision(tmp_path / "box", fresh=True, model="opencode-go/m1")

    assert list(box.workdir.iterdir()) == []


def test_missing_credential_names_the_available_providers(
    auth_store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(sandbox.SandboxError, match="Available: opencode-go, other-provider"):
        sandbox.provision(tmp_path / "box", fresh=True, model="opencode/unknown")


def test_absent_credential_store_is_a_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "empty"))

    with pytest.raises(sandbox.SandboxError, match="opencode providers"):
        sandbox.provision(tmp_path / "box", fresh=True, model="opencode-go/m1")


def test_previous_run_cannot_leak_into_the_next(auth_store: Path, tmp_path: Path) -> None:
    root = tmp_path / "box"
    first = sandbox.provision(root, fresh=True, model="opencode-go/m1")
    (first.workdir / "stale.txt").write_text("from a previous run")

    second = sandbox.provision(root, fresh=True, model="opencode-go/m1")

    assert list(second.workdir.iterdir()) == []


# --- profile ----------------------------------------------------------------


def test_profile_denies_the_repository_and_credential_stores(
    auth_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = Path("/private/tmp/box/home")
    profile = sandbox._seatbelt_profile(home=home, workdir=Path("/private/tmp/box/work"))
    home_root = str(Path.home())

    assert f'(deny file-read*\n  (subpath "{home_root}/Documents")' in profile
    assert f'(subpath "{home_root}/.ssh")' in profile
    assert f'(subpath "{home_root}/.aws")' in profile
    assert f'(subpath "{home_root}/.config/opencode")' in profile
    assert f'(subpath "{home_root}/.local/share/opencode")' in profile
    assert f'(subpath "{home_root}/.config/gh")' in profile
    assert "(deny file-write*" in profile


def test_wrap_command_is_a_no_op_without_a_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "sandbox_available", lambda: False)
    box = sandbox.Sandbox(
        home=Path("/h"), workdir=Path("/w"), root=Path("/r"), profile_path=None, env={}
    )

    assert sandbox.wrap_command(box, ["opencode", "run"]) == ["opencode", "run"]


# --- the boundary, executed -------------------------------------------------


@pytest.mark.skipif(not sandbox.sandbox_available(), reason="requires macOS Seatbelt")
def test_sandboxed_process_cannot_read_the_repository_or_secrets(
    auth_store: Path, tmp_path: Path
) -> None:
    """Execute real binaries under the real profile and check what is reachable.

    Asserting on profile text would not catch a profile that denies nothing. This
    runs the same reads a prompt injection would attempt.
    """
    # Plant the things an injection would go after, inside the operator's home.
    operator_home = Path.home()
    secret_dir = operator_home / ".ssh"
    secret_dir.mkdir(parents=True, exist_ok=True)
    secret = secret_dir / "id_rsa_probe"
    secret.write_text("PRIVATE-KEY-MATERIAL")
    repo_file = operator_home / "Documents" / "kojutsu-probe-file.txt"
    repo_file.parent.mkdir(parents=True, exist_ok=True)
    repo_file.write_text("REPOSITORY-SOURCE")
    try:
        box = sandbox.provision(tmp_path / "box", fresh=True, model="opencode-go/m1")
        reads = [
            ("ssh key", str(secret)),
            ("operator repository file", str(repo_file)),
        ]
        for label, target in reads:
            result = sandbox.run(box, ["/bin/cat", target], timeout_seconds=30)
            combined = (result.stdout + result.stderr).strip()
            assert "PRIVATE-KEY-MATERIAL" not in combined, f"{label} was readable"
            assert "REPOSITORY-SOURCE" not in combined, f"{label} was readable"
            assert combined, f"{label} produced no output, so the probe proved nothing"

        # The scratch home must remain readable, or the model call cannot work.
        allowed = sandbox.run(
            box, ["/bin/cat", str(box.home / sandbox.OPENCODE_AUTH_PATH)], timeout_seconds=30
        )
        assert "model-key" in allowed.stdout
    finally:
        secret.unlink(missing_ok=True)
        repo_file.unlink(missing_ok=True)
        for directory in (secret_dir, repo_file.parent):
            with contextlib.suppress(OSError):
                directory.rmdir()


@pytest.mark.skipif(not sandbox.sandbox_available(), reason="requires macOS Seatbelt")
def test_sandboxed_process_cannot_write_outside_the_scratch(
    auth_store: Path, tmp_path: Path
) -> None:
    operator_home = Path.home()
    target = operator_home / "Documents" / "kojutsu-probe-write.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    try:
        box = sandbox.provision(tmp_path / "box", fresh=True, model="opencode-go/m1")
        sandbox.run(box, ["/usr/bin/touch", str(target)], timeout_seconds=30)

        assert not target.exists()
    finally:
        target.unlink(missing_ok=True)


@pytest.mark.skipif(not sandbox.sandbox_available(), reason="requires macOS Seatbelt")
def test_run_passes_the_scrubbed_environment(auth_store: Path, tmp_path: Path) -> None:
    """The sandbox does not filter the environment; run() must."""
    os.environ["GITHUB_TOKEN"] = "ghp_leak"
    try:
        box = sandbox.provision(tmp_path / "box", fresh=True, model="opencode-go/m1")
        result = sandbox.run(
            box, ["/bin/sh", "-c", "printenv GITHUB_TOKEN || true"], timeout_seconds=30
        )

        assert result.stdout.strip() == ""
    finally:
        os.environ.pop("GITHUB_TOKEN", None)


def test_run_reports_a_nonzero_exit_without_raising(auth_store: Path, tmp_path: Path) -> None:
    box = sandbox.provision(tmp_path / "box", fresh=True, model="opencode-go/m1")

    result = sandbox.run(box, ["/bin/sh", "-c", "exit 3"], timeout_seconds=30)

    assert result.returncode == 3
    assert isinstance(result.stdout, str)


def test_provisioning_rejects_a_non_dict_credential_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "operator-home"
    auth = home / sandbox.OPENCODE_AUTH_PATH
    auth.parent.mkdir(parents=True)
    auth.write_text("[]")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    with pytest.raises(sandbox.SandboxError, match="Unexpected opencode credential format"):
        sandbox.provision(tmp_path / "box", fresh=True, model="opencode-go/m1")


# --- per-principal isolation ------------------------------------------------
#
# Everything above pins what a single sandbox denies. What follows pins that two
# principals never share one, because a home that is shared is still enforcing a
# profile while handing the wrong credential to the model, and that failure is
# silent.


def test_each_model_gets_its_own_root_and_credential(auth_store: Path, tmp_path: Path) -> None:
    first = sandbox.provision(
        sandbox.root_for_model("opencode-go/m1", base=tmp_path), model="opencode-go/m1"
    )
    second = sandbox.provision(
        sandbox.root_for_model("other-provider/m2", base=tmp_path), model="other-provider/m2"
    )

    assert first.root != second.root
    assert first.home != second.home
    assert first.workdir != second.workdir
    if first.profile_path is not None and second.profile_path is not None:
        assert first.profile_path != second.profile_path

    first_auth = (first.home / sandbox.OPENCODE_AUTH_PATH).read_text()
    second_auth = (second.home / sandbox.OPENCODE_AUTH_PATH).read_text()
    assert set(json.loads(first_auth)) == {"opencode-go"}
    assert set(json.loads(second_auth)) == {"other-provider"}
    assert "unrelated-secret" not in first_auth, "one principal could read another's credential"
    assert "model-key" not in second_auth, "one principal could read another's credential"


def test_two_providers_offering_the_same_model_name_stay_separate(
    auth_store: Path, tmp_path: Path
) -> None:
    """The key is the full model id, so the provider prefix is not discardable."""
    left = sandbox.root_for_model("opencode-go/shared", base=tmp_path)
    right = sandbox.root_for_model("other-provider/shared", base=tmp_path)

    assert left != right


def test_agent_definition_names_the_model_it_was_provisioned_for(
    auth_store: Path, tmp_path: Path
) -> None:
    box = sandbox.provision(
        sandbox.root_for_model("opencode-go/m1", base=tmp_path), model="opencode-go/m1"
    )

    agent = box.home / sandbox.OPENCODE_AGENT_DIR / sandbox.AGENT_FILENAME
    assert f"model: {box.root.name and 'opencode-go/m1'}" in agent.read_text()


def test_reprovisioning_a_root_replaces_the_credential(auth_store: Path, tmp_path: Path) -> None:
    """A credential from a previous model must not stay usable in the same home."""
    root = tmp_path / "box"
    sandbox.provision(root, model="opencode-go/m1")
    assert "model-key" in (root / "home" / sandbox.OPENCODE_AUTH_PATH).read_text()

    second = sandbox.provision(root, model="other-provider/m2")

    stored = json.loads((second.home / sandbox.OPENCODE_AUTH_PATH).read_text())
    assert set(stored) == {"other-provider"}
    assert "model-key" not in json.dumps(stored)
    agent = (second.home / sandbox.OPENCODE_AGENT_DIR / sandbox.AGENT_FILENAME).read_text()
    assert "model: other-provider/m2" in agent
    assert "model: opencode-go/m1" not in agent


def test_derived_root_cannot_escape_its_parent(tmp_path: Path) -> None:
    """A model id is untrusted input to the filesystem, not a filename."""
    base = tmp_path / "sandbox"
    max_length = sandbox.MODEL_SLUG_MAX_LENGTH + 2 + 16

    for model in (
        "../../../etc/passwd",
        "/etc/passwd",
        "..",
        "opencode-go/../../../../escape",
        "a" * 5_000,
        "op encode/../../escape",
        "\n\rtrailing",
    ):
        root = sandbox.root_for_model(model, base=base)
        assert root.parent == base, f"{model!r} escaped the sandbox parent"
        assert ".." not in root.name
        assert "/" not in root.name
        assert root.name == root.name.strip()
        assert 0 < len(root.name) <= max_length, f"{model!r} produced an unbounded directory name"

    # Identical readable slugs that differ only past the truncation point still
    # get separate homes, because the digest covers the whole id.
    long_prefix = "provider/" + "x" * 500
    assert sandbox.root_for_model(long_prefix + "-one", base=base) != sandbox.root_for_model(
        long_prefix + "-two", base=base
    )
    # A hostile id that slugifies to nothing still yields a usable name.
    assert sandbox.root_for_model("..", base=base).name.startswith("model--")


# --- provisioning concurrency ------------------------------------------------


def test_concurrent_provisioning_of_one_root_does_not_interleave(
    auth_store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two provisions of the same root must serialise, not interleave.

    The write is slowed and instrumented so the critical section is observable.
    Without the lock this reliably reports a depth above one; with it, exactly one
    provision is ever inside.
    """
    root = (tmp_path / "box").resolve()
    state = {"depth": 0, "max_depth": 0}
    guard = threading.Lock()
    real_write = sandbox._write_atomic

    def slow_write(path: Path, text: str, *, mode: int = 0o600) -> None:
        with guard:
            state["depth"] += 1
            state["max_depth"] = max(state["max_depth"], state["depth"])
        time.sleep(0.02)
        try:
            real_write(path, text, mode=mode)
        finally:
            with guard:
                state["depth"] -= 1

    monkeypatch.setattr(sandbox, "_write_atomic", slow_write)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(sandbox.provision, root, model="opencode-go/m1") for _ in range(4)
        ]
        boxes = [future.result() for future in futures]

    assert len(boxes) == 4, "the four provisions must actually have run"
    assert state["max_depth"] == 1, "two provisions were inside the critical section at once"
    assert {box.root for box in boxes} == {root}
    # The home is coherent, not a mixture of two runs.
    assert json.loads((boxes[0].home / sandbox.OPENCODE_AUTH_PATH).read_text()) == {
        "opencode-go": {"type": "api", "key": "model-key"}
    }
    agent = (boxes[0].home / sandbox.OPENCODE_AGENT_DIR / sandbox.AGENT_FILENAME).read_text()
    assert "model: opencode-go/m1" in agent


def test_a_held_provisioning_lock_reports_a_clear_busy_error(
    auth_store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = (tmp_path / "box").resolve()
    holding = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with sandbox._provision_lock(root):
            holding.set()
            release.wait(10)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    try:
        assert holding.wait(10), "the holder never took the lock"
        monkeypatch.setattr(sandbox, "PROVISION_LOCK_TIMEOUT_SECONDS", 0.2)
        with pytest.raises(sandbox.SandboxError, match="provisioning the sandbox"):
            sandbox.provision(root, model="opencode-go/m1")
    finally:
        release.set()
        holder.join(10)

    # Once the lock is free the same call succeeds, so the refusal is a retry, not
    # a permanent poisoning of the root.
    monkeypatch.setattr(sandbox, "PROVISION_LOCK_TIMEOUT_SECONDS", 5.0)
    assert sandbox.provision(root, model="opencode-go/m1").root == root


def test_a_provision_lock_that_is_not_a_regular_file_is_refused(
    auth_store: Path, tmp_path: Path
) -> None:
    root = (tmp_path / "box").resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    (root.parent / f"{root.name}.provision.lock").mkdir()

    with pytest.raises(sandbox.SandboxError, match="could not be opened"):
        sandbox.provision(root, model="opencode-go/m1")


def test_atomic_writes_leave_no_partial_file_behind(auth_store: Path, tmp_path: Path) -> None:
    root = tmp_path / "box"
    box = sandbox.provision(root, model="opencode-go/m1")

    leftovers = [path.name for path in root.rglob(".auth.json.*")]
    assert leftovers == [], f"atomic writes left temporary files: {leftovers}"
    assert json.loads((box.home / sandbox.OPENCODE_AUTH_PATH).read_text())


# --- async dispatch ----------------------------------------------------------


async def test_run_async_leaves_the_event_loop_free_and_the_blocking_call_does_not(
    auth_store: Path, tmp_path: Path
) -> None:
    """The blocking form must genuinely block, or the async assertion means nothing.

    Both halves run the same real subprocess. The control establishes that a
    ticker coroutine makes no progress while ``run`` is called inline; only then
    does progress under ``run_async`` prove the loop stayed free.
    """
    box = sandbox.provision(tmp_path / "box", model="opencode-go/m1")

    async def ticks_while(blocking: bool) -> int:
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        task = asyncio.create_task(ticker())
        # Let the ticker establish itself, then measure only the window in which
        # the subprocess call is in flight.
        await asyncio.sleep(0.05)
        before = ticks
        if blocking:
            assert sandbox.run(box, ["/bin/sleep", "0.3"], timeout_seconds=30).returncode == 0
        else:
            result = await sandbox.run_async(box, ["/bin/sleep", "0.3"], timeout_seconds=30)
            assert result.returncode == 0
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return ticks - before

    blocking_ticks = await ticks_while(True)
    async_ticks = await ticks_while(False)

    assert blocking_ticks == 0, "the control did not block, so this test proves nothing"
    assert async_ticks >= 5, "the event loop was blocked for the whole subprocess call"
