"""The unattended worker, as a runnable command.

The loop, the steps and the queue all existed and were tested, but nothing a person
could start. This is the entrypoint that turns them into a process: it builds the
worker from configuration, recovers the claims a previous boot left behind, and then
polls until it is told to stop.

Two decisions are made here rather than buried in the loop, because both are choices
an operator has to be able to see and reverse.

**Capture-only is explicit.** A worker reviewing somebody else's pull request has
nothing to implement, so it runs without an implementer -- but it says so, in
``--capture-only`` on the command line, in ``worker status``, and in every record the
cycle writes. The alternative, a worker that opens an empty pull request so the loop
can tick over, is the failure the rest of this codebase is built to avoid.

**Shutdown is a first-class path.** ``run_forever`` sleeps between polls, and a
process killed during a sleep is a process that may have been holding a claim. So the
sleep here is interruptible and the loop is stopped, not abandoned: a second worker
starting while the first is exiting would otherwise find live leases and wait out
their whole duration before doing anything.
"""

from __future__ import annotations

import signal
import threading
from pathlib import Path

import typer

from kojutsu.config import Settings, get_settings
from kojutsu.core.gates import (
    DEFAULT_GATES,
    Gate,
    GateAction,
    GateRegistry,
    LifecyclePoint,
)
from kojutsu.integrations.github import GitHubClient
from kojutsu.worker import (
    CAPTURE_ONLY_STEPS,
    DELEGATED_STEPS,
    CaptureCycleSteps,
    CaptureOnlyCycleSteps,
    Worker,
    WorkerConfig,
)
from kojutsu.worker.sources import OpenPullRequestSource

app = typer.Typer(help="Run the unattended capture loop.")

#: Placeholders, not working defaults. Both are overridable (``--repo``,
#: ``--model``, or the matching settings) and both are deliberately not a real
#: repository or a real model: a shipped default that names somebody's project is a
#: personal reference in the source, and one that silently works is worse. Left
#: unset they fail loudly -- a repository that is not there, or a provider with no
#: credential, the second of which names the providers that are.
DEFAULT_REPO = "acme/widgets"
DEFAULT_MODEL = "opencode/model"

#: The default policy blocks a cycle at ``plan.proposed`` until a person approves,
#: which is correct for a worker that writes tickets and impossible for one that
#: only reviews a pull request somebody else opened. ``record`` is that same policy
#: with the approval requirement demoted to a recorded decision: the gate is still
#: evaluated and still written to the decision stream, it simply stops the loop no
#: longer. It is the arrangement ``GateAction`` describes for staging a change --
#: watch before enforcing -- and the policy stays one edit from being strict again.
CAPTURE_POLICY: tuple[Gate, ...] = tuple(
    gate
    if gate.point is not LifecyclePoint.PLAN_PROPOSED
    else Gate(
        name=gate.name,
        point=gate.point,
        action=GateAction.NOTIFY,
        predicate=gate.predicate,
    )
    for gate in DEFAULT_GATES
)

GATE_POLICIES: dict[str, tuple[Gate, ...]] = {
    "enforce": DEFAULT_GATES,
    "record": CAPTURE_POLICY,
}


def _state_dir(settings: Settings) -> Path:
    return Path(settings.kojutsu_registry_path).expanduser().parent


def _require(settings: Settings) -> None:
    if not settings.github_token:
        typer.echo("Set GITHUB_TOKEN; the worker reads pull requests and posts answers.", err=True)
        raise typer.Exit(1)


def _build(
    settings: Settings,
    *,
    repo: str,
    model: str,
    agent: str,
    capture_only: bool,
    apply: bool,
    poll_interval: float,
    state_dir: Path,
    policy: str = "record",
) -> Worker:
    steps: CaptureCycleSteps
    # One client for the whole worker: the source reads pull requests through it
    # and the answer step posts through it, and two clients would mean two places
    # for the credential to be configured. Owned by the returned Worker and
    # released by Worker.close() (via the source) -- never closed here, which
    # would end the worker before its first poll.
    client = GitHubClient(settings.github_token)
    if capture_only:
        steps = CaptureOnlyCycleSteps(
            client=client,
            apply=apply,
            answer_model=model,
            answer_agent=agent,
            github_token=settings.github_token,
            settings=settings,
        )
    else:
        # A full cycle has no implementer available yet, so say that at build time
        # rather than letting every cycle fail on the same missing injection.
        # Closed before refusing: the client was built for a worker that will
        # never exist, so leaving it open would leak its pool to GC.
        client.close()
        typer.echo(
            "A full cycle needs an implementer, and none is wired up. "
            "Use --capture-only to review pull requests without one.",
            err=True,
        )
        raise typer.Exit(1)
    return Worker(
        config=WorkerConfig(
            name=agent,
            poll_interval_seconds=poll_interval,
            answer_model=model,
            answer_agent=agent,
            state_path=state_dir / "worker-state.json",
            steps=CAPTURE_ONLY_STEPS,
        ),
        source=OpenPullRequestSource(
            client,
            repo=repo,
            state_path=state_dir / f"worker-queue-{repo.replace('/', '-')}.json",
        ),
        steps=steps,
        gates=GateRegistry(GATE_POLICIES[policy]),
    )


@app.command()
def run(
    repo: str = typer.Option(DEFAULT_REPO, "--repo", help="owner/name to review."),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="provider/model that answers."),
    agent: str = typer.Option("kojutsu-worker", "--agent", help="Name recorded as author."),
    capture_only: bool = typer.Option(
        False,
        "--capture-only",
        help="Review pull requests without an implement step. Required today.",
    ),
    apply: bool = typer.Option(
        False,
        "--apply/--plan",
        help="Post questions and answers to GitHub. Defaults to drafting only.",
    ),
    poll_interval: float = typer.Option(60.0, "--poll-interval", help="Seconds between polls."),
    policy: str = typer.Option(
        "record",
        "--gates",
        help="enforce: block at plan.proposed until a person approves. "
        "record: evaluate that gate and write the decision without stopping.",
    ),
    cycles: int = typer.Option(
        None, "--cycles", help="Stop after this many polls. Default: run until stopped."
    ),
) -> None:
    """Poll for open pull requests and capture knowledge from each one."""
    settings = get_settings()
    _require(settings)
    if policy not in GATE_POLICIES:
        typer.echo(f"--gates must be one of {', '.join(sorted(GATE_POLICIES))}.", err=True)
        raise typer.Exit(1)
    if not capture_only:
        typer.echo(
            "Refusing to start: the full cycle has no implementer. Pass --capture-only, "
            "which reviews pull requests other people wrote.",
            err=True,
        )
        raise typer.Exit(1)

    state_dir = _state_dir(settings)
    worker = _build(
        settings,
        repo=repo,
        model=model,
        agent=agent,
        capture_only=capture_only,
        apply=apply,
        poll_interval=poll_interval,
        state_dir=state_dir,
        policy=policy,
    )

    released = worker.release_orphans()
    if released:
        typer.echo(f"Recovered {len(released)} orphaned claim(s): {', '.join(released)}")

    status = worker.status()
    typer.echo(
        f"Worker '{status['worker']}' on {repo} | steps: {', '.join(status['steps'])} | "
        f"{'posting' if apply else 'plan only'} | every {poll_interval:g}s"
    )
    if status["capture_only"]:
        typer.echo("Capture-only: this worker reviews pull requests, it does not write code.")
    if policy == "record":
        # Said out loud every start, because this is the one setting that relaxes a
        # default block, and a demo that quietly stopped requiring approval would be
        # the least honest thing this worker could do.
        typer.echo(
            "Gates: 'record' — the plan-approval gate is evaluated and its decision "
            "written, but it does not block. Use --gates enforce to require it."
        )

    stop = threading.Event()

    def _handle(signum: int, _frame: object) -> None:
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle)

    def _on_report(report: object) -> None:
        summary = getattr(report, "summary", str(report))
        typer.echo(f"  {summary}")

    typer.echo("Polling. Ctrl-C to stop.")
    try:
        worker.run_forever(
            cycles=cycles,
            sleep=stop.wait,  # type: ignore[arg-type]
            on_report=_on_report,
        )
    finally:
        # Stop the timer regardless of how the loop ended, so a Ctrl-C that lands
        # during a long poll is not followed by a full interval of silence.
        stop.set()
        # Then hand back the forge connection. In this order and not the reverse:
        # the client is closed after the loop that uses it has stopped, never at
        # the point it was built, which would end the worker's ability to read on
        # its first poll.
        worker.close()
        typer.echo("Worker stopped.")


@app.command()
def status(
    repo: str = typer.Option(DEFAULT_REPO, "--repo", help="owner/name to report on."),
) -> None:
    """Show the loop's durable state: claims, blocks, resume points, dead letters."""
    settings = get_settings()
    state_dir = _state_dir(settings)
    report: dict[str, object] = {}
    try:
        worker = _build(
            settings,
            repo=repo,
            model=DEFAULT_MODEL,
            agent="kojutsu-worker",
            capture_only=True,
            apply=False,
            poll_interval=60.0,
            state_dir=state_dir,
        )
        try:
            report = worker.status()
        finally:
            worker.close()
    except Exception as exc:  # status must report, not crash
        typer.echo(f"Could not read worker state: {type(exc).__name__}: {exc}", err=True)

    for key in (
        "worker",
        "steps",
        "capture_only",
        "gates",
        "answer_model",
        "single_model",
        "durable",
        "in_flight_attempts",
        "blocked",
        "resuming",
        "dead_letter_ids",
    ):
        if key in report:
            typer.echo(f"{key}: {report[key]}")

    try:
        # Scoped to the reporting below, which is the only thing this uses the
        # source for. The client belongs to the source, so the source is what
        # closes -- hence the context manager rather than a bare client.
        with OpenPullRequestSource(
            GitHubClient(settings.github_token),
            repo=repo,
            state_path=state_dir / f"worker-queue-{repo.replace('/', '-')}.json",
        ) as source:
            typer.echo(f"reviews_done: {len(source.reviewed())}")
            for item_id in source.in_flight():
                typer.echo(f"in_flight: {item_id}")
    except Exception as exc:  # the queue is optional context here
        typer.echo(f"queue unavailable: {type(exc).__name__}", err=True)


@app.command()
def requeue(
    item_id: str = typer.Argument(..., help="Item id to return to the queue, e.g. owner/repo#1."),
    repo: str = typer.Option(DEFAULT_REPO, "--repo", help="owner/name the item belongs to."),
) -> None:
    """Return a completed or dead-lettered review to the queue.

    Without this a review that was completed wrongly is permanently unrepeatable,
    which is the one way a finished-looking queue can still be hiding lost work.
    """
    settings = get_settings()
    state_dir = _state_dir(settings)
    with OpenPullRequestSource(
        GitHubClient(settings.github_token),
        repo=repo,
        state_path=state_dir / f"worker-queue-{repo.replace('/', '-')}.json",
    ) as source:
        if not source.forget(item_id):
            typer.echo(f"{item_id} is not recorded as reviewed; nothing to requeue.", err=True)
            raise typer.Exit(1)
        typer.echo(f"{item_id} returned to the queue.")


def register(parent: typer.Typer) -> None:
    """Attach the worker commands to the main Typer app as a sub-group."""
    parent.add_typer(app, name="worker")


__all__ = ["DEFAULT_MODEL", "DEFAULT_REPO", "DELEGATED_STEPS", "app", "register"]
