"""CLI commands: backfill history from the forge API."""

import json
from pathlib import Path

import httpx
import typer

from kojutsu.cli_shared import _require_github_token, _settings
from kojutsu.config import Settings
from kojutsu.core.backfill import run_backfill
from kojutsu.core.backfill_reviews import (
    MAX_OBJECTS_PER_RUN,
    build_plan,
)
from kojutsu.core.backfill_reviews import run_backfill as run_review_backfill
from kojutsu.core.backfill_reviews_client import GitHubHistoryReader, RateLimitedError
from kojutsu.core.outbox import OutboxOwnershipError
from kojutsu.integrations.github import (
    GitHubClient,
    GitHubIntegrationError,
)
from kojutsu.integrations.tanseki import TansekiError
from kojutsu.runtime import build_runtime


def _parse_associations(value: str | None) -> frozenset[str] | None:
    """Read ``--authorized-associations`` into the set the collector expects.

    ``None`` for unset, which downstream means **no restriction** —
    :data:`kojutsu.allowlist.ADMIT_ALL_ASSOCIATIONS` is ``None``, and
    that is the whole default policy rather than a stand-in for a set. An explicit
    set means *only those*. The two are kept apart here so that "the operator did not
    say" and "the operator said something restrictive" cannot arrive downstream as
    the same value by accident.

    Values are case-folded upward because GitHub reports them capitalised and a
    hand-typed ``member`` should not silently refuse every review it was meant to
    admit.
    """
    if value is None or not value.strip():
        return None
    parsed = {part.strip().upper() for part in value.split(",") if part.strip()}
    if not parsed:
        return None
    return frozenset(parsed)


def backfill(
    repo: str = typer.Option(..., "--repo", help="Repository owner/name, inside the allowlist"),
    since: str = typer.Option(..., "--since", help="Inclusive start date, YYYY-MM-DD"),
    until: str = typer.Option(..., "--until", help="Inclusive end date, YYYY-MM-DD"),
    limit: int = typer.Option(
        100, "--limit", "-n", help="Max pull requests to enumerate (GitHub caps at 1000)"
    ),
    report: Path | None = typer.Option(
        None, "--report", help="Write the coverage record to this JSON file"
    ),
) -> None:
    """Collect from every pull request created in a date range. Read-only.

    The counterpart to `collect`, which reads one pull request's comments by hand.
    This one enumerates a range, which means it can reach pull requests that were
    already merged when they were written -- anything the webhook, which only
    ever sees live events, could not have caught.

    It writes nothing to GitHub. There is no `--apply`, and that is the point:
    every request is a `GET`, and a historical ingest has no reason to comment on
    a pull request from three months ago. What it writes is to the store, through
    the ordinary collectors, keyed on the comment -- so re-running an overlapping
    range stores nothing the first run stored, and a run interrupted half way is
    resumed by running the same range again.

    The range must be inside the configured allowlist, and both ends must be
    `YYYY-MM-DD` with `since` no later than `until`. A truncated range says so
    rather than reading as a complete one; see `docs/github-seam.md`.
    """
    settings = _settings()
    _require_github_token(settings)
    try:
        runtime = build_runtime(settings)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    try:
        with runtime, GitHubClient(token=settings.github_token) as gh:
            coverage = run_backfill(
                gh,
                runtime.registry,
                runtime.sink,
                repository=repo,
                since=since,
                until=until,
                settings=settings,
                limit=limit,
            )
    except GitHubIntegrationError as exc:
        # A rate limit is retryable and is named as such, so the operator is not
        # left reading a transient failure as a permanent one. The rate-limited
        # read is refused rather than turned into a short range.
        retryable = " (retryable)" if exc.retryable else ""
        typer.echo(f"Backfill failed{retryable}: {exc}", err=True)
        raise typer.Exit(1) from None
    except TansekiError:
        typer.echo("Backfill failed: Tanseki is unavailable or rejected the write.", err=True)
        raise typer.Exit(1) from None
    except ValueError as exc:
        typer.echo(f"Backfill refused: {exc}", err=True)
        raise typer.Exit(1) from None

    typer.echo(coverage.summary())
    if report is not None:
        report.write_text(json.dumps(coverage.as_dict(), indent=2) + "\n", encoding="utf-8")
        typer.echo(f"Coverage record written to {report}.")
    # A range that could not be fully enumerated exits non-zero even though the
    # run stored whatever it found. A truncation is a result to act on, not a
    # success, and an exit code of 0 is what a script is looking for.
    if not coverage.complete:
        raise typer.Exit(2)


#: Cap on gaps listed per backfill run; the rest are counted, not shown.
MAX_REPORTED_BACKFILL_GAPS = 20


def backfill_reviews(
    repository: list[str] = typer.Option(
        None,
        "--repo",
        metavar="OWNER/NAME",
        help="Repository to reconstruct. Repeatable; each must be allow-listed.",
    ),
    since: str = typer.Option(
        ...,
        "--since",
        metavar="YYYY-MM-DD",
        help="Date floor for what is worth reconstructing. Required.",
    ),
    until: str | None = typer.Option(
        None,
        "--until",
        metavar="YYYY-MM-DD",
        help="Inclusive date ceiling. Omitted means up to now, which is rarely what is meant.",
    ),
    max_objects: int = typer.Option(
        ...,
        "--max-objects",
        help=(
            f"New objects this run may store. Required, capped at {MAX_OBJECTS_PER_RUN}. "
            "Objects already stored are not charged again."
        ),
    ),
    authorized_associations: str = typer.Option(
        None,
        "--authorized-associations",
        help=(
            "Comma-separated GitHub author_association values to admit, narrowing "
            "the default, which admits all of them. Name one only to restrict a run "
            "to project members or collaborators; the named set replaces the policy "
            "rather than adding to it, so CONTRIBUTOR alone also stops admitting "
            "MEMBER. Unset admits every association, which is the point: the field "
            "sorts automated reviewers into CONTRIBUTOR, so filtering on it discarded "
            "far more human review activity than machine. Automation is recorded on "
            "every stored comment instead, and the repository allowlist is "
            "unchanged."
        ),
    ),
    pr: list[str] = typer.Option(
        None,
        "--pr",
        metavar="NUMBER",
        help=(
            "Pull request number to reconstruct. Repeatable and comma-separated, "
            "so --pr 2829 --pr 2830,1 is three PRs. Reads only those PRs' own "
            "pages and no listing pages at all; the window still applies and is "
            "reported, never widened. A PR with no reviews is an empty range, "
            "not an error."
        ),
    ),
) -> None:
    """Reconstruct pre-deployment history as backfilled records.

    Reads the pull request reviews and comments that predate this deployment,
    and turns them into records marked `capture_source: backfilled` through the
    same collectors the webhook handler uses. Each record is anchored to the
    read that produced it and to nothing else: it shows what the forge says
    now, not what it said at the time.

    Bounded on three axes, all required, none with a default. `--repo` may
    only name repositories the capture allowlist already permits
    (GITHUB_WEBHOOK_ALLOWED_REPOSITORIES), `--since` is a date floor, and
    `--max-objects` caps the new objects one run may store. The bounds are not
    caution for its own sake: history is unbounded, the token is
    per-repository, and the cost of a runaway run lands on everyone else
    using the forge.

    `--since` is a policy decision, not a tuning knob. Pre-deployment history
    contains the pull requests nobody ever reviewed, and reconstructing those
    as empty observations would assert a completeness that was never true.
    Choose the era of history worth reconstructing deliberately; this command
    will not choose one for you, and it reports the floor it ran under.

    `--until` bounds the same range at the other end, and it is optional
    because "up to now" is a real answer. It is not a tuning knob either:
    without it, enumeration necessarily includes everything up to the present
    moment, so asking a busy repository for the last quarter returns the last
    page of it rather than the quarter — a different question, answered
    without saying so. Both ends are `YYYY-MM-DD`, `since` no later than
    `until`, and `until` means the whole of the day it names.

    `--max-objects` bounds new work, not reads. An object the store already
    holds is charged nothing, and neither is one that turns out to have
    nothing to capture, which is what lets a re-run walk past what the last
    one stored instead of spending its budget on it. The forge reads a run
    performs are bounded by the page ceilings rather than by this flag, so a
    re-run over a range you have already walked re-reads it: what it costs is
    reads, and `objects-read` counts every object examined.

    It does not make the corpus representative. A backfill removes the
    deployment date from the picture and makes the corpus larger. It does
    nothing about the behavioural bias: a repository where nobody comments
    still produces nothing, and a backfill over that repository faithfully
    reconstructs the fact that there was nothing to reconstruct. That is a
    true statement about a biased sample, and the sample is still biased.

    Answers are not reconstructed. An answer requires a question to have been
    asked, and kojutsu only started asking when it was installed, so a
    historical comment with no kojutsu marker is somebody talking to a
    colleague and is never stored as a conclusion.

    Re-running the same range is safe, and is how an interrupted run
    continues: record identity is derived from the forge's own object
    identity, so an object seen twice collides and writes nothing. There is
    no cursor file, because a cursor can drift, can be lost, and cannot tell
    you that two runs saw the same object. Safe is not the same as advancing,
    which is why a re-run charges only for what it stores: charged per read,
    it would re-read the same first page and stop in the same place forever.

    Token scope: `Pull requests: Read` and `Issues: Read`, the minimum this
    pipeline needs. `Contents` is deliberately not used, and no diff is read.

    Reported on completion: objects read, records written, new objects stored,
    objects skipped as already present, objects unreadable, and objects read
    that had nothing to say. The unreadable count is the one to watch: it is
    the part of history that was never reconstructed and never will be. A run
    that stopped at its budget says so as a truncation and exits 2, because a
    truncated range is indistinguishable from a complete one to anything
    reading the exit code — and a run whose enumeration ran out before it
    reached the floor says that too, because the forge will not list a
    repository's history for ever and a range quietly missing its own floor
    looks identical from the outside.
    """
    settings = _settings()
    _require_github_token(settings)

    plan = _review_backfill_plan(
        settings,
        repository=repository,
        since=since,
        until=until,
        max_objects=max_objects,
        authorized_associations=authorized_associations,
        pr=pr,
    )
    report = _run_review_backfill_plan(settings, plan)
    _echo_review_backfill_report(plan, report)


def _parse_pr_numbers(value: list[str] | None) -> list[int] | None:
    """Read ``--pr`` into numbers: repeatable and comma-separated.

    ``None`` (flag absent) means the listing walk. Anything else must name at
    least one positive number, or the run is refused rather than read as the
    whole listing -- a ``--pr`` that silently meant everything would be the
    walk this bound exists to avoid.
    """
    if not value:
        return None
    numbers: list[int] = []
    for entry in value:
        for part in str(entry).split(","):
            text = part.strip()
            if not text:
                continue
            try:
                number = int(text)
            except ValueError:
                raise ValueError(f"--pr must name pull request numbers, got {text!r}.") from None
            if number < 1:
                raise ValueError(f"--pr must name pull request numbers, got {text!r}.")
            numbers.append(number)
    if not numbers:
        raise ValueError("--pr names no pull request; omit it to walk the listing.")
    return numbers


def _review_backfill_plan(
    settings: Settings,
    *,
    repository: list[str],
    since: str,
    until: str | None,
    max_objects: int,
    authorized_associations: str,
    pr: list[str] | None = None,
):
    """Validate the run's three bounds, or exit naming the problem."""
    try:
        pr_numbers = _parse_pr_numbers(pr)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    try:
        return build_plan(
            settings=settings,
            repositories=repository,
            since=since,
            until=until,
            max_objects=max_objects,
            authorized_associations=_parse_associations(authorized_associations),
            pr_numbers=pr_numbers,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None


def _run_review_backfill_plan(settings: Settings, plan):
    """Execute the plan, translating transport failures into exit codes."""
    try:
        reader = GitHubHistoryReader(token=settings.github_token)
        runtime = build_runtime(settings)
    except (ValueError, OutboxOwnershipError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None

    with runtime:
        try:
            return run_review_backfill(
                reader=reader,
                registry=runtime.registry,
                sink=runtime.sink,
                plan=plan,
                # The bound an operator set on a shared token, not the default: this is
                # the one place the run's concurrency is decided, and passing it here
                # rather than reading it inside the walk is what keeps the walk
                # testable at concurrency 1.
                read_concurrency=settings.github_history_concurrency,
            )
        except (GitHubIntegrationError, httpx.HTTPError, TansekiError, RateLimitedError) as exc:
            # A rate limit that outlived its backoff stops the run rather than
            # skipping the object it was reading. The re-run is safe because
            # identity makes the work idempotent, so failing loudly costs a re-run
            # and continuing would cost a hole in the corpus with nothing in it
            # recording the hole.
            typer.echo(f"Backfill stopped: {exc}", err=True)
            raise typer.Exit(1) from None
        finally:
            # Hands back the connection pool the run held open. A deep walk opens one
            # pool and keeps it for the length of the run rather than re-handshaking
            # per request, which is only an improvement if something closes it.
            close = getattr(reader, "close", None)
            if close is not None:
                close()


def _echo_review_backfill_report(plan, report) -> None:
    """Print what the run did, and exit non-zero when the range is truncated."""
    ceiling = f" until={plan.until.date().isoformat()}" if plan.until is not None else ""
    # The effective gate, always printed and not only when it was narrowed. The flag
    # replaces the policy rather than adding to it, so `--authorized-associations
    # CONTRIBUTOR` also stops admitting MEMBER -- which is how narrowing is expressed,
    # and which would be a silent surprise if the summary did not say so. `all` is
    # printed rather than an empty value, because an empty line beside this label reads
    # as a bug and "all" is the actual answer: the default admits every association,
    # and the reason is in `ADMIT_ALL_ASSOCIATIONS` for whoever asks next.
    admitted = plan.authorized_associations
    pr_bound = (
        f" prs={','.join(str(n) for n in plan.pr_numbers)}" if plan.pr_numbers is not None else ""
    )
    typer.echo(
        f"backfill: {', '.join(plan.repositories)} "
        f"since={plan.since.date().isoformat()}{ceiling} max-objects={plan.max_objects}{pr_bound}"
    )
    typer.echo(
        "admitting-author-associations: "
        + (",".join(sorted(admitted)) if admitted is not None else "all")
    )
    typer.echo(f"objects-read: {report.objects_read}")
    typer.echo(f"records-written: {report.records_written}")
    typer.echo(f"new-objects: {report.objects_new} (charged against max-objects)")
    typer.echo(f"already-present: {report.already_present}")
    typer.echo(f"unreadable: {report.unreadable}")
    typer.echo(f"silent: {report.silent}")
    for reason, count in report.refusals.items():
        # Broken out because ``silent`` merges two facts with opposite implications:
        # an object with nothing capturable is an absence of review, while one a gate
        # refused is review activity the corpus is discarding. An operator reading a
        # single number cannot tell which they have.
        typer.echo(f"  refused: {count:>6} {reason}")
    for gap in report.gaps[:MAX_REPORTED_BACKFILL_GAPS]:
        location = f"{gap.repository}#{gap.pr_number if gap.pr_number is not None else '-'}"
        typer.echo(f"  gap: {location} {gap.what}: {gap.reason}")
    omitted = len(report.gaps) - MAX_REPORTED_BACKFILL_GAPS
    if omitted > 0:
        typer.echo(f"  ... and {omitted} more unreadable object(s) not listed")
    if report.budget_exhausted:
        # Named as a truncation, because a run that stopped at its bound and a run
        # that finished the range report the same counts otherwise, and the whole
        # cost of a truncated backfill is that it is indistinguishable from a
        # complete one. The re-run it recommends is now true rather than
        # misleading: what the store already holds is charged nothing, so the next
        # run spends its budget further back instead of on the same page.
        ceiling_hit = any("page ceiling" in gap.reason for gap in report.gaps)
        typer.echo(
            f"range: TRUNCATED — stored {report.objects_new} new object(s) of the "
            f"{report.plan.max_objects} allowed, then stopped. This range is not complete.",
            err=True,
        )
        if ceiling_hit:
            typer.echo(
                "  a read also hit the page ceiling, which another identical run will hit "
                "again: raise the window's bound or the page ceiling rather than re-running",
                err=True,
            )
        else:
            typer.echo(
                "  re-run the same range to continue; objects already stored are not charged "
                "again, so it advances. It does re-read them, so expect reads without records.",
                err=True,
            )
    elif report.objects_new == 0:
        typer.echo(
            "range: walked to its end; nothing new to store in it. A re-run of this range would "
            "read it again and store nothing.",
            err=True,
        )
    else:
        typer.echo("range: walked to its end; the budget did not cut it short.")
    if report.unreadable:
        # On stderr, because this is the line that matters most and it must not be
        # the one a log tail drops: it is the part of history that was never
        # reconstructed and never will be.
        typer.echo(
            f"{report.unreadable} object(s) were read but not reconstructed; nothing in this "
            "run or any later one will recover them.",
            err=True,
        )
    if report.floor_unreached:
        # A range missing its own floor, named for the same reason a budget stop is:
        # the counts above look identical either way. The forge caps how deep a
        # pull request listing goes — on pingdotgg/t3code it stops after about a
        # thousand changes — so this is reachable by asking for a floor the
        # enumeration cannot deliver, and nothing in the counts would have said so.
        typer.echo(
            f"range: NOT COVERED TO ITS FLOOR — the forge's listing ran out above "
            f"{plan.since.date().isoformat()}, so any history older than that was never "
            "listed. If this repository has none, this run was complete; if it has some, "
            "this range does not contain it and the sibling `backfill` reads by creation "
            "date instead.",
            err=True,
        )
    if report.budget_exhausted:
        # A truncation is a result to act on, not a success, and an exit code of 0
        # is what a script is looking for. The sibling `backfill` command exits 2 for
        # the same reason, and a truncated range that exits 0 is the failure this
        # report exists to name.
        raise typer.Exit(2)


def register(app: typer.Typer) -> None:
    """Attach these commands to a Typer app."""
    app.command()(backfill)
    app.command()(backfill_reviews)
