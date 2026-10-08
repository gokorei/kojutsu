"""CLI commands: ask questions, collect answers, and post them."""

import fcntl
import json
import os
import stat
import tempfile
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Literal

import httpx
import typer
from pydantic import BaseModel, ConfigDict, Field

from kojutsu.cli_shared import _require_github_token, _settings
from kojutsu.config import Settings
from kojutsu.core.answer_collector import process_comment_reply
from kojutsu.core.answerer import (
    MAX_ANSWERS_PER_RUN,
    AnswerSelectionError,
    draft_answers,
    post_answers,
    select_questions,
)
from kojutsu.core.question_generator import generate_questions_for_pr
from kojutsu.core.question_registry import build_registry
from kojutsu.integrations.github import (
    GitHubClient,
    GitHubIntegrationError,
    canonical_pr_url,
    extract_agent_claim,
    extract_answer_question_id_from_comment_body,
    extract_question_id_from_comment_body,
    parse_pr_identifier,
)
from kojutsu.integrations.jira_client import JiraIntegrationError
from kojutsu.integrations.llm import (
    LLMError,
)
from kojutsu.integrations.tanseki import TansekiError
from kojutsu.models import Question, QuestionCategory, ReviewSession
from kojutsu.runtime import build_runtime

_MAX_ASK_PLAN_BYTES = 1024 * 1024

#: Bound on distinct plan files held in-process. Unbounded growth is a slowly
#: leaking file-descriptor-adjacent table: every new ``--plan-file`` path
#: inserts an RLock that is never removed. 128 entries covers concurrent CLI use
#: with headroom; the oldest entry is evicted under the guard when full. The
#: on-disk ``.apply.lock`` remains the cross-process mutual exclusion -- this
#: table only serialises threads inside this process.
_MAX_ASK_PLAN_LOCKS = 128
_ask_plan_locks: dict[Path, threading.RLock] = {}


_ask_plan_locks_guard = threading.Lock()


def _get_ask_plan_lock(target: Path) -> threading.RLock:
    """Return the in-process lock for ``target``, evicting the oldest if full."""
    with _ask_plan_locks_guard:
        existing = _ask_plan_locks.get(target)
        if existing is not None:
            # Re-insert to mark recent use so eviction drops the stalest.
            del _ask_plan_locks[target]
            _ask_plan_locks[target] = existing
            return existing
        if len(_ask_plan_locks) >= _MAX_ASK_PLAN_LOCKS:
            oldest = next(iter(_ask_plan_locks))
            del _ask_plan_locks[oldest]
        lock = threading.RLock()
        _ask_plan_locks[target] = lock
        return lock


class _AskPlanQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=500)
    category: QuestionCategory


class _AskPlanContext(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    pr_url: str = Field(min_length=1, max_length=500)
    pr_number: int = Field(gt=0)
    repo: str = Field(min_length=3, max_length=300)
    owner: str = Field(min_length=1, max_length=100)
    repo_name: str = Field(min_length=1, max_length=100)
    branch_name: str = Field(max_length=500)
    jira_ticket_key: str | None = Field(default=None, max_length=100)
    files_changed: list[str] = Field(max_length=10_000)
    #: The commit the diff these questions were generated from was taken at, when
    #: the generator knows it. Carried in the artifact rather than re-derived on
    #: apply, because the artifact is the record of what was reviewed: by the time
    #: ``--plan-file`` is applied the head may have moved, and reading the head then
    #: would anchor the questions to a commit they were not asked about. ``None`` is
    #: a real value here and stays ``None`` -- it says the artifact does not know,
    #: which is the answer the column is honest enough to carry.
    head_sha: str | None = Field(default=None, max_length=64)


class _AskPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    context: _AskPlanContext
    questions: list[_AskPlanQuestion] = Field(min_length=1, max_length=6)


def _validate_ask_plan(plan: _AskPlan) -> None:
    context = plan.context
    if context.repo != f"{context.owner}/{context.repo_name}":
        raise ValueError("Plan repository identity is inconsistent")
    if canonical_pr_url(context.repo, context.pr_number) != context.pr_url:
        raise ValueError("Plan pull request identity is inconsistent")
    question_ids = [question.id for question in plan.questions]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Plan question IDs must be unique")
    if any(question.category is QuestionCategory.SYSTEM_EVENT for question in plan.questions):
        raise ValueError("Plan contains an invalid question category")
    if any(any(ord(character) < 32 for character in question.id) for question in plan.questions):
        raise ValueError("Plan question IDs contain control characters")


#: Namespace for question ids derived from a question's content. Distinct from
#: the session namespace so a derived id can never collide with a derived
#: session id, and distinct from ``uuid4`` in that it is reproducible.
_QUESTION_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "kojutsu:question-id:v1")


def derived_question_id(pr_url: str, question: Question) -> str:
    """Return the id this question carries for this PR, every time.

    Generated questions arrive with a fresh ``uuid4`` per run, which makes the
    marker in the posted comment unrepeatable: a re-run cannot recognise the
    comment its own previous run left, and posts a second one. Deriving the id
    from what the question *is* — the PR and the text — makes the marker stable
    instead, so the marker-keyed reconcile in ``post_questions_as_pr_comments``
    matches and the run is a no-op.

    Deriving rather than looking is the load-bearing choice. Asking the forge
    "do you already have a comment with this marker" needs the marker first, and
    the marker is the thing being derived. Asking the registry does not help
    either: the crash this exists to survive is a post that succeeded and a
    record that did not, so the registry is exactly what is missing.

    The category is deliberately **not** part of the preimage. A posted comment
    carries a marker and the question text and nothing else, so two questions
    that differ only in category are indistinguishable on the pull request — and
    keying them apart would put two registry rows on one comment, both pending,
    with only one answer to give. What the reviewer is asked is the text, so the
    text is the identity.

    The bound this does not cover: a reworded question derives a new marker and
    is asked again. Recognising a paraphrase would mean guessing that two
    different sentences want the same answer, and a wrong guess here silently
    drops a question nobody asked.
    """
    preimage = f"{pr_url}\x1f{question.text.strip()}"
    return str(uuid.uuid5(_QUESTION_ID_NAMESPACE, preimage))


def _keyed_questions(questions: list[Question], *, pr_url: str) -> list[Question]:
    """Re-key generated questions to derived ids, dropping ones that collide.

    A generator that returns the same question twice would otherwise produce two
    identical comments on a pull request and two registry rows competing for one
    answer — and the reviewer would be asked the same thing twice, which is
    enough to make the whole thread stop being read.
    """
    keyed: list[Question] = []
    seen: set[str] = set()
    for question in questions:
        question_id = derived_question_id(pr_url, question)
        if question_id in seen:
            continue
        seen.add(question_id)
        keyed.append(question.model_copy(update={"id": question_id}))
    return keyed


def _build_ask_plan(questions: list[Question], context: dict[str, Any]) -> _AskPlan:
    try:
        plan = _AskPlan(
            context=_AskPlanContext.model_validate(context),
            questions=[
                _AskPlanQuestion.model_validate(
                    {"id": question.id, "text": question.text, "category": question.category}
                )
                for question in questions
            ],
        )
        _validate_ask_plan(plan)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Generated ask plan is invalid: {exc}") from None
    return plan


def _write_ask_plan(path: Path, plan: _AskPlan) -> None:
    target = path.expanduser().resolve()
    parent_created = not target.parent.exists()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if parent_created:
        target.parent.chmod(0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(plan.model_dump(mode="json"), stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            raise ValueError("Plan file already exists; choose a new path") from None
        target.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _load_ask_plan(path: Path) -> _AskPlan:
    target = path.expanduser().resolve()
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags)
        with os.fdopen(descriptor, "rb") as stream:
            file_stat = os.fstat(stream.fileno())
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError("Plan path is not a regular file")
            if hasattr(os, "geteuid") and file_stat.st_uid != os.geteuid():
                raise ValueError("Plan file is not owned by the current user")
            if stat.S_IMODE(file_stat.st_mode) & 0o077:
                raise ValueError("Plan file permissions are too broad")
            if file_stat.st_size > _MAX_ASK_PLAN_BYTES:
                raise ValueError("Plan file is too large")
            payload = stream.read(_MAX_ASK_PLAN_BYTES + 1)
        if len(payload) > _MAX_ASK_PLAN_BYTES:
            raise ValueError("Plan file is too large")
        plan = _AskPlan.model_validate_json(payload)
        _validate_ask_plan(plan)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"Unable to load safe ask plan: {exc}") from None
    return plan


@contextmanager
def _ask_plan_apply_lock(path: Path) -> Iterator[None]:
    target = path.expanduser().resolve()
    process_lock = _get_ask_plan_lock(target)
    with process_lock:
        lock_path = target.with_name(f".{target.name}.apply.lock")
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(lock_path, flags, 0o600)
        try:
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError("Ask plan lock is not a regular file")
            if hasattr(os, "geteuid") and file_stat.st_uid != os.geteuid():
                raise ValueError("Ask plan lock is not owned by the current user")
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def ask(
    pr: str | None = typer.Argument(None, help="PR URL or owner/repo#number"),
    pr_number: int | None = typer.Option(None, "--pr", help="PR number (use with --repo)"),
    repo: str | None = typer.Option(None, "--repo", help="Repository owner/name"),
    save_session: bool = typer.Option(
        True, "--save-session/--no-save-session", help="Save review session and question markers"
    ),
    apply: bool = typer.Option(
        False, "--apply/--plan", help="Apply the plan and post marked questions to GitHub"
    ),
    plan_file: Path | None = typer.Option(
        None,
        "--plan-file",
        help="Persist a plan for exact later apply; existing files are never overwritten",
        dir_okay=False,
    ),
) -> None:
    """Generate knowledge-capture questions for a PR and post them as GitHub comments."""
    settings = _settings()
    _require_github_token(settings)
    if apply and not save_session:
        typer.echo(
            "--no-save-session is preview-only; use --apply only with session saving.", err=True
        )
        raise typer.Exit(1)
    apply_lock: Any = nullcontext()
    if apply and plan_file is not None:
        apply_lock = _ask_plan_apply_lock(plan_file)
    try:
        with apply_lock:
            _run_ask(
                settings,
                pr=pr,
                pr_number=pr_number,
                repo=repo,
                save_session=save_session,
                apply=apply,
                plan_file=plan_file,
            )
    except (OSError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None


def _resolve_ask_questions(
    settings: Settings,
    *,
    pr: str | None,
    pr_number: int | None,
    repo: str | None,
    apply: bool,
    plan_file: Path | None,
) -> tuple[list[Question], dict[str, Any]]:
    """Produce the questions and their context, from a plan file or the generator."""
    requested_pr = pr
    if not requested_pr and pr_number is not None and repo:
        requested_pr = f"{repo}#{pr_number}"

    questions: list[Question]
    context: dict[str, Any]
    if apply and plan_file is not None:
        plan = _load_ask_plan(plan_file)
        if requested_pr:
            parsed_requested = parse_pr_identifier(requested_pr)
            if not parsed_requested or (
                parsed_requested[0],
                parsed_requested[1],
            ) != (plan.context.repo, plan.context.pr_number):
                typer.echo("PR argument does not match the plan artifact.", err=True)
                raise typer.Exit(1)
        context = plan.context.model_dump()
        questions = [
            Question(id=item.id, text=item.text, category=item.category) for item in plan.questions
        ]
    else:
        if not requested_pr:
            typer.echo("Provide PR as argument (e.g. owner/repo#123) or --pr and --repo.", err=True)
            raise typer.Exit(1)
        parsed_pr = parse_pr_identifier(requested_pr)
        if not parsed_pr:
            typer.echo(f"Invalid PR identifier. Got: '{requested_pr}'.", err=True)
            raise typer.Exit(1)
        pr_spec = canonical_pr_url(parsed_pr[0], parsed_pr[1])
        try:
            questions, context = generate_questions_for_pr(
                pr_spec=pr_spec,
                github_token=settings.github_token,
                jira_url=settings.jira_url or "",
                jira_username=settings.jira_username or "",
                jira_api_token=settings.jira_api_token or "",
                llm_provider=settings.llm_provider,
                llm_model=settings.llm_model,
                llm_api_key=settings.llm_api_key or None,
                llm_external_enabled=settings.llm_external_enabled,
                llm_allowed_repositories=settings.llm_allowed_repositories,
                ollama_url=settings.ollama_url,
                llm_timeout_seconds=settings.llm_timeout_seconds,
                llm_retries=settings.llm_retries,
            )
        except (
            ValueError,
            LLMError,
            JiraIntegrationError,
            GitHubIntegrationError,
            httpx.HTTPError,
        ) as exc:
            message = (
                str(exc)
                if isinstance(
                    exc, (ValueError, LLMError, JiraIntegrationError, GitHubIntegrationError)
                )
                else "GitHub is unavailable; verify GITHUB_TOKEN and connectivity."
            )
            typer.echo(message, err=True)
            raise typer.Exit(1) from None
        # Applied to generated questions only. A plan artifact's ids are what the
        # operator reviewed, and re-deriving them would make the artifact a
        # description of the run rather than the thing being applied.
        questions = _keyed_questions(questions, pr_url=context["pr_url"])
    return questions, context


def _apply_asked_questions(
    settings: Settings, questions: list[Question], context: dict[str, Any]
) -> None:
    """Record the session and post (or reconcile) the questions on the PR."""
    owner = context["owner"]
    repo_name = context["repo_name"]
    # Read once, from whichever source produced ``context``: the generator's payload
    # on a live run, the reviewed artifact under ``--plan-file``. Both name the head
    # their own diff was taken at, which is the only head that answers "what was
    # this question asked about". See the note where it is recorded.
    ask_time_head_sha = context.get("head_sha")
    pairs = [(question.id, question.text) for question in questions]
    question_by_text = {question.text: question for question in questions}
    session_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"kojutsu:review:{context['pr_url']}"))
    session = ReviewSession(
        session_id=session_id,
        context_url=context["pr_url"],
        context_id=str(context["pr_number"]),
        scope=context["repo"],
        metadata={
            "pr_url": context["pr_url"],
            "pr_number": context["pr_number"],
            "repo": context["repo"],
            "jira_ticket_key": context.get("jira_ticket_key"),
            "branch_name": context["branch_name"],
            "files_changed": context.get("files_changed", []),
        },
        created_by=None,
    )
    try:
        with build_registry(settings) as registry, GitHubClient(token=settings.github_token) as gh:
            registry.record_session(session)
            # Read before posting, and reconcile against it. That read is also the
            # boundary of the idempotency: if a previous run's comment is not in
            # it — deleted by a human, edited so the marker is gone, or authored
            # by a token other than this one — this run cannot see it and posts
            # its own. That is the honest outcome. The alternative, guessing from
            # the registry, would suppress a question whose comment is genuinely
            # gone and leave the pull request with a marker and no question.
            existing_comments = gh.list_issue_comments(owner, repo_name, context["pr_number"])
            if existing_comments.truncated:
                typer.echo(
                    "Warning: comment listing hit the page cap; reconciling "
                    "against a partial list.",
                    err=True,
                )
            existing_comments = existing_comments.items
            existing_comment_ids = {comment.id for comment in existing_comments}

            def checkpoint(resolved_id: str, text: str, created: Any) -> None:
                question = question_by_text[text]
                registry.record_question(
                    question_id=resolved_id,
                    github_comment_id=created.id,
                    repo=context["repo"],
                    pr_number=context["pr_number"],
                    pr_url=context["pr_url"],
                    question_text=text,
                    question_category=question.category.value,
                    jira_ticket_key=context.get("jira_ticket_key"),
                    session_id=session_id,
                    question_author=created.user.login,
                    # The commit the question was asked about, taken from the same
                    # payload the diff was generated from -- and only from there.
                    # Re-reading the head here would be a *later* head whenever the
                    # branch moved between generating and posting, and under
                    # ``--plan-file`` it is arbitrarily later still; both are
                    # different commits from the ones a reviewer is being asked
                    # about, so recording either would be an anchor that reads as
                    # fact and is not. Absent is the answer the registry stores for
                    # a question whose commit nobody recorded.
                    head_sha=ask_time_head_sha,
                )

            comments = gh.post_questions_as_pr_comments(
                owner,
                repo_name,
                context["pr_number"],
                pairs,
                existing_comments=existing_comments,
                on_comment=checkpoint,
            )
    except (GitHubIntegrationError, httpx.HTTPError):
        typer.echo(
            "Unable to post questions to GitHub; verify GITHUB_TOKEN and connectivity.", err=True
        )
        raise typer.Exit(1) from None
    # Reported as posted vs already present rather than as a single count. The
    # reconciled list includes comments this run found rather than created, so
    # "Applied 2" on a re-run that posted nothing reads as two new comments on a
    # pull request that already had them — which is the duplicate this command
    # is supposed to be avoiding, reported as success.
    already_present = sum(1 for comment in comments if comment.id in existing_comment_ids)
    typer.echo(
        f"Applied {len(comments)} question(s) to PR #{context['pr_number']}: "
        f"{len(comments) - already_present} posted, {already_present} already present."
    )
    typer.echo("Review session and question markers saved.")


def _run_ask(
    settings: Settings,
    *,
    pr: str | None,
    pr_number: int | None,
    repo: str | None,
    save_session: bool,
    apply: bool,
    plan_file: Path | None,
) -> None:
    questions, context = _resolve_ask_questions(
        settings,
        pr=pr,
        pr_number=pr_number,
        repo=repo,
        apply=apply,
        plan_file=plan_file,
    )
    if not questions:
        typer.echo("No questions generated.")
        return

    if not apply:
        mode = "Preview only" if not save_session else "Plan only"
        typer.echo(f"{mode}: no GitHub comments or session state were written.")
        for question in questions:
            typer.echo(f"- {question.text}")
        if plan_file is not None:
            plan = _build_ask_plan(questions, context)
            _write_ask_plan(plan_file, plan)
            typer.echo(f"Plan artifact written: {plan_file.expanduser().resolve()}")
        return

    _apply_asked_questions(settings, questions, context)


def answer(
    pr: str | None = typer.Argument(None, help="PR URL or owner/repo#number"),
    pr_number: int | None = typer.Option(None, "--pr", help="PR number (use with --repo)"),
    repo_opt: str | None = typer.Option(None, "--repo", help="Repository owner/name"),
    model: str = typer.Option(..., "--model", help="Model to review with, provider/model."),
    agent: str = typer.Option(..., "--agent", help="Agent name recorded as the author."),
    question_id: list[str] = typer.Option(
        None, "--question-id", help="Answer only this registered question. Repeatable."
    ),
    limit: int = typer.Option(
        MAX_ANSWERS_PER_RUN, "--limit", help="Maximum answers to post in this run."
    ),
    apply: bool = typer.Option(False, "--apply/--plan", help="Post the answers (default: plan)."),
) -> None:
    """Draft model answers to outstanding questions and post them as review comments.

    Plans by default, exactly like `ask`: without --apply nothing is written, no
    comment is posted, and no question is consumed. The reviewer prompt asks for a
    verdict rather than help, so an answer is expected to say when a change is wrong
    or cannot be verified. Every posted comment carries the agent marker and model,
    so capture records who wrote it and how independent it is.
    """
    settings = _settings()
    _require_github_token(settings)

    pr_spec = pr or (f"{repo_opt}#{pr_number}" if repo_opt and pr_number is not None else None)
    if not pr_spec:
        typer.echo("Provide PR as argument or --pr and --repo.", err=True)
        raise typer.Exit(1)
    parsed = parse_pr_identifier(pr_spec)
    if not parsed:
        typer.echo(f"Invalid PR identifier. Got: '{pr_spec}'.", err=True)
        raise typer.Exit(1)
    owner, repo_name = parsed[0].split("/", 1)
    number = int(parsed[1])

    with GitHubClient(token=settings.github_token) as client:
        try:
            pull = client.get_pull_request(owner, repo_name, number)
            diff = client.get_pull_diff(owner, repo_name, number)
        except (GitHubIntegrationError, httpx.HTTPError, ValueError) as exc:
            typer.echo(f"Unable to read the pull request: {exc}", err=True)
            raise typer.Exit(1) from None

    with build_registry(settings) as registry:
        try:
            questions = select_questions(
                registry,
                repo=parsed[0],
                pr_number=number,
                question_ids=list(question_id) if question_id else None,
                limit=limit,
            )
        except AnswerSelectionError as exc:
            typer.echo(f"Nothing to answer: {exc}", err=True)
            raise typer.Exit(1) from None

    if not questions:
        typer.echo(f"No outstanding questions on {parsed[0]}#{number}.")
        return

    try:
        plans = draft_answers(
            questions,
            diff=diff,
            pr_title=pull.title,
            model=model,
            agent=agent,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    except (LLMError, AnswerSelectionError) as exc:
        # Nothing has been posted at this point, so there is no partial comment to
        # clean up: the whole set is drafted before any of it is written.
        typer.echo(f"Unable to draft answers: {exc}", err=True)
        raise typer.Exit(1) from None

    for plan in plans:
        typer.echo(f"- [{plan.question_id}] {plan.question_text}")
        typer.echo(f"  {plan.answer_text.splitlines()[0][:160]}")

    if not apply:
        typer.echo(
            f"\nPlan only: {len(plans)} answer(s) drafted, none posted. Re-run with --apply."
        )
        return

    try:
        posted = post_answers(plans, client=client, owner=owner, repo=repo_name, pr_number=number)
    except (GitHubIntegrationError, httpx.HTTPError, ValueError) as exc:
        typer.echo(f"Unable to post answers: {exc}", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"Posted {len(posted)} answer(s) to {parsed[0]}#{number}.")


def collect(
    pr: str | None = typer.Argument(None, help="PR URL or owner/repo#number"),
    pr_number: int | None = typer.Option(None, "--pr", help="PR number (use with --repo)"),
    repo_opt: str | None = typer.Option(None, "--repo", help="Repository owner/name"),
) -> None:
    """Manually collect answers from PR comments (fallback when webhook is not configured)."""
    settings = _settings()
    _require_github_token(settings)

    pr_spec = pr or (f"{repo_opt}#{pr_number}" if repo_opt and pr_number is not None else None)
    if not pr_spec:
        typer.echo("Provide PR as argument or --pr and --repo.", err=True)
        raise typer.Exit(1)

    parsed = parse_pr_identifier(pr_spec)
    if not parsed:
        typer.echo(f"Invalid PR identifier. Got: '{pr_spec}'.", err=True)
        raise typer.Exit(1)

    owner, repo_name = parsed[0].split("/", 1)
    number = int(parsed[1])
    try:
        with GitHubClient(token=settings.github_token) as client:
            listed = client.list_issue_comments(owner, repo_name, number)
    except (GitHubIntegrationError, httpx.HTTPError, ValueError) as exc:
        typer.echo(f"Unable to read GitHub comments: {type(exc).__name__}.", err=True)
        raise typer.Exit(1) from None
    if listed.truncated:
        typer.echo(
            "Warning: comment listing hit the page cap; answers past the cap were not collected.",
            err=True,
        )
    comments = listed.items
    comments.sort(key=lambda c: c.created_at)
    questions = {
        extract_question_id_from_comment_body(comment.body): comment
        for comment in comments
        if extract_question_id_from_comment_body(comment.body)
    }

    try:
        runtime = build_runtime(settings)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    collected = 0
    try:
        with runtime:
            for comment in comments:
                question_id = extract_answer_question_id_from_comment_body(comment.body)
                parent = questions.get(question_id)
                if not question_id or parent is None:
                    continue
                created = process_comment_reply(
                    new_comment_id=comment.id,
                    new_comment_body=comment.body,
                    new_comment_author=comment.user.login,
                    new_comment_created_at=comment.created_at,
                    parent_comment_id=parent.id,
                    new_comment_author_association=comment.author_association,
                    repo=parsed[0],
                    pr_number=number,
                    registry=runtime.registry,
                    sink=runtime.sink,
                    question_id=question_id,
                    parent_agent_claim=extract_agent_claim(parent.body),
                )
                if created:
                    collected += 1
    except TansekiError:
        typer.echo(
            "Unable to collect answers: Tanseki is unavailable or rejected the write.", err=True
        )
        raise typer.Exit(1) from None

    typer.echo(f"Collected {collected} new answer(s).")


def register(app: typer.Typer) -> None:
    """Attach these commands to a Typer app."""
    app.command()(ask)
    app.command()(answer)
    app.command()(collect)
