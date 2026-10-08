"""A real implementation of the cycle, wired to the real capture system.

Until this existed, ``CycleSteps`` was a Protocol that only fakes satisfied. The
loop was a well-tested engine with nothing in it: every test drove it with a stub,
so nothing had ever been proved about the product, only about the control flow.

So this wires the parts that already exist and are already tested -- question
generation, the adversarial answerer, posting through the ordinary authenticated
comment path -- into the four steps the loop drives. A cycle that reaches the end
now produces comments on a pull request and records a capture, which is the whole
thesis of the unattended programme.

``implement`` is the honest exception. Producing a code change is a different
subsystem: a working tree, an agent that edits it, a diff, a push, a pull request.
That does not exist yet, and pretending it does by shipping a stub that opens an
empty branch would be the exact failure this project exists to prevent. So it is a
required injection, and a worker built without one refuses to run a cycle rather
than completing a ticket with nothing behind it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from kojutsu.config import Settings
from kojutsu.core.answerer import (
    AnswerPlan,
    QuestionSource,
    draft_answers,
    post_answers,
    select_questions,
)
from kojutsu.core.question_generator import generate_questions_for_pr
from kojutsu.core.question_registry import build_registry
from kojutsu.integrations.github import GitHubClient
from kojutsu.repo_name import split_repo

from .loop import CAPTURE_ONLY_STEPS, WorkItem

#: The seam a real implementer plugs into: given the item, return the URL of the
#: pull request it opened. Named so the gap is a named gap.
Implementer = Callable[[WorkItem], str]


class ImplementerRequiredError(RuntimeError):
    """No implementer was supplied, so no cycle can honestly be run."""


@dataclass
class CaptureCycleSteps:
    """The real cycle: implement, ask, answer, capture.

    ``implementer`` is required. ``apply`` is required too, and defaults off,
    because every write here reaches a real pull request. A worker that posts
    comments because nobody passed a flag is not an unattended worker, it is an
    incident.
    """

    implementer: Implementer | None = None
    client: GitHubClient | None = None
    question_source: QuestionSource | None = None
    apply: bool = False
    implement_model: str = ""
    answer_model: str = ""
    implement_agent: str = "kojutsu-worker"
    answer_agent: str = "kojutsu-worker"
    #: Explicit, defaulting to empty -- never read from the environment here.
    #: The builder (``cli_worker._build``) copies ``settings.github_token`` in,
    #: so one settings object decides and two steps cannot disagree. A default
    #: factory reading ``os.environ`` would re-read behind the caller's back.
    github_token: str = ""
    #: Same: empty means "derive from the work item" (see ``ask``), not "read
    #: ``KOJUTSU_PR_SPEC`` ambiently". Callers that need a fixed spec pass it.
    question_spec: str = ""
    answer_timeout_seconds: float = 120.0
    #: Comment ids posted by the last run, and the answers they carry, so the
    #: capture step has something concrete to confirm.
    _posted: list[int] = field(default_factory=list)
    _plans: list[AnswerPlan] = field(default_factory=list)
    _questions: list[dict[str, Any]] = field(default_factory=list)

    #: Injected, never defaulted: a default here would re-read the environment
    #: behind the caller's back, so two steps built from one settings object
    #: could disagree about the configuration. The runtime builds settings once
    #: and hands them down; tests pass them explicitly.
    settings: Settings = field(kw_only=True)

    def __post_init__(self) -> None:
        if self.question_source is None:
            self.question_source = build_registry(self.settings)

    # -- the four steps ------------------------------------------------------

    def implement(self, item: WorkItem) -> str:
        if self.implementer is None:
            raise ImplementerRequiredError(
                "no implementer is configured, so this worker cannot honestly run a "
                "cycle. Pass one that opens a pull request and returns its URL; a "
                "worker that completes a ticket without having changed anything is "
                "worse than a worker that refuses to start."
            )
        return self.implementer(item)

    def ask(self, item: WorkItem) -> str:
        """Generate and register questions for the pull request.

        Returns the pull request number, which is what the later steps need and
        what a resumed cycle already has in its produced values.

        Every LLM setting is passed through explicitly. The generator's own
        defaults are a different provider, a different model, and external
        processing switched off, so a call that relied on them would quietly
        ignore the configured model and then refuse to run -- which reads as a
        privacy error rather than as a wiring mistake.
        """
        if not self.apply:
            return str(item.pr_number or "")
        if not self.github_token:
            raise RuntimeError("GITHUB_TOKEN is not set; refusing to post questions")
        generate_questions_for_pr(
            self.question_spec or f"{item.repo}#{item.pr_number}",
            self.github_token,
            self.settings.jira_url,
            self.settings.jira_username,
            self.settings.jira_api_token,
            self.settings.llm_provider,
            self.settings.llm_model,
            self.settings.llm_api_key or None,
            llm_external_enabled=self.settings.llm_external_enabled,
            llm_allowed_repositories=self.settings.llm_allowed_repositories,
            ollama_url=self.settings.ollama_url,
            llm_timeout_seconds=self.settings.llm_timeout_seconds,
            llm_retries=self.settings.llm_retries,
        )
        return str(item.pr_number or "")

    def answer(self, item: WorkItem) -> str:
        """Answer the outstanding questions with the answerer model.

        Nothing is posted without ``apply``. The returned value is a short
        description of what was drafted, never the answer text: the prose belongs
        in the comment, and copying it into ticket metadata would put unreviewed
        model output somewhere an operator would read it as a result.
        """
        if self.client is None:
            raise RuntimeError("no GitHub client is configured")
        owner, repo = split_repo(item.repo)
        if self.question_source is None:
            raise RuntimeError("no question source is configured")
        questions = select_questions(
            self.question_source, repo=item.repo, pr_number=int(item.pr_number or 0)
        )
        self._questions = questions
        if not questions:
            return "no outstanding questions"
        plans = draft_answers(
            questions,
            diff=self._diff_for(item),
            pr_title=item.subject,
            model=self.answer_model,
            agent=self.answer_agent,
            timeout_seconds=self.answer_timeout_seconds,
        )
        self._plans = plans
        if not self.apply:
            return f"drafted {len(plans)} answer(s), not posted"
        self._posted = post_answers(
            plans,
            client=self.client,
            owner=owner,
            repo=repo,
            pr_number=int(item.pr_number or 0),
        )
        return f"posted {len(self._posted)} answer(s)"

    def capture(self, item: WorkItem) -> str:
        """Confirm what the cycle produced, in the store.

        Capture is the webhook's job, not this step's: an answer posted through
        the ordinary comment path is collected by the ordinary collector, and
        re-capturing it here would create a second, privileged record. So this
        returns what was posted and stops, and the ledger fills in on its own.
        """
        if not self._posted:
            return "nothing posted; capture will follow the webhook"
        return f"{len(self._posted)} comment(s) posted for capture"

    # -- helpers -------------------------------------------------------------

    def _diff_for(self, item: WorkItem) -> str:
        if self.client is None:
            return ""
        owner, repo = split_repo(item.repo)
        try:
            pr = self.client.get_pull_request(owner, repo, int(item.pr_number or 0))
        except Exception:  # a missing diff must not invent one
            return ""
        return getattr(pr, "diff", "") or ""


@dataclass
class CaptureOnlyCycleSteps(CaptureCycleSteps):
    """A cycle that reviews work it did not write: ask, answer, capture.

    This exists because the two halves of the programme are different jobs.
    Opening a pull request is real work with a real deliverable, and the loop is
    right to insist on an implementer before it will run ``IMPLEMENT``. But
    capturing knowledge from a review is work on *someone else's* pull request:
    there is no branch to push and no diff to author, so requiring an implementer
    here would not make the cycle more honest, it would only make it impossible.

    What it deliberately does not do is pretend. ``implement`` still raises if it
    is ever called, and the class is only useful in combination with
    :data:`CAPTURE_ONLY_STEPS`, so a run that skipped implementation is visible in
    ``Worker.status`` and in every record the cycle produces, rather than being
    indistinguishable from one that implemented something.
    """

    #: Stated on the steps so a caller building a worker from these cannot
    #: accidentally pair capture-only steps with the full step set and get a
    #: cycle that dispatches ``IMPLEMENT`` into a class that refuses it.
    steps: tuple = CAPTURE_ONLY_STEPS

    def implement(self, item: WorkItem) -> str:
        raise ImplementerRequiredError(
            "this worker is capture-only: it reviews pull requests other people wrote, "
            "so it has nothing to implement. Use CaptureCycleSteps with an implementer "
            "if the cycle is meant to open a pull request."
        )
