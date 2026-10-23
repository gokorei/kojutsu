"""Orchestrates context gathering and LLM question generation for a PR."""

import logging
from typing import Any

from kojutsu.chapters import Chapter, Prologue, build_chapters, build_prologue
from kojutsu.integrations.github import (
    GitHubClient,
    extract_jira_key_from_branch,
    extract_jira_key_from_text,
    parse_pr_identifier,
)
from kojutsu.integrations.jira_client import JiraClient, JiraIntegrationError, JiraIssueFields
from kojutsu.integrations.llm import (
    ChapteredQuestion,
    build_chaptered_questions_prompt,
    build_questions_prompt,
    generate_chaptered_questions_sync,
    generate_questions_sync,
    questions_to_models,
    validate_llm_privacy,
)
from kojutsu.models import Question

logger = logging.getLogger(__name__)

#: Hard ceiling on questions from one chapter. Two lets a dense area receive the
#: second question the budget has left over; three would let one chapter consume a
#: six-question budget and defeat the coverage rule the split exists for.
MAX_QUESTIONS_PER_CHAPTER = 2


def _max_questions_for_diff(diff_length: int, files_count: int) -> int:
    """Suggest max questions based on diff size and file count."""
    if diff_length < 500 and files_count <= 2:
        return 2
    if diff_length < 2000 and files_count <= 5:
        return 3
    if diff_length < 8000:
        return 5
    return 6


def _max_questions_per_chapter(chapter_count: int, max_questions: int) -> int:
    """Split the question budget across chapters, coverage first.

    One per chapter is the shape that makes a gap visible: a chapter that gets no
    question is reported, and the reader can see that the budget went somewhere
    rather than everywhere. With one or two chapters that rule would leave most
    of the budget unused, so a chapter may receive a second question when the
    chapters are at most half as numerous as the questions. More than two per
    chapter reintroduces the clustering this split exists to prevent.
    """
    if chapter_count <= 0:
        return 1
    return max(1, min(MAX_QUESTIONS_PER_CHAPTER, max_questions // chapter_count))


def _warn_about_uncovered_chapters(
    chapters: tuple[Chapter, ...],
    parsed: list[ChapteredQuestion],
    *,
    repo: str,
    pr_number: int,
) -> None:
    """Report chapters that received no question, when the response was anchored.

    Nothing is logged when every question arrived unlabelled: the coverage of that
    response is unknown, and reporting every chapter as uncovered would state a
    fact the parser does not have. The absence of a label is a formatting miss,
    not a claim that those areas were skipped.
    """
    anchored = {question.chapter_id for question in parsed if question.chapter_id is not None}
    if not anchored:
        return
    uncovered = [
        chapter for chapter in chapters if chapter.id is not None and chapter.id not in anchored
    ]
    if not uncovered:
        return
    logger.warning(
        "Question generation for %s#%d left %d chapter(s) uncovered: %s",
        repo,
        pr_number,
        len(uncovered),
        ", ".join(f"{chapter.id} ({chapter.title})" for chapter in uncovered),
    )


def generate_questions_for_pr(
    pr_spec: str,
    github_token: str,
    jira_url: str = "",
    jira_username: str = "",
    jira_api_token: str = "",
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o",
    llm_api_key: str | None = None,
    *,
    llm_external_enabled: bool = False,
    llm_allowed_repositories: str | list[str] | tuple[str, ...] = "",
    ollama_url: str = "http://localhost:11434",
    llm_timeout_seconds: float = 30.0,
    llm_retries: int = 1,
) -> tuple[list[Question], dict]:
    """
    Given a PR identifier (URL or owner/repo#123), fetch PR diff and optional Jira context,
    call LLM to generate questions, and return (list of Question models, context dict for session).
    Context dict includes: pr_url, pr_number, repo, branch_name, jira_ticket_key,
    files_changed, head_sha.
    """
    parsed = parse_pr_identifier(pr_spec)
    if not parsed:
        raise ValueError(f"Invalid PR identifier: {pr_spec}")

    owner, repo_name = parsed[0].split("/", 1)
    pr_number = parsed[1]
    repo = f"{owner}/{repo_name}"
    validate_llm_privacy(
        repo,
        llm_provider,
        llm_external_enabled,
        llm_allowed_repositories,
        base_url=ollama_url if llm_provider.strip().lower() == "ollama" else "",
    )
    pr, diff, files_changed = _fetch_pr_context(github_token, owner, repo_name, pr_number, repo)
    jira_key = _resolve_jira_key(pr)
    pr_url = f"https://github.com/{owner}/{repo_name}/pull/{pr_number}"
    jira_context = _fetch_jira_context(jira_url, jira_username, jira_api_token, jira_key)
    prologue = build_prologue(pr.title, pr.body, jira_context)
    questions = _generate_pr_questions(
        diff,
        files_changed,
        prologue,
        repo=repo,
        pr_number=pr_number,
        jira_key=jira_key,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_external_enabled=llm_external_enabled,
        llm_allowed_repositories=llm_allowed_repositories,
        ollama_url=ollama_url,
        llm_timeout_seconds=llm_timeout_seconds,
        llm_retries=llm_retries,
    )
    context = _pr_question_context(
        pr, owner, repo_name, pr_number, repo, pr_url, jira_key, files_changed
    )
    return questions, context


def _fetch_pr_context(
    github_token: str, owner: str, repo_name: str, pr_number: int, repo: str
) -> tuple[Any, str, list[str]]:
    """Fetch the PR, its diff and its file list through one connection pool."""
    # Scoped to the three reads below, and closed by the block rather than left to
    # be dropped: the client owns one connection pool now, so a client that simply
    # goes out of scope holds its sockets until the process does. The three reads
    # share a pool on purpose -- a unified diff is megabytes, and rehandshaking
    # before each of these would cost more than the reads.
    with GitHubClient(token=github_token) as gh:
        pr = gh.get_pull_request(owner, repo_name, pr_number)
        diff = gh.get_pull_diff(owner, repo_name, pr_number)
        files_result = gh.get_pr_files(owner, repo_name, pr_number)
    if files_result.truncated:
        logger.warning(
            "Changed files for %s#%d hit the page cap; sizing the prompt from a partial file list.",
            repo,
            pr_number,
        )
    return pr, diff, files_result.items


def _resolve_jira_key(pr: Any) -> str | None:
    branch_name = pr.head["ref"] if pr.head else ""
    jira_key = extract_jira_key_from_branch(branch_name)
    # Fallback to PR description if not found in branch name
    if not jira_key and pr.body:
        jira_key = extract_jira_key_from_text(pr.body)
    return jira_key


def _fetch_jira_context(
    jira_url: str, jira_username: str, jira_api_token: str, jira_key: str | None
) -> JiraIssueFields | None:
    if jira_url and jira_username and jira_api_token and jira_key:
        try:
            jira = JiraClient(base_url=jira_url, username=jira_username, api_token=jira_api_token)
            return jira.get_issue_fields(jira_key)
        except JiraIntegrationError as exc:
            logger.warning(
                "Jira enrichment unavailable for %s; continuing without Jira context: %s",
                jira_key,
                exc,
            )
    return None


def _generate_pr_questions(
    diff: str,
    files_changed: list[str],
    prologue: Prologue,
    *,
    repo: str,
    pr_number: int,
    jira_key: str | None,
    llm_provider: str,
    llm_model: str,
    llm_api_key: str | None,
    llm_external_enabled: bool,
    llm_allowed_repositories: str | list[str] | tuple[str, ...],
    ollama_url: str,
    llm_timeout_seconds: float,
    llm_retries: int,
) -> list[Question]:
    max_q = _max_questions_for_diff(len(diff), len(files_changed))
    base_url = ollama_url if llm_provider.strip().lower() == "ollama" else ""
    context = {
        "repo": repo,
        "pr_number": pr_number,
        "jira_ticket_key": jira_key,
    }

    chapters = build_chapters(diff, files_changed)
    if not chapters:
        # No diff and no file list. The flat prompt is the honest fallback: there
        # is nothing to chapter, and refusing here would make an empty diff the one
        # input the generator cannot process rather than an input with no questions.
        prompt = build_questions_prompt(diff, prologue.jira or {}, max_questions=max_q)
        parsed_questions = generate_questions_sync(
            prompt,
            provider=llm_provider,
            model=llm_model,
            api_key=llm_api_key,
            repository=repo,
            external_enabled=llm_external_enabled,
            allowed_repositories=llm_allowed_repositories,
            base_url=base_url,
            timeout_seconds=llm_timeout_seconds,
            max_retries=llm_retries,
            max_questions=max_q,
        )
        return questions_to_models(parsed_questions, context=context)

    offered = [chapter for chapter in chapters if chapter.id is not None]
    per_chapter = _max_questions_per_chapter(len(offered), max_q)
    prompt = build_chaptered_questions_prompt(
        prologue,
        chapters,
        max_questions=max_q,
        max_questions_per_chapter=per_chapter,
    )
    parsed = generate_chaptered_questions_sync(
        prompt,
        provider=llm_provider,
        model=llm_model,
        api_key=llm_api_key,
        repository=repo,
        external_enabled=llm_external_enabled,
        allowed_repositories=llm_allowed_repositories,
        base_url=base_url,
        timeout_seconds=llm_timeout_seconds,
        max_retries=llm_retries,
        chapter_ids=[chapter.id for chapter in offered if chapter.id is not None],
        max_questions=max_q,
        max_questions_per_chapter=per_chapter,
    )
    _warn_about_uncovered_chapters(chapters, parsed, repo=repo, pr_number=pr_number)
    return questions_to_models(
        [(question.category, question.text) for question in parsed],
        context=context,
    )


def _pr_question_context(
    pr: Any,
    owner: str,
    repo_name: str,
    pr_number: int,
    repo: str,
    pr_url: str,
    jira_key: str | None,
    files_changed: list[str],
) -> dict:
    branch_name = pr.head["ref"] if pr.head else ""
    return {
        "pr_url": pr_url,
        "pr_number": pr_number,
        "repo": f"{owner}/{repo_name}",
        "owner": owner,
        "repo_name": repo_name,
        "branch_name": branch_name,
        "jira_ticket_key": jira_key,
        "files_changed": files_changed,
        # The commit the diff above was read at. Taken from the same fetch as
        # ``pr`` and ``diff`` rather than re-read at post time, so the anchor and
        # the questions it describes cannot come from two different heads: the
        # questions are about *this* diff, so a head that moves during the LLM
        # call afterwards is irrelevant, while a re-read would record a later
        # commit and quietly relabel questions nobody read.
        #
        # ``.get`` rather than ``["sha"]``: ``head`` is an untyped optional dict,
        # and GitHub omitting the sha is a weaker fact about the anchor, not a
        # reason to abandon a question the reviewer already generated. Absent
        # stays absent -- the same rule the v7 migration follows by leaving old
        # rows NULL rather than reconstructing a head for them.
        "head_sha": pr.head.get("sha") if pr.head else None,
    }
