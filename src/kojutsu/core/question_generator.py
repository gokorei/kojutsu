"""Orchestrates context gathering and LLM question generation for a PR."""

import logging

from kojutsu.integrations.github import (
    GitHubClient,
    extract_jira_key_from_branch,
    extract_jira_key_from_text,
    parse_pr_identifier,
)
from kojutsu.integrations.jira_client import JiraClient, JiraIntegrationError, JiraIssueFields
from kojutsu.integrations.llm import (
    build_questions_prompt,
    generate_questions_sync,
    questions_to_models,
    validate_llm_privacy,
)
from kojutsu.models import Question

logger = logging.getLogger(__name__)


def _max_questions_for_diff(diff_length: int, files_count: int) -> int:
    """Suggest max questions based on diff size and file count."""
    if diff_length < 500 and files_count <= 2:
        return 2
    if diff_length < 2000 and files_count <= 5:
        return 3
    if diff_length < 8000:
        return 5
    return 6


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
    files_changed = files_result.items

    branch_name = pr.head["ref"] if pr.head else ""
    jira_key = extract_jira_key_from_branch(branch_name)

    # Fallback to PR description if not found in branch name
    if not jira_key and pr.body:
        jira_key = extract_jira_key_from_text(pr.body)

    pr_url = f"https://github.com/{owner}/{repo_name}/pull/{pr_number}"

    jira_context: JiraIssueFields | None = None
    if jira_url and jira_username and jira_api_token and jira_key:
        try:
            jira = JiraClient(base_url=jira_url, username=jira_username, api_token=jira_api_token)
            jira_context = jira.get_issue_fields(jira_key)
        except JiraIntegrationError as exc:
            logger.warning(
                "Jira enrichment unavailable for %s; continuing without Jira context: %s",
                jira_key,
                exc,
            )

    max_q = _max_questions_for_diff(len(diff), len(files_changed))
    prompt = build_questions_prompt(diff, jira_context or {}, max_questions=max_q)
    parsed_questions = generate_questions_sync(
        prompt,
        provider=llm_provider,
        model=llm_model,
        api_key=llm_api_key,
        repository=repo,
        external_enabled=llm_external_enabled,
        allowed_repositories=llm_allowed_repositories,
        base_url=ollama_url if llm_provider.strip().lower() == "ollama" else "",
        timeout_seconds=llm_timeout_seconds,
        max_retries=llm_retries,
        max_questions=max_q,
    )
    questions = questions_to_models(
        parsed_questions,
        context={
            "repo": repo,
            "pr_number": pr_number,
            "jira_ticket_key": jira_key,
        },
    )

    context = {
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
    return questions, context
