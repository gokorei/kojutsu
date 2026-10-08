"""The repository scope, and the two case rules that look contradictory.

kojutsu folds case when *authorising* and not when *filtering*, which reads
like an inconsistency until you know that GitHub defines repository names as
case-insensitive and that the filter exists to keep an index usable. Both halves
are pinned here, because they are the two things a reader is most likely to
"fix" into agreeing with each other -- and doing so silently drops real reviews
or mixes another repository's work into a page.

These tests also pin the defect that motivated the shared module: the webhook
and the MCP server used to parse the same environment variable with different
acceptance criteria, so a repository could be captured and then not readable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kojutsu import allowlist
from kojutsu.config import Settings
from kojutsu.core.question_registry import SqliteQuestionRegistry


def _settings(allowlist_value: str) -> Settings:
    return Settings(github_webhook_allowed_repositories=allowlist_value)


def test_authorisation_folds_case_because_github_defines_names_that_way() -> None:
    """The allowlist folds case, and that is the correct rule for a forge.

    GitHub treats ``owner/repo`` case-insensitively, so a repository genuinely
    called ``Owner/Name`` must be authorised by an allowlist reading
    ``owner/name``. Refusing to fold would silently drop real reviews.

    The opposite error is the more dangerous one to reason about, so it is worth
    stating: folding when it should not have accepts a repository the operator
    did not name. That is a real risk, and it is the price of agreeing with the
    platform rather than imposing a stricter spelling than the platform has.
    """
    settings = _settings("owner/name")

    assert allowlist.repository_allowed("Owner/Name", settings)
    assert allowlist.repository_allowed("owner/name", settings)
    assert allowlist.repository_allowed("OWNER/NAME", settings)


def test_enumeration_does_not_fold_case_because_it_is_an_index_filter(
    tmp_path: Path,
) -> None:
    """The ``list_questions`` filter matches exactly, and for a stated reason.

    Matching exactly keeps ``questions_work_idx`` usable, and a mis-cased filter
    shows *less* work rather than another repository's questions mixed into the
    page. Failing toward less is the safe direction for an enumeration surface.
    """
    with SqliteQuestionRegistry(tmp_path / "registry.db") as registry:
        registry.record_question(
            question_id="q1",
            github_comment_id=100,
            repo="org/repo",
            pr_number=1,
            pr_url="https://github.com/org/repo/pull/1",
            question_text="Why?",
            question_category="design_decision",
        )
        assert len(registry.list_questions(repo="org/repo")) == 1
        assert registry.list_questions(repo="Org/Repo") == []


def test_a_wildcard_is_never_authorisation() -> None:
    settings = _settings("*")

    assert allowlist.configured_repositories(settings) == frozenset()
    assert not allowlist.repository_allowed("org/repo", settings)
    assert not allowlist.repository_allowed("*", settings)


def test_an_empty_allowlist_denies_everything() -> None:
    for value in ("", "   ", ",,,"):
        settings = _settings(value)
        assert allowlist.configured_repositories(settings) == frozenset()
        assert not allowlist.repository_allowed("org/repo", settings)


def test_a_malformed_entry_is_refused_rather_than_skipped() -> None:
    """A typo must not become a repository the operator thinks is collected.

    Silently dropping an unparseable entry would leave a configuration that reads
    as correct and captures nothing.
    """
    for bad in ("org/repo/extra", "justarepo", "org/", "/repo", "org/re po"):
        with pytest.raises(allowlist.AllowlistError):
            allowlist.configured_repositories(_settings(f"org/repo,{bad}"))


def test_the_allowlist_is_bounded() -> None:
    entries = ",".join(f"org{i}/repo" for i in range(allowlist.MAX_REPOSITORIES + 1))

    with pytest.raises(allowlist.AllowlistError, match="misconfiguration"):
        allowlist.configured_repositories(_settings(entries))


def test_the_allowlist_is_not_a_glob_or_a_prefix_rule() -> None:
    """A scope that a character can widen is not a scope."""
    for pattern in ("org/*", "*/repo", "org/re*", "org"):
        with pytest.raises(allowlist.AllowlistError):
            allowlist.configured_repositories(_settings(pattern))


def test_capture_and_read_authorise_the_same_repositories() -> None:
    """The defect this module exists to fix.

    The webhook and the MCP server authorise against the same environment
    variable. They used to parse it separately, and the webhook applied no token
    validation at all -- so a repository the webhook accepted and captured could
    be rejected by the read path, leaving a review that existed and could not be
    read back.
    """
    from mcp_server import server as mcp_server_module

    from kojutsu.webhook import server as webhook_server

    for value in ("org/repo", "Org/Repo,other/repo", "*", "", "org/repo,bad-token"):
        settings = _settings(value)
        webhook_scope = _webhook_scope_or_error(webhook_server, settings)
        mcp_scope = mcp_server_module._allowed_repositories(settings)

        assert webhook_scope == mcp_scope, f"capture and read disagree on {value!r}"
        for repository in ("org/repo", "Org/Repo", "other/repo", "bad-token"):
            assert allowlist.repository_allowed(
                repository, settings
            ) == mcp_server_module._repository_allowed(settings, repository), (
                f"the two surfaces disagree about {repository!r} for {value!r}"
            )


def _webhook_scope_or_error(webhook_server: object, settings: Settings) -> frozenset[str]:
    """The webhook's view of the scope, whether it is usable or refused."""
    from fastapi import HTTPException

    try:
        return webhook_server._require_repository_allowlist(settings)  # type: ignore[attr-defined]
    except HTTPException:
        return frozenset()


def test_parse_repository_list_normalises_csv_and_sequences() -> None:
    """One lenient parser for scopes that are compared, not validated: blanks
    are dropped, case is folded, and a CSV string and a sequence agree."""
    assert allowlist.parse_repository_list("org/repo, Other/Repo,, ") == frozenset(
        {"org/repo", "other/repo"}
    )
    assert allowlist.parse_repository_list(["org/repo", "OTHER/REPO"]) == frozenset(
        {"org/repo", "other/repo"}
    )
    assert allowlist.parse_repository_list(("org/repo",)) == frozenset({"org/repo"})
    assert allowlist.parse_repository_list("") == frozenset()
