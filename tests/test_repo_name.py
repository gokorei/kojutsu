"""A repository token splits exactly one way, everywhere."""

import pytest

from kojutsu.repo_name import split_repo


def test_well_formed_token_splits() -> None:
    assert split_repo("org/repo") == ("org", "repo")


@pytest.mark.parametrize("bad", ["", "noslash", "/name", "owner/", "a/b/c", "a//b"])
def test_malformed_tokens_are_refused(bad: str) -> None:
    """Every historical variant disagreed about at least one of these: an empty
    owner passed in one, ``a/b/c`` silently dropped a segment in four."""
    with pytest.raises(ValueError, match="repo must be 'owner/name'"):
        split_repo(bad)
