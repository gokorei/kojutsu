"""Owner/name repository identifiers, parsed in exactly one place.

A leaf module: stdlib only, imported by the webhook, the backfills, the
worker and the MCP servers alike. Six copies of ``owner, _, name =
repo.partition("/")`` used to disagree about what counts as a repository --
some refused an empty owner, some did not, one silently read ``a/b/c`` as
owner ``a`` and name ``b/c``. A repository token with two slashes is not a
repository with a funny name; dropping a segment silently is how a read lands
on the wrong repository, so the strictest historical reading wins and every
caller shares it.
"""

__all__ = ["split_repo"]


def split_repo(repo: str) -> tuple[str, str]:
    """Split an ``owner/name`` repository token.

    Exactly one slash, non-empty owner, non-empty name. Anything else raises
    :class:`ValueError` naming what was received, because a malformed
    repository that parses anyway parses *wrong* -- and a wrong repository is
    a read or a write against somebody else's code.
    """
    owner, separator, name = repo.partition("/")
    if not separator or not owner or not name or "/" in name:
        raise ValueError(f"repo must be 'owner/name', got {repo!r}")
    return owner, name
