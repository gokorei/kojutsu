"""One definition of the repository allowlist, shared by capture and read.

The webhook and the MCP server both authorise against
``GITHUB_WEBHOOK_ALLOWED_REPOSITORIES``. They used to parse it separately, and
not identically: the webhook applied no token validation at all, while the MCP
server required a pattern. That made the *write* scope looser than the *read*
scope, so kojutsu could capture a review from a repository and then refuse to
read it back. One definition, two callers, is the fix.

**Case folding is deliberate, and is the opposite of what a pure ledger would
choose.** A system that both writes and reads its own repository tokens has one
canonical spelling, and refusing to fold is the stricter, safer rule. kojutsu
does not: repository names arrive from GitHub, which defines them as
case-insensitive. Folding here is what makes the allowlist agree with the
platform it reads from. The cost of being wrong is asymmetric, and that is the
whole argument -- folding when it should not have accepts a repository the
operator did not name, whereas refusing to fold when it should have silently
drops a real review that nobody notices is missing.

The contrast with the *filter* surfaces is deliberate too. ``list_questions``
matches its ``repo`` argument exactly, so the work index stays usable and a
mis-cased filter shows less work rather than another repository's. Authorisation
folds; enumeration does not. See the note on
:meth:`kojutsu.core.question_registry.SqliteQuestionRegistry.list_questions`.

``*`` is never authorisation. It is dropped from the scope, and the existing
behaviour — an empty or missing allowlist denies every repository — is preserved
here rather than re-implemented at each call site.
"""

from __future__ import annotations

import re

from kojutsu.config import Settings

__all__ = [
    "ADMIT_ALL_ASSOCIATIONS",
    "MAX_REPOSITORIES",
    "REPOSITORY_TOKEN",
    "AllowlistError",
    "association_admitted",
    "configured_repositories",
    "is_valid_repository",
    "parse_repository_list",
    "repository_allowed",
]

#: ``owner/name``, both segments limited to the characters GitHub permits in a
#: repository name. Deliberately not a glob, a prefix rule, or a pattern with
#: alternatives: a scope that can be widened by a character in an environment
#: variable is not a scope.
REPOSITORY_TOKEN = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")

#: A scope with thousands of entries is a misconfiguration, not a scope. The
#: bound exists so an operator typo cannot turn the allowlist into something
#: unbounded; nobody is expected to approach it.
MAX_REPOSITORIES = 256


class AllowlistError(ValueError):
    """The configured repository allowlist is not usable as written."""


def is_valid_repository(repository: str) -> bool:
    """Return whether ``repository`` is one well-formed ``owner/name`` token."""
    return bool(repository) and REPOSITORY_TOKEN.fullmatch(repository) is not None


def parse_repository_list(value: str | list[str] | tuple[str, ...]) -> frozenset[str]:
    """Split a CSV-or-sequence repository scope into a case-folded set.

    Lenient by design: blank entries are dropped and nothing is validated.
    Validation lives in :func:`configured_repositories`, which names the bad
    entry; this is for scopes that are *compared* (membership tests), where a
    blank entry or a case difference must not decide the answer. Case-folded
    because repository identity is case-insensitive and ``Org/Repo`` in one
    setting and ``org/repo`` in another is one repository, not two.
    """
    items = value.split(",") if isinstance(value, str) else list(value)
    return frozenset(item.strip().casefold() for item in items if item.strip())


def configured_repositories(settings: Settings) -> frozenset[str]:
    """Return the case-folded, validated, non-wildcard repository scope.

    Raises:
        AllowlistError: if a configured entry is not a well-formed
            ``owner/name`` token, or if more than :data:`MAX_REPOSITORIES` are
            configured. Failing loudly matters more than tolerating a typo: an
            entry that silently failed to parse would be an entry the operator
            believes is being collected.
    """
    raw = settings.github_webhook_allowed_repositories
    entries = [item.strip() for item in raw.split(",") if item.strip()]

    if len(entries) > MAX_REPOSITORIES:
        raise AllowlistError(
            f"Repository allowlist has {len(entries)} entries, more than the "
            f"{MAX_REPOSITORIES} supported. This is a misconfiguration, not a scope."
        )

    scope: set[str] = set()
    for entry in entries:
        if entry == "*":
            # A wildcard narrows to nothing rather than widening to everything.
            # See the module docstring: it is never authorisation.
            continue
        if not is_valid_repository(entry):
            raise AllowlistError(
                f"Repository allowlist entry {entry!r} is not a valid owner/name value."
            )
        scope.add(entry.casefold())
    return frozenset(scope)


def repository_allowed(repository: str | None, settings: Settings) -> bool:
    """Return whether ``repository`` is inside the configured scope.

    Comparison is case-folded for the reason given in the module docstring.

    This is a predicate, so it does not raise. A malformed allowlist denies
    rather than propagating, which means every caller fails closed and the
    capture and read paths cannot disagree about what a broken configuration
    permits. The startup and readiness paths use :func:`configured_repositories`
    directly, which does raise, because refusing to serve is the correct
    response there and a boolean is not.
    """
    if not isinstance(repository, str):
        return False
    try:
        scope = configured_repositories(settings)
    except AllowlistError:
        return False
    return repository.casefold() in scope


#: Which GitHub ``author_association`` values are admitted by default: **all of
#: them**, which is why this is ``None`` and not a set.
#:
#: ``None`` is the whole policy and it is not a placeholder for a set that has not
#: been written down yet. It has to be the meaning, because the override is a *set*
#: and an operator who passes one means *only these*: a default that enumerated
#: every value GitHub can send would have to be edited the day GitHub adds another,
#: and would be indistinguishable from an explicit narrowing to a reader holding a
#: record. The distinction is stated once, here, and applied once, by
#: :func:`association_admitted`.
#:
#: **It was ``{OWNER, MEMBER, COLLABORATOR}``, and on the evidence it refused 28
#: human comments to admit 21 bot ones.** Measured live against ``pingdotgg/t3code``
#: and ``fastapi/fastapi``: 400 of the 439 stored review captures re-fetched from
#: the forge were 278 ``MEMBER`` and 118 ``COLLABORATOR``, with **zero**
#: ``CONTRIBUTOR`` or ``NONE``. So the gate had never once refused a review, and
#: there were no bot reviews in the corpus to refuse either — it cost nothing on the
#: path it was written for. The damage was on the comment path. Issue comments on
#: t3code's PR #2829:
#:
#: | association | bots | humans |
#: | --- | --- | --- |
#: | ``MEMBER`` | 0 | 12 |
#: | ``CONTRIBUTOR`` | **21** — cursor[bot] ×12, macroscopeapp[bot] ×6, github-actions[bot] ×2, coderabbitai[bot] ×1 | 6 |  # noqa: RUF003
#: | ``NONE`` | 0 | ~28 |
#:
#: **The field is anti-correlated with automation.** A bot is by definition not a
#: member or collaborator of anything, so ``CONTRIBUTOR`` is where automated
#: reviewers land — every one of the twenty-one above is a ``CONTRIBUTOR``. Widening
#: to ``CONTRIBUTOR`` therefore admits every automated reviewer on that change, and
#: widening to ``NONE`` admits only humans. One field cannot answer "may this
#: account write?" and "is this automated?" at the same time, and using it for both
#: is what produced the exchange rate above.
#:
#: **What is load-bearing now, and had better hold.** The repository allowlist
#: above, checked before anything is written, and failing closed on a malformed
#: configuration; the automation flag recorded on every stored comment
#: (:func:`kojutsu.core.answer_collector.is_machine_account`, whose verdict is
#: provenance rather than a filter); and ``independence``, which is what still
#: lets a reader see how far a record is from checking itself. Admission is no
#: longer a claim about standing, so the reasons a reader needs are recorded
#: instead of enforced.
#:
#: Lives here, beside the repository allowlist, rather than in any one collector:
#: this codebase prefers a single definition imported by several callers, and a
#: policy constant declared in two modules is the exact failure this module was
#: added to prevent.
ADMIT_ALL_ASSOCIATIONS: frozenset[str] | None = None


def association_admitted(
    association: str | None,
    authorized_associations: frozenset[str] | None = ADMIT_ALL_ASSOCIATIONS,
) -> bool:
    """Whether ``association`` is admitted under ``authorized_associations``.

    The one place the ``None``-means-everything rule is applied, because it was
    previously spelled out at each of five gates and the spelling included a bug:
    ``authorized_associations or AUTHORIZED_COMMENT_ASSOCIATIONS`` read an empty
    set as "no opinion" rather than "admit nothing", so a caller narrowing to
    ``frozenset()`` silently got the default. ``None`` is now distinguished from
    empty by identity, and an empty set refuses everything.

    Association values are compared case-folded upward because GitHub capitalises
    them and a hand-typed ``member`` should not refuse every review it was meant to
    admit. A comment GitHub attributes to nobody is ``""`` after folding, so it is
    admitted only by an unrestricted policy — which is the honest reading, since the
    forge declined to characterise the account rather than characterising it as an
    outsider.
    """
    if authorized_associations is None:
        return True
    return (association or "").strip().upper() in authorized_associations
