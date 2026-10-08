"""Tests for the declaration posting tool and the read-server freeze it enabled.

Two halves, and the second matters as much as the first.

The tool tests cover the properties that make this a *posting* tool rather than a
second write path: it cannot name a repository outside the allowlist, it cannot
post twice for one declaration, and it releases its claim when a post fails.

The freeze tests live in ``test_mcp_server.py`` because they are a statement about
the read server. They are the test
``docs/design-review/open-questions.md`` asked for, and the existence of this
write server is the change they make legitimate.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

from kojutsu.config import Settings

SERVER_MODULE = "mcp_server.capture_server"


def load_capture_server():
    return importlib.reload(importlib.import_module(SERVER_MODULE))


class FakeComment:
    def __init__(self, comment_id: int) -> None:
        self.id = comment_id


class FakeGitHub:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.posted: list[tuple[str, str, int, str]] = []

    def post_issue_comment(
        self, owner: str, repo: str, issue_number: int, body: str
    ) -> FakeComment:
        if self.fail:
            raise RuntimeError("403 Forbidden")
        self.posted.append((owner, repo, issue_number, body))
        return FakeComment(1000 + len(self.posted))


class FakeRegistry:
    def __init__(self) -> None:
        self.claimed: list[dict[str, Any]] = []
        self.completed: list[str] = []
        self.released: list[tuple[str, str]] = []
        self.closed = False
        self.hold: set[str] = set()

    def claim_rationale(self, **kwargs: Any) -> str | None:
        entry_id = str(kwargs["entry_id"])
        if entry_id in self.hold:
            return None
        self.hold.add(entry_id)
        self.claimed.append(kwargs)
        return "token-for-" + entry_id

    def complete_rationale(self, entry_id: str, claim_token: str) -> bool:
        self.completed.append(entry_id)
        return True

    def release_rationale(self, entry_id: str, claim_token: str, error: str) -> bool:
        self.released.append((entry_id, error))
        self.hold.discard(entry_id)
        return True

    def close(self) -> None:
        self.closed = True


class FakeSink:
    def __init__(self) -> None:
        self.entries: list[Any] = []

    def store(self, entry: Any):
        self.entries.append(entry)

    def close(self) -> None:
        pass


@pytest.fixture
def capture():
    module = load_capture_server()
    settings = Settings(
        github_token="ghp_faketokenfortests0000000000000000000000",
        github_webhook_allowed_repositories="org/repo",
    )
    github = FakeGitHub()
    registry = FakeRegistry()
    sink = FakeSink()
    module.configure(
        settings=settings,
        client_factory=lambda _: github,
        registry_factory=lambda _: registry,
        sink_factory=lambda _: sink,
    )
    return module, github, registry, sink


def _record(capture, **overrides: Any):
    module, _, _, _ = capture
    fields: dict[str, Any] = {
        "repo": "org/repo",
        "reason": "Chose exponential backoff because the API rate-limits on 429.",
        "agent_id": "opencode",
        "branch": "feat/backoff",
        "pr_number": 42,
        "model": "opencode/model",
    }
    fields.update(overrides)
    return module.record_decision_rationale(**fields)


# --- posting through the ordinary path ---------------------------------------


def test_a_declaration_is_posted_as_a_marked_comment(capture) -> None:
    result = _record(capture)

    assert result.ok is True
    _module, github, registry, _sink = capture
    assert len(github.posted) == 1
    owner, repo, number, body = github.posted[0]
    assert (owner, repo, number) == ("org", "repo", 42)
    assert "kojutsu:rationale:opencode" in body
    assert "Chose exponential backoff" in body
    assert registry.completed, (
        "a posted declaration whose claim was never finalised would block its own retry"
    )


def test_the_marker_precedes_the_reason(capture) -> None:
    """Attribution first, so a reader skimming meets who is claiming what first."""
    _, github, _, _ = capture
    _record(capture)
    body = github.posted[0][3]
    assert body.index("kojutsu:rationale:") < body.index("Chose exponential backoff")


def test_a_declaration_is_published_and_stored(capture) -> None:
    """Both halves, because a change on a forge is worth publishing on it.

    The record is written here rather than by a later webhook, so the tool's
    success message is true at the moment it is returned. Storing here and then
    letting the webhook capture the same comment does not double it: both derive
    the same entry id from the anchor, and the second one finds the claim
    completed.
    """
    result = _record(capture)

    assert result.ok is True
    _module, github, registry, sink = capture
    assert len(github.posted) == 1
    owner, repo, number, body = github.posted[0]
    assert (owner, repo, number) == ("org", "repo", 42)
    assert "kojutsu:rationale:opencode" in body
    assert "Chose exponential backoff" in body
    assert len(sink.entries) == 1
    assert sink.entries[0].channel.value == "forge_comment"
    assert registry.completed, (
        "a stored declaration whose claim was never finalised would block its own retry"
    )


def test_the_marker_carries_the_branch_so_the_collector_can_derive_the_same_id(capture) -> None:
    """The webhook has nothing but the comment body to identify the record by.

    If the branch were missing here the collector would fall back to a different
    anchor, and one declaration would become two records.
    """
    _record(capture, branch="feat/backoff")
    _module, github, _registry, sink = capture
    body = github.posted[0][3]
    assert "branch=feat/backoff" in body
    assert sink.entries[0].branch == "feat/backoff"


# --- the direct route, for an agent with nowhere to publish -------------------


def test_a_declaration_with_no_pull_request_is_stored_without_publishing(capture) -> None:
    """The case the comment-only design made impossible.

    An agent working from a local repository has no forge to post to, and a
    rationale it could not record is a rationale nobody will read. So the record
    is written directly, through the same claim and the same store.
    """
    result = _record(capture, pr_number=None, branch="feat/local-only")

    assert result.ok is True
    _module, github, _registry, sink = capture
    assert github.posted == [], "there is nowhere to publish to, so nothing is posted"
    assert len(sink.entries) == 1
    entry = sink.entries[0]
    assert entry.channel.value == "capture_server"
    assert entry.pr_number is None
    assert entry.branch == "feat/local-only"
    assert "without publishing" in (result.result or ""), (
        "the caller has to be told the reason was not put in front of a human, or "
        "it will assume a reader saw it"
    )


def test_both_routes_derive_the_same_identity_for_the_same_declaration(capture) -> None:
    """Two routes, one identity, one record.

    This is what makes the direct path a transport choice rather than a second
    privileged write path: the same declaration cannot become two records, and a
    comment posted for it later is recognised rather than stored again.
    """
    from kojutsu.core.question_registry import stable_rationale_entry_id

    _record(capture, pr_number=None, branch="feat/local-only")
    _module, _, registry, sink = capture

    direct_id = sink.entries[0].entry_id
    assert direct_id == stable_rationale_entry_id(
        repo="org/repo",
        pr_number=None,
        branch="feat/local-only",
        declared_by="opencode",
        revision=1,
    )
    assert registry.claimed[0]["entry_id"] == direct_id


def test_a_declaration_naming_no_change_at_all_is_refused(capture) -> None:
    """A rationale attached to nothing is one nobody will find again."""
    result = _record(capture, pr_number=None, branch="")

    assert result.ok is False
    assert result.code == "invalid_input"
    _module, github, _registry, sink = capture
    assert github.posted == []
    assert sink.entries == []


def test_neither_route_reaches_a_second_delivery_path(capture) -> None:
    """The direct route uses the store's own sink, not a queue of its own.

    A rationale that reached the store by a different mechanism would be a thing
    that could be lost differently from everything else, which is the failure the
    outbox exists to prevent.
    """
    module, _, _, _ = capture
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "TansekiKnowledgeSink" in source, "it must use the shared sink, not a bespoke one"
    assert source.count("TansekiOutbox(") == 1, "one outbox, built from configuration only"
    for forbidden in ("subprocess", "requests.post"):
        assert forbidden not in source


# --- the allowlist, enforced on the way IN -----------------------------------


def test_a_repository_outside_the_allowlist_is_refused(capture) -> None:
    """The write-time check, which the read path does not have.

    The read server checks the same variable, but only on the way out. A
    declaration arrives from an agent that has read attacker-controlled issue and
    pull request text, so the decision has to be made before anything is written.
    """
    result = _record(capture, repo="other/repo")

    assert result.ok is False
    assert result.code == "repository_not_authorized"
    assert result.retryable is False, "a policy refusal is not a transient fault"


def test_nothing_is_posted_or_claimed_for_a_refused_repository(capture) -> None:
    _, github, registry, _ = capture
    _record(capture, repo="other/repo")

    assert github.posted == []
    assert registry.claimed == [], (
        "a refused declaration must not take a claim it will never finish"
    )


def test_a_malformed_allowlist_is_reported_rather_than_reading_as_nothing(capture) -> None:
    """A broken configuration must not become an empty scope that reads as a denial."""
    module, github, registry, _sink = capture
    module.configure(
        settings=Settings(
            github_token="ghp_faketokenfortests0000000000000000000000",
            github_webhook_allowed_repositories="not-a-repo",
        )
    )

    result = _record(capture)

    assert result.ok is False
    assert result.code == "allowlist_invalid"
    assert github.posted == []
    assert registry.claimed == []


# --- bounded input, because this is a write ----------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"reason": ""},
        {"repo": ""},
        {"agent_id": ""},
        {"reason": "x" * 10_000},
        {"repo": "org/" + "x" * 300},
        {"agent_id": "a" * 100},
        {"model": "m" * 200},
        {"branch": "b" * 400},
        {"revision": 0},
        {"revision": -1},
        {"revision": True},
        {"pr_number": 0},
        {"pr_number": "42"},
        {"repo": "nodelimiter"},
        {"repo": "a/b/c"},
    ],
)
def test_malformed_input_is_refused_rather_than_posted(capture, overrides: dict[str, Any]) -> None:
    """A write tool on a stdio transport is reachable by anything speaking the protocol.

    Unbounded input here is the same denial-of-service primitive ``open-questions.md``
    discusses for the webhook, so the bound goes where the input arrives.
    """
    result = _record(capture, **overrides)
    _, github, registry, _ = capture

    assert result.ok is False
    assert result.code == "invalid_input"
    assert result.retryable is False, "the caller sent too much; sending it again will not help"
    assert github.posted == []
    assert registry.claimed == []


# --- exactly once, across failures --------------------------------------------


def test_a_repeat_call_posts_nothing_and_says_so(capture) -> None:
    """A second call for the same declaration is a no-op, not a second comment.

    The second comment would be a second record saying the same thing, and the
    reader would have no way to tell the duplication from two real declarations.
    """
    first = _record(capture)
    second = _record(capture)
    _, github, _, _ = capture

    assert first.ok is True
    assert second.ok is True
    assert len(github.posted) == 1
    assert "already recorded" in (second.result or "")


def test_a_higher_revision_posts_again_and_supersedes_the_first(capture) -> None:
    """Intent captured early goes stale, so a later rationale is appended."""
    _record(capture)
    second = _record(capture, revision=2, reason="Dropped the jitter; it hid the latency.")
    _, github, registry, _ = capture

    assert second.ok is True
    assert len(github.posted) == 2
    assert registry.claimed[1]["revises"] == registry.claimed[0]["entry_id"]
    assert registry.claimed[1]["revision"] == 2


def test_a_failed_post_releases_the_claim_so_the_retry_can_work(capture) -> None:
    """A claim left held by a failed post would silently refuse the retry that works."""
    module, _, registry, _sink = capture
    github = FakeGitHub(fail=True)
    module.configure(client_factory=lambda _: github)

    result = _record(capture)

    assert result.ok is False
    assert result.code == "capture_unavailable"
    assert result.retryable is True
    assert registry.released, "a claim must not outlive a post that never happened"
    assert registry.completed == []


def test_a_failure_response_does_not_leak_the_underlying_error(capture) -> None:
    """A GitHub error body can echo the URL, and the URL is where a token would be.

    So the cause is dropped entirely rather than redacted and returned.
    """
    module, _, _, _ = capture
    module.configure(client_factory=lambda _: FakeGitHub(fail=True))

    result = _record(capture)

    assert "403" not in (result.error or "")
    assert "Forbidden" not in (result.error or "")
    assert result.error and "Verify the store and service health" in result.error


def test_the_registry_is_closed_even_when_the_post_fails(capture) -> None:
    module, _, registry, _sink = capture
    module.configure(client_factory=lambda _: FakeGitHub(fail=True))

    _record(capture)

    assert registry.closed is True, "a leaked registry handle survives as an open WAL file"


# --- the honest limit ---------------------------------------------------------


def test_the_module_states_that_it_has_no_caller_identity(capture) -> None:
    """The docstring is the only place this claim is made, so assert it is there.

    A stdio server with a single trust domain cannot know who invoked it. What
    can honestly be claimed is narrow: this build exposes one tool, it can only
    post on an allow-listed repository with the capture process's token, and the
    declaration is a self-assertion by the posting account. That is not the same
    as "no caller can induce a write", and the module must not imply it is.
    """
    module, _, _, _ = capture
    docstring = " ".join((module.__doc__ or "").lower().split())

    assert "no caller identity" in docstring
    assert "not the same" in docstring


def test_the_write_tool_is_not_annotated_as_read_only(capture) -> None:
    """Marking a write tool read-only would defeat the freeze test's purpose."""
    module, _, _, _ = capture
    assert module._WRITE_ANNOTATIONS.read_only_hint is False
    assert module._WRITE_ANNOTATIONS.idempotent_hint is True


# --- one declaration, two routes, and the policy that has to cover both -------
#
# A rationale reaches one document id from two places: out of the text this tool was
# given, and out of the comment this tool posts, read later by
# ``core/rationale_collector.py``. They are mutually exclusive per declaration --
# ``claim_rationale`` refuses a row that is already ``completed``, and this module
# completes before it stores -- so the route that writes the surviving document is
# this one, almost always. That is what makes the character policy here load-bearing
# rather than redundant: the collector's copy of it is unreachable for the common
# case.
#
# The argument is in the module docstring. What these pin is that both routes really
# do agree, which is a stronger claim than "both call ``sanitise``" and the one that
# was not true before.

RLO = "\u202e"
BELL = "\u0007"


def test_both_routes_store_the_same_wording_for_one_declaration(capture) -> None:
    """The real round trip, not a comparison of two calls to the same function.

    This posts the declaration through the tool, then takes the comment it actually
    posted and reads it back the way the collector does -- marker stripped, prose
    normalised by ``extract_rationale_text_from_comment_body`` -- and asserts the
    result is character-for-character what the direct route stored.

    The route that would make the two disagree is the collector's, because it applies
    its own narrower normalisation inside the extractor before ``sanitise`` runs: it
    stores ``sanitise(normalise(x))`` where this route stores ``sanitise(x)``. They
    are equal because the refused set is a *superset* of the extractor's -- composing
    the narrower pass with the wider one removes the same set as the wider one alone.

    That is a property of two enumerated sets, not of this function, which is exactly
    why it is asserted here rather than asserted in a docstring and trusted. A future
    edit that widened the refused set past the extractor's, or narrowed the
    extractor's without saying so, breaks the equality and this fails.
    """
    from kojutsu.core.text_hygiene import sanitise
    from kojutsu.integrations.github import extract_rationale_text_from_comment_body

    reason = f"Chose exponential backoff because the API rate-limits on 429.{RLO}{BELL}"
    _record(capture, reason=reason)
    _, github, _, sink = capture

    stored = sink.entries[0]
    posted = github.posted[0][3]

    # Both passes the collector applies, in its order: the extractor's narrower
    # normalisation, then the wider policy on the extracted prose.
    collector_side = sanitise(extract_rationale_text_from_comment_body(posted)).text

    assert stored.rationale_text == "Chose exponential backoff because the API rate-limits on 429."
    assert collector_side == stored.rationale_text, (
        "one declaration must store one wording, whichever route writes the document"
    )
    # The comment itself is *not* sanitised: this process is a courier, and a courier
    # that edits the text becomes its author. The store holds a disclosed rendering of
    # the comment, and the comment stays the thing a reader can re-fetch.
    assert RLO in posted and BELL in posted


def test_both_routes_record_what_the_policy_took_out(capture) -> None:
    """Same key, same wording, so a reader need not know which route wrote the document.

    The collector measures its note against the delivered comment body and this route
    against the text it was handed; after the ordering above both are looking at a
    string carrying the same refused characters, so the two notes are equal rather
    than merely similar. Asserted as equality because a note that differed between
    routes would leave a reader comparing two documents for one declaration and
    finding two accounts of what was removed from it.
    """
    from kojutsu.core.text_hygiene import SANITISATION_KEY, describe_removals

    reason = f"Chose exponential backoff because the API rate-limits on 429.{RLO}{BELL}"
    _record(capture, reason=reason)
    _, github, _, sink = capture

    note = sink.entries[0].metadata[SANITISATION_KEY]
    assert note == describe_removals(reason)
    # What the collector computes from the comment it receives.
    assert note == describe_removals(github.posted[0][3])


def test_a_clean_declaration_carries_no_sanitisation_note(capture) -> None:
    """Absence is the claim, and it has to be reachable from this route too.

    The note is omitted rather than written empty: ``""`` would say sanitisation ran
    and found nothing, which is a different statement from a declaration that never
    needed it, and the frontmatter writer skips empty strings anyway -- so an empty
    value would vanish silently and take the distinction with it.
    """
    from kojutsu.core.text_hygiene import SANITISATION_KEY

    _record(capture)
    _, _, _, sink = capture

    assert SANITISATION_KEY not in sink.entries[0].metadata


def test_the_direct_route_records_the_comment_it_posted(capture) -> None:
    """A rationale's only re-fetchable anchor, which used to stop at the model.

    The entry id is derived from the anchor and not from the comment, so nothing
    before the post could carry the id: the call has to return first. Without it in
    the stored document, the one record kind that is a self-assertion arrives with
    nothing to check it against -- which is what ``docs/tanseki-seam.md`` had to
    record as a limit rather than as a property.
    """
    _record(capture)
    _, github, _, sink = capture

    # ``FakeGitHub`` numbers its first comment 1001, so this is the real id and not a
    # coincidence: it has to be the comment that was actually posted.
    assert sink.entries[0].metadata["github_comment_id"] == 1001
    assert github.posted[0][3].startswith("<!--")


def test_a_declaration_posted_nowhere_records_no_comment_id(capture) -> None:
    """Absent rather than empty, because there is no comment to name.

    A branch-only declaration was never published, so it has no forge-side anchor at
    all. No channel test is needed to get that right: the id is in ``metadata`` only
    when a post actually happened, and the frontmatter writer skips ``None``, so the
    two cases cannot be confused by a later edit to either.
    """
    _record(capture, pr_number=None, branch="feat/local-only")
    _, github, _, sink = capture

    assert github.posted == []
    assert "github_comment_id" not in sink.entries[0].metadata


def test_a_declaration_of_only_display_controls_is_refused_and_nothing_is_written(
    capture,
) -> None:
    """The record would exist and read as nothing, which is worse than not existing.

    ``RationaleEntry`` has no emptiness validator -- unlike ``ClarificationEntry`` --
    so nothing downstream would have caught this: the document would be written, the
    declaration marked complete, and the claim closed against a record with no reason
    in it. Refused here instead, after the policy, which is the only point at which it
    is knowable.

    Refused *before* the allowlist, so a caller whose reason is entirely display
    controls is told what was wrong with the reason rather than being told the
    repository is not allowed -- the same ordering argument the repository check
    already makes.
    """
    result = _record(capture, reason=RLO * 60)

    assert result.ok is False
    assert result.code == "invalid_input"
    # A refusal carries its reason in ``error``; ``result`` is for the success text.
    assert "empty once the display-deceptive characters are removed" in (result.error or "")
    # And what was found is named, rather than the whole declaration vanishing as
    # "nothing to record".
    assert "U+202E RIGHT-TO-LEFT OVERRIDE" in (result.error or "")
    _, github, registry, sink = capture
    assert github.posted == []
    assert sink.entries == []
    assert registry.claimed == [], "nothing may be claimed for a declaration with no reason"


def test_sanitising_the_reason_cannot_move_a_declaration(capture) -> None:
    """The ordering constraint, stated as an identity rather than as prose.

    ``stable_rationale_entry_id`` hashes the repository, change, branch, agent and
    revision -- never the prose -- and the marker's values are composed from those
    same fields. So rewriting the reason cannot re-identify a stored document, orphan
    one, or give the writer and the collector different ids for one comment, which is
    the precise loss the marker carrying a branch exists to prevent.
    """
    from kojutsu.core.question_registry import stable_rationale_entry_id

    _record(capture, reason=f"Chose exponential backoff because it 429s.{RLO}")
    _, github, _, sink = capture

    assert sink.entries[0].entry_id == stable_rationale_entry_id(
        repo="org/repo",
        pr_number=42,
        branch="feat/backoff",
        declared_by="opencode",
        revision=1,
    )
    # The marker's own values are untouched by the policy, which is what lets the
    # collector derive the same id out of the comment this tool posted.
    assert "branch=feat/backoff" in github.posted[0][3]


def test_the_module_states_that_both_routes_write_the_record(capture) -> None:
    """The docstring claimed this tool posts a comment and stores nothing at all.

    It has stored the record since the branch-only route was added, so the claim was
    false, and it was load-bearing: a reader deciding whether the direct write was a
    privileged second path would have been reading fiction, directly above the code
    that contradicts it. Pinned because the correction is a paragraph of prose and
    nothing else would notice its removal -- and because the policy paragraph beside
    it is the whole argument for applying the character policy on this route.
    """
    module, _, _, _ = capture
    docstring = " ".join((module.__doc__ or "").lower().split())

    assert "not store a record" not in docstring
    assert "write the record" in docstring
    # The specific fact that makes the policy here load-bearing rather than redundant.
    assert "claim_rationale" in docstring
    assert "sanitise" in docstring
