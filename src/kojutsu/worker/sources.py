"""A work source over real open pull requests.

The loop's other :class:`~kojutsu.worker.loop.WorkSource` is the ticket system,
which is the right shape when agents are doing the writing. Knowledge capture is
the other half: the pull request already exists, someone human wrote it, and the
work is to review it and record what the author knew. So ready work here is an
open pull request that has not been reviewed yet.

Two properties matter more than throughput.

**A review is not repeated silently.** A cycle that completes must not be handed
the same pull request forever, and one that is mid-flight must not be handed to a
second worker. Both are recorded durably, so a restart resumes rather than
re-reviews everything.

**A pull request that vanished is not an error.** A source that reads live
repository state will eventually see a pull request that was closed, merged, or
made inaccessible between listing and claiming. That is ordinary drift, not a
failure, so it drops the item and says so instead of burning a cycle attempt on
work that no longer exists.
"""

from __future__ import annotations

import json
import os
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any

from kojutsu.integrations.github import GitHubClient
from kojutsu.repo_name import split_repo

from .loop import WorkItem


def _write_atomic(path: Path, text: str) -> None:
    """Replace ``path`` in one step, or leave it as it was.

    Same reasoning as the sandbox's writer: a half-written queue file is a source
    that has forgotten which reviews it already did, which is the one failure this
    class exists to prevent.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class OpenPullRequestSource:
    """Ready work is an open pull request nobody has reviewed yet.

    ``state_path`` persists both the reviews already done and the ones in flight.
    Without it an unattended worker re-reviews every open pull request on every
    restart, which is both wasteful and -- because each review posts comments --
    actively noisy on somebody else's pull request.
    """

    def __init__(
        self,
        client: GitHubClient,
        *,
        repo: str,
        state_path: Path,
        lease_seconds: float = 900.0,
    ) -> None:
        owner, name = split_repo(repo)
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self.client = client
        self.repo = repo
        self.owner = owner
        self.name = name
        self.lease_seconds = lease_seconds
        self.path = state_path
        self._done: dict[str, dict[str, Any]] = {}
        self._claims: dict[str, dict[str, Any]] = {}
        self._load()

    def close(self) -> None:
        """Release the forge client this source was handed.

        The client is passed in rather than built here, so it outlives any single
        call by design -- the worker builds one and reads through it for as long
        as it runs. That makes this the only place that can hand it back: whoever
        holds it cannot close it at construction, and a source that quietly kept
        it would hold a connection pool for the life of the process.

        Safe to call twice, and safe on a source whose client was never read.
        """
        self.client.close()

    def __enter__(self) -> OpenPullRequestSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- durable record --------------------------------------------------------

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A missing or unreadable file is a queue that has not started, not a
            # corrupt one. Treating it as fatal would make a fresh checkout
            # unrunnable; treating it as empty and overwriting is the safe read,
            # because the alternative is re-reviewing, which is recoverable.
            self._done, self._claims = {}, {}
            return
        if not isinstance(raw, dict):
            self._done, self._claims = {}, {}
            return
        done = raw.get("done")
        claims = raw.get("claims")
        self._done = done if isinstance(done, dict) else {}
        self._claims = claims if isinstance(claims, dict) else {}

    def _save(self) -> None:
        payload = {
            "repo": self.repo,
            "done": self._done,
            "claims": self._claims,
        }
        _write_atomic(self.path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @staticmethod
    def item_id(repo: str, pr_number: int) -> str:
        """The stable identity of a review, independent of its title."""
        return f"{repo}#{pr_number}"

    def _expired(self, claim: dict[str, Any]) -> bool:
        try:
            return time.monotonic() - float(claim.get("claimed_at", 0.0)) > self.lease_seconds
        except (TypeError, ValueError):
            return True

    # -- WorkSource ------------------------------------------------------------

    def ready_work(self, *, limit: int) -> list[WorkItem]:
        """Open pull requests that are neither reviewed nor currently claimed."""
        pulls = self.client.list_open_pull_requests(self.owner, self.name, limit=max(limit, 1) * 4)
        items: list[WorkItem] = []
        for pull in pulls:
            if pull.state.casefold() != "open" or pull.merged_at is not None:
                continue
            item_id = self.item_id(self.repo, pull.number)
            if item_id in self._done:
                continue
            claim = self._claims.get(item_id)
            if claim is not None and not self._expired(claim):
                continue
            if claim is not None:
                # A lease from a process that died is not one this process can
                # honour, and holding it would park the review forever.
                self._claims.pop(item_id, None)
            items.append(
                WorkItem(
                    item_id=item_id,
                    repo=self.repo,
                    # No branch is opened by a review, so the head ref is recorded
                    # for context only; nothing is ever pushed to it.
                    branch=_head_ref(pull.head),
                    pr_number=pull.number,
                    subject=pull.title,
                )
            )
            if len(items) >= limit:
                break
        return items

    def claim(self, item: WorkItem) -> str | None:
        """Take the review, unless another live lease already holds it."""
        existing = self._claims.get(item.item_id)
        if existing is not None and not self._expired(existing):
            return None
        token = secrets.token_hex(16)
        self._claims[item.item_id] = {
            "claim_token": token,
            "claimed_at": time.monotonic(),
            "pr_number": item.pr_number,
            "repo": item.repo,
        }
        self._save()
        return token

    def release(self, item: WorkItem, claim_token: str, reason: str) -> bool:
        """Give the review back so the next cycle can pick it up."""
        held = self._claims.get(item.item_id)
        if held is None or held.get("claim_token") != claim_token:
            return False
        self._claims.pop(item.item_id, None)
        self._save()
        return True

    def complete(self, item: WorkItem, claim_token: str, deliverable: str, notes: str) -> bool:
        """Record the review as done, so it is never handed out again."""
        held = self._claims.get(item.item_id)
        if held is None or held.get("claim_token") != claim_token:
            return False
        if not deliverable.strip():
            # A completion that points nowhere must not be recorded as done:
            # the item stays claimed so it is retried rather than forgotten.
            return False
        self._done[item.item_id] = {
            "deliverable": deliverable,
            "notes": notes,
            "completed_at": time.time(),
        }
        self._claims.pop(item.item_id, None)
        self._save()
        return True

    def record_branch(self, item: WorkItem, branch: str) -> bool:
        """Nothing to record: a review opens no branch.

        Returning ``True`` rather than raising keeps the loop's contract -- the
        branch is recorded so an operator can spot a conflicting in-flight change
        -- without inventing a branch that was never created.
        """
        return True

    def release_by_token(self, item_id: str, claim_token: str, reason: str) -> bool:
        """Release a lease recovered from another process's durable state."""
        held = self._claims.get(item_id)
        if held is None or held.get("claim_token") != claim_token:
            return False
        self._claims.pop(item_id, None)
        self._save()
        return True

    # -- operator views --------------------------------------------------------

    def reviewed(self) -> dict[str, dict[str, Any]]:
        """Reviews recorded as done, by item id."""
        return dict(sorted(self._done.items()))

    def in_flight(self) -> dict[str, dict[str, Any]]:
        """Live leases held by this process, by item id."""
        return {
            item_id: claim
            for item_id, claim in sorted(self._claims.items())
            if not self._expired(claim)
        }

    def forget(self, item_id: str) -> bool:
        """Return a completed review to the queue.

        The operator escape hatch, and the counterpart to the worker's
        dead-letter requeue: without it a review that was completed wrongly is
        permanently unrepeatable.
        """
        if item_id not in self._done:
            return False
        self._done.pop(item_id)
        self._save()
        return True


def _head_ref(head: dict | None) -> str:
    if not isinstance(head, dict):
        return ""
    ref = head.get("ref")
    return ref if isinstance(ref, str) else ""


__all__ = ["OpenPullRequestSource"]
