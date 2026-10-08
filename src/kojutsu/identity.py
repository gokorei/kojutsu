"""How every identity and request key in this system is derived, in one place.

This module exists because the same three decisions were being made in several
files and each of them could drift: **what bytes get hashed**, **which namespace
the value belongs to**, and **which fields must never reach the hash at all**. The
first two arrived together — a preimage helper and a set of domain labels — and the
third was requested by `94A8A96W` and never written down, so nothing asserted it.

It lives apart from ``question_registry`` and ``answer_collector`` because it is
none of their business. Neither owns identity; they use it. ``core/allowlist.py`` is
the precedent for a single definition imported by two callers rather than each
caller carrying its own copy, and a policy constant declared twice is the exact
failure that module was added to prevent — which is why ``TRANSPORT_ASSIGNED_FIELDS``
lives here rather than beside whichever collector happened to need it first.

Adding a derivation here means adding a label here. A domain that is not in this
module is a value with no namespace, and two of those can collide without anything
noticing.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

#: Domain label for the Tanseki ``Idempotency-Key`` derivation.
#:
#: Named separately from the record identities because it answers a different
#: question. An entry id says *which record this is*; an idempotency key says *have I
#: already sent this request*. They are computed from overlapping inputs — the
#: document id and the content — so sharing a namespace would let a record identity
#: and a request key collide on the same bytes.
#:
#: Note this is the only label with no record behind it, and that is deliberate: it
#: protects a header, not a stored row. Nothing is keyed on it, so it carries no
#: migration consequence — see ``docs/design-review/identity-and-limits.md``.
IDEMPOTENCY_IDENTITY_DOMAIN = "kojutsu.idempotency.v1"

#: Fields that transport assigns to a row after capture, and which therefore must
#: never reach an identity or request-key preimage.
#:
#: These are the fields whose values change every time a row is retried: the attempt
#: count, the next retry time, whether it has been dead-lettered, and the lease state
#: of a claim. A preimage that folded any of them in would derive a *new* key for
#: the same logical write on each attempt, which defeats the mechanism entirely —
#: a retry that cannot be recognised as a retry produces a second delivery rather
#: than a confirmation, and the store ends up holding the same knowledge twice under
#: two identities.
#:
#: It is a constant rather than a comment because a comment is checked by nothing,
#: and the failure it describes is silent: nothing raises, the write succeeds, and
#: the duplication only becomes visible as a count that looks wrong much later.
#: :func:`kojutsu.models.capture_anchor_gaps`'s siblings in the registry assert
#: similar rules about anchors; this asserts itself in
#: ``tests/test_identity_derivations.py``.
TRANSPORT_ASSIGNED_FIELDS: frozenset[str] = frozenset(
    {
        # Retry accounting. Changes on every attempt by construction.
        "attempts",
        "next_attempt_at",
        "last_error",
        "last_attempt_at",
        # Terminal transport state. A dead-lettered row is the same row.
        "dead_lettered_at",
        # Lease state. A claim is held and released over a row's life; the row does
        # not change when it is.
        "lease_expires_at",
        "leased_by",
        "claim_token",
        # Delivery timestamps, which record when transport moved the row rather
        # than anything about what the row says.
        "delivered_at",
        "captured_by_transport_at",
    }
)


def identity_preimage(
    domain: str,
    fields: tuple[object, ...],
    *,
    default: Callable[[Any], Any] | None = None,
) -> bytes:
    """Build the exact bytes one identity derivation hashes.

    Both properties here are load-bearing, and neither is about which hash function
    is used.

    **Unambiguous.** The fields go in as a JSON array rather than delimiter-joined,
    so no field can contain its own separator: ``("a|b", "c")`` and ``("a", "b|c")``
    serialise differently. These components are attacker-influenced -- a repository
    name, a branch name, a delivery action -- so the delimiter form's safety rested
    on the *forge* refusing a character in the wrong field, which is a property of
    the input rather than of the derivation.

    **Domain-separated.** The domain is element zero, so what is digested is *these
    fields of this kind of record* and not merely a list of values. Two record
    kinds with the same field layout then cannot derive the same value, and a
    change to the encoding or the field order yields a different value rather than
    one that looks comparable and is not.

    ``default`` exists for the one derivation whose preimage carries a ``datetime``.
    Every other caller passes JSON-native values, so leaving it at ``None`` means an
    unserialisable component raises at the derivation rather than being coerced into
    a digest that no longer describes the record.

    ``None`` and ``""`` are deliberately distinct, because they are different facts.
    A caller that passed ``body.get("id", "")`` could not tell an absent field from
    an empty one, and two different requests then derived one key — which for an
    idempotency key means a genuine write being reported as a replay of nothing.
    JSON's ``null`` against ``""`` keeps them apart for free, provided the caller
    does not flatten them first.
    """
    return json.dumps([domain, *fields], separators=(",", ":"), default=default).encode("utf-8")
