"""What happens to text a contributor wrote before Kojutsu stores or shows it.

Kojutsu stores and then displays other people's prose: review bodies, inline
comments, pull request titles, check names. That gives it two questions to ask of
every character in that text, and the questions are unrelated to each other.

**Is this the same text as last time?** ``é`` reaches a comment as one code point
or as ``e`` plus a combining acute. The forge treats them as the same characters
and so does a reader, while everything downstream sees two byte strings -- so one
sentence is stored twice, matches twice and counts twice, with nothing in the
corpus able to say which copy was the original. :func:`nfc` answers that.

**Will this render as what it says?** A right-to-left override makes the tail of
a stored path, or a reviewer's name, display as something else entirely. Nothing
downstream can notice, because every renderer and every terminal applies the
override faithfully: the deception works by being correct.
:func:`sanitise` answers that, and it *records what it removed*, so the answer is
checkable rather than merely claimed.

The two halves share characters, which is why they live together: a maintainer
tightening one of them is editing the other. So the reasoning that justifies each
character is written out here rather than left to whoever reads the list next.

**The characters are removed; the record is kept.** Refusing the record is the
obvious reading of "reject the dangerous characters" and it is the wrong one. A
hostile comment is *evidence of a hostile comment*, so dropping it destroys
precisely what this project exists to preserve, and nothing at the boundary can
tell a reader that something was dropped rather than never said. The text is
sanitised, the record is written, and :data:`SANITISATION_KEY` says what was taken
out of it. The forge-issued comment, review or check id is stored alongside, so
the original can be re-fetched with everything its author put there. Visible,
rather than silent.

**Where this is applied, and the two places it must not be.** At the boundary
where untrusted text becomes *stored* text -- the collectors in
:mod:`kojutsu.core.answer_collector` -- and never before, for two reasons that are
load-bearing rather than stylistic:

- **The marker grammar.** ``kojutsu:answer:q-100`` is what makes a comment a
  Kojutsu record, and :func:`kojutsu.integrations.github.normalise_captured_text`
  is explicit that the grammar is read *verbatim* while a marker's value is
  normalised: a token whose own punctuation had to be repaired in order to parse
  is not one this system wrote, so parsing fails closed. Sanitising a payload
  before its markers are read would repair exactly those markers and defeat the
  check. The policy therefore runs on a marker's value, after extraction.
- **The signed delivery bytes.** The webhook's HMAC covers the raw request body
  (:func:`kojutsu.webhook.server._verify_signature`), so a byte rewritten before
  the delivery claim is computed invalidates the signature against itself.

**This is not a "reject suspicious input" filter.** There is no score, no
heuristic and no threshold. The refused set is a closed enumeration of named code
points, and adding one is a decision with a consequence to weigh rather than a
tightening to perform.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from kojutsu.text_limits import INVISIBLE_FORMATTING_CHARACTERS

#: The metadata key a sanitised record carries what was removed under.
#:
#: A string, not a list or a mapping, and that is not a simplification: the
#: frontmatter writer in :mod:`kojutsu.core.tanseki_mapping` stringifies every
#: value it is given except ``files``, so a list here would be written as a Python
#: repr -- a value no consumer can filter on, which is the outcome that key was
#: designed to avoid. A flat, greppable sentence survives that boundary intact.
#:
#: Absent rather than empty when nothing was removed. ``"text_sanitisation": ""``
#: would be a claim that sanitisation happened and found nothing, which is a
#: different statement from a record that never needed it, and the frontmatter
#: writer skips empty strings -- so an empty value would silently vanish anyway.
SANITISATION_KEY = "text_sanitisation"


def nfc(text: str) -> str:
    """Fold ``text`` to its canonical composed form.

    One function, named, so that "normalise before hashing" is a call rather than
    a convention somebody re-implements. Applied to the string a derivation hashes
    rather than to its serialised preimage, and that ordering is not a detail:
    :func:`kojutsu.identity.identity_preimage` serialises with ``json.dumps``,
    which escapes every non-ASCII code point, so its output is pure ASCII by
    construction and normalising *it* would be a provable no-op. The characters
    have to be folded before the escaping, in the field.

    ``NFC`` and not ``NFKC``: compatibility decomposition folds characters that are
    not the same character -- ligatures, fullwidth forms, superscripts -- so a
    compatibility fold makes two genuinely different author names collide. Canonically
    equivalent text is the only case where collapsing is correct.
    """
    return unicodedata.normalize("NFC", text)


#: The refused C0 and C1 control characters: TAB and LF survive, nothing else does.
#:
#: TAB and LF are what a comment body is *made of*. A record whose answer is a
#: single line, or a pasted diff, cannot be written without them, so refusing them
#: would refuse ordinary prose.
#:
#: The rest are refused because each one can terminate, overwrite or reorder a line
#: in a terminal or in a rendered Markdown body, and none of them can be the
#: difference between two meanings of an author's words. CR is in here, which is a
#: real cost: a progress report pasted from a Windows terminal loses its line breaks.
#: It is paid because the alternative is a stored record that renders as something
#: its author did not write, and because the removal is recorded rather than silent.
REFUSED_CONTROL_CHARACTERS: frozenset[str] = frozenset(
    chr(code) for code in (*range(0x00, 0x20), 0x7F, *range(0x80, 0xA0)) if code not in (0x09, 0x0A)
)

#: The invisible and bidirectional display controls, as one set.
#:
#: Assembled rather than written out: the fourteen invisible and bidirectional
#: characters the marker extractors already apply come from
#: :data:`kojutsu.integrations.github.INVISIBLE_FORMATTING_CHARACTERS`, and this adds
#: the word joiner that enumeration is missing. Two copies of the fourteen would be
#: two answers to "what is a valid comment body" that agree until the day they do
#: not, and nothing would report the disagreement.
REFUSED_INVISIBLE_CHARACTERS: frozenset[str] = INVISIBLE_FORMATTING_CHARACTERS | {
    # U+2060 WORD JOINER. Invisible, carries no display meaning of its own, and is
    # used to glue a line together so it cannot be broken -- which is the property
    # this whole set exists to refuse. It is not in the borrowed enumeration above.
    "⁠"
}

#: Every character removed from text before it is stored. The union, and the only
#: set a caller should consult: the two halves above are named so each can be
#: reasoned about, not so each can be applied.
REFUSED_CHARACTERS: frozenset[str] = REFUSED_CONTROL_CHARACTERS | REFUSED_INVISIBLE_CHARACTERS


def _character_pattern(characters: frozenset[str]) -> re.Pattern[str]:
    """Compile a set of code points into one character class.

    Derived from the set rather than written beside it, so the pattern cannot
    describe a different set than the one the module refuses. Sorting makes the
    compiled source stable, which keeps two interpreters from producing different
    patterns for the same frozenset.
    """
    codes = sorted(ord(character) for character in characters)
    return re.compile("[" + "".join(f"\\u{code:04X}" for code in codes) + "]")


#: Matcher for :data:`REFUSED_CONTROL_CHARACTERS`.
CONTROL_RE: re.Pattern[str] = _character_pattern(REFUSED_CONTROL_CHARACTERS)

#: Matcher for :data:`REFUSED_INVISIBLE_CHARACTERS`.
INVISIBLE_RE: re.Pattern[str] = _character_pattern(REFUSED_INVISIBLE_CHARACTERS)

#: Matcher for :data:`REFUSED_CHARACTERS`. The one the sanitiser uses; the two above
#: are the named halves, and are what a reader checks a single character against.
REFUSED_RE: re.Pattern[str] = re.compile(f"{CONTROL_RE.pattern}|{INVISIBLE_RE.pattern}")

#: Invisible characters that carry meaning in honest content and are therefore
#: **not** refused.
#:
#: This set exists because the instinct to tighten :data:`REFUSED_CHARACTERS` is
#: exactly backwards, and it is the single most likely thing for the next maintainer
#: to "fix". The reason is written here rather than left to whoever reads the list
#: next, because a test that merely asserts these characters survive looks like an odd
#: thing to protect and says nothing about why. A comment nothing checks is only a
#: comment, so ``tests/test_answer_collector.py`` reads *this* text back and fails if
#: the code points it names stop being named -- which is what keeps the explanation
#: from drifting away from the set it explains.
#:
#: - **U+200D ZERO WIDTH JOINER** builds every multi-person emoji. Removing it does
#:   not shorten a name, it destroys the emoji: a family, a rainbow flag and a
#:   profession all collapse into a single unidentified person.
#: - **U+200C ZERO WIDTH NON-JOINER** is a letter in Persian, Urdu and several Indic
#:   scripts. Removing it changes the spelling of a name -- ``مي‌رود`` and
#:   ``ميرود`` are two different words -- so a contributor with one of these names
#:   is a contributor whose name is mangled by the filter.
#: - **U+E0020-U+E007F, the emoji tag block**, is how a subdivision flag is written:
#:   the England, Scotland and Wales flags are single code points followed by
#:   ``<tag>``s naming the subdivision. Refusing the tag characters does not
#:   degrade those flags, it deletes them, and unlike the joiners there is no
#:   fallback rendering at all.
#: - **U+061C ARABIC LETTER MARK** and **U+2064 INVISIBLE PLUS** are the two
#:   counter-examples that shaped the rest of the set. Both are ``Cf`` format
#:   characters, so the obvious implementation -- refuse every ``Cf`` -- takes them,
#:   and neither is decoration: the first attaches an Arabic-Indic number to the digit
#:   before it, and the second glues a number together in accounting. A filter that
#:   reaches further out still, into every non-ASCII digit, would take Arabic-Indic and
#:   Devanagari numerals with them.
#:
#: None of the five can reorder displayed text the way an override or an embedding
#: control does, which is the only harm this filter exists to prevent. Refusing them
#: would buy no display fidelity and would cost real names -- and a control that blocks
#: legitimate content gets switched off, which is worse than never having had one. So
#: they are a named set, the reason is written next to it, and a test reads that reason
#: back: the failure mode of getting this wrong is not a broken filter, it is *no*
#: filter, and nothing about that failure looks like a bug at runtime.
PRESERVED_INVISIBLE_CHARACTERS: frozenset[str] = frozenset(
    {
        "‌",  # U+200C ZERO WIDTH NON-JOINER
        "‍",  # U+200D ZERO WIDTH JOINER
        "؜",  # U+061C ARABIC LETTER MARK  # noqa: PLE2502 - the filter's own table is the subject
        "⁤",  # U+2064 INVISIBLE PLUS
        *(chr(code) for code in range(0xE0020, 0xE0080)),  # emoji tag characters
    }
)

# A tripwire, not a validation of the text. It fires at import rather than at
# capture because the thing it detects is a bug in *this file*: the failure it
# prevents is a filter that silently mangles real names in production, which is the
# outcome the set above is written to avoid and which no test run would notice if
# the character was added to both sets at once. Failing to import is loud, happens
# in the first second of any run, and cannot be deployed quietly.
_REFUSED_BY_MISTAKE: frozenset[str] = REFUSED_CHARACTERS & PRESERVED_INVISIBLE_CHARACTERS
if _REFUSED_BY_MISTAKE:  # pragma: no cover - unreachable unless the policy is edited
    raise RuntimeError(
        "the refused character set now contains characters that carry meaning in honest "
        "content: "
        + ", ".join(f"U+{ord(character):04X}" for character in sorted(_REFUSED_BY_MISTAKE))
        + ". Read PRESERVED_INVISIBLE_CHARACTERS before editing REFUSED_CHARACTERS; "
        "refusing these mangles real names, and a filter that mangles real names gets "
        "switched off entirely."
    )


@dataclass(frozen=True)
class RemovedCharacter:
    """One code point taken out of a string, by name rather than by presence.

    ``codepoint`` is a formatted ``U+XXXX`` string rather than a ``chr``, so the
    whole record is safe to write into a note that gets stored and displayed --
    naming the character cannot reintroduce it. ``count`` is there because "a
    right-to-left override was removed" and "four hundred were" are different
    statements about a record, and only the second tells a reader the text was
    adversarial rather than merely untidy.
    """

    codepoint: str
    name: str
    count: int


@dataclass(frozen=True)
class SanitisedText:
    """A string after :func:`sanitise`, and what was taken out of it.

    The two halves travel together because the second is the only way to check the
    first. A sanitiser that returned a string alone would leave a reader unable to
    distinguish a record stored verbatim from one that was rewritten at the
    boundary -- which is exactly the ambiguity this ticket exists to remove, one
    level down.
    """

    text: str
    removed: tuple[RemovedCharacter, ...] = ()

    @property
    def was_modified(self) -> bool:
        """Whether anything was removed. False is the common case and means nothing."""
        return bool(self.removed)

    def note(self) -> str | None:
        """The one-line description of what was removed, or ``None`` if nothing was.

        ``None`` rather than an empty string, so that a record which needed no
        sanitisation carries no key at all -- see :data:`SANITISATION_KEY`.

        Describes the removals from *this* value, so a caller whose input has already
        been through a narrower pass wants :func:`describe_removals` against the
        original instead. Getting that wrong under-reports, which is the one failure
        mode this whole module exists to avoid.
        """
        return combine_notes(self)


def combine_notes(*fields: SanitisedText | None) -> str | None:
    """Describe every removal across several fields of one record, as one line.

    One record can hold more than one piece of captured prose -- an inline comment
    carries a body, a file path and a diff hunk -- and three separate notes would
    leave a reader to work out whether they are describing one attack or three.
    Counts are summed per code point rather than per field, so the note answers
    "how many of these did this record lose" instead of restating the schema.

    ``None`` is accepted for a field the payload did not carry, so a caller can
    pass what it has rather than filtering the arguments itself.

    The result is deliberately flat text: it crosses a storage boundary that
    stringifies, it is greppable by a console, and it cannot itself carry a
    character that would have to be sanitised again.
    """
    return _format_note(
        _merge_removals(removed for field in fields if field for removed in field.removed)
    )


def describe_removals(*sources: str | None) -> str | None:
    """One note covering what the policy removes from each string **as it arrived**.

    Takes the payload rather than the values that will be stored, and that difference
    is the entire reason this function exists. Kojutsu already runs captured text
    through a narrower pass on its way in --
    :func:`kojutsu.integrations.github.normalise_captured_text`, applied by the marker
    extractors -- which removes fourteen of the same characters and hands back a bare
    string. A note computed from the *stored* value therefore under-reports: it names
    the controls this module took out and silently omits the ones the extractor took,
    which is a false statement by omission in the one field whose whole job is to say
    what was done to the record.

    Measuring over the raw source is exact rather than approximate. Every refused
    character in a delivered string is removed on the way to storage, with one
    harmless exception: the ``kojutsu:`` grammar is read *verbatim*, so a refused
    character inside a marker can survive. That is the fail-closed path rather than a
    hole in it -- a marker is parsed rather than stored, and a marker whose value
    changes when the character is normalised no longer matches what the registry
    recorded, so no record is written and there is nothing to report on.

    Costs one pass over a string that has already been read. Nothing is stored and
    nothing is compared against a stored value.
    """
    return _format_note(
        _merge_removals(
            removed for source in sources if source is not None for removed in _removed_from(source)
        )
    )


def _merge_removals(
    removed: Iterable[RemovedCharacter],
) -> Counter[tuple[str, str]]:
    """Sum counts per code point, so one character seen twice is one entry."""
    counts: Counter[tuple[str, str]] = Counter()
    for entry in removed:
        counts[(entry.codepoint, entry.name)] += entry.count
    return counts


def _format_note(counts: Counter[tuple[str, str]]) -> str | None:
    if not counts:
        return None
    listed = ", ".join(
        f"{codepoint} {name} (x{count})" for (codepoint, name), count in sorted(counts.items())
    )
    total = sum(counts.values())
    return f"removed {total} character{'' if total == 1 else 's'} before storing: {listed}"


def _name(character: str) -> str:
    """The Unicode name of one refused character, or its category if it has none.

    ``unicodedata.name`` *raises* rather than returning ``None`` for an unnamed
    character, and the refused set is full of them: the C0 controls carry
    abbreviations (BEL, ESC, ...) rather than names, so a note that said
    ``U+0007 UNNAMED`` would read like a bug in this module rather than like a
    fact about the record. The general category is always available and is the
    thing a reader wants to know about a character the database does not name.
    """
    try:
        return unicodedata.name(character)
    except ValueError:
        return f"unnamed {unicodedata.category(character)}"


def sanitise(text: str) -> SanitisedText:
    """Return ``text`` in the form Kojutsu stores and shows, and what was removed.

    The order is NFC first and refused characters second, and it is the order
    :func:`kojutsu.integrations.github.normalise_captured_text` already documents:
    the refused set is checked against the canonical form a reader will actually
    see, rather than against a decomposition that may be hiding one. A combining
    sequence either folds to a precomposed character or leaves nothing behind; it
    does not manufacture a bidi control.

    Idempotent, which is what makes it safe to apply at more than one seam: running
    :func:`sanitise` over already-sanitised text removes nothing and reports
    nothing, so a caller that cannot tell whether an upstream boundary already ran
    it does not have to know.
    """
    return SanitisedText(text=REFUSED_RE.sub("", nfc(text)), removed=_removed_from(text))


def _removed_from(text: str) -> tuple[RemovedCharacter, ...]:
    """What :func:`sanitise` would remove from ``text``, without rewriting it."""
    counts = Counter(match.group() for match in REFUSED_RE.finditer(nfc(text)))
    return tuple(
        RemovedCharacter(
            codepoint=f"U+{ord(character):04X}",
            # C0 and C1 controls have abbreviations (BEL, ESC, ...) rather than
            # names, so ``unicodedata.name`` raises for them. The general category is
            # always there and is what a reader actually wants to know about a
            # character the database does not name.
            name=_name(character),
            count=count,
        )
        for character, count in sorted(counts.items())
    )
