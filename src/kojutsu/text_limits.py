"""Shared text bounds for machine-authored markers and hygiene.

A leaf module: stdlib only, imported by ``core`` and ``integrations`` alike.
These bounds lived in ``integrations/github.py`` next to the code that enforces
them, which made ``core`` import ``integrations`` for constants alone -- one of
the two back-edges that turned the intended one-way layering
(``core`` -> ``integrations``) into a cycle. The enforcement stays where it
was; only the numbers moved, so nothing that was bounded became unbounded.
"""

#: Branch names are bounded like every other marker field: configuration reaching
#: a stored provenance field, on a path that ends up in a document id.
MAX_BRANCH_CHARS = 255
# An agent identifier is short and machine-authored; keep it bounded so a comment
# body cannot smuggle an unbounded string into a stored provenance field.
MAX_AGENT_ID_CHARS = 64
# A model id is longer and contains provider prefixes and separators, but it is
# still configuration reaching a stored provenance field, so it is bounded too.
MAX_MODEL_ID_CHARS = 128

#: Longest repository token an MCP tool argument may carry. The tools receive
#: ``owner/name`` from untrusted callers, and the bound applies before the
#: allowlist is consulted so a megabyte-long string never reaches it.
MAX_REPOSITORY_CHARS = 200

#: Invisible and bidirectional *display* controls: characters that change what a
#: stored comment renders as without changing what it says. The reason to
#: enumerate them is that the obvious implementation -- strip every ``Cf``
#: format character -- takes honest content with it. ``U+0600`` and ``U+061C``
#: are Arabic-Indic digits and the Arabic letter mark; ``U+2064`` is the
#: invisible plus used in accounting. A filter that blocks real content gets
#: switched off, and then the bidi controls it was there for come back too.
INVISIBLE_FORMATTING_CHARACTERS: frozenset[str] = frozenset(
    {
        "\u00ad",  # soft hyphen
        "\u200b",  # zero width space
        "\u200e",  # left-to-right mark
        "\u200f",  # right-to-left mark
        "\u202a",  # left-to-right embedding
        "\u202b",  # right-to-left embedding
        "\u202c",  # pop directional formatting
        "\u202d",  # left-to-right override
        "\u202e",  # right-to-left override
        "\u2066",  # left-to-right isolate
        "\u2067",  # right-to-left isolate
        "\u2068",  # first strong isolate
        "\u2069",  # pop directional isolate
        "\ufeff",  # zero width no-break space, also the byte order mark
    }
)

# Two joiners are deliberately **absent** from that set, and the absence is the point.
#
# ``U+200C`` (ZWNJ) and ``U+200D`` (ZWJ) are invisible too, and stripping them would
# break honest content: ZWJ builds every multi-person emoji, and ZWNJ carries meaning
# in Persian, Urdu and several Indic scripts, where removing it changes the spelling
# of a name. Neither can reorder displayed text the way the overrides above do, so
# neither does the thing this filter exists for. Refusing them would buy no display
# fidelity and would cost real names -- and a filter that costs real names gets
# removed, which is how a system ends up with no filter at all.
#
# Spelled out rather than left implicit because the instinct to tighten this set is
# exactly backwards, and the next reader cannot see the content these two characters
# carry.
