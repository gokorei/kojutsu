"""The dashboard's inline JavaScript must parse.

A `const` redeclaration inside one function is a *parse* error, not a runtime one:
the whole script fails to load and the page renders permanently empty, with the
status line stuck on "loading…" and every panel reading "No data." Nothing in the
Python suite can see that, and every API call still returns correct data, so a
broken page is indistinguishable from a quiet one unless something actually parses
the JavaScript.

This is here because that exact failure shipped. A second ``const parts`` in one
function scope blanked the dashboard; the API returned all 11 documents, every
Python test passed, and only loading the page in a browser showed it dead.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

DASHBOARD = Path(__file__).resolve().parents[1] / "scripts" / "knowledge_dashboard.html"
_SCRIPT_RE = re.compile(r"<script[^>]*>(.*?)</script>", re.DOTALL)


def _inline_script() -> str:
    return "\n".join(_SCRIPT_RE.findall(DASHBOARD.read_text(encoding="utf-8")))


def test_the_dashboard_has_inline_script() -> None:
    assert _inline_script().strip(), "the dashboard lost its script"


def test_the_inline_script_parses(tmp_path: Path) -> None:
    """Parse it the way a browser would.

    Deliberately not a Python-side approximation. An earlier attempt counted
    braces to find duplicate declarations and reported false positives for
    ``for (const k of ...)`` and for words inside comments -- a check that cries
    wolf is worse than none, because it trains you to ignore it. ``node --check``
    is the real parser; where it is unavailable the honest answer is that this is
    unverified rather than a guess.
    """
    if shutil.which("node") is None:
        pytest.skip("node is required to parse the dashboard's JavaScript")
    script = tmp_path / "dashboard.js"
    script.write_text(_inline_script(), encoding="utf-8")

    result = subprocess.run(  # noqa: S603 - node checks a file this test just wrote
        ["node", "--check", str(script)],  # noqa: S607 - node via PATH is the documented check
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        "the dashboard's script does not parse, so the page renders empty:\n"
        f"{result.stderr.strip()}"
    )


def test_the_dashboard_polls_the_knowledge_endpoint() -> None:
    """Without this the page renders, and renders nothing at all."""
    script = _inline_script()

    assert "/api/knowledge" in script
    assert "setInterval" in script, "the dashboard must update on its own to be a live view"


def test_the_dashboard_says_whether_a_pairing_was_inferred() -> None:
    """A payload field nobody renders is a field nobody reads.

    The console always sends a ``structure``; if the page stops reading it, a
    model-matched question and answer renders identically to a conversation a human
    had, which is the one reading this axis exists to prevent. Asserted against the
    script text because that is the only surface the Python suite can see -- the
    alternative is a browser, which nothing here runs.
    """
    script = _inline_script()

    assert "c.structure" in script
    assert "inferred" in script


def test_the_inferred_badge_names_the_model_in_the_row_not_only_in_a_tooltip() -> None:
    """A distinction that requires a hover is a distinction most readers never see.

    Two things have to hold, and only the first is obvious. The badge must be
    appended to the row rather than set as a ``title``, and the model must be part of
    the chip's own text -- ``inferred`` on its own says a guess happened without
    saying who to ask about it, which is the part a reader weighing the record
    actually needs.
    """
    script = _inline_script()

    assert "structure_inferred_by" in script, (
        "the badge cannot name the model it does not receive; the console sends "
        "this field on every row"
    )
    badge = re.search(r"el\('span', 'chip warn', `inferred · \$\{([^}]+)\}`\)", script)
    assert badge, (
        "the inferred badge must put the model in the chip's own text, not in a "
        "title attribute, so a reader who never hovers still sees it"
    )
    assert "model not named" in script, (
        "an inferred row whose model is missing must say so rather than render a "
        "badge with nothing after the dot"
    )


def test_the_dashboard_offers_a_structure_filter_and_counts_what_it_excludes() -> None:
    """Filtering must never read as absence.

    A structure filter that dropped inferred records without counting them turns
    "no anchored records" into a claim about an empty store. The MCP read path
    already established the rule with ``anchored_only`` -- exclusions are counted and
    named -- and the dashboard's filter has to obey the same rule or it becomes the
    one surface where the claim goes unqualified.
    """
    html = DASHBOARD.read_text(encoding="utf-8")
    script = _inline_script()

    assert 'name="structure"' in html, "the dashboard offers no structure filter at all"
    assert "excluded" in script, "the filter must report the count it hid"
    # Counted, not just totalled: a bare number still leaves the reader unable to
    # tell an inferred pairing from a structure this build cannot read.
    assert "structureOf" in script


def test_no_asked_or_answered_metric_counts_an_inferred_pairing() -> None:
    """An inferred pairing is not a question anybody was asked.

    The console's KPI labels are the only place a reader skims for "how many
    questions were asked and answered", so that figure has to exist and has to be
    the honest one. This pins the predicate the figure is computed from rather than
    the label text: a rename to a different phrasing would not change what is
    counted, and only the count is the claim.
    """
    script = _inline_script()

    assert "isAskedQuestion" in script
    assert re.search(
        r"const isAskedQuestion = \(c\) => c\.kind === 'capture' && isAnchored\(c\)", script
    ), "an 'asked & answered' figure that admits an inferred pairing is the wrong figure"
    assert re.search(r"const isAnchored = \(c\) => c\.structure === 'anchored'", script), (
        "the predicate must resolve to the anchored axis; defaulting an unreadable "
        "structure to anchored would count a record this page cannot interpret"
    )


# --- running it, rather than reading it -----------------------------------------
#
# Everything above asserts against the script's text, which is all a Python test can
# see. Two of the things this ticket exists to get right are arithmetic and rendered
# output -- a count that excludes the wrong record, and a note that never appears --
# and both are invisible to a grep. So the script is actually run, through a DOM stub
# just large enough for it to render, and the numbers and the note are read back.
#
# The stub's control list is read out of the dashboard rather than written down. A
# hand-kept copy of the form's control names is a second source of truth for the
# form: when the two drift, the page reads a control the stub never created and
# throws a TypeError while the API keeps returning 200. That is not hypothetical --
# it is how adding the structure filter broke `test_clarification_record.py` first.

_FILTERS_FORM_RE = re.compile(r'<form id="filters">(.*?)</form>', re.DOTALL)
_CONTROL_NAME_RE = re.compile(r'<(?:input|select|textarea)[^>]*\bname="([^"]+)"')

_STUB = """
function __makeNode(tag) {
  return {
    tag: tag, className: '', textContent: '', innerHTML: '', value: '',
    style: {}, dataset: {}, childNodes: [],
    addEventListener() {},
    classList: { add() {}, remove() {} },
    append(...kids) { for (const k of kids) if (k != null) this.childNodes.push(k); },
    replaceChildren(...kids) { this.childNodes = []; this.append(...kids); },
    get options() {
      const found = this.childNodes.filter((n) => n && n.tag === 'option');
      return found.length ? found : [{ value: '' }];
    },
  };
}
const __nodes = new Map();
globalThis.document = {
  querySelector(selector) {
    if (!__nodes.has(selector)) {
      const node = __makeNode('div');
      if (selector === '#filters') {
        for (const name of __CONTROLS__) node[name] = __makeNode(name);
      } else if (selector === '#source') {
        node.base = __makeNode('input');
        node.every = __makeNode('select');
      }
      __nodes.set(selector, node);
    }
    return __nodes.get(selector);
  },
  createElement: (tag) => __makeNode(tag),
  createTextNode: (text) => ({ tag: '#text', textContent: String(text), childNodes: [] }),
};
globalThis.location = { protocol: 'http:', origin: 'http://console.test' };
globalThis.setInterval = () => 0;
globalThis.clearInterval = () => {};
globalThis.fetch = () => new Promise(() => {});
function __textOf(node) {
  if (node == null) return '';
  if (typeof node === 'string') return node;
  if (!Array.isArray(node.childNodes)) return String(node.textContent || '');
  let out = String(node.textContent || '') + (node.innerHTML ? String(node.innerHTML) : '');
  for (const child of node.childNodes) out += __textOf(child);
  return out;
}
"""


def _filter_control_names() -> list[str]:
    form = _FILTERS_FORM_RE.search(DASHBOARD.read_text(encoding="utf-8"))
    assert form is not None, "the dashboard lost its filters form"
    return _CONTROL_NAME_RE.findall(form.group(1))


def _render(captures: list[dict[str, object]], structure: str, tmp_path: Path) -> dict[str, object]:
    """Render ``captures`` with ``structure`` applied, and read the page back.

    ``tmp_path`` is the pytest one so a failing run leaves the script behind to read.
    """
    if shutil.which("node") is None:
        pytest.skip("node is required to render the dashboard")
    driver = f"""
const __items = {json.dumps(captures)};
render({{ captures: __items, meta: {{}} }});
$('#filters').structure.value = {json.dumps(structure)};
renderCaptures();
const kpis = {{}};
for (const card of $('#kpis').childNodes) {{
  kpis[__textOf(card.childNodes[1])] = __textOf(card.childNodes[0]);
}}
const rows = $('#captures').childNodes.filter((n) => n.className === 'cap');
// renderBars lays each row out as [key, track, value]; the value is the count a
// reader would read off the bar, so it is read from its own node rather than
// scraped out of the concatenated text.
const barValues = (sel) => $('#' + sel).childNodes
  .filter((r) => r.className === 'bar-row')
  .map((r) => [__textOf(r.childNodes[0]), __textOf(r.childNodes[2])]);
process.stdout.write(JSON.stringify({{
  kpis,
  filter_note: __textOf($('#filter-note')),
  kpi_note: __textOf($('#kpi-note')),
  by_agent_note: __textOf($('#by-agent-note')),
  row_count: rows.length,
  row_badges: rows.map((r) => r.childNodes[0].childNodes.map(__textOf).join(' | ')),
  by_agent: barValues('by-agent'),
}}));
"""
    script_file = tmp_path / "dashboard_render.js"
    script_file.write_text(
        _STUB.replace("__CONTROLS__", json.dumps(_filter_control_names()))
        + "\n"
        + _inline_script()
        + "\n"
        + driver,
        encoding="utf-8",
    )
    result = subprocess.run(  # noqa: S603 - node checks a file this test just wrote
        ["node", str(script_file)],  # noqa: S607 - node via PATH is the documented check
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"the dashboard did not render:\n{result.stderr.strip()}"
    return json.loads(result.stdout)


def _store() -> list[dict[str, object]]:
    """Two answers (one of them inferred), a rationale and a clarification."""
    return [
        {
            "id": "org/repo/pr-1/asked",
            "kind": "capture",
            "category": "trade_off",
            "repo": "org/repo",
            "pr": 1,
            "author": "opencode",
            "answered_by_agent": "opencode",
            "capture_source": "webhook",
            "agent_authored": True,
            "structure": "anchored",
            "title": "Why is the window narrow?",
            "answer": "Because of the rebuild.",
            "answered_at": "2026-03-04T12:00:00+00:00",
        },
        {
            "id": "org/repo/pr-1/guessed",
            "kind": "capture",
            "category": "trade_off",
            "repo": "org/repo",
            "pr": 1,
            "author": "opencode",
            "answered_by_agent": "opencode",
            "capture_source": "webhook",
            "agent_authored": True,
            "structure": "inferred",
            "structure_inferred_by": "anthropic/claude-opus-5",
            "title": "Why is the ledger append-only?",
            "answer": "Probably so a row cannot be edited.",
            "answered_at": "2026-03-04T12:00:00+00:00",
        },
        {
            "id": "org/repo/pr-1/rationale/r1",
            "kind": "rationale",
            "category": "declared",
            "repo": "org/repo",
            "pr": 1,
            "author": "kojutsu-pilot",
            "answered_by_agent": "kojutsu-pilot",
            "rationale_model": "opencode/model",
            "capture_source": "asserted",
            "structure": "anchored",
            "title": "Rationale r1",
            "answer": "Used a lease token.",
            "answered_at": "2026-03-04T12:00:00+00:00",
        },
        {
            "id": "org/repo/pr-1/clarification/c1",
            "kind": "clarification",
            "category": "",
            "repo": "org/repo",
            "pr": 1,
            "author": "davy",
            "comment_author": "davy",
            "clarification_model": "opencode/model",
            "clarification_comment_id": 5,
            "capture_source": "collect",
            "structure": "anchored",
            "title": "Clarification c1",
            "answer": "The narrow window is deliberate.",
            "answered_at": "2026-03-04T12:00:00+00:00",
        },
    ]


def test_the_asked_and_answered_figure_leaves_out_the_inferred_pairing(tmp_path: Path) -> None:
    """One, read off the rendered page rather than asserted about the source.

    Four records: one anchored answer, one inferred pairing, one rationale, one
    clarification. Only the first is a question somebody is on record as having been
    asked, so ``asked & answered`` is 1 -- and the two record kinds that answer no
    question are not quietly added to make the number look busier.
    """
    rendered = _render(_store(), "", tmp_path)

    assert rendered["kpis"]["asked & answered"] == "1"
    assert rendered["kpis"]["captures"] == "3", "captures count the reasoning, not the rationales"
    assert rendered["kpis"]["rationales"] == "1"
    assert rendered["kpis"]["clarifications"] == "1"
    assert "1 inferred pairing is not in" in rendered["kpi_note"], (
        "the exclusion has to be visible next to the number, or the honest count and "
        "the flattering one are the same shape"
    )
    # The "Answered by" panel is a count of answers too, so it gets the same
    # treatment and the same number. One bar, one answer; the other three records
    # are named as excluded rather than quietly left out of the total.
    assert rendered["by_agent"] == [["opencode", "1"]]
    assert "3 records excluded" in rendered["by_agent_note"]


def test_the_structure_filter_reports_what_it_excluded(tmp_path: Path) -> None:
    """Filtering for anchored records hides the inferred one and says so.

    The whole point: a filter that dropped the inferred record silently would let a
    reader conclude the store holds one answered question, which is the flattering
    and wrong reading the axis exists to prevent. The count is named, and broken
    down, because "1 excluded" still does not say whether it was a guess or a
    record this build could not read.
    """
    rendered = _render(_store(), "anchored", tmp_path)

    assert rendered["row_count"] == 3, "the inferred pairing is filtered out"
    assert not any("inferred" in badge for badge in rendered["row_badges"]), (
        "the badge is the row's own marking; a filter that leaves it visible is not filtering"
    )
    assert rendered["filter_note"] == (
        "Showing anchored only — 1 of 4 records excluded (1 inferred). "
        "They are in the store; widen the filter to see them."
    )


def test_an_unreadable_structure_is_counted_apart_from_an_inference(tmp_path: Path) -> None:
    """Two different claims about the store, and the filter must not merge them.

    "Inferred" is a writer saying it guessed. "unknown" is the console reporting a
    value it could not resolve -- a renamed axis, or a hand-edited document.
    Reported as one number, a rename would look like the readers' caution; reported
    as two, it is a store this build cannot read, and the dashboard says so.

    ``unknown`` is the value the console actually sends, which is the contract this
    page is written against: it never has to resolve a structure itself.
    """
    store = _store()
    store.append(
        {
            "id": "org/repo/pr-1/garbled",
            "kind": "capture",
            "category": "edge_case",
            "repo": "org/repo",
            "pr": 1,
            "author": "opencode",
            "capture_source": "webhook",
            "structure": "unknown",
            "title": "Something.",
            "answer": "Something.",
            "answered_at": "2026-03-04T12:00:00+00:00",
        }
    )
    rendered = _render(store, "anchored", tmp_path)

    assert "1 inferred, 1 unknown" in rendered["filter_note"]
    assert rendered["kpis"]["asked & answered"] == "1", (
        "a structure this page cannot read is not an answered question either"
    )
    # And unfiltered, the row is badged rather than passed over. A page that simply
    # did not badge it would leave a reader assuming anchored, which is the one
    # reading it must never support.
    unfiltered = _render(store, "", tmp_path)
    assert any("structure · unknown" in badge for badge in unfiltered["row_badges"])
    assert unfiltered["kpis"]["asked & answered"] == "1"
