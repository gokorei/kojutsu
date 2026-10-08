"""Dashboard rendering for the census, question and backfilled records.

**Every test in this module that needs the client side is marked ``xfail`` and is
known not to run.** They
were written against this branch's own dashboard rewrite (``#records``,
``#by-kind``, census KPI tiles), and the merge took main's page as its base
because that is the page main ships. The behaviour they guard -- a census record
counted apart from captures, a question rendered as a question, an unaccounted
kind surfaced rather than folded in, a kind filter, and the backfilled flag --
**is implemented** and is covered at the layer below: every one of those
assertions is also made against the endpoint payload in
``tests/test_dev_console_record_kinds.py``, which passes.

What is missing is the client-side half: a harness driving main's markup rather
than the other implementation's. This is tracked rather than deleted, and rather
than quietly made to pass, because a dashboard assertion that has stopped running
is worse than one that is visibly broken.

The tests that do not need the rewritten markup still run.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

XFAIL_REASON = (
    "The harness drives main's page and the fixtures now use the merged console's "
    "field names. What is left is this test's assertion, which was written against "
    "this branch's own page: it expects figures, row strings and note prose that "
    "main's design puts elsewhere (main reports record kinds as KPI tiles rather "
    "than a chart, and its own notes explain different things). The behaviour is "
    "implemented and is asserted at the payload layer in "
    "test_dev_console_record_kinds.py. Porting the assertions is what remains."
)

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


# --------------------------------------------------------------------------
# Running the script
# --------------------------------------------------------------------------
#
# A stub DOM rather than a headless browser, on purpose: jsdom is not a dependency
# and adding one to assert on eleven numbers would be a poor trade. What the script
# actually touches is a short, closed list of properties, and the stub below models
# exactly those — every write is recorded so the assertions read the rendered page
# rather than the script's own intentions.

_STUB = """
class StubElement {
  constructor(tag) {
    this.tagName = tag;
    this.className = '';
    this.textContent = '';
    this.innerHTML = '';
    this.dataset = {};
    this.style = {};
    this.childNodes = [];
    this.options = [];
    this.value = '';
    this.files = [];
    this.classList = { add: () => {}, remove: () => {} };
  }
  append(...kids) { for (const k of kids) this.childNodes.push(k); }
  replaceChildren(...kids) { this.childNodes = [...kids]; }
  addEventListener() {}
}

// Every id the page declares, which is every id the script can reach: a selector
// that resolves to null here is a selector that would throw in a browser.
const ids = [...new Set([...html.matchAll(/id="([^"]+)"/g)].map((m) => m[1]))];
const registry = new Map(ids.map((id) => [`#${id}`, new StubElement('div')]));
const filters = registry.get('#filters');
// Every filter the page reads. A selector missing here is not a test-only gap:
// the harness throws on the first one it cannot resolve, which is the check
// that keeps a rename from shipping.
for (const name of ['q', 'kind', 'category', 'source', 'human', 'agent', 'structure']) {
  const input = new StubElement('input');
  input.name = name;
  if (name !== 'q') {
    const opt = new StubElement('option');
    opt.value = '';
    input.options = [opt];
    // The same object `#filters select[name=...]` resolves to, so a filter read
    // through the form and one written through the selector are the same value.
    registry.set(`#filters select[name=${name}]`, input);
  }
  filters[name] = input;
}
const sourceForm = registry.get('#source');
for (const name of ['base', 'every']) {
  const input = new StubElement('input');
  input.name = name;
  sourceForm[name] = input;
}
sourceForm.every.value = '0';

const document = {
  querySelector: (sel) => {
    const found = registry.get(sel);
    if (found === undefined) throw new Error('unreachable selector: ' + sel);
    return found;
  },
  createElement: (tag) => new StubElement(tag),
};
const location = { protocol: 'http:' };
const setInterval = () => 0;
const clearInterval = () => {};
// The script calls refresh() as it boots; this stands in for the network so the
// page settles without one, and render() is called explicitly below.
const fetch = async () => ({ ok: false, status: 0, json: async () => ({ error: 'offline' }) });
"""

# Runs the dashboard against a snapshot and reports what it rendered. Everything it
# returns is read back out of the stub DOM, never out of the script's internals.
_PROBE = """
const snapshot = JSON.parse(payload);
render(normalize(snapshot));

// Flatten an element's subtree to the text a reader would see.
const shown = (node) => {
  const out = [];
  const walk = (n) => {
    if (n instanceof StubElement) {
      if (n.textContent) out.push(n.textContent);
      if (n.innerHTML) out.push(n.innerHTML);
      n.childNodes.forEach(walk);
    }
  };
  walk(node);
  return out;
};

// A KPI tile is (number, label, sub-line); key it by label.
const kpi = {};
for (const card of registry.get('#kpis').childNodes) {
  const parts = card.childNodes.map((c) => c.textContent);
  kpi[parts[1]] = { value: parts[0], sub: parts[2] || '' };
}

// A bar panel is (value, count) pairs, keyed on the record's own value rather than
// the prose label, so these assertions are about counts and not about wording.
//
// There is deliberately no `#by-kind` panel here. main's page charts the axes a
// reader filters on -- source, category, agent, human -- and reports the record kinds
// in the KPI row instead, one tile each. That is the page's design, so the kind
// breakdown is asserted from `#kpis` rather than from a chart this branch once had.
// main's bar rows carry their key as the text of a `.k` child rather than in a
// dataset attribute, so the key is read from the rendered label.
const bars = (sel) => Object.fromEntries(registry.get(sel).childNodes
  .filter((r) => r.className === 'bar-row')
  .map((r) => [r.childNodes[0].textContent, r.childNodes[2].textContent]));

// The rendered list, one entry per record, in the order the script put them there.
const rows = registry.get('#captures').childNodes
  .filter((li) => li.className === 'cap')
  .map((li) => shown(li).join(' | '));

const options = (name) => registry.get(`#filters select[name=${name}]`).childNodes
  .map((o) => o.value || o.textContent);

console.log(JSON.stringify({
  kpi,
  kinds: Object.fromEntries(Object.entries(kpi)
    .filter(([k]) => ['captures', 'observations', 'questions', 'unaccounted',
      'rationales', 'clarifications', 'evaluations'].includes(k))
    .map(([k, v]) => [k, v.value])),
  sources: bars('#by-source'),
  categories: bars('#by-category'),
  agents: bars('#by-agent'),
  humans: bars('#by-human'),
  kindOptions: options('kind'),
  timelineDays: registry.get('#timeline').childNodes.length,
  note: registry.get('#kpi-note').textContent,
  sourceNote: registry.get('#source-note').textContent,
  rows,
  // The status line is where the page reports that it could not read a snapshot.
  // Without it in the probe a render that throws halfway is indistinguishable from a
  // render that simply had fewer rows -- and the first is a bug in the page while the
  // second is the fixture. That ambiguity is how a broken page reads as a small corpus.
  sourceStatus: registry.get('#source-status').textContent,
}));
"""


def _render(snapshot: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    """Boot the dashboard against one snapshot and return what it rendered."""
    if shutil.which("node") is None:
        pytest.skip("node is required to run the dashboard's JavaScript")
    driver = tmp_path / "driver.js"
    driver.write_text(
        f"const html = {json.dumps(DASHBOARD.read_text(encoding='utf-8'))};\n"
        f"const payload = {json.dumps(json.dumps(snapshot))};\n"
        f"{_STUB}\n{_inline_script()}\n{_PROBE}",
        encoding="utf-8",
    )
    result = subprocess.run(["node", str(driver)], capture_output=True, text=True, check=False)  # noqa: S603, S607 - node checks a file this test just wrote
    assert result.returncode == 0, f"the dashboard threw while rendering:\n{result.stderr.strip()}"
    return json.loads(result.stdout)


def _record(kind: str, doc_id: str, **overrides: Any) -> dict[str, Any]:
    """One record as ``capture_of`` emits it -- merged field names, not this
    branch's earlier rewrite.

    The rename matters and is the reason these tests could not run: the console
    reports ``answer``/``answer_truncated``/``answered_at``/``backfilled``, and the
    page reads those. A fixture still spelling ``text``/``timestamp`` produced
    records the page could not render, so every assertion was looking for content
    that was never put in. Keep this in step with ``dev_console.capture_of``.
    """
    record: dict[str, Any] = {
        "id": doc_id,
        "repo": "org/repo",
        "pr": 1,
        "jira": "",
        "kind": kind,
        "declared_kind": kind if kind not in {"undeclared", "capture"} else "",
        "title": f"a {kind}",
        # Only the sources that were actually captured carry one. A question and a
        # rationale are not captures and the console emits an empty string for both,
        # which is what makes them excluded from the source panel.
        "capture_source": "" if kind in {"question", "rationale", "census"} else "webhook",
        "backfilled": False,
        "structure": "anchored",
        "structure_stated": False,
        "structure_inferred_by": "",
        "category": "" if kind in {"question", "rationale", "census"} else "trade_off",
        "author": "davy",
        "comment_author": "",
        "answered_by_agent": "",
        "agent_authored": False,
        "answer": f"text for {kind}",
        "answer_truncated": False,
        "answered_at": "2026-09-30T12:00:00+00:00",
    }
    if kind == "census":
        record.update(
            observation_delivery_id="delivery-1",
            observation_action="opened",
        )
    if kind == "question":
        record.update(question_id="q-7", question_status="answered")
    record.update(overrides)
    return record


_ALL_KINDS = [
    "answer",
    "review_verdict",
    "inline_review_comment",
    "pr_lifecycle",
    "check_run",
    "census",
    "rationale",
    "question",
    "undeclared",
    "unrecognised",
]
_FULL_CORPUS = [_record(kind, f"org/repo/pr-1/{kind}") for kind in _ALL_KINDS]


def test_one_observation_does_not_truncate_the_list(tmp_path: Path) -> None:
    """A record after an observation must still be rendered.

    Regression, and the worst failure on this page while it stood: the census row
    branch ended in ``return`` while sitting directly inside ``for (const c of
    rows)``, so it returned from ``renderCaptures`` rather than from the iteration.
    Every record after the first observation was dropped from the list.

    It was invisible because the list is rebuilt from scratch on every render, so the
    symptom was not a stale page but a count that did not match the corpus -- a
    census record silently deleting the records that followed it. The payload was
    correct throughout, so nothing below this layer could see it.

    The order matters: the observation sits in the middle of the corpus on purpose.
    Placed last, a ``return`` at the end of the loop looks exactly like ``continue``.
    """
    corpus = [
        _record("answer", "org/repo/pr-1/a"),
        _record("census", "org/repo/pr-1/c", observation_delivery_id="delivery-9"),
        _record("question", "org/repo/pr-1/q"),
        _record("rationale", "org/repo/pr-1/r"),
    ]
    rendered = _render({"captures": corpus}, tmp_path)

    assert len(rendered["rows"]) == len(corpus), (
        "a record rendered after an observation was dropped; the census branch is "
        "using `return` where it needs `continue`, which ends the whole loop"
    )
    assert any("delivery-9" in row for row in rendered["rows"])
    assert any("question" in row for row in rendered["rows"])


def test_the_dashboard_counts_every_kind_apart(tmp_path: Path) -> None:
    """No kind is added to another, and every record reaches the page.

    Asserted as the whole KPI breakdown summing to the corpus rather than "the
    capture count is plausible". A plausible total is what silent absorption produces,
    and this page had exactly that bug once already: an observation used ``return``
    inside the render loop and every record after it vanished from the list while the
    capture count stayed plausible.

    The five knowledge kinds share one tile, because on this page they are one thing
    to a reader -- evidence somebody captured -- and main reports them that way. The
    kinds that are *not* that get a tile each.
    """
    rendered = _render({"captures": _FULL_CORPUS}, tmp_path)
    kpi = {name: tile["value"] for name, tile in rendered["kpi"].items()}

    assert len(rendered["rows"]) == len(_FULL_CORPUS), "a record never reached the page"
    assert int(kpi["captures"]) == 5, "the five knowledge kinds are one tile"
    for kind in ("observations", "questions", "rationales"):
        assert kpi[kind] == "1", f"{kind} was folded into another tile"
    assert kpi["unaccounted"] == "2", (
        "a record whose kind this build cannot resolve must be counted on its own "
        "tile, never inside the capture total"
    )
    counted = sum(
        int(kpi[k]) for k in ("captures", "observations", "questions", "rationales", "unaccounted")
    )
    assert counted == len(_FULL_CORPUS), (
        f"the tiles account for {counted} of {len(_FULL_CORPUS)} records"
    )


def test_a_census_record_is_never_counted_as_a_capture(tmp_path: Path) -> None:
    """The renderer's boundary, and the rate's two terms.

    Five captures and two observations must read as five and two. A renderer that
    included the observation in the capture count would show 7 and 100% -- the exact
    claim the record exists to prevent: "we saw seven things and captured all of
    them".
    """
    five = [
        _record(kind, f"org/repo/pr-1/{kind}")
        for kind in (
            "answer",
            "review_verdict",
            "inline_review_comment",
            "pr_lifecycle",
            "check_run",
        )
    ]
    two = [
        _record("census", f"org/repo/pr-{n}/c", observation_delivery_id=f"delivery-{n}")
        for n in (2, 3)
    ]
    rendered = _render({"captures": five + two}, tmp_path)

    assert rendered["kpi"]["captures"]["value"] == "5"
    assert rendered["kpi"]["observations"]["value"] == "2"
    # The rate names both terms rather than standing alone as a percentage, because
    # "71%" over what is exactly the misreading the census was added to prevent.
    assert rendered["kpi"]["capture rate"]["value"] == "71% of 7 observed"


def test_the_observation_count_sits_next_to_the_capture_count(tmp_path: Path) -> None:
    """Adjacent in the row, not in a footnote.

    The two are one ratio split in two. A reader who has to go looking for the second
    term will read the first one on its own, which is the number this whole effort
    exists to make impossible to quote.
    """
    corpus = [_record("answer", "org/repo/pr-1/a"), _record("census", "org/repo/pr-2/c")]
    labels = list(_render({"captures": corpus}, tmp_path)["kpi"])

    assert labels.index("observations") == labels.index("captures") + 1
    assert labels.index("capture rate") <= 3, (
        f"the rate is the {labels.index('capture rate')}th tile; its two terms are "
        f"the first three and it belongs with them"
    )


def test_a_census_record_renders_as_an_observation_not_an_empty_capture(tmp_path: Path) -> None:
    """As an observation, carrying what makes it checkable.

    The failure this guards is a record rendered as a capture with nothing in it --
    a conclusion with nothing behind it, which is how a rationale used to render. The
    delivery id is the anchor: "we processed this and kept nothing" is unverifiable
    without it, and an empty row would carry no way to check the claim at all.
    """
    record = _record(
        "census",
        "org/repo/pr-2/c",
        title="nothing captured: delivery processed",
        observation_delivery_id="delivery-1",
        observation_action="opened",
    )
    rendered = _render({"captures": [record]}, tmp_path)

    row = rendered["rows"][0]
    assert "nothing captured" in row
    assert "delivery delivery-1" in row, "the anchor is what makes the row checkable"
    assert "opened" in row, "the action says what was observed"
    # Named nobody as answering anything, because nobody did.
    assert "answered by" not in row
    assert "attributed to" not in row
    # And it is not counted as a capture, which is the point of rendering it apart.
    assert rendered["kpi"]["captures"]["value"] == "0"
    assert rendered["kpi"]["observations"]["value"] == "1"


def test_a_question_renders_as_a_question(tmp_path: Path) -> None:
    """A request, named as one, and attributed to whoever made it.

    Rendering it as a capture that found nothing to say is the specific misreading the
    projection exists to prevent, and a request with no asker on the row is a request
    floating free of the person who made it -- which is the one fact a reader needs
    before deciding whether to answer.
    """
    record = _record("question", "org/repo/pr-3/q", author="octocat", question_id="q-7")
    rendered = _render({"captures": [record]}, tmp_path)

    row = rendered["rows"][0]
    assert "question" in row
    assert "asked by octocat" in row
    assert "answered by" not in row
    assert rendered["kpi"]["captures"]["value"] == "0"
    assert rendered["kpi"]["observations"]["value"] == "0"
    assert rendered["kpi"]["questions"]["value"] == "1"


def test_an_unaccounted_record_is_surfaced_and_left_out_of_every_total(tmp_path: Path) -> None:
    """Reported, and counted nowhere it would imply something.

    Two different failures are kept apart deliberately. ``undeclared`` is a store that
    stopped labelling; ``unrecognised`` is a kind somebody added and did not finish
    wiring. Only the second is about the store being broken, and folding either into
    the capture total would report a corpus as something it is not.
    """
    corpus = [
        _record("undeclared", "org/repo/pr-1/u"),
        _record("unrecognised", "org/repo/pr-1/x", declared_kind="verification_attestation"),
        _record("answer", "org/repo/pr-1/a"),
    ]
    rendered = _render({"captures": corpus}, tmp_path)
    kpi = {name: tile["value"] for name, tile in rendered["kpi"].items()}

    assert kpi["unaccounted"] == "2"
    assert kpi["captures"] == "1", "an unaccounted record was counted as evidence"
    # Surfaced, not hidden: both still render, and the one whose key this build knows
    # is contradicted keeps the kind it declared.
    assert len(rendered["rows"]) == 3
    # Both still render. The `structure` badge is deliberately not asserted here: that
    # badge is about a structure this build cannot read, not about a kind it cannot,
    # and conflating the two would be a new claim of the same family as the bug this
    # test exists to catch. The kind shows in the record's own tile instead.
    assert any("unrecognised" in row for row in rendered["rows"])
    assert any("undeclared" in row for row in rendered["rows"])


def test_a_backfilled_record_is_distinguishable_from_a_witnessed_one(tmp_path: Path) -> None:
    """A reader must be able to tell, without opening a payload.

    ``backfilled`` is the one source that makes a weaker guarantee: it shows what the
    forge says now, not what it said then. Folding it into the webhook bar would let
    a reconstructed record read as witnessed evidence, which is the same failure as the
    census being counted inside the capture total.
    """
    corpus = [
        _record("answer", "org/repo/pr-1/a", capture_source="backfilled", backfilled=True),
        _record("answer", "org/repo/pr-1/b", capture_source="webhook"),
    ]
    rendered = _render({"captures": corpus}, tmp_path)

    assert rendered["sources"] == {"backfilled": "1", "webhook": "1"}, (
        "the two sources are counted on their own bars, so neither absorbs the other"
    )
    backfilled_row = next(r for r in rendered["rows"] if "backfilled" in r)
    assert "unverified" in backfilled_row, (
        "the row is badged; a reconstructed record must not read as witnessed evidence"
    )


def test_a_record_with_no_axis_of_its_own_is_not_bucketed_as_unknown(tmp_path: Path) -> None:
    """ "No source" is a fact about the record; "unknown" is a fact about the store.

    A question, a rationale and an observation carry no capture source because none
    was captured. Filing them under a source called ``unknown`` would assert that the
    store lost track of something it was never given -- and would put them in the
    source panel, where a reader filtering by provenance would find them.
    """
    corpus = [
        _record("answer", "org/repo/pr-1/a"),
        _record("question", "org/repo/pr-3/q"),
        _record("rationale", "org/repo/pr-1/r"),
    ]
    rendered = _render({"captures": corpus}, tmp_path)

    assert rendered["sources"] == {"webhook": "1"}, (
        "only the record that was actually captured appears in the source panel"
    )
    assert rendered["categories"] == {"trade_off": "1"}, (
        "a record with no question has no category, and is not filed under one"
    )
    assert len(rendered["rows"]) == 3, "they are still on the page, just on no source axis"


def test_every_kind_is_offered_as_a_filter(tmp_path: Path) -> None:
    """Otherwise a record kind exists that a reader cannot narrow to.

    Compared as sets: the page orders the filter by its own vocabulary so the
    order stays stable between refreshes, and that ordering is presentation rather
    than behaviour.
    """
    rendered = _render({"captures": _FULL_CORPUS}, tmp_path)

    # [1:] drops the "all kinds" option every filter keeps first.
    assert sorted(rendered["kindOptions"][1:]) == sorted(_ALL_KINDS)


def test_observations_and_questions_reach_the_timeline(tmp_path: Path) -> None:
    """Dated, so they reach the timeline.

    A per-kind time field would leave them undated, which drops them off the chart
    without saying so -- the same silent absorption, in a chart rather than in a total.
    Four records on four days, three of them non-captures: a timeline showing one
    column would mean three of them never got a date.
    """
    corpus = [
        _record("answer", "org/repo/pr-1/a", answered_at="2026-09-27T12:00:00+00:00"),
        _record("census", "org/repo/pr-2/c", answered_at="2026-09-28T12:00:00+00:00"),
        _record("question", "org/repo/pr-3/q", answered_at="2026-09-29T12:00:00+00:00"),
        _record("rationale", "org/repo/pr-1/r", answered_at="2026-09-30T12:00:00+00:00"),
    ]
    rendered = _render({"captures": corpus}, tmp_path)

    assert rendered["timelineDays"] == 4, (
        "a record with no date of its own drops off the timeline silently"
    )
