"""The import layering, checked rather than asserted.

The intended shape is one-directional: entry points (``cli``, ``webhook``,
``worker``, ``mcp_server``) -> ``runtime`` -> ``core`` -> ``integrations`` ->
leaves (``config``, ``models``, ``allowlist``, ``identity``, ``text_limits``,
``webhook.urls``). The disease these tests prevent is a back-edge --
``integrations`` importing ``core`` -- which once forced the layering to be
held together by function-level imports. Every test here parses the AST, so a
new edge fails the build rather than waiting for a reviewer to notice it.

``cli.py`` is deliberately out of scope for the no-lazy-import test: deferring
heavy command imports to call time is startup latency, not cycle avoidance.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "kojutsu"
INTEGRATIONS = SRC / "integrations"

#: What ``integrations`` may reach upward. Everything else under ``kojutsu``
#: (``core``, ``runtime``, ``worker``, ``webhook``, tools) is a back-edge.
_INTEGRATIONS_ALLOWLIST = frozenset(
    {
        "kojutsu.config",
        "kojutsu.models",
        "kojutsu.text_limits",
        "kojutsu.allowlist",
        "kojutsu.identity",
        "kojutsu.net",
    }
)

#: Leaf modules and the internal imports each is allowed. A leaf that reaches
#: outside this table is not a leaf, and whatever imports it inherits the edge.
_LEAF_ALLOWLIST = {
    "kojutsu.config": frozenset(),
    "kojutsu.models": frozenset(),
    "kojutsu.allowlist": frozenset({"kojutsu.config"}),
    "kojutsu.identity": frozenset(),
    "kojutsu.net": frozenset(),
    "kojutsu.repo_name": frozenset(),
    "kojutsu.text_limits": frozenset(),
    "kojutsu.webhook.urls": frozenset({"kojutsu.net"}),
}

#: Files where a function-level ``kojutsu`` import is a layering workaround
#: rather than startup latency. ``cli*.py``, ``compare.py`` and ``dev_console.py``
#: defer heavy imports to command time on purpose and are not covered.
_NO_LAZY_SCOPE = (
    *(SRC / "core").glob("*.py"),
    *(SRC / "integrations").glob("*.py"),
    SRC / "runtime.py",
    SRC / "relay_worker.py",
    SRC / "worker" / "loop.py",
    SRC / "worker" / "steps.py",
    SRC / "worker" / "sources.py",
    SRC / "worker" / "state.py",
    SRC / "webhook" / "lifecycle.py",
    SRC / "webhook" / "server.py",
    SRC / "webhook" / "urls.py",
)


def _absolute_imports(tree: ast.Module) -> list[str]:
    """Every absolute ``kojutsu.*`` import in the tree, including nested ones."""
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("kojutsu"):
            found.append(node.module)
        elif isinstance(node, ast.Import):
            found.extend(name.name for name in node.names if name.name.startswith("kojutsu"))
    return found


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_integrations_imports_no_higher_layer() -> None:
    """``integrations`` reaching into ``core`` (or above) is the back-edge that
    once turned one-way layering into a cycle. Leaves and self-imports only."""
    offenders: list[str] = []
    for path in sorted(INTEGRATIONS.glob("*.py")):
        for imported in _absolute_imports(_parse(path)):
            if imported == "kojutsu.integrations" or imported.startswith("kojutsu.integrations."):
                continue
            if imported not in _INTEGRATIONS_ALLOWLIST:
                offenders.append(f"{path.name} imports {imported}")
    assert not offenders, "integrations must not reach above itself:\n" + "\n".join(offenders)


def test_leaf_modules_import_nothing_internal() -> None:
    """A leaf that imports the layers above it is not a leaf."""
    offenders: list[str] = []
    for dotted, allowed in sorted(_LEAF_ALLOWLIST.items()):
        path = ROOT / "src" / Path(*dotted.split(".")).with_suffix(".py")
        for imported in _absolute_imports(_parse(path)):
            if imported not in allowed:
                offenders.append(f"{dotted} imports {imported}")
    assert not offenders, "leaf modules must stay leaves:\n" + "\n".join(offenders)


def test_mcp_servers_do_not_rewrite_sys_path() -> None:
    """The wheel ships both ``kojutsu`` and ``mcp_server`` (``pyproject``
    ``[tool.hatch.build.targets.wheel]``), and the ``kojutsu-mcp`` entry point
    resolves without path surgery. A ``sys.path.insert`` in the servers would
    silently redirect imports to a checkout when the installed package was
    meant, which is the opposite of the isolation the sandbox work buys."""
    offenders: list[str] = []
    for path in sorted((ROOT / "mcp_server").glob("*.py")):
        tree = _parse(path)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"insert", "append"}
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "path"
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "sys"
            ):
                offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, "mcp_server must not rewrite sys.path:\n" + "\n".join(offenders)


def _inside_function(parents: dict[int, ast.AST], node: ast.AST) -> bool:
    """True when ``node`` sits inside a function, stopping at TYPE_CHECKING."""
    current: ast.AST | None = parents.get(id(node))
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return True
        if (
            isinstance(current, ast.If)
            and isinstance(current.test, ast.Name)
            and current.test.id == "TYPE_CHECKING"
        ):
            return False
        current = parents.get(id(current))
    return False


def test_no_function_level_kojutsu_imports_in_layered_code() -> None:
    """A deferred ``kojutsu`` import in layered code is how a cycle hides: it
    works until someone imports the modules in the other order. TYPE_CHECKING
    blocks are exempt -- they emit no runtime import."""
    offenders: list[str] = []
    for path in _NO_LAZY_SCOPE:
        if not path.exists():
            continue
        tree = _parse(path)
        parents: dict[int, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[id(child)] = parent
        for node in ast.walk(tree):
            if not isinstance(node, (ast.ImportFrom, ast.Import)):
                continue
            modules = (
                [node.module]
                if isinstance(node, ast.ImportFrom)
                else [name.name for name in node.names]
            )
            if not any((module or "").startswith("kojutsu") for module in modules):
                continue
            if _inside_function(parents, node):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, "function-level kojutsu imports in layered code:\n" + "\n".join(offenders)
