from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

from kojutsu import models
from kojutsu.models import STRUCTURE_INFERRED_BY

LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
SHELL_BLOCK_PATTERN = re.compile(r"```(?:bash|sh)\n(.*?)```", re.DOTALL)
STALE_TERMS = ("mongodb", "mongodb_uri", "neo4j", "mongodb://")
DOCKER_GUIDANCE_FILES = ("README.md", "docs/webhook-integration.md")
DOCKER_SQLITE_ENV = {
    "TANSEKI_OUTBOX_PATH": "/data/tanseki-outbox.db",
    "KOJUTSU_REGISTRY_PATH": "/data/registry.db",
}
DOCKER_SERVICE_URLS = {
    "TANSEKI_URL": "http://host.docker.internal:8088",
    "OLLAMA_URL": "http://host.docker.internal:11434",
}
WEBHOOK_ALLOWLIST_DOCS = (
    ".env.example",
    "README.md",
    "docs/webhook-integration.md",
)

#: The frontmatter keys the read path has to surface for a stored fact not to be
#: invisible to whoever reads it.
#:
#: Deliberately not "every key a mapper writes". Most of what a mapper emits is
#: identity or presentation -- ``title``, ``author``, ``tags``, the timestamps -- and
#: a check that demanded all of it would report every run and therefore report
#: nothing, which is worse than having no check. These are the keys where omission
#: changes what a reader is allowed to conclude:
#:
#: - the structure axis, in both halves. ``structure`` is what says a pairing was
#:   established rather than matched by a model, and ``structure_inferred_by_model``
#:   is what says which one. Serve the first without the second and an inferred
#:   record is an unattributed guess; serve neither and it is indistinguishable from
#:   a conversation somebody had, which is the whole failure the axis was added for.
#: - the two record kinds that are not answers. A rationale is a claim by an agent
#:   about its own work and a clarification is a quotation from a comment. Neither
#:   is readable as either unless the reader can see who said it.
#: - what a measurement is a measurement of. A report about a classifier carries
#:   the most quotable numbers the store holds and the least self-evident
#:   provenance, and the failure mode is specific: a precision figure served
#:   without its subject is indistinguishable from a finding about the pull
#:   request the harness was pointed at. ``structure`` is what caught the same
#:   conflation one axis earlier, for the same reason.
#:
#: This is the one place the read path's key list is pinned. ``check_docs.py`` used
#: to enumerate no frontmatter key at all, so a rename on the write side could leave
#: a fact stored and unservable with nothing failing.
READ_PATH_REQUIRED_KEYS = (
    "structure",
    "capture_source",
    "rationale_source",
    "declared_by",
    "declared_by_model",
    "clarified_by_agent",
    "clarified_by_model",
    "evaluation_target",
    "evaluated_model",
)


def _markdown_files(root: Path) -> list[Path]:
    ignored = {
        ".git",
        ".venv",
        ".package-venv",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".pijul",
        ".playwright-mcp",
    }
    return [path for path in root.rglob("*.md") if not any(part in ignored for part in path.parts)]


def _check_links(root: Path, path: Path, text: str, errors: list[str]) -> None:
    for raw_target in LINK_PATTERN.findall(text):
        target = raw_target.strip().split("#", 1)[0].split("?", 1)[0]
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        candidate = (path.parent / target).resolve()
        if not candidate.is_relative_to(root) or not candidate.exists():
            errors.append(f"{path.relative_to(root)}: missing link target {target}")


def _check_shell_scripts(root: Path, errors: list[str]) -> None:
    ignored = {
        ".git",
        ".venv",
        ".package-venv",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".pijul",
        ".playwright-mcp",
    }
    for path in root.rglob("*.sh"):
        if any(part in ignored for part in path.parts):
            continue
        result = subprocess.run(  # noqa: S603 - argv is built from repo .sh paths, never caller input
            ["bash", "-n", str(path)],  # noqa: S607 - bash resolved via PATH is the documented check
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            errors.append(f"{path.relative_to(root)}: invalid shell syntax")


def _check_docker_guidance(root: Path, errors: list[str]) -> None:
    dockerfile = root / "Dockerfile"
    dockerfile_text = dockerfile.read_text(encoding="utf-8")
    if 'VOLUME ["/data"]' not in dockerfile_text:
        errors.append("Dockerfile: missing /data volume")
    for key, value in DOCKER_SQLITE_ENV.items():
        if f"{key}={value}" not in dockerfile_text:
            errors.append(f"Dockerfile: missing persistent {key}={value}")

    for relative_path in DOCKER_GUIDANCE_FILES:
        path = root / relative_path
        text = path.read_text(encoding="utf-8")
        blocks = [
            block
            for block in SHELL_BLOCK_PATTERN.findall(text)
            if re.search(r"^\s*docker run\b", block, re.MULTILINE)
        ]
        if not blocks:
            errors.append(f"{relative_path}: missing docker run example")
            continue
        for block in blocks:
            for key, value in DOCKER_SQLITE_ENV.items():
                if f"-e {key}={value}" not in block:
                    errors.append(f"{relative_path}: docker run must set {key}={value}")
            for key, value in DOCKER_SERVICE_URLS.items():
                if f"-e {key}={value}" not in block:
                    errors.append(f"{relative_path}: docker run must set {key}={value}")
            if re.search(
                r"-(?:e|--env)\s+(?:TANSEKI_URL|OLLAMA_URL)=https?://(?:localhost|127\.0\.0\.1)",
                block,
            ):
                errors.append(f"{relative_path}: docker run uses container-local service URL")
        lowered = re.sub(r"\s+", " ", text.lower())
        for term in ("host.docker.internal", "host-gateway", "macos", "linux", "service dns"):
            if term not in lowered:
                errors.append(f"{relative_path}: missing Docker networking guidance {term}")


def _check_webhook_allowlist_guidance(root: Path, errors: list[str]) -> None:
    for relative_path in WEBHOOK_ALLOWLIST_DOCS:
        path = root / relative_path
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        for term in (
            "github_webhook_allowed_repositories",
            "required",
            "empty",
            "public",
            "*",
        ):
            if term not in lowered:
                errors.append(f"{relative_path}: missing webhook allowlist guidance {term}")
        if "reject" not in lowered and "deni" not in lowered:
            errors.append(f"{relative_path}: missing empty-allowlist deny behavior")


def _check_read_path_keys(root: Path, errors: list[str]) -> None:
    """Pin the frontmatter keys the MCP read path copies onto what it renders.

    Read from the source with ``ast`` rather than grepped, so a key that moves into
    a variable, is built by a comprehension, or appears only inside a comment is not
    counted as surfaced. A grep here would pass on a comment saying the key is
    handled, which is the failure this is meant to catch.
    """
    path = root / "mcp_server" / "server.py"
    if not path.exists():
        errors.append("mcp_server/server.py: missing, so the read path cannot be checked")
        return
    tree = ast.parse(path.read_text(encoding="utf-8"))

    literal_keys: set[str] = set()
    referenced_names: set[str] = set()
    for node in ast.walk(tree):
        # The list the read path copies key by key onto what it returns.
        if (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "key"
        ):
            for element in ast.walk(node.iter):
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    literal_keys.add(element.value)
                elif isinstance(element, ast.Name):
                    referenced_names.add(element.id)
        # Keys stated one at a time, resolved rather than copied. ``structure`` is
        # one of these: it is reported through a default so a document written
        # before the axis existed is told what it resolves to, rather than being
        # passed through with the key simply absent.
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id.startswith("provenance")
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            literal_keys.add(node.slice.value)

    # A key the read path names by importing it rather than by spelling it. Resolved
    # through the module it came from, so the spelling lives in one place: comparing
    # the identifier instead would make this file a second source of the name, which
    # is how one of the two eventually drifts.
    surfaced = literal_keys | {
        value for name in referenced_names if isinstance(value := getattr(models, name, None), str)
    }

    for key in READ_PATH_REQUIRED_KEYS:
        if key not in surfaced:
            errors.append(
                f"mcp_server/server.py: the read path does not surface {key!r}, so a "
                "stored fact is invisible to a reader of search_knowledge"
            )
    if STRUCTURE_INFERRED_BY not in surfaced:
        errors.append(
            f"mcp_server/server.py: the read path does not surface "
            f"{STRUCTURE_INFERRED_BY!r}, so an inferred pairing is served as an "
            "unattributed guess"
        )


def _check_closed_vocabularies_documented(root: Path, errors: list[str]) -> None:
    """Require every member of a closed vocabulary to be described in the seam document.

    Three axes are closed sets in code -- ``CaptureSource``, ``RecordKind`` and
    ``RecordStructure`` -- and ``docs/tanseki-seam.md`` is where a reader deciding what
    a stored record means looks. A member that exists in the enum but not in the
    document is invisible there: the reader does not know the value is possible, so
    they cannot discover that a record they are holding carries a label they have
    never been told about.

    This has already happened twice. ``backfilled`` existed before the seam
    document named it, and ``RecordStructure`` had no section at all -- so the
    newest axis was entirely absent from the document that describes the storage
    contract. Both passed every check, because prose that enumerates a set the code
    also enumerates has no protection of its own. The point of the check is that
    adding a member to an enum is now *sufficient* to make it fail.

    The comparison is against the enums themselves rather than a list written here.
    A list in this script would be the same duplication that let the drift happen,
    and it would drift again the next time someone adds a member.

    A bare mention is enough for the check to pass. What cannot be checked
    mechanically is whether the document says what the value *means*, and this
    deliberately does not pretend to: a token-matching check that also tried to
    judge the prose would report a confident pass on a document that names the
    value and explains it wrongly.
    """
    seam_path = root / "docs" / "tanseki-seam.md"
    if not seam_path.exists():
        errors.append("docs/tanseki-seam.md: missing, so the storage contract cannot be checked")
        return
    seam = seam_path.read_text(encoding="utf-8")

    vocabularies: list[tuple[str, list[str]]] = [
        ("CaptureSource", [source.value for source in models.CaptureSource]),
        ("RecordKind", [kind.value for kind in _record_kinds()]),
        ("RecordStructure", [structure.value for structure in models.RecordStructure]),
    ]
    for name, members in vocabularies:
        undocumented = [value for value in members if value not in seam]
        if undocumented:
            errors.append(
                f"docs/tanseki-seam.md: {name} member(s) {', '.join(sorted(undocumented))} are "
                "not described in the seam document, so a reader of that document cannot "
                "tell that a stored record may carry them. Add the value, its anchor and "
                "the guarantee it does not make."
            )


def _record_kinds() -> tuple[object, ...]:
    """Import ``RecordKind`` lazily, so this script still runs if the module moves."""
    from kojutsu.core.answer_collector import RecordKind

    return tuple(RecordKind)


#: Modules allowed to build a GitHub API path without appearing in ``github-seam.md``.
#: ``github.py`` is the capture client itself, so naming it in a document *about* the
#: seam it implements is circular.
_GITHUB_API_MODULES = ("integrations/github.py",)

#: A GitHub API path as it appears in source: ``repos/{owner}/...``, ``search/issues``.
_GITHUB_PATH_RE = re.compile(r"""repos/\{|"repos/|f"repos/|search/issues""")


def _check_github_seam_callers(root: Path, errors: list[str]) -> None:
    """Require every module that builds a GitHub API path to be named in the seam doc.

    ``docs/github-seam.md`` is the reference consulted when deciding what the token
    may reach, so a module issuing requests the document does not mention is a
    request nobody reviewed. The failure is invisible at runtime -- the scope really
    is sufficient, so the call succeeds -- which is exactly why it needs a check
    rather than a reader.

    Detection is by *path literal*, not by importing a client. A module that imports
    the GitHub models is not issuing requests, and a module that constructs
    ``httpx.Client`` may be talking to Jira, Tanseki or the webhook API; both are false
    positives that would make the check noise nobody keeps. The literal is the thing
    that is only ever true of a GitHub request.

    Compiled bytecode is skipped for the same reason: ``__pycache__`` copies of these
    files satisfy every text search and are not modules anybody reads.
    """
    seam_path = root / "docs" / "github-seam.md"
    if not seam_path.exists():
        errors.append("docs/github-seam.md: missing, so the GitHub seam cannot be checked")
        return
    seam = seam_path.read_text(encoding="utf-8")

    src = root / "src" / "kojutsu"
    for path in sorted(src.rglob("*.py")):
        relative = path.relative_to(src).as_posix()
        if relative in _GITHUB_API_MODULES:
            continue
        if not _GITHUB_PATH_RE.search(path.read_text(encoding="utf-8")):
            continue
        module = path.stem
        if module not in seam:
            errors.append(
                f"src/kojutsu/{relative} builds a GitHub API path but is not named in "
                "docs/github-seam.md, so a reader deciding what the token may reach will "
                "not see the request. Name the module and list the endpoints it issues."
            )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    errors: list[str] = []
    for path in _markdown_files(root):
        text = path.read_text(encoding="utf-8")
        _check_links(root, path, text, errors)
        lowered = text.lower()
        for term in STALE_TERMS:
            if term in lowered:
                errors.append(f"{path.relative_to(root)}: stale term {term}")
    _check_shell_scripts(root, errors)
    _check_docker_guidance(root, errors)
    _check_webhook_allowlist_guidance(root, errors)
    _check_read_path_keys(root, errors)
    _check_closed_vocabularies_documented(root, errors)
    _check_github_seam_callers(root, errors)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
