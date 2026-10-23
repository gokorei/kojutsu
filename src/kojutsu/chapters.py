"""Decompose a pull request diff into chapters, with a prologue.

The unit this module produces is an *orientation*, not a summary. A unified diff
is ordered by filename, which is the order git stores trees in and not the order
a reviewer needs: a lockfile bump sits between two changes to the same subsystem,
and on a large pull request the first fifteen thousand characters -- the only
part a bounded prompt can carry -- are usually the least interesting files. A
chapter is a group of changed files that share a place in the codebase, so a
question budget can be spent once per area instead of clustering wherever the
diff happens to start.

Three decisions are deliberate.

**Deterministic, not model-authored.** Chapter names are derived from paths.
Asking a model to name the chapters would produce a more pleasant narrative and a
less auditable one: the grouping would differ between runs, and a chapter that
silently swallowed half the diff would be indistinguishable from one that did
not. A path group can be recomputed and checked.

**Noise is excluded, and named.** Lockfiles, generated output, vendored trees,
binaries and pure renames carry no decision to capture. Dropping them silently
would make the prompt look complete when it is not, so the prompt names them
separately instead.

**Absence is visible.** A chapter that receives no question is reported by the
caller rather than treated as covered. That is the same rule as the read path: an
answer shortened by a bound must be visible as shortened, because a caller's
conclusion from a truncated result is much stronger than the evidence supports.

This is a leaf module: it reads stdlib only, so ``integrations`` may import it
without creating the back-edge the layering test exists to prevent.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

#: Most chapters offered to one prompt. Beyond this the per-chapter diff budget
#: is too small to show a reviewer anything, so the smallest groups are merged
#: into an ``other areas`` chapter rather than each getting a sliver.
MAX_CHAPTERS = 8

#: Lockfiles by every name the common ecosystems use. Compared case-folded: the
#: file name is not a place a decision is recorded, whichever spelling it uses.
_LOCKFILE_NAMES = frozenset(
    {
        "package-lock.json",
        "npm-shrinkwrap.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "bun.lockb",
        "bun.lock",
        "pipfile.lock",
        "poetry.lock",
        "uv.lock",
        "pdm.lock",
        "cargo.lock",
        "go.sum",
        "composer.lock",
        "gemfile.lock",
        "mix.lock",
        "pubspec.lock",
        "packages.lock.json",
        "gradle.lockfile",
    }
)

#: Directories whose contents are produced rather than written. A file under one
#: of these is not a design decision at the source level, even when the diff
#: shows a large textual change.
_SUPPORT_DIRS = frozenset(
    {
        "node_modules",
        "vendor",
        "third_party",
        "dist",
        "build",
        "coverage",
        "__snapshots__",
        ".venv",
        "venv",
        "target",
        ".next",
        ".nuxt",
        ".pytest_cache",
    }
)

#: Generated-file suffixes, matched case-folded. ``.map`` is included because a
#: source map is emitted, never hand-written.
_GENERATED_SUFFIXES = (
    ".min.js",
    ".min.mjs",
    ".min.css",
    ".map",
    ".pb.go",
    ".pb.gw.go",
    "pb2.py",
    "pb2_grpc.py",
    ".generated.ts",
    ".generated.js",
    ".generated.go",
    ".snap",
)

#: Binary media and artifacts. A binary diff states that it is binary, but a file
#: listed without a diff section (a page-capped file list) has no text to check,
#: so the suffix is the only signal left.
_BINARY_SUFFIXES = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".ico",
        ".pdf",
        ".zip",
        ".gz",
        ".tgz",
        ".bz2",
        ".xz",
        ".7z",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
        ".mp3",
        ".mp4",
        ".mov",
        ".avi",
        ".webm",
        ".so",
        ".dylib",
        ".dll",
        ".exe",
        ".bin",
        ".class",
        ".jar",
        ".wasm",
        ".docx",
        ".xlsx",
        ".pptx",
        ".sqlite",
        ".db",
    }
)

#: One git path piece: either a quoted path (which may contain spaces) or a run
#: of non-space characters. Git quotes a path rather than changing its identity.
_GIT_PATH_PART = r'(?:"(?:[^"\\]|\\.)*"|\S+)'
_DIFF_HEADER = re.compile(rf"^diff --git (?P<a>{_GIT_PATH_PART}) (?P<b>{_GIT_PATH_PART})\s*$")
_RENAME_TO = re.compile(r"^rename to (?P<path>.+?)\s*$")
_PLUS_PATH = re.compile(r"^\+\+\+ (?P<path>.+?)\s*$")
_MINUS_PATH = re.compile(r"^--- (?P<path>.+?)\s*$")


@dataclass(frozen=True)
class DiffFile:
    """One changed file, with the diff section that describes it.

    ``diff`` may be empty: a file can be listed by ``get_pr_files`` while its
    section is absent from the fetched diff (a truncated or binary response).
    Empty is kept rather than dropped because the chapter should still say the
    file changed; a prompt that omits it would understate the change.
    """

    path: str
    diff: str
    #: Why this file is not worth a question, or ``None`` when it is. Naming the
    #: reason rather than only a boolean lets the prompt explain the exclusion.
    support_reason: str | None = None

    @property
    def is_support(self) -> bool:
        return self.support_reason is not None


@dataclass(frozen=True)
class Chapter:
    """A group of changed files that share a place in the codebase.

    ``id`` is ``None`` for a chapter made entirely of support files. Those are
    still chapters -- the prompt names them -- but they are not offered as a
    place to ask a question, so they need no label the response can refer to.
    """

    id: str | None
    title: str
    files: tuple[DiffFile, ...]

    @property
    def is_support(self) -> bool:
        """Whether every file here is support. A mixed chapter is reviewable."""
        return all(file.is_support for file in self.files)

    @property
    def reviewable_files(self) -> tuple[DiffFile, ...]:
        return tuple(file for file in self.files if not file.is_support)

    @property
    def support_files(self) -> tuple[DiffFile, ...]:
        return tuple(file for file in self.files if file.is_support)


@dataclass(frozen=True)
class Prologue:
    """What the author says the change is for, before any question is asked.

    Kept as source fields rather than rendered text because bounding and
    redaction belong at the prompt boundary, where every other piece of
    untrusted source data is bounded. ``description`` is the pull request body;
    ``jira`` is the optional ticket context the generator already fetched.
    """

    title: str = ""
    description: str = ""
    jira: Mapping[str, object] | None = None


def build_prologue(
    title: str | None,
    description: str | None,
    jira_context: Mapping[str, object] | None,
) -> Prologue:
    """Assemble the prologue, treating absent text as absent rather than empty.

    A pull request with no body is not a pull request whose body said nothing
    useful; it is one where the field was not written. The prompt makes that
    distinction, so ``""`` is preserved instead of being filled with a summary
    this module invented.
    """
    return Prologue(
        title=(title or "").strip(),
        description=(description or "").strip(),
        jira=dict(jira_context) if jira_context else None,
    )


def _clean_git_path(token: str) -> str:
    """Normalise the path forms git writes in a diff header.

    Git quotes a path with special characters and prefixes it with ``a/`` or
    ``b/``. Both are transport details, not the file's name, so both are removed
    before the path is used as a grouping key.
    """
    token = token.strip()
    if len(token) >= 2 and token[0] == '"' and token[-1] == '"':
        token = token[1:-1]
        token = token.replace('\\"', '"').replace("\\\\", "\\")
        token = token.replace("\\t", "\t").replace("\\n", "\n")
    if token.startswith(("a/", "b/")):
        token = token[2:]
    return token


def _section_path(header: str, body: str) -> str:
    """The file a diff section is about, from the most specific evidence available.

    ``rename to`` wins because the new path is the one the chapter should name:
    a rename's old path no longer exists. The ``+++`` line follows, then ``---``
    for a deletion, and the ``diff --git`` header is the last resort.
    """
    for line in body.splitlines():
        match = _RENAME_TO.match(line)
        if match:
            return _clean_git_path(match.group("path"))
    for line in body.splitlines():
        match = _PLUS_PATH.match(line)
        if match:
            path = _clean_git_path(match.group("path"))
            if path and path != "/dev/null":
                return path
    for line in body.splitlines():
        match = _MINUS_PATH.match(line)
        if match:
            path = _clean_git_path(match.group("path"))
            if path and path != "/dev/null":
                return path
    match = _DIFF_HEADER.match(header)
    if match:
        return _clean_git_path(match.group("b"))
    return "(unknown)"


def split_diff_sections(diff: str) -> list[tuple[str, str]]:
    """Split a unified diff into ``(path, section)`` pairs, in diff order.

    The section text includes the ``diff --git`` header so a renamed file can
    still be recognised from its body. Text before the first header is dropped:
    it is not a file's change, and attributing it to the first file would make
    that file's diff claim more than it shows.
    """
    sections: list[tuple[str, str]] = []
    header: str | None = None
    lines: list[str] = []
    for line in diff.splitlines(keepends=True):
        if line.startswith("diff --git "):
            if header is not None:
                sections.append((_section_path(header, "".join(lines)), "".join(lines)))
            header = line.rstrip("\r\n")
            lines = [line]
        elif header is not None:
            lines.append(line)
    if header is not None:
        sections.append((_section_path(header, "".join(lines)), "".join(lines)))
    return sections


def _support_reason(path: PurePosixPath, text: str) -> str | None:
    """Why a file carries no decision, or ``None`` when it might.

    Ordered from the most certain signal to the least. A file name is stronger
    evidence than a directory (``build/`` can be a real package), and a diff that
    says it is binary is stronger than any suffix.
    """
    name = path.name.casefold()
    if name in _LOCKFILE_NAMES:
        return "lockfile"
    if any(part.casefold() in _SUPPORT_DIRS for part in path.parts[:-1]):
        return "vendored or generated"
    if name.endswith(_GENERATED_SUFFIXES):
        return "generated"
    if path.suffix.casefold() in _BINARY_SUFFIXES:
        return "binary"
    if "Binary files " in text or "GIT binary patch" in text:
        return "binary"
    # A pure rename (``similarity index 100%`` with no hunk) moves a file without
    # changing it. There is nothing in the change to ask about, and treating the
    # move as reviewable spends a chapter on a path edit.
    if "similarity index 100%" in text and "@@" not in text:
        return "rename only"
    return None


def chapter_title(path: str) -> str:
    """The directory that names a file's chapter.

    Two components are enough to separate the top-level areas this project
    keeps, and three for ``src/`` so that ``src/kojutsu/core`` and
    ``src/kojutsu/integrations`` are different chapters rather than one
    ``src/kojutsu`` bucket. Deeper paths stay in their two- or three-component
    group so a chapter remains a place a reviewer can name, not a directory
    listing.
    """
    parts = PurePosixPath(path).parts
    if not parts:
        return "(unknown)"
    directories = parts[:-1]
    if not directories:
        return "(root)"
    if directories[0] == "src" and len(directories) > 2:
        return "/".join(directories[:3])
    if len(directories) > 2:
        return "/".join(directories[:2])
    return "/".join(directories)


def _merge_smallest(
    groups: list[tuple[str, list[DiffFile]]], max_chapters: int
) -> list[tuple[str, list[DiffFile]]]:
    """Keep the largest groups and merge the rest into one ``other areas`` chapter.

    The merged chapter stays last rather than being interleaved: it is a bucket,
    not a place, and giving it a position in the reading order would imply a
    relationship to the chapter before it.
    """
    if max_chapters <= 1:
        return [("other areas", [file for _, files in groups for file in files])]
    keep_count = max(1, max_chapters - 1)
    ranked = sorted(range(len(groups)), key=lambda index: (-len(groups[index][1]), index))
    keep = set(ranked[:keep_count])
    merged: list[DiffFile] = []
    for index, (_, files) in enumerate(groups):
        if index not in keep:
            merged.extend(files)
    kept = [group for index, group in enumerate(groups) if index in keep]
    if merged:
        kept.append(("other areas", merged))
    return kept


def build_chapters(
    diff: str,
    files_changed: Sequence[str] = (),
    *,
    max_chapters: int = MAX_CHAPTERS,
) -> tuple[Chapter, ...]:
    """Group a diff into reviewable chapters, preserving diff order.

    ``files_changed`` supplies paths that the diff itself does not describe (a
    page-capped file list or a binary response). They are kept as empty-diff
    files so a chapter still reports that they changed; dropping them would make
    the chapter map disagree with the change it claims to describe.
    """
    limit = max(1, max_chapters)
    sections = split_diff_sections(diff)
    order: list[str] = []
    section_text: dict[str, list[str]] = {}
    for path, text in sections:
        if path not in section_text:
            order.append(path)
            section_text[path] = []
        section_text[path].append(text)
    for path in files_changed:
        if path not in section_text:
            order.append(path)
            section_text[path] = []

    files = [
        DiffFile(
            path=path,
            diff="\n".join(section_text[path]),
            support_reason=_support_reason(PurePosixPath(path), "\n".join(section_text[path])),
        )
        for path in order
    ]

    grouped: dict[str, list[DiffFile]] = {}
    for file in files:
        grouped.setdefault(chapter_title(file.path), []).append(file)
    groups = list(grouped.items())
    if len(groups) > limit:
        groups = _merge_smallest(groups, limit)

    chapters: list[Chapter] = []
    reviewable_index = 0
    for title, chapter_files in groups:
        if all(file.is_support for file in chapter_files):
            chapters.append(Chapter(id=None, title=title, files=tuple(chapter_files)))
            continue
        reviewable_index += 1
        chapters.append(Chapter(id=f"c{reviewable_index}", title=title, files=tuple(chapter_files)))
    return tuple(chapters)


__all__ = [
    "MAX_CHAPTERS",
    "Chapter",
    "DiffFile",
    "Prologue",
    "build_chapters",
    "build_prologue",
    "chapter_title",
    "split_diff_sections",
]
