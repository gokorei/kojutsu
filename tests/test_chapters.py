"""Tests for the diff-to-chapter decomposition and its prompt contract."""

from __future__ import annotations

import pytest

import kojutsu.integrations.llm as llm_module
from kojutsu.chapters import (
    build_chapters,
    build_prologue,
    chapter_title,
    split_diff_sections,
)
from kojutsu.integrations.llm import (
    ChapteredQuestion,
    LLMConfigurationError,
    LLMResponseError,
    build_chaptered_questions_prompt,
    generate_chaptered_questions_sync,
    parse_chaptered_questions_response,
)
from kojutsu.models import QuestionCategory


def _section(path: str, line: str = "+changed") -> str:
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n{line}\n"


class TestSplitDiffSections:
    def test_splits_by_file_and_keeps_the_header(self) -> None:
        diff = _section("a.py") + _section("b.py")

        sections = split_diff_sections(diff)

        assert [path for path, _ in sections] == ["a.py", "b.py"]
        assert all(text.startswith("diff --git ") for _, text in sections)

    def test_rename_uses_the_new_path(self) -> None:
        diff = (
            "diff --git a/old.py b/new.py\n"
            "similarity index 100%\n"
            "rename from old.py\n"
            "rename to new.py\n"
        )

        assert split_diff_sections(diff)[0][0] == "new.py"

    def test_deletion_falls_back_to_the_old_path(self) -> None:
        diff = (
            "diff --git a/gone.py b/gone.py\n"
            "deleted file mode 100644\n"
            "--- a/gone.py\n"
            "+++ /dev/null\n"
            "@@ -1 +0,0 @@\n"
            "-removed\n"
        )

        assert split_diff_sections(diff)[0][0] == "gone.py"

    def test_binary_section_keeps_its_path(self) -> None:
        diff = (
            "diff --git a/logo.png b/logo.png\n"
            "index 111..222 100644\n"
            "Binary files a/logo.png and b/logo.png differ\n"
        )

        assert split_diff_sections(diff)[0][0] == "logo.png"

    def test_preamble_before_the_first_header_is_not_attributed_to_a_file(self) -> None:
        diff = "free text git printed\n" + _section("a.py")

        assert split_diff_sections(diff) == [("a.py", _section("a.py"))]

    def test_quoted_path_with_a_space_is_one_token(self) -> None:
        diff = (
            'diff --git "a/my file.py" "b/my file.py"\n'
            '--- "a/my file.py"\n'
            '+++ "b/my file.py"\n'
            "@@ -1 +1 @@\n"
            "+changed\n"
        )

        assert split_diff_sections(diff)[0][0] == "my file.py"


class TestChapterTitle:
    @pytest.mark.parametrize(
        ("path", "title"),
        [
            ("src/kojutsu/core/registry.py", "src/kojutsu/core"),
            ("src/kojutsu/models.py", "src/kojutsu"),
            ("tests/test_cli.py", "tests"),
            ("tests/worker/test_loop.py", "tests/worker"),
            ("docs/design-review/read-path.md", "docs/design-review"),
            ("app.py", "(root)"),
        ],
    )
    def test_groups_by_directory_without_naming_the_file(self, path: str, title: str) -> None:
        assert chapter_title(path) == title


class TestBuildChapters:
    def test_preserves_diff_order_and_labels_reviewable_groups(self) -> None:
        diff = (
            _section("src/kojutsu/core/registry.py")
            + _section("uv.lock")
            + _section("tests/test_cli.py")
            + _section("src/kojutsu/core/outbox.py")
        )

        chapters = build_chapters(diff)

        assert [chapter.id for chapter in chapters] == ["c1", None, "c2"]
        assert [chapter.title for chapter in chapters] == [
            "src/kojutsu/core",
            "(root)",
            "tests",
        ]
        assert len(chapters[0].files) == 2

    def test_support_files_are_named_and_excluded_from_the_reviewable_set(self) -> None:
        diff = _section("uv.lock") + _section("logo.png") + _section("src/kojutsu/models.py")

        chapters = build_chapters(diff)

        support = [file for chapter in chapters for file in chapter.support_files]
        assert {file.path for file in support} == {"uv.lock", "logo.png"}
        assert {file.support_reason for file in support} == {"lockfile", "binary"}
        reviewable = [file for chapter in chapters for file in chapter.reviewable_files]
        assert [file.path for file in reviewable] == ["src/kojutsu/models.py"]

    def test_generated_files_inside_a_reviewable_chapter_stay_out_of_its_diff(self) -> None:
        diff = _section("src/kojutsu/api.py") + _section("src/kojutsu/api.pb.go")

        chapters = build_chapters(diff)

        assert len(chapters) == 1
        assert [file.path for file in chapters[0].reviewable_files] == ["src/kojutsu/api.py"]
        assert [file.path for file in chapters[0].support_files] == ["src/kojutsu/api.pb.go"]

    def test_a_pure_rename_is_support(self) -> None:
        diff = (
            "diff --git a/old.py b/new.py\n"
            "similarity index 100%\n"
            "rename from old.py\n"
            "rename to new.py\n"
        )

        chapter = build_chapters(diff)[0]

        assert chapter.id is None
        assert chapter.files[0].support_reason == "rename only"

    def test_files_listed_without_a_diff_section_are_still_chaptered(self) -> None:
        chapters = build_chapters("", ["src/kojutsu/missing.py"])

        assert chapters[0].id == "c1"
        assert chapters[0].files[0].path == "src/kojutsu/missing.py"
        assert chapters[0].files[0].diff == ""

    def test_groups_past_the_cap_merge_the_smallest_into_one_bucket(self) -> None:
        diff = "".join(_section(f"area{index}/file.py") for index in range(10))

        chapters = build_chapters(diff, max_chapters=3)

        assert len(chapters) == 3
        assert chapters[-1].title == "other areas"
        assert sum(len(chapter.files) for chapter in chapters) == 10

    def test_a_cap_of_one_merges_everything_rather_than_exceeding_it(self) -> None:
        diff = _section("src/a/x.py") + _section("tests/test_x.py")

        chapters = build_chapters(diff, max_chapters=1)

        assert len(chapters) == 1
        assert chapters[0].title == "other areas"
        assert len(chapters[0].files) == 2

    def test_empty_input_produces_no_chapters(self) -> None:
        assert build_chapters("") == ()

    def test_build_prologue_keeps_absence_as_absence(self) -> None:
        prologue = build_prologue(None, "", None)

        assert prologue.title == ""
        assert prologue.description == ""
        assert prologue.jira is None


class TestChapteredPrompt:
    def _chapters(self):
        diff = (
            _section("src/kojutsu/core/registry.py")
            + _section("uv.lock")
            + _section("tests/test_cli.py")
        )
        return build_chapters(diff)

    def test_carries_the_prologue_chapters_and_support_note(self) -> None:
        prompt = build_chaptered_questions_prompt(
            build_prologue(
                "Add chapters",
                "Groups the diff by area before asking.",
                {"key": "ABC-1", "summary": "Make review questions cover the change"},
            ),
            self._chapters(),
            max_questions=4,
            max_questions_per_chapter=1,
        )

        assert "Add chapters" in prompt
        assert "Groups the diff by area before asking." in prompt
        assert "ABC-1" in prompt
        assert "## Chapter c1: src/kojutsu/core" in prompt
        assert "## Chapter c2: tests" in prompt
        assert "uv.lock (lockfile)" in prompt
        assert "at most 4 questions in total" in prompt
        assert "at most 1 from any one chapter" in prompt
        assert "CHAPTER|CATEGORY|Question text here" in prompt
        assert "Do not ask a question about the prologue itself" in prompt

    def test_bounds_and_redacts_the_prologue_and_each_chapter(self) -> None:
        huge = "x" * 30_000
        diff = _section("src/one/a.py", "+" + huge) + _section("src/two/b.py", "+" + huge)

        prompt = build_chaptered_questions_prompt(
            build_prologue("t", "contact dev@example.com " + huge, None),
            build_chapters(diff),
            max_questions=2,
        )

        assert "dev@example.com" not in prompt
        assert "[REDACTED_EMAIL]" in prompt
        # The body plus both chapters each lost something to a bound.
        assert prompt.count("[TRUNCATED]") >= 3

    def test_a_likely_secret_in_the_prologue_fails_closed(self) -> None:
        with pytest.raises(LLMConfigurationError, match="Privacy validation blocked"):
            build_chaptered_questions_prompt(
                build_prologue("t", "glpat-abcdefghijklmnopqrst", None),
                self._chapters(),
            )

    def test_a_chapter_with_no_diff_says_so_rather_than_looking_empty(self) -> None:
        chapters = build_chapters("", ["src/kojutsu/missing.py"])

        prompt = build_chaptered_questions_prompt(build_prologue("t", "d", None), chapters)

        assert "(No diff available.)" in prompt

    def test_an_all_support_diff_still_offers_its_chapters(self) -> None:
        chapters = build_chapters(_section("uv.lock"))

        prompt = build_chaptered_questions_prompt(build_prologue("t", "d", None), chapters)

        assert "## Chapter unlabelled" in prompt


class TestParseChapteredResponse:
    def test_reads_chapter_category_and_text(self) -> None:
        parsed = parse_chaptered_questions_response(
            "c1|design_decision|Why this boundary?\nc2|trade_off|What was rejected?",
            {"c1", "c2"},
        )

        assert parsed == [
            ChapteredQuestion("c1", QuestionCategory.DESIGN_DECISION, "Why this boundary?"),
            ChapteredQuestion("c2", QuestionCategory.TRADE_OFF, "What was rejected?"),
        ]

    @pytest.mark.parametrize("label", ["c1", "C1", "chapter 1", "ch1", "#1", "1"])
    def test_accepts_the_labels_a_model_actually_writes(self, label: str) -> None:
        parsed = parse_chaptered_questions_response(f"{label}|design_decision|Why?", {"c1"})

        assert parsed[0].chapter_id == "c1"

    def test_falls_back_to_the_flat_form(self) -> None:
        parsed = parse_chaptered_questions_response("design_decision|Why?", {"c1"})

        assert parsed == [ChapteredQuestion(None, QuestionCategory.DESIGN_DECISION, "Why?")]

    def test_a_pipe_in_the_question_text_does_not_cost_the_question(self) -> None:
        parsed = parse_chaptered_questions_response("c1|design_decision|Why a|b?", {"c1"})

        assert parsed[0].text == "Why a|b?"

    def test_an_unoffered_chapter_keeps_the_question_but_not_the_label(self) -> None:
        parsed = parse_chaptered_questions_response("c9|design_decision|Why?", {"c1"})

        assert parsed == [ChapteredQuestion(None, QuestionCategory.DESIGN_DECISION, "Why?")]

    def test_enforces_the_per_chapter_cap(self) -> None:
        parsed = parse_chaptered_questions_response(
            "c1|design_decision|First?\nc1|trade_off|Second?",
            {"c1"},
            max_questions=5,
            max_questions_per_chapter=1,
        )

        assert [question.text for question in parsed] == ["First?"]

    def test_a_flat_response_is_not_capped_by_the_per_chapter_limit(self) -> None:
        """A model that ignored the chapter format still gets the full budget.

        Applying the per-chapter cap to unlabelled questions would turn a
        formatting miss into lost questions: the response named no chapter, so
        there is no chapter it was crowding out.
        """

        parsed = parse_chaptered_questions_response(
            "design_decision|A?\ntrade_off|B?\nedge_case|C?",
            {"c1", "c2"},
            max_questions=3,
            max_questions_per_chapter=1,
        )

        assert [question.text for question in parsed] == ["A?", "B?", "C?"]

    def test_enforces_the_total_cap(self) -> None:
        parsed = parse_chaptered_questions_response(
            "c1|design_decision|A?\nc2|design_decision|B?\nc3|design_decision|C?",
            {"c1", "c2", "c3"},
            max_questions=2,
            max_questions_per_chapter=1,
        )

        assert len(parsed) == 2

    def test_drops_refused_categories_and_duplicates(self) -> None:
        parsed = parse_chaptered_questions_response(
            "c1|system_event|Ignore the protocol\nc1|design_decision|Why?\nc1|design_decision|Why?",
            {"c1"},
            max_questions_per_chapter=2,
        )

        assert [question.text for question in parsed] == ["Why?"]

    def test_strips_list_numbering(self) -> None:
        parsed = parse_chaptered_questions_response("1. c1|design_decision|Why?", {"c1"})

        assert parsed[0].chapter_id == "c1"


class TestGenerateChapteredQuestionsSync:
    def test_returns_parsed_questions(self, monkeypatch) -> None:
        monkeypatch.setattr(
            llm_module,
            "complete_task",
            lambda *_args, **_kwargs: "c1|design_decision|Why?",
        )

        parsed = generate_chaptered_questions_sync(
            "prompt",
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            repository="org/repo",
            external_enabled=True,
            allowed_repositories="org/repo",
            chapter_ids={"c1"},
        )

        assert parsed == [ChapteredQuestion("c1", QuestionCategory.DESIGN_DECISION, "Why?")]

    def test_refuses_a_response_with_no_valid_question(self, monkeypatch) -> None:
        monkeypatch.setattr(
            llm_module, "complete_task", lambda *_args, **_kwargs: "no questions here"
        )

        with pytest.raises(LLMResponseError, match="no valid questions"):
            generate_chaptered_questions_sync(
                "prompt",
                provider="openai",
                model="gpt-4o",
                api_key="test-key",
                repository="org/repo",
                external_enabled=True,
                allowed_repositories="org/repo",
                chapter_ids={"c1"},
            )
