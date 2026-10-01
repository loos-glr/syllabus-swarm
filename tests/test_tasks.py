"""
test_tasks.py — Tests for syllabus and lab generation task factories
====================================================================

Validates that ``create_syllabus_generation_task`` and
``create_lab_generation_task`` pass the correct arguments to the
``crewai.Task`` constructor.

Because ``crewai.Task`` uses Pydantic validation and requires a real
``BaseAgent`` instance (not a MagicMock), we patch the ``Task``
reference in each task module and inspect the keyword arguments.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ===================================================================
# create_syllabus_generation_task
# ===================================================================


class TestCreateSyllabusGenerationTask:
    """Tests for the syllabus generation task factory."""

    @pytest.fixture(autouse=True)
    def _patch_task(self) -> None:
        """Patch Task in the syllabus_generation module."""
        with patch("src.tasks.syllabus_generation.Task") as self.mock_task:
            yield

    def _call_factory(self, **overrides) -> dict:
        """Call create_syllabus_generation_task and return Task kwargs."""
        from src.tasks.syllabus_generation import create_syllabus_generation_task

        kwargs: dict = {
            "course_name": "Python for Data Engineering",
            "agent": MagicMock(),
        }
        kwargs.update(overrides)
        create_syllabus_generation_task(**kwargs)
        return self.mock_task.call_args.kwargs

    def test_course_name_in_description(self) -> None:
        """The course name appears in the task description."""
        kwargs = self._call_factory()
        assert "Python for Data Engineering" in kwargs["description"]

    def test_course_name_in_expected_output(self) -> None:
        """The expected_output mandates an H1 with the course name."""
        kwargs = self._call_factory()
        assert "# Python for Data Engineering — Course Syllabus" in kwargs["expected_output"]

    def test_output_file_not_set(self) -> None:
        """The orchestrator writes the syllabus; output_file is not on the Task."""
        kwargs = self._call_factory()
        assert "output_file" not in kwargs

    def test_async_execution_is_false(self) -> None:
        """Tasks are synchronous by default."""
        kwargs = self._call_factory(course_name="Test Course")
        assert kwargs["async_execution"] is False

    def test_agent_is_passed_through(self) -> None:
        """The agent parameter is passed through to the Task."""
        agent = MagicMock()
        kwargs = self._call_factory(agent=agent)
        assert kwargs["agent"] is agent

    # -- Optional fields -------------------------------------------------

    def test_course_context_injected_when_provided(self) -> None:
        """course_context appears in the description when supplied."""
        kwargs = self._call_factory(
            course_name="Test Course",
            course_context="This is a test context about Python and data.",
        )
        assert "This is a test context about Python and data." in kwargs["description"]
        assert "Course Context (from Intake Specialist)" in kwargs["description"]

    def test_course_description_injected_when_provided(self) -> None:
        """course_description appears in the description when supplied."""
        kwargs = self._call_factory(
            course_name="Test Course",
            course_description="A comprehensive course on data engineering.",
        )
        assert "A comprehensive course on data engineering." in kwargs["description"]
        assert "Course Description:" in kwargs["description"]

    def test_course_duration_injected_when_provided(self) -> None:
        """course_duration appears in the description when supplied."""
        kwargs = self._call_factory(
            course_name="Test Course",
            course_duration="12 weeks",
        )
        assert "12 weeks" in kwargs["description"]
        assert "Course Duration:" in kwargs["description"]

    def test_target_audience_injected_when_provided(self) -> None:
        """target_audience appears in the description when supplied."""
        kwargs = self._call_factory(
            course_name="Test Course",
            target_audience="Career changers with basic programming experience",
        )
        assert "Career changers with basic programming experience" in kwargs["description"]
        assert "Target Audience:" in kwargs["description"]

    def test_optional_fields_omitted_when_none(self) -> None:
        """When optional fields are None, their labels do not appear."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Course Context (from Intake Specialist)" not in kwargs["description"]
        assert "Course Description:" not in kwargs["description"]
        assert "Course Duration:" not in kwargs["description"]
        assert "Target Audience:" not in kwargs["description"]

    # -- Mandate injection -----------------------------------------------

    def test_humanics_mandate_in_description(self) -> None:
        """The Humanics embedding mandate is present in the description."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Joseph Aoun's Humanics framework" in kwargs["description"]
        assert "Technological Literacy [T]" in kwargs["description"]
        assert "Data Literacy [D]" in kwargs["description"]
        assert "Human Literacy [H]" in kwargs["description"]

    def test_experiential_learning_mandate_in_description(self) -> None:
        """The experiential learning mandate is present in the description."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Experiential Learning & Industry Integration" in kwargs["description"]
        assert "Co-operative Education" in kwargs["description"]
        assert "Capstone Project" in kwargs["description"]
        assert "Industry Partnerships" in kwargs["description"]

    def test_markdown_structure_in_description(self) -> None:
        """The Markdown formatting requirements are present."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Markdown Formatting Requirements" in kwargs["description"]
        assert "ATX-style headings" in kwargs["description"]

    def test_backward_design_mentioned(self) -> None:
        """The description references backward design principles."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "backward design" in kwargs["description"].lower()

    # -- Expected output content -----------------------------------------

    def test_expected_output_requires_nine_sections(self) -> None:
        """The expected_output references all nine required sections."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Course Overview" in kwargs["expected_output"]
        assert "Instructor Notes" in kwargs["expected_output"]

    def test_expected_output_requires_humanics_tags(self) -> None:
        """The expected_output mandates Humanics tags."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "[T]" in kwargs["expected_output"]
        assert "[D]" in kwargs["expected_output"]
        assert "[H]" in kwargs["expected_output"]

    def test_expected_output_requires_week_by_week_table(self) -> None:
        """The expected_output mandates a week-by-week schedule table."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "week-by-week" in kwargs["expected_output"].lower()


# ===================================================================
# create_lab_generation_task
# ===================================================================


class TestCreateLabGenerationTask:
    """Tests for the lab generation task factory."""

    @pytest.fixture(autouse=True)
    def _patch_task(self) -> None:
        """Patch Task in the lab_generation module."""
        with patch("src.tasks.lab_generation.Task") as self.mock_task:
            yield

    def _call_factory(self, **overrides) -> dict:
        """Call create_lab_generation_task and return Task kwargs."""
        from src.tasks.lab_generation import create_lab_generation_task

        kwargs: dict = {
            "course_name": "Python for Data Engineering",
            "agent": MagicMock(),
        }
        kwargs.update(overrides)
        create_lab_generation_task(**kwargs)
        return self.mock_task.call_args.kwargs

    def test_course_name_in_description(self) -> None:
        """The course name appears in the task description."""
        kwargs = self._call_factory()
        assert "Python for Data Engineering" in kwargs["description"]

    def test_language_in_description(self) -> None:
        """The primary language appears in the description."""
        kwargs = self._call_factory(course_name="Test Course", language="Python")
        assert "Python" in kwargs["description"]
        assert "Primary Language" in kwargs["description"]

    def test_default_language_is_python(self) -> None:
        """When no language is specified, Python is the default."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Python" in kwargs["description"]

    def test_output_file_follows_expected_pattern(self) -> None:
        """The output_file is output/labs/README.md."""
        kwargs = self._call_factory()
        assert kwargs["output_file"] == "output/labs/README.md"

    def test_async_execution_is_false(self) -> None:
        """Tasks are synchronous by default."""
        kwargs = self._call_factory(course_name="Test Course")
        assert kwargs["async_execution"] is False

    def test_agent_is_passed_through(self) -> None:
        """The agent parameter is passed through to the Task."""
        agent = MagicMock()
        kwargs = self._call_factory(agent=agent)
        assert kwargs["agent"] is agent

    # -- Optional fields -------------------------------------------------

    def test_syllabus_context_injected_when_provided(self) -> None:
        """syllabus_context appears in the description when supplied."""
        kwargs = self._call_factory(
            course_name="Test Course",
            syllabus_context="# Sample Syllabus\n\n## Module 1: Intro\n...",
        )
        assert "# Sample Syllabus" in kwargs["description"]
        assert "Syllabus Context:" in kwargs["description"]

    def test_topic_focus_injected_when_provided(self) -> None:
        """topic_focus appears in the description when supplied."""
        kwargs = self._call_factory(
            course_name="Test Course",
            topic_focus="ETL pipelines, REST APIs, Docker",
        )
        assert "ETL pipelines, REST APIs, Docker" in kwargs["description"]
        assert "Topic Focus:" in kwargs["description"]

    def test_optional_fields_omitted_when_none(self) -> None:
        """When optional fields are None, their labels do not appear."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Syllabus Context:" not in kwargs["description"]
        assert "Topic Focus:" not in kwargs["description"]

    # -- Mandate injection -----------------------------------------------

    def test_tier_definitions_in_description(self) -> None:
        """The tier definitions are present in the description."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Tier 1 — Foundations" in kwargs["description"]
        assert "Tier 2 — Application" in kwargs["description"]
        assert "Tier 3 — Architecture" in kwargs["description"]

    def test_directory_structure_in_description(self) -> None:
        """The directory structure requirements are present."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Directory Structure Requirements" in kwargs["description"]
        assert "tier1_foundations" in kwargs["description"]

    def test_readme_template_in_description(self) -> None:
        """The README template requirements are present."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "README.md Requirements" in kwargs["description"]
        assert "Learning Objectives" in kwargs["description"]

    def test_humanics_lab_mandate_in_description(self) -> None:
        """The Humanics lab mandate is present."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Humanics Literacies in Labs" in kwargs["description"]
        assert "Technological Literacy [T]" in kwargs["description"]

    def test_code_quality_mandate_in_description(self) -> None:
        """The code quality mandate is present."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Code Quality Standards" in kwargs["description"]
        assert "type hints" in kwargs["description"]

    # -- Expected output content -----------------------------------------

    def test_expected_output_requires_three_tiers(self) -> None:
        """The expected_output mandates all three tiers."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Tier 1, Tier 2, and Tier 3" in kwargs["expected_output"]

    def test_expected_output_requires_top_level_readme(self) -> None:
        """The final response must be a top-level Markdown README index."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "top-level index" in kwargs["expected_output"]
        assert "Markdown README" in kwargs["expected_output"]

    # -- Language-specific tooling ----------------------------------------

    def test_language_config_javascript(self) -> None:
        """JavaScript labs use .js, eslint, and package.json."""
        kwargs = self._call_factory(course_name="Test Course", language="JavaScript")
        desc = kwargs["description"]
        assert ".js" in desc
        assert "eslint" in desc
        assert "package.json" in desc
        assert ".py" not in desc
        assert "ruff check" not in desc

    def test_language_config_typescript(self) -> None:
        """TypeScript labs use .ts, eslint, package.json."""
        kwargs = self._call_factory(course_name="Test Course", language="TypeScript")
        desc = kwargs["description"]
        assert ".ts" in desc
        assert "eslint" in desc
        assert "package.json" in desc

    def test_language_config_go(self) -> None:
        """Go labs use .go, golangci-lint, go.mod."""
        kwargs = self._call_factory(course_name="Test Course", language="Go")
        desc = kwargs["description"]
        assert ".go" in desc
        assert "golangci-lint" in desc
        assert "go.mod" in desc

    def test_language_specific_extension_in_description(self) -> None:
        """description references the correct file extension and linter."""
        kwargs = self._call_factory(course_name="Test Course", language="JavaScript")
        assert ".js" in kwargs["description"]
        assert "eslint" in kwargs["description"]

    # -- Tool usage mandate -----------------------------------------------

    def test_tool_usage_mandate_in_description(self) -> None:
        """The description mandates use of output_export_tool."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "output_export_tool" in kwargs["description"]
        assert "write-labs" in kwargs["description"]

    def test_tool_usage_mandate_in_expected_output(self) -> None:
        """The expected_output mandates use of output_export_tool."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "output_export_tool" in kwargs["expected_output"]
        assert "write-labs" in kwargs["expected_output"]

    def test_expected_output_mentions_final_textual_response(self) -> None:
        """The expected_output instructs the agent to produce a final textual response."""
        kwargs = self._call_factory(course_name="Test Course")
        eo = kwargs["expected_output"]
        assert "FINAL textual response" in eo
        assert "write-labs" in eo


# ===================================================================
# _get_lang_config helper
# ===================================================================


class TestGetLangConfig:
    """Tests for the _get_lang_config helper in lab_generation."""

    @staticmethod
    def _call(language: str) -> dict:
        from src.tasks.lab_generation import _get_lang_config

        return _get_lang_config(language)

    def test_known_language_python(self) -> None:
        cfg = self._call("Python")
        assert cfg["ext"] == ".py"
        assert cfg["linter"] == "ruff check"

    def test_known_language_javascript(self) -> None:
        cfg = self._call("JavaScript")
        assert cfg["ext"] == ".js"
        assert cfg["linter"] == "eslint"

    def test_known_language_typescript(self) -> None:
        cfg = self._call("TypeScript")
        assert cfg["ext"] == ".ts"

    def test_known_language_java(self) -> None:
        cfg = self._call("Java")
        assert cfg["ext"] == ".java"

    def test_known_language_go(self) -> None:
        cfg = self._call("Go")
        assert cfg["ext"] == ".go"

    def test_known_language_rust(self) -> None:
        cfg = self._call("Rust")
        assert cfg["ext"] == ".rs"

    def test_known_language_csharp(self) -> None:
        cfg = self._call("C#")
        assert cfg["ext"] == ".cs"

    def test_known_language_php(self) -> None:
        cfg = self._call("PHP")
        assert cfg["ext"] == ".php"

    def test_case_insensitive_match(self) -> None:
        cfg = self._call("javascript")
        assert cfg["ext"] == ".js"

    def test_unknown_falls_back_to_python(self) -> None:
        cfg = self._call("Brainfuck")
        assert cfg["ext"] == ".py"
        assert cfg["linter"] == "ruff check"

    def test_empty_string_falls_back_to_python(self) -> None:
        cfg = self._call("")
        assert cfg["ext"] == ".py"

    def test_whitespace_string_falls_back_to_python(self) -> None:
        cfg = self._call("   ")
        assert cfg["ext"] == ".py"


# ===================================================================
# create_syllabus_review_task
# ===================================================================


class TestCreateSyllabusReviewTask:
    """Tests for the syllabus review task factory."""

    @pytest.fixture(autouse=True)
    def _patch_task(self) -> None:
        """Patch Task in the syllabus_review module."""
        with patch("src.tasks.syllabus_review.Task") as self.mock_task:
            yield

    def _call_factory(self, **overrides) -> dict:
        """Call create_syllabus_review_task and return Task kwargs."""
        from src.tasks.syllabus_review import create_syllabus_review_task

        kwargs: dict = {
            "course_name": "Python for Data Engineering",
            "agent": MagicMock(),
            "syllabus_context": "# Test Syllabus\n\n## Module 1\n...",
        }
        kwargs.update(overrides)
        create_syllabus_review_task(**kwargs)
        return self.mock_task.call_args.kwargs

    def test_course_name_in_description(self) -> None:
        """The course name appears in the task description."""
        kwargs = self._call_factory()
        assert "Python for Data Engineering" in kwargs["description"]

    def test_syllabus_context_in_description(self) -> None:
        """The syllabus context is injected into the description."""
        kwargs = self._call_factory(
            course_name="Test Course",
            syllabus_context="# Custom Syllabus\n\nContent here.",
        )
        assert "# Custom Syllabus" in kwargs["description"]
        assert "Content here." in kwargs["description"]

    def test_async_execution_is_false(self) -> None:
        """Tasks are synchronous by default."""
        kwargs = self._call_factory(course_name="Test Course")
        assert kwargs["async_execution"] is False

    def test_agent_is_passed_through(self) -> None:
        """The agent parameter is passed through to the Task."""
        agent = MagicMock()
        kwargs = self._call_factory(agent=agent)
        assert kwargs["agent"] is agent

    # -- Mandate injection -----------------------------------------------

    def test_time_budget_check_in_description(self) -> None:
        """The time-budget mathematics check is present."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Time-Budget Mathematics" in kwargs["description"]
        assert "TIME-BUDGET VIOLATION" in kwargs["description"]

    def test_workload_realism_check_in_description(self) -> None:
        """The workload realism check is present."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Workload Realism" in kwargs["description"]
        assert "40 total hours per week" in kwargs["description"]

    def test_scheduling_sanity_check_in_description(self) -> None:
        """The scheduling sanity check is present."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Scheduling Sanity" in kwargs["description"]

    def test_mbo4_appropriateness_check_in_description(self) -> None:
        """The MBO4 appropriateness check is present."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "MBO4 Appropriateness" in kwargs["description"]

    def test_delegation_mandate_in_description(self) -> None:
        """The delegation mandate is present in the description."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Delegation Mandate" in kwargs["description"]
        assert "Curriculum Architect" in kwargs["description"]

    def test_feasibility_audit_in_description(self) -> None:
        """The description references a Feasibility Audit."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Feasibility Audit" in kwargs["description"]

    # -- Expected output content -----------------------------------------

    def test_expected_output_requires_time_budget_math(self) -> None:
        """The expected_output references Time-Budget Mathematics."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Time-Budget Mathematics" in kwargs["expected_output"]

    def test_expected_output_requires_sign_off(self) -> None:
        """The expected_output references a Sign-Off section."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Sign-Off" in kwargs["expected_output"]

    def test_expected_output_requires_delegation_summary(self) -> None:
        """The expected_output references a Delegation Summary."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Delegation Summary" in kwargs["expected_output"]


# ===================================================================
# Theory / lesson-plan / presentation tasks — enforced write commands
# ===================================================================


class TestTheoryTaskUsesWriteTheoryCommand:
    """The theory task must mandate the validated 'write-theory' command."""

    @pytest.fixture(autouse=True)
    def _patch_task(self) -> None:
        with patch("src.tasks.theory_generation.Task") as self.mock_task:
            yield

    def _call_factory(self, **overrides) -> dict:
        from src.tasks.theory_generation import create_theory_task

        kwargs: dict = {
            "agent": MagicMock(),
            "course_name": "Javascript OOP",
            "run_id": "2026-08-23_120000_Javascript_OOP",
            "tier": "tier1_foundations",
        }
        kwargs.update(overrides)
        create_theory_task(**kwargs)
        return self.mock_task.call_args.kwargs

    def test_expected_output_uses_write_theory(self) -> None:
        kwargs = self._call_factory()
        assert "write-theory" in kwargs["expected_output"]

    def test_no_directory_tree_reference(self) -> None:
        kwargs = self._call_factory()
        assert "write-directory-tree" not in kwargs["expected_output"]
        assert "write-directory-tree" not in kwargs["description"]


class TestLessonPlanTaskUsesWriteLessonPlanCommand:
    """The lesson-plan task must mandate the validated command + canonical dir."""

    @pytest.fixture(autouse=True)
    def _patch_task(self) -> None:
        with patch("src.tasks.lesson_plan_generation.Task") as self.mock_task:
            yield

    def _call_factory(self, **overrides) -> dict:
        from src.tasks.lesson_plan_generation import create_lesson_plan_task

        kwargs: dict = {
            "agent": MagicMock(),
            "course_name": "WebXR",
            "run_id": "2026-09-29_053658_WebXR",
            "module_name": "Tier 1 — Foundations",
            "syllabus_context": "# Syllabus\n\nSome syllabus content.",
        }
        kwargs.update(overrides)
        create_lesson_plan_task(**kwargs)
        return self.mock_task.call_args.kwargs

    def test_expected_output_uses_write_lesson_plan(self) -> None:
        kwargs = self._call_factory()
        assert "write-lesson-plan" in kwargs["expected_output"]

    def test_human_label_maps_to_canonical_dir(self) -> None:
        kwargs = self._call_factory(module_name="Tier 1 — Foundations")
        assert "tier1_foundations" in kwargs["expected_output"]
        assert "tier1_foundations" in kwargs["output_file"]

    def test_no_directory_tree_reference(self) -> None:
        kwargs = self._call_factory()
        assert "write-directory-tree" not in kwargs["description"]


class TestPresentationTaskUsesWritePresentationCommand:
    """The presentation task must mandate 'write-presentation' + canonical dir."""

    @pytest.fixture(autouse=True)
    def _patch_task(self) -> None:
        with patch("src.tasks.presentation_generation.Task") as self.mock_task:
            yield

    def _call_factory(self, **overrides) -> dict:
        from src.tasks.presentation_generation import create_presentation_task

        kwargs: dict = {
            "agent": MagicMock(),
            "course_name": "WebXR",
            "run_id": "2026-09-29_053658_WebXR",
            "module_name": "Tier 2 — Application",
            "syllabus_context": "# Syllabus\n\nSome syllabus content.",
        }
        kwargs.update(overrides)
        create_presentation_task(**kwargs)
        return self.mock_task.call_args.kwargs

    def test_expected_output_uses_write_presentation(self) -> None:
        kwargs = self._call_factory()
        assert "write-presentation" in kwargs["expected_output"]

    def test_human_label_maps_to_canonical_dir(self) -> None:
        kwargs = self._call_factory(module_name="Tier 2 — Application")
        assert "tier2_application" in kwargs["expected_output"]

    def test_no_directory_tree_reference(self) -> None:
        kwargs = self._call_factory()
        assert "write-directory-tree" not in kwargs["description"]
