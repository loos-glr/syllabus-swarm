"""
test_main.py — Tests for the main CLI module and CourseSpecification model
==========================================================================

Validates that ``CourseSpecification`` (the Pydantic structured-output
model used by the Intake Specialist) enforces its schema correctly.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.main import CourseSpecification


class TestCourseSpecification:
    """Tests for the CourseSpecification Pydantic model."""

    def test_valid_model_creation(self) -> None:
        """A CourseSpecification can be created with both fields."""
        spec = CourseSpecification(
            course_context="A rich context string about Python.",
            primary_language="Python",
        )
        assert spec.course_context == "A rich context string about Python."
        assert spec.primary_language == "Python"

    def test_javascript_language(self) -> None:
        """primary_language accepts JavaScript."""
        spec = CourseSpecification(
            course_context="Context for JS course.",
            primary_language="JavaScript",
        )
        assert spec.primary_language == "JavaScript"

    def test_typescript_language(self) -> None:
        """primary_language accepts TypeScript."""
        spec = CourseSpecification(
            course_context="Context for TS course.",
            primary_language="TypeScript",
        )
        assert spec.primary_language == "TypeScript"

    def test_course_context_is_required(self) -> None:
        """course_context is a required field."""
        with pytest.raises(ValidationError):
            CourseSpecification(primary_language="Python")

    def test_primary_language_is_required(self) -> None:
        """primary_language is a required field."""
        with pytest.raises(ValidationError):
            CourseSpecification(course_context="Some context")

    def test_both_fields_are_required(self) -> None:
        """Both fields must be provided."""
        with pytest.raises(ValidationError):
            CourseSpecification()

    def test_model_can_be_serialized(self) -> None:
        """The model can be serialized to a dict."""
        spec = CourseSpecification(
            course_context="Test context",
            primary_language="Go",
        )
        data = spec.model_dump()
        assert data == {
            "course_context": "Test context",
            "primary_language": "Go",
            "grading_scale": None,
            "student_pathway": None,
            "year_level": None,
            "hardware_constraints": None,
        }

    def test_model_can_be_deserialized(self) -> None:
        """The model can be created from a dict."""
        data = {
            "course_context": "Deserialized context",
            "primary_language": "Rust",
        }
        spec = CourseSpecification.model_validate(data)
        assert spec.course_context == "Deserialized context"
        assert spec.primary_language == "Rust"

    def test_empty_strings_are_allowed(self) -> None:
        """Empty strings pass Pydantic validation (no min_length constraint)."""
        spec = CourseSpecification(course_context="", primary_language="")
        assert spec.course_context == ""
        assert spec.primary_language == ""

    def test_multiline_course_context(self) -> None:
        """course_context can contain multi-line text."""
        context = (
            "Course Name: Advanced JavaScript\n"
            "Tech Stack: Node.js, Express, React\n"
            "Student Profile: BOL, Year 2\n"
        )
        spec = CourseSpecification(
            course_context=context,
            primary_language="JavaScript",
        )
        assert "Node.js" in spec.course_context
        assert "BOL" in spec.course_context
# ═══════════════════════════════════════════════════════════════════════
# Issue #7 RED PHASE — Interactive CLI Feedback Tests
# These tests MUST fail until prompt_for_feedback is implemented.
# ═══════════════════════════════════════════════════════════════════════


class TestPromptForFeedback:
    """Verify the interactive feedback prompt function for HITL."""

    def test_prompt_for_feedback_exists(self) -> None:
        """src.main MUST expose a prompt_for_feedback function."""
        from src.main import prompt_for_feedback

        assert callable(prompt_for_feedback)

    def test_prompt_returns_approve_for_a(self) -> None:
        """Typing 'A' returns ('APPROVE', None)."""
        from src.main import prompt_for_feedback

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("builtins.input", lambda _: "A")
            action, feedback = prompt_for_feedback()
            assert action == "APPROVE"
            assert feedback is None

    def test_prompt_returns_quit_for_q(self) -> None:
        """Typing 'Q' returns ('QUIT', None)."""
        from src.main import prompt_for_feedback

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("builtins.input", lambda _: "Q")
            action, feedback = prompt_for_feedback()
            assert action == "QUIT"

    def test_prompt_returns_feedback_for_other_input(self) -> None:
        """Any non-A/Q input is treated as feedback text."""
        from src.main import prompt_for_feedback

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("builtins.input", lambda _: "Modules are too long")
            action, feedback = prompt_for_feedback()
            assert action == "FEEDBACK"
            assert feedback == "Modules are too long"

    def test_whitespace_only_input_is_treated_as_feedback(self) -> None:
        """Whitespace-only input is treated as FEEDBACK (stripped to empty)."""
        from src.main import prompt_for_feedback

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("builtins.input", lambda _: "   ")
            action, feedback = prompt_for_feedback()
            assert action == "FEEDBACK"
            # strip() removes whitespace, resulting in empty string
            assert feedback == ""
