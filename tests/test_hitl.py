"""
test_hitl.py — Tests for Human-in-the-Loop Cyclic State Machine
===============================================================

Issue #6: Implement Cyclic State Machine for HITL Feedback

Validates that the syllabus swarm supports cyclic execution with
human feedback injection, as required by Constitution Flow 3:

  *After QA Review, the swarm pauses. The CLI prompts for human
   feedback. If approved, proceed to Export. If rejected/critiqued,
   route feedback back to specific generating agents for a new
   iteration.*
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ═══════════════════════════════════════════════════════════════════════
# Issue #6 RED PHASE — HITL State Machine Tests
# These tests MUST fail until cyclic execution is implemented.
# ═══════════════════════════════════════════════════════════════════════


class TestSwarmStateEnum:
    """Verify the SwarmState enum exists with required states."""

    def test_swarm_state_enum_exists(self) -> None:
        """src.crews.syllabus_crew MUST expose a SwarmState enum."""
        from src.crews.syllabus_crew import SwarmState

        assert SwarmState is not None

    def test_swarm_state_has_required_values(self) -> None:
        """SwarmState MUST include GENERATING, AWAITING_FEEDBACK, EXPORTING."""
        from src.crews.syllabus_crew import SwarmState

        assert hasattr(SwarmState, "GENERATING"), "Missing GENERATING state"
        assert hasattr(SwarmState, "AWAITING_FEEDBACK"), "Missing AWAITING_FEEDBACK state"
        assert hasattr(SwarmState, "EXPORTING"), "Missing EXPORTING state"


class TestHumanFeedbackParameter:
    """Verify run_syllabus_crew accepts human_feedback parameter."""

    def test_run_syllabus_crew_accepts_human_feedback(self) -> None:
        """run_syllabus_crew MUST accept optional human_feedback parameter."""
        import inspect
        from src.crews.syllabus_crew import run_syllabus_crew

        sig = inspect.signature(run_syllabus_crew)
        params = list(sig.parameters.keys())
        assert "human_feedback" in params, (
            f"run_syllabus_crew must accept 'human_feedback' parameter. "
            f"Found: {params}"
        )

    def test_human_feedback_defaults_to_none(self) -> None:
        """human_feedback parameter defaults to None (no feedback)."""
        import inspect
        from src.crews.syllabus_crew import run_syllabus_crew

        sig = inspect.signature(run_syllabus_crew)
        param = sig.parameters["human_feedback"]
        assert param.default is None, (
            f"human_feedback must default to None, got {param.default!r}"
        )


class TestFeedbackInjectionIntoTasks:
    """Verify that human_feedback reaches task descriptions."""

    def test_syllabus_task_includes_feedback_when_provided(self) -> None:
        """Syllabus generation task includes human_feedback in description."""
        from unittest.mock import patch

        mock_agent = MagicMock()
        mock_agent.role = "Test Agent"

        with patch("src.tasks.syllabus_generation.Task") as mock_task_cls:
            from src.tasks.syllabus_generation import create_syllabus_generation_task

            create_syllabus_generation_task(
                agent=mock_agent,
                course_name="Test",
                course_context="Some context",
                human_feedback="The modules are too long.",
            )
            call_kwargs = mock_task_cls.call_args.kwargs
            description = call_kwargs.get("description", "")
            assert "The modules are too long" in description or (
                "human_feedback" in description
            ), f"Task description should include feedback, got: {repr(description[:300])}"

    def test_lab_task_includes_feedback_when_provided(self) -> None:
        """Lab generation task includes human_feedback in description."""
        from unittest.mock import patch

        mock_agent = MagicMock()
        mock_agent.role = "Test Agent"

        with patch("src.tasks.lab_generation.Task") as mock_task_cls:
            from src.tasks.lab_generation import create_lab_generation_task

            create_lab_generation_task(
                agent=mock_agent,
                course_name="Test",
                syllabus_context="# Syllabus",
                language="Python",
                run_id="test_run",
                tier="tier1_foundations",
                human_feedback="Fix the exercise difficulty.",
            )
            call_kwargs = mock_task_cls.call_args.kwargs
            description = call_kwargs.get("description", "")
            assert "Fix the exercise difficulty" in description or (
                "human_feedback" in description
            ), f"Task description should include feedback, got: {repr(description[:300])}"


class TestCrewResultFeedbackField:
    """Verify CrewResult tracks feedback state."""

    def test_crew_result_has_feedback_fields(self) -> None:
        """CrewResult MUST expose feedback-related attributes."""
        from src.crews.syllabus_crew import CrewResult

        # Try to construct with feedback fields
        result = CrewResult(
            syllabus_path=Path("/tmp/test.md"),
            labs_base_path=Path("/tmp/test_labs"),
            human_feedback_requested=True,
            human_feedback_summary="QA suggests: modules need rework",
        )
        assert result.human_feedback_requested is True
        assert result.human_feedback_summary == "QA suggests: modules need rework"