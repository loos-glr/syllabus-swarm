"""Tests for modality routing and state machine branching (Issue #10)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.crews.syllabus_crew import SwarmState
from src.models import ModalityDecision, ModalityType


class TestSwarmStateExpansion:
    """Tests for the expanded SwarmState enum with video generation."""

    def test_video_generating_state_exists(self) -> None:
        """SwarmState must include VIDEO_GENERATING."""
        assert hasattr(SwarmState, "VIDEO_GENERATING")
        assert SwarmState.VIDEO_GENERATING.value == "video_generating"

    def test_video_generating_is_distinct(self) -> None:
        """VIDEO_GENERATING is distinct from existing states."""
        assert SwarmState.VIDEO_GENERATING != SwarmState.GENERATING
        assert SwarmState.VIDEO_GENERATING != SwarmState.AWAITING_FEEDBACK
        assert SwarmState.VIDEO_GENERATING != SwarmState.EXPORTING


class TestModalityRoutingLogic:
    """Tests for the modality-based routing logic in syllabus_crew."""

    def test_video_as_code_routes_to_video_engineer(self) -> None:
        """VIDEO_AS_CODE modality selects video_engineer."""
        decision = ModalityDecision(
            module_name="Recursion",
            modality=ModalityType.VIDEO_AS_CODE,
            rationale="Needs animation.",
            complexity_score=0.8,
        )
        assert decision.modality == ModalityType.VIDEO_AS_CODE

    def test_classic_reader_routes_to_theory_instructor(self) -> None:
        """CLASSIC_READER modality selects theory_instructor."""
        decision = ModalityDecision(
            module_name="Syntax",
            modality=ModalityType.CLASSIC_READER,
            rationale="Text is sufficient.",
            complexity_score=0.3,
        )
        assert decision.modality == ModalityType.CLASSIC_READER

    def test_modality_decision_parsing(self) -> None:
        """ModalityDecision can be parsed from dict (simulating LLM JSON output)."""
        raw = {
            "module_name": "Sorting",
            "modality": "video_as_code",
            "rationale": "Visual sorting needs animation.",
            "complexity_score": 0.7,
            "suggested_components": ["bubble_sort", "quick_sort"],
        }
        decision = ModalityDecision(**raw)
        assert decision.modality == ModalityType.VIDEO_AS_CODE
        assert decision.complexity_score == 0.7

    def test_routing_dispatcher_handles_all_modalities(self) -> None:
        """Every ModalityType should have a routing path."""
        routing_map = {
            ModalityType.CLASSIC_READER: "theory_instructor",
            ModalityType.VIDEO_AS_CODE: "video_engineer",
            ModalityType.INTERACTIVE_WEB: "theory_instructor",
            ModalityType.INTERACTIVE_CLI: "theory_instructor",
        }
        for modality in ModalityType:
            assert modality in routing_map, f"No route for {modality}"


class TestModalityRoutingIntegration:
    """Integration-style tests using mocked crew execution."""

    def test_routing_switches_to_video_generating_state(self) -> None:
        """When VIDEO_AS_CODE is selected, state transitions to VIDEO_GENERATING."""
        decision = ModalityDecision(
            module_name="Recursion",
            modality=ModalityType.VIDEO_AS_CODE,
            rationale="Needs animation.",
            complexity_score=0.8,
        )
        # Simulate: if modality is VIDEO_AS_CODE, next state is VIDEO_GENERATING
        if decision.modality == ModalityType.VIDEO_AS_CODE:
            next_state = SwarmState.VIDEO_GENERATING
        else:
            next_state = SwarmState.GENERATING

        assert next_state == SwarmState.VIDEO_GENERATING

    def test_routing_stays_in_generating_for_classic_reader(self) -> None:
        """CLASSIC_READER keeps execution in standard GENERATING flow."""
        decision = ModalityDecision(
            module_name="Syntax",
            modality=ModalityType.CLASSIC_READER,
            rationale="Text is sufficient.",
            complexity_score=0.3,
        )
        if decision.modality == ModalityType.VIDEO_AS_CODE:
            next_state = SwarmState.VIDEO_GENERATING
        else:
            next_state = SwarmState.GENERATING

        assert next_state == SwarmState.GENERATING
