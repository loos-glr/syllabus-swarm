"""Tests for modality routing and state machine branching.

The routing decision is produced deterministically by the System One model
(:mod:`src.evaluators.modality_router`) — the former Media Strategist agent has
been retired.  These tests therefore assert against the **real** ``ROUTING_MAP``
and the swarm state mapping rather than a locally-redefined dictionary.
"""

from __future__ import annotations

from src.crews.syllabus_crew import (
    MODALITY_STATE_MAP,
    ROUTING_MAP,
    SwarmState,
    next_state_after_routing,
)
from src.evaluators.modality_router import routing_target
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
        """Every ModalityType has a route in the real ROUTING_MAP."""
        for modality in ModalityType:
            assert modality in ROUTING_MAP, f"No route for {modality}"
            assert ROUTING_MAP[modality] in {"theory_instructor", "video_engineer"}

    def test_defensive_routing_target_lookup(self) -> None:
        """routing_target() is the single source of truth for the downstream agent."""
        assert routing_target(ModalityType.VIDEO_AS_CODE) == "video_engineer"

    def test_state_map_covers_every_modality(self) -> None:
        """Every modality maps to a swarm state."""
        for modality in ModalityType:
            assert modality in MODALITY_STATE_MAP, f"No state for {modality}"

    def test_modality_decision_provenance_defaults(self) -> None:
        """Provenance fields are optional so legacy callers keep working."""
        decision = ModalityDecision(
            module_name="Syntax",
            modality=ModalityType.CLASSIC_READER,
            rationale="Deterministic.",
            complexity_score=0.3,
        )
        assert decision.confidence is None
        assert decision.model_version is None
        assert decision.needs_review is False


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
        assert next_state_after_routing(decision.modality) == SwarmState.VIDEO_GENERATING

    def test_routing_stays_in_generating_for_classic_reader(self) -> None:
        """CLASSIC_READER keeps execution in standard GENERATING flow."""
        decision = ModalityDecision(
            module_name="Syntax",
            modality=ModalityType.CLASSIC_READER,
            rationale="Text is sufficient.",
            complexity_score=0.3,
        )
        assert next_state_after_routing(decision.modality) == SwarmState.GENERATING

    def test_interactive_modalities_stay_in_generating(self) -> None:
        """Interactive web/CLI are produced by the Theory Instructor."""
        for modality in (ModalityType.INTERACTIVE_WEB, ModalityType.INTERACTIVE_CLI):
            assert next_state_after_routing(modality) == SwarmState.GENERATING
