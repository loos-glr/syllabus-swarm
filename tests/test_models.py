"""Tests for Layer 1 Domain Entities - Modality Routing and VaC (Issue #8)."""

from __future__ import annotations

import json
from enum import Enum

import pytest
from pydantic import ValidationError

from src.models import ModalityDecision, ModalityType, RemotionManifest


class TestModalityType:
    """Tests for the ModalityType enum."""

    def test_has_exactly_four_members(self) -> None:
        assert len(list(ModalityType)) == 4

    def test_classic_reader_member(self) -> None:
        assert ModalityType.CLASSIC_READER in ModalityType

    def test_interactive_web_member(self) -> None:
        assert ModalityType.INTERACTIVE_WEB in ModalityType

    def test_interactive_cli_member(self) -> None:
        assert ModalityType.INTERACTIVE_CLI in ModalityType

    def test_video_as_code_member(self) -> None:
        assert ModalityType.VIDEO_AS_CODE in ModalityType

    def test_is_string_enum(self) -> None:
        assert issubclass(ModalityType, str)
        assert issubclass(ModalityType, Enum)

    def test_classic_reader_value(self) -> None:
        assert ModalityType.CLASSIC_READER.value == "classic_reader"

    def test_video_as_code_value(self) -> None:
        assert ModalityType.VIDEO_AS_CODE.value == "video_as_code"


class TestModalityDecision:
    """Tests for the ModalityDecision Pydantic model."""

    def test_valid_decision_with_classic_reader(self) -> None:
        d = ModalityDecision(
            module_name="Python Fundamentals",
            modality=ModalityType.CLASSIC_READER,
            rationale="Text-based theory is sufficient.",
            complexity_score=0.3,
            suggested_components=["markdown_reader"],
        )
        assert d.module_name == "Python Fundamentals"
        assert d.modality == ModalityType.CLASSIC_READER
        assert d.complexity_score == 0.3

    def test_valid_decision_with_video_as_code(self) -> None:
        d = ModalityDecision(
            module_name="Recursion Trees",
            modality=ModalityType.VIDEO_AS_CODE,
            rationale="Visual recursion depth needs animation.",
            complexity_score=0.85,
            suggested_components=["tree_animation"],
        )
        assert d.modality == ModalityType.VIDEO_AS_CODE
        assert d.complexity_score == 0.85

    def test_valid_decision_with_interactive_web(self) -> None:
        d = ModalityDecision(
            module_name="Sorting Algorithms",
            modality=ModalityType.INTERACTIVE_WEB,
            rationale="Interactive sorting needs browser.",
            complexity_score=0.6,
            suggested_components=["bubble_sort_viz"],
        )
        assert d.modality == ModalityType.INTERACTIVE_WEB

    def test_valid_decision_with_interactive_cli(self) -> None:
        d = ModalityDecision(
            module_name="Git Workflows",
            modality=ModalityType.INTERACTIVE_CLI,
            rationale="CLI git practice needs terminal.",
            complexity_score=0.5,
        )
        assert d.modality == ModalityType.INTERACTIVE_CLI

    def test_suggested_components_defaults_to_empty(self) -> None:
        d = ModalityDecision(
            module_name="Intro",
            modality=ModalityType.CLASSIC_READER,
            rationale="Simple.",
            complexity_score=0.2,
        )
        assert d.suggested_components == []

    def test_rejects_empty_module_name(self) -> None:
        with pytest.raises(ValidationError, match="module_name"):
            ModalityDecision(
                module_name="",
                modality=ModalityType.CLASSIC_READER,
                rationale="Valid.",
                complexity_score=0.5,
            )

    def test_rejects_empty_rationale(self) -> None:
        with pytest.raises(ValidationError, match="rationale"):
            ModalityDecision(
                module_name="Valid",
                modality=ModalityType.CLASSIC_READER,
                rationale="",
                complexity_score=0.5,
            )

    def test_rejects_missing_modality(self) -> None:
        with pytest.raises(ValidationError, match="modality"):
            ModalityDecision(
                module_name="Valid",
                rationale="Valid.",
                complexity_score=0.5,
            )

    def test_rejects_complexity_above_1(self) -> None:
        with pytest.raises(ValidationError, match="complexity_score"):
            ModalityDecision(
                module_name="Valid",
                modality=ModalityType.CLASSIC_READER,
                rationale="Valid.",
                complexity_score=1.5,
            )

    def test_rejects_complexity_below_0(self) -> None:
        with pytest.raises(ValidationError, match="complexity_score"):
            ModalityDecision(
                module_name="Valid",
                modality=ModalityType.CLASSIC_READER,
                rationale="Valid.",
                complexity_score=-0.1,
            )

    def test_allows_complexity_0(self) -> None:
        d = ModalityDecision(
            module_name="Boundary",
            modality=ModalityType.CLASSIC_READER,
            rationale="Testing.",
            complexity_score=0.0,
        )
        assert d.complexity_score == 0.0

    def test_allows_complexity_1(self) -> None:
        d = ModalityDecision(
            module_name="Boundary",
            modality=ModalityType.CLASSIC_READER,
            rationale="Testing.",
            complexity_score=1.0,
        )
        assert d.complexity_score == 1.0

    def test_rejects_invalid_modality_string(self) -> None:
        with pytest.raises(ValidationError):
            ModalityDecision(
                module_name="Test",
                modality="INVALID",
                rationale="Testing.",
                complexity_score=0.5,
            )

    def test_decision_json_serializable(self) -> None:
        d = ModalityDecision(
            module_name="Python Fundamentals",
            modality=ModalityType.VIDEO_AS_CODE,
            rationale="Visual explanation needed.",
            complexity_score=0.75,
            suggested_components=["intro_sequence", "code_highlight"],
        )
        data = d.model_dump()
        parsed = json.loads(json.dumps(data, default=str))
        assert parsed["module_name"] == "Python Fundamentals"
        assert parsed["modality"] == "video_as_code"
        assert parsed["complexity_score"] == 0.75
        assert "intro_sequence" in parsed["suggested_components"]


class TestRemotionManifest:
    """Tests for the RemotionManifest Pydantic model."""

    def test_valid_manifest(self) -> None:
        m = RemotionManifest(
            composition_id="recursion_basics",
            duration_in_frames=300,
            fps=30,
            width=1920,
            height=1080,
            components=[
                {"type": "sequence", "name": "intro", "from": 0, "duration": 90},
                {"type": "sequence", "name": "main", "from": 90, "duration": 210},
            ],
            module_name="Recursion Trees",
        )
        assert m.composition_id == "recursion_basics"
        assert m.duration_in_frames == 300
        assert m.module_name == "Recursion Trees"

    def test_manifest_minimal_components(self) -> None:
        m = RemotionManifest(
            composition_id="simple_intro",
            duration_in_frames=150,
            fps=24,
            components=[{"type": "title_card", "text": "Welcome"}],
            module_name="Intro Module",
        )
        assert len(m.components) == 1

    def test_manifest_empty_components(self) -> None:
        m = RemotionManifest(
            composition_id="placeholder",
            duration_in_frames=60,
            fps=30,
            components=[],
            module_name="Stub Module",
        )
        assert m.components == []

    def test_default_fps_30(self) -> None:
        m = RemotionManifest(
            composition_id="test",
            duration_in_frames=100,
            components=[],
            module_name="Test",
        )
        assert m.fps == 30

    def test_default_width_1920(self) -> None:
        m = RemotionManifest(
            composition_id="test",
            duration_in_frames=100,
            components=[],
            module_name="Test",
        )
        assert m.width == 1920

    def test_default_height_1080(self) -> None:
        m = RemotionManifest(
            composition_id="test",
            duration_in_frames=100,
            components=[],
            module_name="Test",
        )
        assert m.height == 1080

    def test_rejects_empty_composition_id(self) -> None:
        with pytest.raises(ValidationError, match="composition_id"):
            RemotionManifest(
                composition_id="",
                duration_in_frames=100,
                components=[],
                module_name="Test",
            )

    def test_rejects_empty_module_name(self) -> None:
        with pytest.raises(ValidationError, match="module_name"):
            RemotionManifest(
                composition_id="test",
                duration_in_frames=100,
                components=[],
                module_name="",
            )

    def test_rejects_duration_zero(self) -> None:
        with pytest.raises(ValidationError, match="duration_in_frames"):
            RemotionManifest(
                composition_id="test",
                duration_in_frames=0,
                components=[],
                module_name="Test",
            )

    def test_rejects_duration_negative(self) -> None:
        with pytest.raises(ValidationError, match="duration_in_frames"):
            RemotionManifest(
                composition_id="test",
                duration_in_frames=-1,
                components=[],
                module_name="Test",
            )

    def test_rejects_fps_zero(self) -> None:
        with pytest.raises(ValidationError, match="fps"):
            RemotionManifest(
                composition_id="test",
                duration_in_frames=100,
                fps=0,
                components=[],
                module_name="Test",
            )

    def test_rejects_width_zero(self) -> None:
        with pytest.raises(ValidationError, match="width"):
            RemotionManifest(
                composition_id="test",
                duration_in_frames=100,
                width=0,
                components=[],
                module_name="Test",
            )

    def test_rejects_height_zero(self) -> None:
        with pytest.raises(ValidationError, match="height"):
            RemotionManifest(
                composition_id="test",
                duration_in_frames=100,
                height=0,
                components=[],
                module_name="Test",
            )
