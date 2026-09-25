"""Tests for Layer 1 Domain Entities - Modality Routing and VaC (Issue #8)."""

from __future__ import annotations

import json
from enum import Enum

import pytest
from pydantic import ValidationError

from src.models import (
    LessonPlanManifest,
    ModalityDecision,
    ModalityType,
    PresentationManifest,
    RemotionManifest,
    SessionBlock,
    Slide,
)


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

# ===================================================================
# LessonPlanManifest & SessionBlock (Issue #12)
# ===================================================================


class TestSessionBlock:
    """Tests for the SessionBlock Pydantic model."""

    def test_valid_session_block(self) -> None:
        sb = SessionBlock(
            session_number=1,
            title="Introduction to OOP",
            duration_minutes=45,
        )
        assert sb.session_number == 1
        assert sb.title == "Introduction to OOP"
        assert sb.duration_minutes == 45
        assert sb.learning_objectives == []
        assert sb.activities == []

    def test_full_session_block(self) -> None:
        sb = SessionBlock(
            session_number=2,
            title="Inheritance Deep Dive",
            duration_minutes=60,
            learning_objectives=["Understand inheritance", "Apply super()"],
            activities=["Live coding demo", "Pair programming exercise"],
            resources=["IDE", "Slides", "Starter code repo"],
            differentiation="Advanced students: add multiple inheritance",
            assessment="Exit ticket: write a class hierarchy",
        )
        assert len(sb.learning_objectives) == 2
        assert len(sb.activities) == 2
        assert len(sb.resources) == 3
        assert "Advanced" in sb.differentiation

    def test_rejects_invalid_duration_minutes(self) -> None:
        with pytest.raises(ValidationError, match="duration_minutes"):
            SessionBlock(
                session_number=1,
                title="Test",
                duration_minutes=0,
            )

    def test_rejects_negative_duration(self) -> None:
        with pytest.raises(ValidationError, match="duration_minutes"):
            SessionBlock(
                session_number=1,
                title="Test",
                duration_minutes=-1,
            )

    def test_rejects_empty_title(self) -> None:
        with pytest.raises(ValidationError, match="title"):
            SessionBlock(
                session_number=1,
                title="",
                duration_minutes=45,
            )

    def test_session_number_must_be_positive(self) -> None:
        with pytest.raises(ValidationError, match="session_number"):
            SessionBlock(
                session_number=0,
                title="Test",
                duration_minutes=45,
            )

    def test_json_roundtrip(self) -> None:
        sb = SessionBlock(
            session_number=1,
            title="Test Session",
            duration_minutes=45,
            learning_objectives=["LO1"],
        )
        raw = sb.model_dump_json()
        reloaded = SessionBlock.model_validate_json(raw)
        assert reloaded.title == sb.title
        assert reloaded.learning_objectives == sb.learning_objectives


class TestLessonPlanManifest:
    """Tests for the LessonPlanManifest Pydantic model."""

    def test_valid_manifest(self) -> None:
        lp = LessonPlanManifest(
            module_name="Python Fundamentals",
            sessions=[
                SessionBlock(
                    session_number=1,
                    title="Variables and Types",
                    duration_minutes=45,
                )
            ],
            total_duration_minutes=45,
        )
        assert lp.module_name == "Python Fundamentals"
        assert len(lp.sessions) == 1
        assert lp.total_duration_minutes == 45

    def test_rejects_empty_module_name(self) -> None:
        with pytest.raises(ValidationError, match="module_name"):
            LessonPlanManifest(module_name="")

    def test_multiple_sessions(self) -> None:
        lp = LessonPlanManifest(
            module_name="Advanced JS",
            sessions=[
                SessionBlock(session_number=1, title="S1", duration_minutes=45),
                SessionBlock(session_number=2, title="S2", duration_minutes=50),
                SessionBlock(session_number=3, title="S3", duration_minutes=40),
            ],
            total_duration_minutes=135,
        )
        assert len(lp.sessions) == 3
        assert lp.total_duration_minutes == 135

    def test_prerequisites_defaults_to_empty(self) -> None:
        lp = LessonPlanManifest(module_name="Test")
        assert lp.prerequisites == []
        assert lp.sessions == []

    def test_json_roundtrip(self) -> None:
        lp = LessonPlanManifest(
            module_name="Test Module",
            sessions=[
                SessionBlock(session_number=1, title="S1", duration_minutes=45),
            ],
            total_duration_minutes=45,
            prerequisites=["Basic Python"],
        )
        raw = lp.model_dump_json()
        reloaded = LessonPlanManifest.model_validate_json(raw)
        assert reloaded.module_name == lp.module_name
        assert reloaded.prerequisites == lp.prerequisites
        with pytest.raises(ValidationError, match="height"):
            RemotionManifest(
                composition_id="test",
                duration_in_frames=100,
                height=0,
                components=[],
                module_name="Test",
            )
# ===================================================================
# PresentationManifest & Slide (Issue #12)
# ===================================================================


class TestSlide:
    """Tests for the Slide Pydantic model."""

    def test_valid_slide(self) -> None:
        s = Slide(
            slide_number=0,
            slide_type="title",
            title="Welcome to Python",
        )
        assert s.slide_number == 0
        assert s.slide_type == "title"
        assert s.title == "Welcome to Python"
        assert s.content == ""

    def test_full_slide(self) -> None:
        s = Slide(
            slide_number=5,
            slide_type="code",
            title="Functions Example",
            content="def hello():\n    print('Hello')",
            speaker_notes="Walk through function syntax step by step.",
            transition="fade",
        )
        assert s.speaker_notes == "Walk through function syntax step by step."
        assert s.transition == "fade"

    def test_rejects_empty_slide_type(self) -> None:
        with pytest.raises(ValidationError, match="slide_type"):
            Slide(
                slide_number=1,
                slide_type="",
                title="Test",
            )

    def test_rejects_empty_title(self) -> None:
        with pytest.raises(ValidationError, match="title"):
            Slide(
                slide_number=1,
                slide_type="bullets",
                title="",
            )

    def test_slide_number_non_negative(self) -> None:
        """slide_number with ge=0 accepts zero and positive."""
        s = Slide(slide_number=0, slide_type="title", title="Start")
        assert s.slide_number == 0

    def test_slide_number_cannot_be_negative(self) -> None:
        with pytest.raises(ValidationError):
            Slide(slide_number=-1, slide_type="title", title="Test")

    def test_json_roundtrip(self) -> None:
        s = Slide(
            slide_number=3,
            slide_type="bullets",
            title="Key Points",
            content="- Point 1\n- Point 2",
            speaker_notes="Emphasize point 2",
            transition="none",
        )
        raw = s.model_dump_json()
        reloaded = Slide.model_validate_json(raw)
        assert reloaded.slide_number == s.slide_number
        assert reloaded.speaker_notes == s.speaker_notes


class TestPresentationManifest:
    """Tests for the PresentationManifest Pydantic model."""

    def test_valid_manifest(self) -> None:
        pm = PresentationManifest(
            module_name="Python Basics",
            slides=[
                Slide(slide_number=0, slide_type="title", title="Python Basics"),
                Slide(
                    slide_number=1,
                    slide_type="bullets",
                    title="Learning Goals",
                    content="- Variables\n- Functions",
                ),
            ],
            total_estimated_minutes=15,
        )
        assert pm.module_name == "Python Basics"
        assert len(pm.slides) == 2
        assert pm.total_estimated_minutes == 15

    def test_rejects_empty_module_name(self) -> None:
        with pytest.raises(ValidationError, match="module_name"):
            PresentationManifest(module_name="")

    def test_marp_frontmatter_defaults_to_empty(self) -> None:
        pm = PresentationManifest(module_name="Test")
        assert pm.marp_frontmatter == {}
        assert pm.slides == []

    def test_marp_frontmatter_with_config(self) -> None:
        pm = PresentationManifest(
            module_name="Test",
            marp_frontmatter={
                "theme": "gaia",
                "paginate": "true",
                "size": "16:9",
            },
        )
        assert pm.marp_frontmatter["theme"] == "gaia"
        assert pm.marp_frontmatter["size"] == "16:9"

    def test_json_roundtrip(self) -> None:
        pm = PresentationManifest(
            module_name="Test Module",
            slides=[
                Slide(slide_number=0, slide_type="title", title="Title Slide"),
            ],
            total_estimated_minutes=5,
            marp_frontmatter={"theme": "default"},
        )
        raw = pm.model_dump_json()
        reloaded = PresentationManifest.model_validate_json(raw)
        assert reloaded.module_name == pm.module_name
        assert reloaded.marp_frontmatter == pm.marp_frontmatter
