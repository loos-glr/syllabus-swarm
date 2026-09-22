"""
test_generation_state.py — Tests for generation state management.

Covers:
- ``GenerationState`` and ``TierState`` model creation and serialisation.
- ``load_generation_state`` auto-detection from filesystem.
- ``save_generation_state`` atomic write and reload round-trip.
- ``_is_tier_labs_complete`` and ``_is_tier_theory_complete`` checks.
- ``_count_lab_files`` counts only real (non-.gitkeep) files.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.crews.syllabus_crew import (
    _count_lab_files,
    _is_tier_labs_complete,
    _is_tier_theory_complete,
    load_generation_state,
    save_generation_state,
)
from src.models import GenerationState, TierState

# ---------------------------------------------------------------------------
# TierState / GenerationState model tests
# ---------------------------------------------------------------------------


class TestTierState:
    """Unit tests for the TierState Pydantic model."""

    def test_defaults(self) -> None:
        ts = TierState()
        assert ts.status == "incomplete"
        assert ts.files == 0
        assert ts.error is None

    def test_complete_state(self) -> None:
        ts = TierState(status="complete", files=42)
        assert ts.status == "complete"
        assert ts.files == 42

    def test_failed_state(self) -> None:
        ts = TierState(status="failed", error="API timeout")
        assert ts.status == "failed"
        assert ts.error == "API timeout"

    def test_json_roundtrip(self) -> None:
        ts = TierState(status="complete", files=12)
        raw = ts.model_dump_json()
        reloaded = TierState.model_validate_json(raw)
        assert reloaded.status == ts.status
        assert reloaded.files == ts.files


class TestGenerationState:
    """Unit tests for the GenerationState Pydantic model."""

    def test_minimal(self) -> None:
        gs = GenerationState(run_id="2026-01-01_000000_Test", course_name="Test Course")
        assert gs.run_id == "2026-01-01_000000_Test"
        assert gs.course_name == "Test Course"
        assert gs.tiers == {}
        assert gs.theory == {}
        assert gs.syllabus_review == "incomplete"
        assert gs.qa_review == "incomplete"

    def test_with_tiers(self) -> None:
        gs = GenerationState(
            run_id="run1",
            course_name="Test",
            tiers={
                "tier1_foundations": TierState(status="complete", files=10),
                "tier2_application": TierState(status="incomplete"),
            },
            theory={
                "tier1_foundations": "complete",
                "tier2_application": "incomplete",
            },
        )
        assert gs.tiers["tier1_foundations"].status == "complete"
        assert gs.tiers["tier1_foundations"].files == 10
        assert gs.tiers["tier2_application"].status == "incomplete"

    def test_json_roundtrip(self) -> None:
        gs = GenerationState(
            run_id="run2",
            course_name="Test",
            tiers={
                "tier1_foundations": TierState(status="complete", files=5),
            },
        )
        raw = gs.model_dump_json()
        reloaded = GenerationState.model_validate_json(raw)
        assert reloaded.run_id == gs.run_id
        assert reloaded.tiers["tier1_foundations"].status == "complete"
        assert reloaded.tiers["tier1_foundations"].files == 5

    def test_json_serializes_indent(self) -> None:
        gs = GenerationState(run_id="r", course_name="c")
        raw = gs.model_dump_json(indent=2)
        parsed = json.loads(raw)
        assert parsed["run_id"] == "r"
        assert parsed["course_name"] == "c"
        assert "tiers" in parsed


# ---------------------------------------------------------------------------
# Filesystem check tests
# ---------------------------------------------------------------------------


class TestIsTierLabsComplete:
    """Tests for _is_tier_labs_complete."""

    def test_empty_dirs(self, tmp_path: Path) -> None:
        tier_path = tmp_path / "tier1_foundations"
        (tier_path / "starter").mkdir(parents=True)
        (tier_path / "solution").mkdir(parents=True)
        assert not _is_tier_labs_complete(tier_path)

    def test_only_gitkeep_files(self, tmp_path: Path) -> None:
        tier_path = tmp_path / "tier1_foundations"
        (tier_path / "starter").mkdir(parents=True)
        (tier_path / "solution").mkdir(parents=True)
        (tier_path / "starter" / ".gitkeep").write_text("")
        (tier_path / "solution" / ".gitkeep").write_text("")
        assert not _is_tier_labs_complete(tier_path)

    def test_real_files(self, tmp_path: Path) -> None:
        tier_path = tmp_path / "tier1_foundations"
        (tier_path / "starter").mkdir(parents=True)
        (tier_path / "solution").mkdir(parents=True)
        (tier_path / "starter" / "lab1.js").write_text("// TODO")
        (tier_path / "solution" / "lab1.js").write_text("// Done")
        assert _is_tier_labs_complete(tier_path)

    def test_missing_starter(self, tmp_path: Path) -> None:
        tier_path = tmp_path / "tier1_foundations"
        (tier_path / "solution").mkdir(parents=True)
        (tier_path / "solution" / "lab1.js").write_text("// Done")
        assert not _is_tier_labs_complete(tier_path)

    def test_missing_solution(self, tmp_path: Path) -> None:
        tier_path = tmp_path / "tier1_foundations"
        (tier_path / "starter").mkdir(parents=True)
        (tier_path / "starter" / "lab1.js").write_text("// TODO")
        assert not _is_tier_labs_complete(tier_path)

    def test_hidden_files_excluded(self, tmp_path: Path) -> None:
        tier_path = tmp_path / "tier1_foundations"
        (tier_path / "starter").mkdir(parents=True)
        (tier_path / "solution").mkdir(parents=True)
        (tier_path / "starter" / ".hidden.js").write_text("// hidden")
        (tier_path / "solution" / ".also_hidden.js").write_text("// hidden")
        assert not _is_tier_labs_complete(tier_path)


class TestIsTierTheoryComplete:
    """Tests for _is_tier_theory_complete."""

    def test_no_theory_dir(self, tmp_path: Path) -> None:
        tier_path = tmp_path / "tier1_foundations"
        assert not _is_tier_theory_complete(tier_path)

    def test_empty_theory_dir(self, tmp_path: Path) -> None:
        tier_path = tmp_path / "tier1_foundations"
        (tier_path / "theory").mkdir(parents=True)
        assert not _is_tier_theory_complete(tier_path)

    def test_has_theory_file(self, tmp_path: Path) -> None:
        tier_path = tmp_path / "tier1_foundations"
        (tier_path / "theory").mkdir(parents=True)
        (tier_path / "theory" / "visualizer.html").write_text("<html></html>")
        assert _is_tier_theory_complete(tier_path)

    def test_only_hidden_file(self, tmp_path: Path) -> None:
        tier_path = tmp_path / "tier1_foundations"
        (tier_path / "theory").mkdir(parents=True)
        (tier_path / "theory" / ".hidden.html").write_text("<html></html>")
        assert not _is_tier_theory_complete(tier_path)


class TestCountLabFiles:
    """Tests for _count_lab_files."""

    def test_empty(self, tmp_path: Path) -> None:
        tier_path = tmp_path / "tier1_foundations"
        (tier_path / "starter").mkdir(parents=True)
        (tier_path / "solution").mkdir(parents=True)
        assert _count_lab_files(tier_path) == 0

    def test_counts_real_files(self, tmp_path: Path) -> None:
        tier_path = tmp_path / "tier1_foundations"
        (tier_path / "starter").mkdir(parents=True)
        (tier_path / "solution").mkdir(parents=True)
        (tier_path / "starter" / "lab1.js").write_text("// TODO")
        (tier_path / "starter" / "lab2.js").write_text("// TODO")
        (tier_path / "solution" / "lab1.js").write_text("// Done")
        assert _count_lab_files(tier_path) == 3

    def test_excludes_gitkeep_and_hidden(self, tmp_path: Path) -> None:
        tier_path = tmp_path / "tier1_foundations"
        (tier_path / "starter").mkdir(parents=True)
        (tier_path / "solution").mkdir(parents=True)
        (tier_path / "starter" / ".gitkeep").write_text("")
        (tier_path / "starter" / ".hidden.js").write_text("")
        (tier_path / "starter" / "lab1.js").write_text("// TODO")
        assert _count_lab_files(tier_path) == 1


# ---------------------------------------------------------------------------
# State I/O tests (load_generation_state / save_generation_state)
# ---------------------------------------------------------------------------


class TestLoadGenerationState:
    """Tests for load_generation_state."""

    def test_missing_run_dir(self, tmp_path: Path) -> None:
        assert load_generation_state(tmp_path / "nonexistent") is None

    def test_auto_generate_all_incomplete(self, tmp_path: Path) -> None:
        """When only scaffolding exists, all tiers should be incomplete."""
        run_dir = tmp_path / "2026-01-01_000000_TestCourse"
        labs_path = run_dir / "labs"
        for tier in ["tier1_foundations", "tier2_application", "tier3_architecture"]:
            (labs_path / tier / "starter").mkdir(parents=True)
            (labs_path / tier / "solution").mkdir(parents=True)
            (labs_path / tier / "starter" / ".gitkeep").write_text("")
            (labs_path / tier / "solution" / ".gitkeep").write_text("")

        state = load_generation_state(run_dir)
        assert state is not None
        assert state.run_id == run_dir.name
        for tier in ["tier1_foundations", "tier2_application", "tier3_architecture"]:
            assert state.tiers[tier].status == "incomplete"
            assert state.theory[tier] == "incomplete"

    def test_auto_generate_detects_complete(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "2026-01-01_000000_TestCourse"
        labs_path = run_dir / "labs"

        # tier1: complete labs + theory
        tier1 = labs_path / "tier1_foundations"
        (tier1 / "starter").mkdir(parents=True)
        (tier1 / "solution").mkdir(parents=True)
        (tier1 / "theory").mkdir(parents=True)
        (tier1 / "starter" / "lab1.js").write_text("// TODO")
        (tier1 / "solution" / "lab1.js").write_text("// Done")
        (tier1 / "theory" / "visualizer.html").write_text("<html>")

        # tier2: scaffolding only
        tier2 = labs_path / "tier2_application"
        (tier2 / "starter").mkdir(parents=True)
        (tier2 / "solution").mkdir(parents=True)
        (tier2 / "starter" / ".gitkeep").write_text("")
        (tier2 / "solution" / ".gitkeep").write_text("")

        # tier3: scaffolding only
        tier3 = labs_path / "tier3_architecture"
        (tier3 / "starter").mkdir(parents=True)
        (tier3 / "solution").mkdir(parents=True)

        state = load_generation_state(run_dir)
        assert state is not None

        # tier1 should be detected as complete
        assert state.tiers["tier1_foundations"].status == "complete"
        assert state.tiers["tier1_foundations"].files == 2
        assert state.theory["tier1_foundations"] == "complete"

        # tier2 and tier3 should be incomplete
        assert state.tiers["tier2_application"].status == "incomplete"
        assert state.tiers["tier3_architecture"].status == "incomplete"
        assert state.theory["tier2_application"] == "incomplete"
        assert state.theory["tier3_architecture"] == "incomplete"

    def test_load_existing_state_file(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "2026-01-01_000000_TestCourse"
        run_dir.mkdir(parents=True)
        (run_dir / "labs").mkdir()

        gs = GenerationState(
            run_id=run_dir.name,
            course_name="Test Course",
            tiers={
                "tier1_foundations": TierState(status="complete", files=5),
            },
            theory={"tier1_foundations": "complete"},
        )
        save_generation_state(run_dir, gs)

        loaded = load_generation_state(run_dir)
        assert loaded is not None
        assert loaded.run_id == run_dir.name
        assert loaded.tiers["tier1_foundations"].status == "complete"
        assert loaded.tiers["tier1_foundations"].files == 5

    def test_corrupt_state_falls_back_to_auto(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "2026-01-01_000000_TestCourse"
        run_dir.mkdir(parents=True)
        labs_path = run_dir / "labs"

        # Create real lab files so auto-detection finds them
        tier1 = labs_path / "tier1_foundations"
        (tier1 / "starter").mkdir(parents=True)
        (tier1 / "solution").mkdir(parents=True)
        (tier1 / "starter" / "lab1.js").write_text("// TODO")
        (tier1 / "solution" / "lab1.js").write_text("// Done")

        # Write corrupt JSON as the state file
        (run_dir / "_generation_state.json").write_text("this is not valid json")

        loaded = load_generation_state(run_dir)
        assert loaded is not None
        # Falls back to auto-detection: should find tier1 as complete
        assert loaded.tiers["tier1_foundations"].status == "complete"


class TestSaveGenerationState:
    """Tests for save_generation_state."""

    def test_save_and_reload_roundtrip(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "2026-01-01_000000_TestCourse"
        run_dir.mkdir(parents=True)

        gs = GenerationState(
            run_id=run_dir.name,
            course_name="Test Course",
            tiers={
                "tier1_foundations": TierState(status="complete", files=10),
                "tier2_application": TierState(status="failed", error="timeout"),
            },
            theory={
                "tier1_foundations": "complete",
                "tier2_application": "failed",
            },
            qa_review="complete",
        )
        save_generation_state(run_dir, gs)

        state_file = run_dir / "_generation_state.json"
        assert state_file.exists()

        loaded = GenerationState.model_validate_json(state_file.read_text(encoding="utf-8"))
        assert loaded.run_id == gs.run_id
        assert loaded.course_name == gs.course_name
        assert loaded.tiers["tier1_foundations"].status == "complete"
        assert loaded.tiers["tier1_foundations"].files == 10
        assert loaded.tiers["tier2_application"].status == "failed"
        assert loaded.tiers["tier2_application"].error == "timeout"
        assert loaded.theory["tier1_foundations"] == "complete"
        assert loaded.theory["tier2_application"] == "failed"
        assert loaded.qa_review == "complete"

    def test_save_overwrites(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "2026-01-01_000000_TestCourse"
        run_dir.mkdir(parents=True)

        gs1 = GenerationState(run_id="r", course_name="c")
        save_generation_state(run_dir, gs1)

        gs2 = GenerationState(run_id="r", course_name="c")
        gs2.qa_review = "complete"
        save_generation_state(run_dir, gs2)

        loaded = load_generation_state(run_dir)
        assert loaded is not None
        assert loaded.qa_review == "complete"

    def test_no_tmp_file_left_behind(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "2026-01-01_000000_TestCourse"
        run_dir.mkdir(parents=True)

        gs = GenerationState(run_id="r", course_name="c")
        save_generation_state(run_dir, gs)

        tmp_files = list(run_dir.glob("*.tmp"))
        assert len(tmp_files) == 0
