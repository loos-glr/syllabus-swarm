"""
test_stray_scan.py — Tests for the post-run stray-file scan
===========================================================

Covers :func:`src.crews.syllabus_crew._scan_for_stray_generated_files`,
which detects agent-generated artefacts written directly into the ``output/``
root instead of into a per-run directory (``output/<run_id>/``).

The scan is deliberately scoped to ``OUTPUT_ROOT`` so that permanent
repository files at the project root (``README.md``, ``DESIGN.md``,
``.gitignore``, ``docs/`` …) can never be reported as strays.  These tests
lock in that behaviour and guard against regressions.

All filesystem interaction happens under pytest's ``tmp_path`` — no real
project files are ever touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import src.crews.syllabus_crew as sc_module
from src.crews.syllabus_crew import (
    _OUTPUT_ROOT_IGNORED_FILES,
    _OUTPUT_ROOT_SAFE_FILES,
    _scan_for_stray_generated_files,
)

# Representative run id, matching the ``YYYY-MM-DD_HHMMSS_<slug>`` format
# produced by ``generate_run_id()``.
_RUN_ID = "2026-09-26_030503_Mega_Villains_Inc._Workshop"


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def output_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``syllabus_crew.OUTPUT_ROOT`` to a disposable ``output/`` dir."""
    fake_output = tmp_path / "output"
    fake_output.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(sc_module, "OUTPUT_ROOT", fake_output)
    return fake_output


# ===================================================================
# Scope — only ``output/`` is ever inspected
# ===================================================================


class TestScanScopeIsOutputRootOnly:
    """The scan must never inspect the project root."""

    def test_missing_output_root_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A not-yet-created output/ directory yields no warnings."""
        monkeypatch.setattr(sc_module, "OUTPUT_ROOT", tmp_path / "does_not_exist")

        assert _scan_for_stray_generated_files(run_id=_RUN_ID) == []

    def test_static_project_files_are_never_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Static repo files (README.md, .gitignore, docs/*) are not strays.

        Regression test for the false positives that flagged
        ``README.md``, ``DESIGN.md``, ``.gitignore``, ``docs/ARCHITECTURE.md``
        and ``docs/CONSTITUTION.md`` — these are permanent project files, not
        agent output.
        """
        # Mimic the real layout: static files at the project root, with
        # output/ as a sibling directory.
        (tmp_path / "README.md").write_text("# project readme\n", encoding="utf-8")
        (tmp_path / "DESIGN.md").write_text("# design\n", encoding="utf-8")
        (tmp_path / ".gitignore").write_text("output/\n", encoding="utf-8")
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "ARCHITECTURE.md").write_text("# architecture\n", encoding="utf-8")
        (docs / "CONSTITUTION.md").write_text("# constitution\n", encoding="utf-8")

        fake_output = tmp_path / "output"
        fake_output.mkdir()
        (fake_output / _RUN_ID).mkdir()

        monkeypatch.setattr(sc_module, "OUTPUT_ROOT", fake_output)

        assert _scan_for_stray_generated_files(run_id=_RUN_ID) == []

    def test_files_nested_inside_run_directory_are_not_reported(self, output_root: Path) -> None:
        """Everything inside ``output/<run_id>/`` is already in the right place."""
        run_dir = output_root / _RUN_ID
        starter = run_dir / "labs" / "tier1_foundations" / "starter"
        starter.mkdir(parents=True)
        (starter / "index.html").write_text("<html></html>", encoding="utf-8")
        (starter / "styles.css").write_text("body {}", encoding="utf-8")
        (starter / "app.js").write_text("console.log(1);", encoding="utf-8")

        syllabus_dir = run_dir / "syllabus"
        syllabus_dir.mkdir()
        (syllabus_dir / "Syllabus.md").write_text("# syllabus\n", encoding="utf-8")

        assert _scan_for_stray_generated_files(run_id=_RUN_ID) == []

    def test_other_run_directories_are_not_reported(self, output_root: Path) -> None:
        """Sibling run directories and legacy folders are legitimate entries."""
        (output_root / _RUN_ID).mkdir()
        (output_root / "2025_mega_villains").mkdir()
        (output_root / "2023-10-27_153045_MBO4_Software_Development").mkdir()
        (output_root / "placeholder").mkdir()

        assert _scan_for_stray_generated_files(run_id=_RUN_ID) == []


# ===================================================================
# Output-root whitelist / ignore-list
# ===================================================================


class TestOutputRootSafeFiles:
    """Artefacts that legitimately live at the output root are never strays."""

    def test_manifest_readme_is_not_reported(self, output_root: Path) -> None:
        """``output/README.md`` is the auto-generated output manifest."""
        (output_root / _RUN_ID).mkdir()
        (output_root / "README.md").write_text("# manifest\n", encoding="utf-8")

        assert _scan_for_stray_generated_files(run_id=_RUN_ID) == []

    def test_course_graph_json_is_not_reported(self, output_root: Path) -> None:
        """``output/course_graph.json`` is a legitimate manifest companion."""
        (output_root / _RUN_ID).mkdir()
        (output_root / "course_graph.json").write_text("{}", encoding="utf-8")

        assert _scan_for_stray_generated_files(run_id=_RUN_ID) == []

    def test_safe_files_set_contains_both_manifest_artefacts(self) -> None:
        """The whitelist mirrors what ``update_output_manifest`` writes."""
        assert sorted(_OUTPUT_ROOT_SAFE_FILES) == ["README.md", "course_graph.json"]

    def test_ds_store_is_ignored(self, output_root: Path) -> None:
        """macOS ``.DS_Store`` files are not reported."""
        (output_root / _RUN_ID).mkdir()
        (output_root / ".DS_Store").write_bytes(b"\x00\x01")

        assert _scan_for_stray_generated_files(run_id=_RUN_ID) == []

    def test_ignored_files_set_lists_os_artefacts(self) -> None:
        """The ignore-list explicitly enumerates OS/editor artefacts."""
        assert sorted(_OUTPUT_ROOT_IGNORED_FILES) == [".DS_Store", "Thumbs.db", "desktop.ini"]


# ===================================================================
# Detection of genuine strays
# ===================================================================


class TestStrayDetection:
    """Genuine strays written to the output root are reported."""

    def test_loose_file_in_output_root_is_reported(self, output_root: Path) -> None:
        """A loose ``index.html`` at the output root is a stray file."""
        (output_root / _RUN_ID).mkdir()
        stray = output_root / "index.html"
        stray.write_text("<html></html>", encoding="utf-8")

        warnings_list = _scan_for_stray_generated_files(run_id=_RUN_ID)

        assert len(warnings_list) == 1
        assert "Stray generated file detected" in warnings_list[0]
        assert str(stray) in warnings_list[0]

    def test_loose_tier_directory_in_output_root_is_reported(self, output_root: Path) -> None:
        """A tier directory written to the output root is a stray directory."""
        (output_root / _RUN_ID).mkdir()
        stray_dir = output_root / "tier1_foundations"
        stray_dir.mkdir()

        warnings_list = _scan_for_stray_generated_files(run_id=_RUN_ID)

        assert len(warnings_list) == 1
        assert "Stray generated directory detected" in warnings_list[0]
        assert str(stray_dir) in warnings_list[0]

    def test_stray_gitignore_is_still_detected(self, output_root: Path) -> None:
        """A stray ``.gitignore`` is reported despite being a dotfile.

        Guards the explicit ignore-list design: skipping *all* dotfiles would
        silently disable the ``^\\\\.gitignore$`` entry in ``_STRAY_PATTERNS``.
        """
        (output_root / _RUN_ID).mkdir()
        (output_root / ".gitignore").write_text("node_modules/\n", encoding="utf-8")

        warnings_list = _scan_for_stray_generated_files(run_id=_RUN_ID)

        assert len(warnings_list) == 1
        assert ".gitignore" in warnings_list[0]

    def test_each_stray_is_reported_once(self, output_root: Path) -> None:
        """Multiple strays produce exactly one warning each."""
        (output_root / _RUN_ID).mkdir()
        for name in ("index.html", "main.js", "styles.css", "orphan.md"):
            (output_root / name).write_text("x", encoding="utf-8")

        warnings_list = _scan_for_stray_generated_files(run_id=_RUN_ID)

        assert len(warnings_list) == 4

    def test_message_references_resolved_run_id(self, output_root: Path) -> None:
        """The warning points the operator at the active run directory."""
        (output_root / _RUN_ID).mkdir()
        (output_root / "orphan.md").write_text("x", encoding="utf-8")

        warnings_list = _scan_for_stray_generated_files(run_id=_RUN_ID)

        assert f"output/{_RUN_ID}/" in warnings_list[0]

    def test_message_recommends_write_labs_with_run_id(self, output_root: Path) -> None:
        """The warning tells the operator how to prevent recurrence."""
        (output_root / _RUN_ID).mkdir()
        (output_root / "orphan.md").write_text("x", encoding="utf-8")

        warnings_list = _scan_for_stray_generated_files(run_id=_RUN_ID)

        assert "'write-labs' command with 'run_id'" in warnings_list[0]
        assert "instead of 'write-file'" in warnings_list[0]

    def test_verbose_prints_warning_to_stderr(
        self, output_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """With ``verbose=True`` the warning is echoed to stderr."""
        (output_root / _RUN_ID).mkdir()
        (output_root / "orphan.md").write_text("x", encoding="utf-8")

        warnings_list = _scan_for_stray_generated_files(run_id=_RUN_ID, verbose=True)

        captured = capsys.readouterr()
        assert warnings_list[0] in captured.err

    def test_non_verbose_stays_silent(
        self, output_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """With ``verbose=False`` nothing is printed, but warnings are returned."""
        (output_root / _RUN_ID).mkdir()
        (output_root / "orphan.md").write_text("x", encoding="utf-8")

        warnings_list = _scan_for_stray_generated_files(run_id=_RUN_ID)

        captured = capsys.readouterr()
        assert warnings_list
        assert captured.err == ""
