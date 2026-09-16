"""
test_resume.py — Tests for ``--resume-from`` validation & error handling
========================================================================

Regression tests for the failures documented when ``--resume-from`` was passed
a file (``intake_session.json``) instead of a run directory.  Guards against:

* ``_validate_resume_dir`` accepting a file or a directory without a usable
  ``syllabus/`` subdirectory.
* ``_find_syllabus_in_dir`` failing clearly on non-directory / missing-syllabus
  inputs.
* ``run_syllabus_crew`` raising an actionable ``RuntimeError`` instead of
  fabricating a broken ``output/resume_failed/`` fallback path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.crews.syllabus_crew import _find_syllabus_in_dir, run_syllabus_crew
from src.main import _validate_resume_dir

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run_dir(root: Path, name: str, *, with_syllabus: bool = True) -> Path:
    """Create a synthetic run directory (optionally with a syllabus/ .md)."""
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    if with_syllabus:
        syllabus_dir = run_dir / "syllabus"
        syllabus_dir.mkdir(parents=True, exist_ok=True)
        (syllabus_dir / f"{name}.md").write_text("# Syllabus\n", encoding="utf-8")
    return run_dir


# ---------------------------------------------------------------------------
# _validate_resume_dir
# ---------------------------------------------------------------------------


class TestValidateResumeDir:
    """``_validate_resume_dir`` fails fast for all invalid inputs."""

    def test_returns_resolved_path_for_valid_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A valid run directory resolves to its absolute Path."""
        import src.main as main_module

        monkeypatch.setattr(main_module, "_PROJECT_ROOT", tmp_path)
        _make_run_dir(tmp_path, "2026-08-27_091745_Javascript_OOP")

        result = _validate_resume_dir("2026-08-27_091745_Javascript_OOP")
        assert result == tmp_path / "2026-08-27_091745_Javascript_OOP"

    def test_nonexistent_path_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing directory raises SystemExit with a clear message."""
        import src.main as main_module

        monkeypatch.setattr(main_module, "_PROJECT_ROOT", tmp_path)
        with pytest.raises(SystemExit) as excinfo:
            _validate_resume_dir("does_not_exist")
        assert excinfo.value.code == 1

    def test_file_path_raises_with_session_hint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """Passing a file (e.g. intake_session.json) yields a --load-session hint."""
        import src.main as main_module

        monkeypatch.setattr(main_module, "_PROJECT_ROOT", tmp_path)
        session_file = tmp_path / "intake_session.json"
        session_file.write_text("{}", encoding="utf-8")

        with pytest.raises(SystemExit):
            _validate_resume_dir("intake_session.json")
        captured = capsys.readouterr()
        assert "--load-session" in captured.err

    def test_directory_without_syllabus_subdir_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A directory lacking a syllabus/ subdirectory is rejected."""
        import src.main as main_module

        monkeypatch.setattr(main_module, "_PROJECT_ROOT", tmp_path)
        _make_run_dir(tmp_path, "run_no_syllabus", with_syllabus=False)

        with pytest.raises(SystemExit):
            _validate_resume_dir("run_no_syllabus")

    def test_syllabus_subdir_without_md_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A syllabus/ subdirectory with no .md file is rejected."""
        import src.main as main_module

        monkeypatch.setattr(main_module, "_PROJECT_ROOT", tmp_path)
        run_dir = tmp_path / "run_empty_syllabus"
        (run_dir / "syllabus").mkdir(parents=True)

        with pytest.raises(SystemExit):
            _validate_resume_dir("run_empty_syllabus")


# ---------------------------------------------------------------------------
# _find_syllabus_in_dir
# ---------------------------------------------------------------------------


class TestFindSyllabusInDir:
    """``_find_syllabus_in_dir`` locates the syllabus or fails clearly."""

    def test_returns_syllabus_md(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, "run_with_syllabus")
        expected = run_dir / "syllabus" / "run_with_syllabus.md"
        assert _find_syllabus_in_dir(run_dir) == expected

    def test_nonexistent_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _find_syllabus_in_dir(tmp_path / "missing")

    def test_file_path_raises(self, tmp_path: Path) -> None:
        session_file = tmp_path / "intake_session.json"
        session_file.write_text("{}", encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            _find_syllabus_in_dir(session_file)

    def test_missing_syllabus_subdir_raises(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, "run_no_syllabus", with_syllabus=False)
        with pytest.raises(FileNotFoundError):
            _find_syllabus_in_dir(run_dir)


# ---------------------------------------------------------------------------
# run_syllabus_crew resume-mode error handling
# ---------------------------------------------------------------------------


class TestRunSyllabusCrewResumeError:
    """Resume mode fails loudly instead of crashing in an unrelated spot."""

    def test_invalid_resume_dir_raises_runtime_error(self, tmp_path: Path) -> None:
        """A non-directory resume path surfaces as an actionable RuntimeError."""
        with pytest.raises(RuntimeError) as excinfo:
            run_syllabus_crew(
                "Course Name: Javascript OOP",
                course_name="Javascript OOP",
                resume_dir=tmp_path / "does_not_exist",
            )
        assert "--load-session" in str(excinfo.value)
