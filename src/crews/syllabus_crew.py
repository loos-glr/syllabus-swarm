"""
syllabus_crew.py — Full Syllabus + Labs Orchestration Crew
==========================================================

Issue #2: Core Agent — The Curriculum Architect (Humanics Alignment)
Issue #3: Core Agent — The Lab & Project Developer (Tiered Coding Challenges)
Issue #7: AI-Driven Feedback Loop — QA Reviewer Agent

Wires together three agents into a sequential CrewAI Crew:
  1. **Curriculum Architect** — generates a Humanics-aligned syllabus in
     Markdown and saves it to ``output/syllabus/<course_name>.md``.
  2. **Lab & Project Developer** — receives the syllabus as context and
     generates tiered coding labs saved to ``output/labs/<course_name>/``.
  3. **QA Reviewer** — reviews all generated labs for technical correctness
     and MBO4 didactic appropriateness.  Can delegate fixes back to the
     Lab Developer via CrewAI's delegation mechanism.

The crew runs sequentially so each agent can use the previous agent's
output as grounding context.
"""

from __future__ import annotations

import re
import sys
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from crewai import Agent, Crew, Process

from src.agents.curriculum_architect import get_architect
from src.agents.education_director import get_education_director
from src.agents.lab_developer import get_lab_developer
from src.agents.qa_reviewer import get_qa_reviewer
from src.agents.theory_instructor import get_theory_instructor
from src.exporters import (
    update_output_manifest,
    write_file,
)
from src.exporters.file_writer import _sanitize_filename
from src.exporters.theory_validator import (
    format_validation_report,
    validate_theory_directory,
)
from src.models import GenerationState, TierState
from src.tasks.lab_generation import create_lab_generation_task
from src.tasks.qa_review import create_qa_review_task
from src.tasks.syllabus_generation import create_syllabus_generation_task
from src.tasks.syllabus_review import create_syllabus_review_task
from src.tasks.theory_generation import create_theory_task


class SwarmState(str, Enum):
    """Execution states for the HITL cyclic syllabus swarm.

    .. rubric:: Issue #6 — Cyclic State Machine

    Values
    ------
    GENERATING
        Initial generation in progress (all agents run).
    AWAITING_FEEDBACK
        Generation complete; waiting for human review.
    EXPORTING
        Human approved; proceed to manifest export.
    """

    GENERATING = "generating"
    AWAITING_FEEDBACK = "awaiting_feedback"
    EXPORTING = "exporting"
    VIDEO_GENERATING = "video_generating"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
OUTPUT_ROOT: Path = _PROJECT_ROOT / "output"

# Tier names used for the lab directory scaffolding.
_TIERS: list[tuple[str, str]] = [
    ("tier1_foundations", "Tier 1 — Foundations"),
    ("tier2_application", "Tier 2 — Application"),
    ("tier3_architecture", "Tier 3 — Architecture"),
]

# Known pattern emitted by CrewAI when an agent hits its ``max_iter``
# ceiling.  We scan exception messages for this substring so we can
# surface a clear, actionable hint (which env var to bump) instead of
# forcing the operator to decode three separate error lines.
_MAX_ITER_CREWAI_MARKER: str = "Maximum iterations reached"

# Patterns used by :func:`_scan_for_stray_generated_files` to detect
# agent-generated files that escaped the ``output/`` directory.
_STRAY_PATTERNS: list[tuple[str, str]] = [
    (r"^tier\d", "directory"),
    (r"^tier\d.*", "directory"),
    (r"\.js$", "file"),
    (r"^package\.json$", "file"),
    (r"^Makefile$", "file"),
    (r"^Dockerfile$", "file"),
    (r"^docker-compose\.yml$", "file"),
    (r"^docker-compose\.yaml$", "file"),
    (r"^\.gitignore$", "file"),
    (r"\.html$", "file"),
    (r"\.css$", "file"),
    (r"\.sh$", "file"),
    (r"\.yml$", "file"),
    (r"\.md$", "file"),
]
# Directories that are permanently at the project root and should never
# be flagged as strays.
_STRAY_SAFE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        ".roo",
        ".ruler",
        ".venv",
        "__pycache__",
        "config",
        "output",
        "src",
        "tests",
    }
)


def _annotate_iter_exhaustion(
    raw_error: str,
    agent_role_env_key: str,
    *,
    parent_error: BaseException | None = None,
) -> str:
    """If *raw_error* signals iteration exhaustion, append a hint.

    CrewAI emits ``"Maximum iterations reached"`` (or wraps it in an
    ``Invalid response`` chain) when an agent consumes all of its
    ``max_iter`` budget.  This function detects that marker and appends
    a human-readable suggestion pinned to the appropriate env var::

        AGENT_{agent_role_env_key}_MAX_ITER

    Parameters
    ----------
    raw_error : str
        The original error message text.
    agent_role_env_key : str
        Uppercase role constant used in env-var names
        (e.g. ``"QA_REVIEWER"``).
    parent_error : BaseException or None
        When available, its class name is included in the hint so the
        operator can distinguish a timeout from a max-iter ceiling.

    Returns
    -------
    str
        *raw_error* unchanged if no exhaustion marker is found; otherwise
        *raw_error* plus a newline-separated hint.
    """
    if _MAX_ITER_CREWAI_MARKER not in raw_error:
        return raw_error

    env_var = f"AGENT_{agent_role_env_key}_MAX_ITER"

    parts: list[str] = [
        raw_error,
        "",
        "─" * 60,
        "⚠️  ITERATION LIMIT EXHAUSTION DETECTED",
        "",
        "   The agent exceeded its max_iter budget.  CrewAI emitted:",
        f'   "{_MAX_ITER_CREWAI_MARKER}"',
        "",
        "   👉  Increase the limit by setting this in your .env file:",
        f"       {env_var}=<higher_value>",
    ]

    if parent_error is not None:
        parts.append(f"   (Wrapped exception: {type(parent_error).__name__})")

    return "\n".join(parts)


def _scan_for_stray_generated_files(
    *,
    run_id: str,
    verbose: bool = False,
) -> list[str]:
    """Detect agent-generated files that escaped the ``output/`` tree.

    LLM agents can sometimes write files (via the low-level ``write-file``
    command) to the project root instead of under ``output/<run_id>/``.
    This scanner checks for common generated-file patterns at the project
    root and returns a list of warnings.

    Parameters
    ----------
    run_id : str
        The per-run identifier used for this pipeline execution.
    verbose : bool
        When ``True``, prints the warnings to stderr immediately.

    Returns
    -------
    list[str]
        Human-readable warning strings (one per stray).  Empty if clean.
    """
    warnings_list: list[str] = []
    root = _PROJECT_ROOT

    for entry in sorted(root.iterdir()):
        name = entry.name

        # Skip known project directories and dot-files that are not generated.
        if entry.is_dir() and name in _STRAY_SAFE_DIRS:
            continue
        # Skip hidden files/dirs that aren't in our pattern list.
        if name.startswith(".") and entry.is_dir():
            continue

        matched = False
        for pattern, kind in _STRAY_PATTERNS:
            if re.search(pattern, name):
                matched = True
                msg = (
                    f"⚠️  Stray generated {kind} detected: {entry}\n"
                    f"   This file was likely written by an agent directly to the\n"
                    f"   project root instead of under output/{run_id}/\n"
                    f"   To clean up:  git clean -fd {name}\n"
                    f"   To prevent recurrence: ensure all agents use the\n"
                    f"   'write-labs' command with 'run_id' instead of 'write-file'."
                )
                warnings_list.append(msg)
                if verbose:
                    print(msg, file=sys.stderr)
                break

        if not matched and entry.is_dir():
            # Recurse one level for nested structures like tier2_application/user_data_cli/
            for sub in sorted(entry.iterdir()):
                for pattern, kind in _STRAY_PATTERNS:
                    if re.search(pattern, sub.name):
                        msg = (
                            f"⚠️  Stray generated {kind} detected: {sub}\n"
                            f"   Nested inside stray directory: {entry}\n"
                            f"   To clean up:  git clean -fd {name}/\n"
                            f"   To prevent recurrence: ensure all agents use the\n"
                            f"   'write-labs' command with 'run_id' instead of 'write-file'."
                        )
                        warnings_list.append(msg)
                        if verbose:
                            print(msg, file=sys.stderr)
                        break

    return warnings_list


def _create_lab_scaffolding(labs_base_path: Path) -> Path:
    """Create the tiered lab directory scaffolding under *labs_base_path*."""
    base = labs_base_path
    base.mkdir(parents=True, exist_ok=True)

    for tier_dir_name, tier_label in _TIERS:
        tier_path = base / tier_dir_name
        starter_path = tier_path / "starter"
        solution_path = tier_path / "solution"
        starter_path.mkdir(parents=True, exist_ok=True)
        solution_path.mkdir(parents=True, exist_ok=True)

        (starter_path / ".gitkeep").touch(exist_ok=True)
        (solution_path / ".gitkeep").touch(exist_ok=True)

        tier_readme = tier_path / "README.md"
        if not tier_readme.exists():
            tier_readme.write_text(
                f"# {tier_label}\n\n"
                f"Labs for this tier will be generated here.\n\n"
                f"- **starter/** — Scaffolded exercises with TODO markers.\n"
                f"- **solution/** — Fully-commented reference implementations.\n",
                encoding="utf-8",
            )

    return base


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class CrewResult:
    """Holds the result of a full syllabus + theory + labs crew run."""

    def __init__(
        self,
        syllabus_path: Path,
        labs_base_path: Path,
        syllabus_ok: bool = True,
        labs_ok: bool = True,
        syllabus_error: str | None = None,
        labs_error: str | None = None,
        manifest_path: Path | None = None,
        qa_ok: bool = True,
        qa_error: str | None = None,
        qa_report: str | None = None,
        theory_ok: bool = True,
        theory_error: str | None = None,
        syllabus_review_ok: bool = True,
        syllabus_review_error: str | None = None,
        syllabus_review_report: str | None = None,
        human_feedback_requested: bool = False,
        human_feedback_summary: str | None = None,
    ) -> None:
        self.syllabus_path = syllabus_path
        self.labs_base_path = labs_base_path
        self.syllabus_ok = syllabus_ok
        self.labs_ok = labs_ok
        self.syllabus_error = syllabus_error
        self.labs_error = labs_error
        self.manifest_path = manifest_path
        self.qa_ok = qa_ok
        self.qa_error = qa_error
        self.qa_report = qa_report
        self.theory_ok = theory_ok
        self.theory_error = theory_error
        self.syllabus_review_ok = syllabus_review_ok
        self.syllabus_review_error = syllabus_review_error
        self.syllabus_review_report = syllabus_review_report
        self.human_feedback_requested = human_feedback_requested
        self.human_feedback_summary = human_feedback_summary

    @property
    def all_succeeded(self) -> bool:
        return (
            self.syllabus_ok
            and self.syllabus_review_ok
            and self.theory_ok
            and self.labs_ok
            and self.qa_ok
        )


# ---------------------------------------------------------------------------
# Crew runner
# ---------------------------------------------------------------------------


def generate_run_id(course_safe_name: str) -> str:
    """Generate a unique, human-readable run identifier.

    Format: ``YYYY-MM-DD_HHMMSS_<course_safe_name>``
    """
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")
    return f"{timestamp}_{course_safe_name}"


def _find_syllabus_in_dir(resume_dir: Path) -> Path:
    """Locate the syllabus markdown file inside a resume directory.

    Looks for a ``.md`` file inside ``<resume_dir>/syllabus/``.
    Returns the first match found.

    Raises
    ------
    FileNotFoundError
        If the resume directory or syllabus subdirectory doesn't exist,
        or if no ``.md`` file is found.
    """
    if not resume_dir.exists():
        raise FileNotFoundError(f"Resume directory not found: {resume_dir}")

    if not resume_dir.is_dir():
        raise FileNotFoundError(
            f"Resume path is not a directory (expected a run directory): {resume_dir}"
        )

    syllabus_subdir = resume_dir / "syllabus"
    if not syllabus_subdir.is_dir():
        raise FileNotFoundError(
            f"No 'syllabus/' subdirectory found in resume directory: {resume_dir}"
        )

    md_files = sorted(syllabus_subdir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md syllabus file found in: {syllabus_subdir}")

    return md_files[0]


# ===========================================================================
# Generation State Management — resume/restart progress tracking
# ===========================================================================

_STATE_FILE_NAME: str = "_generation_state.json"


def load_generation_state(run_dir: Path) -> GenerationState | None:
    """Load the generation progress state from a run directory.

    If ``_generation_state.json`` exists, deserialises it.  Otherwise,
    auto-generates a state by scanning the filesystem for existing lab
    and theory files (backward-compatible with pre-state-file runs).

    Parameters
    ----------
    run_dir : Path
        The run output directory (e.g. ``output/2026-09-02_191946_MyCourse/``).

    Returns
    -------
    GenerationState or None
        The loaded or auto-generated state, or ``None`` if the run
        directory does not exist at all.
    """
    if not run_dir.exists():
        return None

    state_path = run_dir / _STATE_FILE_NAME
    course_name = (
        run_dir.name.split("_", 2)[-1] if len(run_dir.name.split("_", 2)) >= 3 else run_dir.name
    )

    if state_path.exists():
        try:
            return GenerationState.model_validate_json(state_path.read_text(encoding="utf-8"))
        except Exception:
            # Corrupt state file — fall back to auto-detection.
            pass

    # ── Auto-generate from filesystem (backward compat) ──────────────
    labs_path = run_dir / "labs"
    state = GenerationState(run_id=run_dir.name, course_name=course_name)

    for tier_dir_name, _label in _TIERS:
        tier_labs_path = labs_path / tier_dir_name

        # Lab files check
        if _is_tier_labs_complete(tier_labs_path):
            state.tiers[tier_dir_name] = TierState(
                status="complete",
                files=_count_lab_files(tier_labs_path),
            )
        else:
            state.tiers[tier_dir_name] = TierState(status="incomplete")

        # Theory files check
        state.theory[tier_dir_name] = (
            "complete" if _is_tier_theory_complete(tier_labs_path) else "incomplete"
        )

    return state


def save_generation_state(run_dir: Path, state: GenerationState) -> None:
    """Persist the generation state to disk atomically.

    Writes to a temp file first, then renames to avoid corruption on
    partial writes.
    """
    state_path = run_dir / _STATE_FILE_NAME
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    tmp_path.rename(state_path)


def _is_tier_labs_complete(tier_labs_path: Path) -> bool:
    """Check whether a tier has real lab files (not just .gitkeep).

    Returns ``True`` when both ``starter/`` and ``solution/`` contain
    at least one file that is not a hidden/dot file.
    """
    for subdir in ("starter", "solution"):
        sub_path = tier_labs_path / subdir
        if not sub_path.exists():
            return False
        real_files = [f for f in sub_path.iterdir() if f.is_file() and not f.name.startswith(".")]
        if not real_files:
            return False
    return True


def _is_tier_theory_complete(tier_labs_path: Path) -> bool:
    """Check whether a tier has theory artifacts.

    Returns ``True`` when ``theory/`` exists and contains at least one
    non-hidden file.
    """
    theory_path = tier_labs_path / "theory"
    if not theory_path.exists():
        return False
    real_files = [f for f in theory_path.iterdir() if f.is_file() and not f.name.startswith(".")]
    return bool(real_files)


def _count_lab_files(tier_labs_path: Path) -> int:
    """Count non-hidden lab files across starter/ and solution/.

    Returns the total number of real (non-``.gitkeep``, non-hidden)
    files found in the tier's lab directories.
    """
    count = 0
    for subdir in ("starter", "solution"):
        sub_path = tier_labs_path / subdir
        if not sub_path.exists():
            continue
        count += sum(1 for f in sub_path.iterdir() if f.is_file() and not f.name.startswith("."))
    return count


def _build_top_level_lab_readme(
    course_name: str,
    primary_language: str,
    labs_base_path: Path,
) -> str:
    """Build a top-level README.md index for all generated labs.

    Scans the lab directory tree and produces a Markdown index with
    a numbered list of every lab, its tier, and a one-line description.
    """
    lines: list[str] = [
        f"# {course_name} — Coding Labs",
        "",
        f"**Primary Language:** {primary_language}",
        "",
        "## Overview",
        "",
        "This directory contains hands-on coding labs organised into three "
        "progressive tiers. Each lab includes a **starter/** scaffold with "
        "TODO markers and a **solution/** reference implementation.",
        "",
        "## Lab Index",
        "",
    ]

    tier_labels = {
        "tier1_foundations": "Tier 1 — Foundations",
        "tier2_application": "Tier 2 — Application",
        "tier3_architecture": "Tier 3 — Architecture",
    }

    lab_num = 0
    for tier_dir_name in ["tier1_foundations", "tier2_application", "tier3_architecture"]:
        tier_path = labs_base_path / tier_dir_name
        if not tier_path.exists():
            continue

        label = tier_labels.get(tier_dir_name, tier_dir_name)
        lines.append(f"### {label}")
        lines.append("")

        # Look for lab files in the solution directory.
        solution_dir = tier_path / "solution"
        if solution_dir.exists():
            for f in sorted(solution_dir.iterdir()):
                if f.name.startswith(".") or f.name == "README.md":
                    continue
                if f.is_file():
                    lab_num += 1
                    # Derive a readable lab name from the filename.
                    lab_name = f.stem.replace("_", " ").replace("-", " ").title()
                    lines.append(
                        f"{lab_num}. **{lab_name}** — "
                        f"`{tier_dir_name}/starter/{f.name}` / "
                        f"`{tier_dir_name}/solution/{f.name}`"
                    )
        lines.append("")

    if lab_num == 0:
        lines.append("*Labs are still being generated. Check back after the pipeline completes.*")

    lines.extend(
        [
            "## Getting Started",
            "",
            "1. Navigate to any lab's `starter/` directory.",
            "2. Read the `README.md` for learning objectives and instructions.",
            "3. Complete the TODO markers in the starter files.",
            "4. Compare your solution with the `solution/` directory.",
            "",
            "## Humanics Literacies",
            "",
            "- **[T] Technological Literacy** — Every lab requires writing, "
            "debugging, and running real code.",
            "- **[D] Data Literacy** — Labs include data processing, analysis, "
            "and evidence-based decision making.",
            "- **[H] Human Literacy** — Every README includes ethics, "
            "accessibility, and collaboration reflection prompts.",
        ]
    )

    return "\n".join(lines) + "\n"


def run_syllabus_crew(
    course_context: str,
    *,
    course_name: str = "",
    primary_language: str = "Python",
    material_language: str = "Dutch",
    verbose: bool = False,
    architect_agent: Agent | None = None,
    lab_dev_agent: Agent | None = None,
    skip_syllabus_review: bool = False,
    skip_theory: bool = False,
    skip_labs: bool = False,
    skip_qa: bool = False,
    resume_dir: str | Path | None = None,
    run_id: str | None = None,
    human_feedback: str | None = None,
) -> CrewResult:
    """Run all agents sequentially and return a full result summary.

    Execution order:
      1. Curriculum Architect generates a syllabus (or loads from disk if
         *resume_dir* is provided).
      2. Education Director audits the syllabus for time-budget math,
         workload realism, scheduling sanity, and MBO4 appropriateness.
         Can delegate fixes back to the Curriculum Architect.
      3. Theory Instructor generates interactive theory artifacts from the
         syllabus.
      4. Lab & Project Developer processes the syllabus to generate tiered labs.
      5. QA Reviewer inspects all generated labs and delegates fixes if needed.

    All output is scoped under ``output/<run_id>/`` where *run_id* is a
    timestamp + course-slug combination, ensuring every pipeline run
    produces a unique, non-overlapping directory.

    Parameters
    ----------
    course_context : str
        Rich course context string (from the Intake Specialist) containing
        tech stack, kerntaken emphasis, student profile, and pedagogical
        notes.  This is the primary input for syllabus generation.
    course_name : str
        Short course name / title used for file naming and directory
        scaffolding.  When empty, extracted from *course_context*.
    primary_language : str
        The exact programming language for coding labs (e.g. 'JavaScript',
        'Python').  Passed through to the Lab Developer task so file
        extensions, linters, and tooling references match the language.
    verbose : bool
        Enable detailed agent and task logging.
    architect_agent : Agent or None
        Pre-built Curriculum Architect agent; lazily created when None.
    lab_dev_agent : Agent or None
        Pre-built Lab Developer agent; lazily created when None.
    skip_syllabus_review : bool
        If True, skip the Education Director feasibility audit step.
    skip_theory : bool
        If True, skip the Theory Instructor step.
    skip_labs : bool
        If True, only run the Curriculum Architect (backward-compatible).
    skip_qa : bool
        If True, skip the QA review step.
    resume_dir : str, Path, or None
        Path to a previous run directory (e.g.
        ``output/2026-08-22_153000_Course_Name``). When provided, the
        Curriculum Architect is **skipped** and the existing syllabus is
        loaded from disk instead. This saves API costs and enables a
        human-in-the-loop workflow where the syllabus can be manually
        edited before generating labs.

    Returns
    -------
    CrewResult
        Container with paths, status flags, and any error messages.
    """
    # Extract course_name from context if not explicitly provided.
    if not course_name:
        # Use the first line or first 80 chars as a fallback name.
        first_line = course_context.strip().split("\n")[0]
        course_name = first_line.replace("Course Name:", "").strip()
        if not course_name or len(course_name) > 100:
            course_name = course_context.strip()[:80]

    safe_name = _sanitize_filename(course_name)

    # ── 0. Handle resume mode ──────────────────────────────────────────
    syllabus_ok = False
    syllabus_error: str | None = None
    syllabus_raw: str = ""
    syllabus_path: Path
    _active_run_id: str  # Always populated below — used for lab task context.

    if resume_dir is not None:
        # ── In-place resume: reuse the SAME run_id and directory ────
        resume_path = Path(resume_dir)
        _active_run_id = resume_path.name
        run_dir = resume_path

        try:
            syllabus_path = _find_syllabus_in_dir(resume_path)
            syllabus_raw = syllabus_path.read_text(encoding="utf-8").strip()

            if not syllabus_raw:
                raise RuntimeError(f"Syllabus file is empty: {syllabus_path}")

            syllabus_ok = True
            if verbose:
                print(f"  📄  Loaded syllabus from: {syllabus_path}")
                print(f"      ({len(syllabus_raw):,} characters)")

        except Exception as exc:
            # Fail loudly instead of fabricating a broken fallback path.  The
            # previous approach (``output/resume_failed/<name>.md``) did not
            # create its parent directory, so the later ``shutil.copy2`` would
            # crash with an unrelated ``FileNotFoundError``.  Surface the real
            # cause with actionable guidance instead.
            raise RuntimeError(
                "Failed to load syllabus from resume directory "
                f"{resume_path}: {exc}\n"
                "  → Pass the run *directory* (e.g. output/<run_id>/) to "
                "--resume-from, not a file such as intake_session.json.  Use "
                "--load-session to reuse a saved intake session."
            ) from exc

        # Use the existing labs directory (files are already in place).
        labs_base_path = resume_path / "labs"
        labs_base_path.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"  📁  Resuming in-place from: {resume_path}")

        # ── In resume mode, create an architect agent so the
        # syllabus review delegation pool has access to it.
        # The architect's LLM is instantiated here but only used
        # if the Education Director delegates back to it.
        if architect_agent is None:
            architect: Agent = get_architect(verbose=verbose)
        else:
            architect = architect_agent

    else:
        # ── Fresh run: create new run directory ────────────────────────
        if run_id is not None:
            # Use the run_id provided by the caller (e.g. main.py for
            # intake-session persistence).  The directory should already
            # exist at this point.
            _active_run_id = run_id
        else:
            _active_run_id = generate_run_id(safe_name)
        run_dir = OUTPUT_ROOT / _active_run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        syllabus_dir = run_dir / "syllabus"
        syllabus_dir.mkdir(parents=True, exist_ok=True)
        syllabus_path = syllabus_dir / f"{safe_name}.md"

        labs_dir = run_dir / "labs"
        labs_dir.mkdir(parents=True, exist_ok=True)
        labs_base_path = labs_dir

        # ── 1. Curriculum Architect ────────────────────────────────────
        if architect_agent is None:
            architect = get_architect(verbose=verbose)
        else:
            architect = architect_agent

        syllabus_task = create_syllabus_generation_task(
            agent=architect,
            course_name=course_name,
            course_context=course_context,
            material_language=material_language,
        )

        try:
            architect_crew = Crew(
                agents=[architect],
                tasks=[syllabus_task],
                process=Process.sequential,
                verbose=verbose,
            )
            architect_result = architect_crew.kickoff()
            syllabus_raw = (
                architect_result.raw if hasattr(architect_result, "raw") else str(architect_result)
            ).strip()

            if not syllabus_raw:
                raise RuntimeError(
                    "Curriculum Architect produced no output. "
                    "Check your API key, model availability, and network."
                )

            write_file(syllabus_path, syllabus_raw, force=True)
            syllabus_ok = True

        except Exception as exc:
            syllabus_error = _annotate_iter_exhaustion(
                str(exc), "CURRICULUM_ARCHITECT", parent_error=exc
            )
            write_file(
                syllabus_path,
                f"# {course_name} — Syllabus Generation Failed\n\n**Error:** {syllabus_error}\n",
                force=True,
            )

    # ── 1.5. Education Director — Syllabus Feasibility Audit ───────────
    syllabus_review_ok = False
    syllabus_review_error: str | None = None
    syllabus_review_report: str | None = None

    if skip_syllabus_review:
        syllabus_review_ok = True
    elif syllabus_raw:
        try:
            education_director = get_education_director(verbose=verbose)
            review_task = create_syllabus_review_task(
                agent=education_director,
                course_name=course_name,
                syllabus_context=syllabus_raw,
                material_language=material_language,
                verbose=verbose,
            )

            # CRITICAL: Both agents must be in the SAME Crew array so
            # the Education Director can delegate fixes back to the
            # Curriculum Architect.
            review_crew = Crew(
                agents=[architect, education_director],
                tasks=[review_task],
                process=Process.sequential,
                verbose=verbose,
            )
            review_result = review_crew.kickoff()
            syllabus_review_report = (
                review_result.raw if hasattr(review_result, "raw") else str(review_result)
            ).strip()

            if syllabus_review_report:
                syllabus_review_ok = True
                if verbose:
                    print("  ✅  Syllabus Feasibility Audit completed.")
            else:
                syllabus_review_error = "Education Director produced no output."

        except Exception as exc:
            syllabus_review_error = _annotate_iter_exhaustion(
                str(exc), "EDUCATION_DIRECTOR", parent_error=exc
            )
            if verbose:
                print(f"  ❌  Syllabus Feasibility Audit failed: {exc}", file=sys.stderr)
    else:
        syllabus_review_error = "Skipped — Curriculum Architect produced no syllabus to audit."

    # ── 2. Theory Instructor (per-tier, with resume detection) ──────────
    theory_ok = False
    theory_error: str | None = None
    theory_instructor: Agent | None = None
    all_theory_tiers_ok = True

    if skip_theory:
        theory_ok = True
    elif syllabus_raw:
        # ── Load generation state for resume detection ────────────────
        state = load_generation_state(run_dir)

        for tier_dir_name, tier_label in _TIERS:
            # ── Skip completed theory ──────────────────────────────────
            if state:
                tier_theory_status = state.theory.get(tier_dir_name)
                if tier_theory_status == "complete":
                    if verbose:
                        print(f"  ⏭️  Theory for {tier_dir_name}: already complete, skipping.")
                    continue

            # Also check filesystem (backward compat / first resume)
            tier_labs_path = labs_base_path / tier_dir_name
            if _is_tier_theory_complete(tier_labs_path):
                if state:
                    state.theory[tier_dir_name] = "complete"
                    save_generation_state(run_dir, state)
                if verbose:
                    print(f"  ⏭️  Theory for {tier_dir_name}: already complete, skipping.")
                continue

            # ── Generate theory for this tier ─────────────────────────
            try:
                if theory_instructor is None:
                    theory_instructor = get_theory_instructor(verbose=verbose)

                tier_theory_task = create_theory_task(
                    agent=theory_instructor,
                    course_name=course_name,
                    syllabus_context=syllabus_raw,
                    run_id=_active_run_id,
                    tier=tier_dir_name,
                    material_language=material_language,
                    verbose=verbose,
                )

                tier_theory_crew = Crew(
                    agents=[theory_instructor],
                    tasks=[tier_theory_task],
                    process=Process.sequential,
                    verbose=verbose,
                )
                tier_theory_crew.kickoff()

                # ── Validate the generated theory file ────────────────
                theory_dir = tier_labs_path / "theory"
                if theory_dir.exists():
                    tier_results = validate_theory_directory(theory_dir)
                    if tier_results:
                        report = format_validation_report(tier_results)
                        report_dir = run_dir / "theory"
                        report_dir.mkdir(parents=True, exist_ok=True)
                        report_path = report_dir / f"VALIDATION_REPORT_{tier_dir_name}.md"
                        report_path.write_text(report, encoding="utf-8")

                        tier_has_errors = any(r.error_count > 0 for r in tier_results)
                        if tier_has_errors:
                            all_theory_tiers_ok = False
                            if verbose:
                                for r in tier_results:
                                    if r.error_count > 0:
                                        print(
                                            f"  ⚠️  Theory validation: "
                                            f"{r.file_path.name} has "
                                            f"{r.error_count} error(s)"
                                        )
                            if state:
                                state.theory[tier_dir_name] = "failed"
                                save_generation_state(run_dir, state)
                            continue

                if state:
                    state.theory[tier_dir_name] = "complete"
                    save_generation_state(run_dir, state)
                if verbose:
                    print(f"  ✅  Theory for {tier_dir_name}: generated successfully.")

            except Exception as exc:
                all_theory_tiers_ok = False
                if state:
                    state.theory[tier_dir_name] = "failed"
                    save_generation_state(run_dir, state)
                exc_msg = _annotate_iter_exhaustion(str(exc), "THEORY_INSTRUCTOR", parent_error=exc)
                if verbose:
                    print(
                        f"  ❌  Theory for {tier_dir_name} failed: {exc_msg}",
                        file=sys.stderr,
                    )

        theory_ok = all_theory_tiers_ok
        if not theory_ok:
            theory_error = "One or more theory tiers failed. See logs above for details."

    else:
        theory_error = "Skipped — Curriculum Architect produced no syllabus to use as context."

    # ── 3. Lab & Project Developer ─────────────────────────────────────
    labs_ok = False
    labs_error: str | None = None

    if skip_labs:
        _create_lab_scaffolding(labs_base_path)
        labs_ok = True
    else:
        _create_lab_scaffolding(labs_base_path)
        lab_dev = None

        try:
            if lab_dev_agent is not None:
                lab_dev = lab_dev_agent
            else:
                lab_dev = get_lab_developer(verbose=verbose)
        except RuntimeError as exc:
            labs_error = str(exc)

        if lab_dev is not None and syllabus_raw:
            # Run three separate per-tier tasks to keep prompt sizes
            # manageable for the LLM.  Each task focuses on a single
            # tier and writes its files via the output_export_tool.

            def _generate_tier(tier_name: str) -> bool:
                tier_task = create_lab_generation_task(
                    agent=lab_dev,
                    course_name=course_name,
                    syllabus_context=syllabus_raw,
                    language=primary_language,
                    run_id=_active_run_id,
                    tier=tier_name,
                    material_language=material_language,
                    verbose=verbose,
                )

                try:
                    tier_crew = Crew(
                        agents=[lab_dev],
                        tasks=[tier_task],
                        process=Process.sequential,
                        verbose=verbose,
                    )
                    tier_result = tier_crew.kickoff()
                    tier_raw = (
                        tier_result.raw if hasattr(tier_result, "raw") else str(tier_result)
                    ).strip()

                    if not tier_raw:
                        print(f"  ⚠️  {tier_name}: produced no output.")
                        return False

                    # Write the tier-level README (summary).
                    tier_readme = labs_base_path / tier_name / "README.md"
                    write_file(tier_readme, tier_raw, force=True)

                    # Verify actual lab files were written, not just text output.
                    starter_glob = list((labs_base_path / tier_name).rglob("starter/**/*"))
                    solution_glob = list((labs_base_path / tier_name).rglob("solution/**/*"))
                    real_files = [
                        f
                        for f in starter_glob + solution_glob
                        if f.is_file() and f.suffix != ".gitkeep" and f.name != ".gitkeep"
                    ]
                    if not real_files:
                        print(
                            f"  ⚠️  {tier_name}: agent produced text but wrote no "
                            f"lab files to starter/ or solution/."
                        )
                        return False

                    print(f"  ✅  {tier_name}: generated successfully ({len(real_files)} files).")
                    return True

                except Exception as exc:
                    print(f"  ❌  {tier_name}: {exc}")
                    return False

            tiers = [
                "tier1_foundations",
                "tier2_application",
                "tier3_architecture",
            ]
            all_tier_ok = True
            # ── Load state for resume detection ────────────────────────
            lab_state = load_generation_state(run_dir)

            # Sequential execution avoids race conditions on the shared
            # Agent singleton (the LLM client and iteration tracker are
            # not thread-safe).
            for tier_name in tiers:
                # ── Skip completed tiers ──────────────────────────────
                if lab_state:
                    tier_state = lab_state.tiers.get(tier_name)
                    if tier_state and tier_state.status == "complete":
                        if verbose:
                            print(
                                f"  ⏭️  {tier_name}: already complete "
                                f"({tier_state.files} files), skipping."
                            )
                        continue

                # Also check filesystem (backward compat)
                tier_labs_path = labs_base_path / tier_name
                if _is_tier_labs_complete(tier_labs_path):
                    if lab_state:
                        lab_state.tiers[tier_name] = TierState(
                            status="complete",
                            files=_count_lab_files(tier_labs_path),
                        )
                        save_generation_state(run_dir, lab_state)
                    if verbose:
                        print(f"  ⏭️  {tier_name}: already complete (filesystem check), skipping.")
                    continue

                if not _generate_tier(tier_name):
                    all_tier_ok = False
                    if lab_state:
                        lab_state.tiers[tier_name] = TierState(
                            status="failed",
                        )
                        save_generation_state(run_dir, lab_state)
                elif lab_state:
                    lab_state.tiers[tier_name] = TierState(
                        status="complete",
                        files=_count_lab_files(tier_labs_path),
                    )
                    save_generation_state(run_dir, lab_state)

            if all_tier_ok:
                # Write a top-level index README.
                top_readme = _build_top_level_lab_readme(
                    course_name, primary_language, labs_base_path
                )
                write_file(labs_base_path / "README.md", top_readme, force=True)
                labs_ok = True
            else:
                labs_error = "One or more tier lab tasks failed.  Check the per-tier output above."
        elif not syllabus_raw:
            labs_error = "Skipped — Curriculum Architect produced no syllabus to use as context."

        if labs_error and not (labs_base_path / "README.md").exists():
            write_file(
                labs_base_path / "README.md",
                f"# {course_name} — Lab Generation Failed\n\n**Error:** {labs_error}\n",
                force=True,
            )

    # ── 4. QA Reviewer ─────────────────────────────────────────────────
    qa_ok = False
    qa_error: str | None = None
    qa_report: str | None = None

    # QA runs when either labs or theory were generated (or both).
    # Skip only if explicitly disabled or nothing was produced.
    if skip_qa or (not labs_ok and not theory_ok):
        qa_ok = True  # Nothing to review, or explicitly skipped.
    elif lab_dev is not None or theory_instructor is not None:
        try:
            qa_reviewer = get_qa_reviewer(verbose=verbose)
            qa_task = create_qa_review_task(
                agent=qa_reviewer,
                course_name=course_name,
                run_id=_active_run_id,
                material_language=material_language,
                lab_developer_role=lab_dev.role if lab_dev is not None else None,
                theory_instructor_role=(
                    theory_instructor.role if theory_instructor is not None else None
                ),
                verbose=verbose,
            )

            # CRITICAL: All agents that may receive delegation MUST be in
            # the SAME Crew array.  The QA Reviewer delegates lab fixes to
            # the Lab Developer and theory fixes to the Theory Instructor.
            qa_agents: list[Agent] = []
            if lab_dev is not None:
                qa_agents.append(lab_dev)
            if theory_instructor is not None:
                qa_agents.append(theory_instructor)
            qa_agents.append(qa_reviewer)

            qa_crew = Crew(
                agents=qa_agents,
                tasks=[qa_task],
                process=Process.sequential,
                verbose=verbose,
            )
            qa_result = qa_crew.kickoff()
            qa_report = (qa_result.raw if hasattr(qa_result, "raw") else str(qa_result)).strip()

            if qa_report:
                qa_ok = True
                if verbose:
                    print("  ✅  QA Review completed.")

                # ── Mark QA as complete in the state file.  Load
                # the existing state file directly (NOT re-scanning the
                # filesystem) to preserve tier statuses set by the
                # pipeline — auto-detection would incorrectly mark
                # QA-found broken files as "complete".
                try:
                    state_path = run_dir / _STATE_FILE_NAME
                    if state_path.exists():
                        existing = GenerationState.model_validate_json(
                            state_path.read_text(encoding="utf-8")
                        )
                        existing.qa_review = "complete"
                        save_generation_state(run_dir, existing)
                except Exception:
                    pass  # Non-critical — state file is best-effort
            else:
                qa_error = _annotate_iter_exhaustion(
                    "QA Reviewer produced no output.",
                    "QA_REVIEWER",
                    parent_error=None,
                )

        except Exception as exc:
            qa_error = _annotate_iter_exhaustion(
                str(exc),
                "QA_REVIEWER",
                parent_error=exc,
            )
            if verbose:
                print(f"  ❌  QA Review failed: {exc}", file=sys.stderr)

    # ── 5. Post-run sanity: scan for stray generated files ─────────────
    strays = _scan_for_stray_generated_files(run_id=run_id, verbose=verbose)
    if strays and verbose:
        print(f"\n  🔍  Stray file scan: {len(strays)} issue(s) found above.", file=sys.stderr)

    # ── 6. Generate output manifest ────────────────────────────────────
    try:
        manifest_path = update_output_manifest(
            course_name,
            syllabus_path=syllabus_path,
            labs_base_path=labs_base_path,
        )
    except Exception as exc:
        if verbose:
            print(f"  [Warning] Manifest generation failed: {exc}", file=sys.stderr)
        manifest_path = None

    # ── 7. Return combined result ──────────────────────────────────────
    return CrewResult(
        syllabus_path=syllabus_path,
        labs_base_path=labs_base_path,
        syllabus_ok=syllabus_ok,
        labs_ok=labs_ok,
        syllabus_error=syllabus_error,
        labs_error=labs_error,
        manifest_path=manifest_path,
        qa_ok=qa_ok,
        qa_error=qa_error,
        qa_report=qa_report,
        theory_ok=theory_ok,
        theory_error=theory_error,
        syllabus_review_ok=syllabus_review_ok,
        syllabus_review_error=syllabus_review_error,
        syllabus_review_report=syllabus_review_report,
    )
