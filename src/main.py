#!/usr/bin/env python3
"""
main.py — Syllabus Swarm CLI Entry Point
=========================================

Issue #2: Core Agent — The Curriculum Architect (Humanics Alignment)
Issue #3: Core Agent — The Lab & Project Developer (Tiered Coding Challenges)

Orchestrates the full syllabus-and-labs generation pipeline:

  0. Runs the **Intake Specialist** agent to interview the user and gather
     rich course context mapped to Dutch SBB Kwalificatiedossiers.
  1. Runs the **Curriculum Architect** agent to generate a Humanics-aligned
     Markdown syllabus saved to ``output/syllabus/<course_name>.md``.
  2. Runs the **Lab & Project Developer** agent using the syllabus as
     context to generate tiered coding labs saved under
     ``output/labs/<course_name>/``.
  3. Prints a clear success/failure summary for all agents.

Usage
-----
  python src/main.py "Data Science with Python"
  python -m src.main "Full-Stack Web Development"
  python src/main.py             # interactive prompt
  python src/main.py "ML Basics" --skip-labs  # syllabus only
  python src/main.py "ML Basics" --resume-from output/2026-08-22_153000_ML_Basics
  python src/main.py "ML Basics" --load-session output/2026-08-22_153000_ML_Basics/intake_session.json

Environment
-----------
  Requires OPENROUTER_API_KEY in .env (copy from .env.example).
"""

from __future__ import annotations

import argparse
import json
import sys

# The project targets Python 3.12+ and relies on stdlib features that did not
# exist in earlier releases (e.g. ``datetime.UTC``, PEP 604 unions).  Fail fast
# with a clear message instead of an opaque ``ImportError`` when an older
# interpreter is used.  This check MUST precede the ``from datetime import UTC``
# import below, otherwise Python 3.9 would raise that ImportError first.
if sys.version_info < (3, 12):
    raise SystemExit(
        "syllabus-swarm requires Python 3.12+ "
        f"(running {'.'.join(map(str, sys.version_info[:3]))}).\n"
        "Activate the project .venv or run with `python3.12`."
    )

from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that `from src.…` imports work
# regardless of whether the script is invoked as `python src/main.py` or
# `python -m src.main`.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml
from dotenv import load_dotenv

# Load environment variables *before* any internal imports that read them.
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)

from crewai import Crew, Process, Task

from src.agents.intake_specialist import get_intake_specialist
from src.crews.syllabus_crew import (
    OUTPUT_ROOT,
    CrewResult,
    generate_run_id,
    run_syllabus_crew,
)
from src.exporters.file_writer import _sanitize_filename
from src.models import CourseSpecification, IntakeSession

# ---------------------------------------------------------------------------
# CLI — Argument Parser
# ---------------------------------------------------------------------------


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the syllabus-swarm CLI.

    Returns
    -------
    argparse.ArgumentParser
        A fully-configured parser with all flags and the optional
        ``course_name`` positional argument.
    """
    parser = argparse.ArgumentParser(
        prog="syllabus-swarm",
        description=(
            "🐝  Syllabus Swarm — Multi-agent curriculum generation pipeline.  "
            "Orchestrates the Intake Specialist, Curriculum Architect, "
            "Lab & Project Developer, and Output Exporter agents to produce "
            "complete course materials from a single command."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  %(prog)s "Data Science with Python"\n'
            '  %(prog)s "Laravel Web Development" --profile config/profiles/program1_profile.yaml\n'
            '  %(prog)s "Advanced PHP" --load-session output/2026-08-22_153000_ML_Basics/intake_session.json\n'
            '  %(prog)s "Period 3 Project" --builds-upon 2026-08-22_153000_ML_Basics\n'
            '  %(prog)s "ML Basics" --resume-from output/2026-08-22_153000_ML_Basics\n'
            '  %(prog)s "Full-Stack Web Development" --skip-labs\n'
            "  %(prog)s  # interactive prompt\n"
        ),
    )

    parser.add_argument(
        "course_name",
        nargs="*",
        help=(
            'Course name / topic (e.g. "Data Science with Python").  '
            "If omitted, prompts interactively."
        ),
    )

    parser.add_argument(
        "--skip-labs",
        action="store_true",
        default=False,
        help="Generate syllabus only — skip lab generation.",
    )

    parser.add_argument(
        "--profile",
        default=None,
        metavar="PATH",
        help=(
            "Load a YAML cohort profile to pre-populate static constraints "
            "(grading scale, student pathway, year level, hardware).  "
            "The Intake Specialist skips questions for pre-populated fields.  "
            "Example: config/profiles/program1_profile.yaml"
        ),
    )

    parser.add_argument(
        "--load-session",
        default=None,
        metavar="PATH",
        help=(
            "Load a saved intake session JSON to skip the interactive "
            "interview entirely.  Example: "
            "output/2026-08-22_153000_ML_Basics/intake_session.json"
        ),
    )

    parser.add_argument(
        "--resume-from",
        default=None,
        metavar="DIR",
        help=(
            "Resume from a previous run directory (skips intake, re-runs "
            "agents).  Example: output/2026-08-22_153000_ML_Basics"
        ),
    )

    parser.add_argument(
        "--builds-upon",
        default=None,
        metavar="SLUG",
        help=(
            "Inject prerequisites from a previous course's output.  "
            "Reads the previous course's learning objectives and key "
            "concepts and passes them as prerequisites to the Curriculum "
            "Architect.  Example: 2026-08-22_153000_ML_Basics"
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# CLI helpers (post-argparse)
# ---------------------------------------------------------------------------


def _gather_course_name(parsed_name: str | None = None) -> str:
    """Return the course name from parsed args or interactive input.

    Parameters
    ----------
    parsed_name : str or None
        The course name extracted from positional CLI arguments by
        argparse.  When ``None`` or empty, prompts interactively.
    """
    if parsed_name:
        return parsed_name.strip()

    try:
        return input("📘 Enter course name / topic: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n⚠️  No input provided. Exiting.", file=sys.stderr)
        sys.exit(1)


def _validate_name(name: str) -> str:
    """Ensure the course name is non-empty and reasonable."""
    if not name:
        print("❌ Course name cannot be empty.", file=sys.stderr)
        sys.exit(1)
    if len(name) > 200:
        print("❌ Course name is too long (max 200 characters).", file=sys.stderr)
        sys.exit(1)
    return name


def _validate_resume_dir(resume_dir: str) -> Path:
    """Validate a ``--resume-from`` path and return it as a resolved ``Path``.

    ``--resume-from`` must point at a previous **run directory** (e.g.
    ``output/2026-08-27_091745_Javascript_OOP``) containing a ``syllabus/``
    subdirectory with at least one ``.md`` file.  Failing fast here — before
    the crew is built — prevents the confusing cascade that otherwise occurs
    when a file (such as ``intake_session.json``) is passed by mistake.

    Parameters
    ----------
    resume_dir : str
        The value of the ``--resume-from`` CLI flag.

    Returns
    -------
    Path
        The resolved, validated run directory.

    Raises
    ------
    SystemExit
        If the path does not exist, is not a directory, or lacks a usable
        ``syllabus/`` subdirectory.
    """
    resume_path = Path(resume_dir)
    if not resume_path.is_absolute():
        resume_path = _PROJECT_ROOT / resume_path

    if not resume_path.exists():
        print(
            f"❌ --resume-from directory not found: {resume_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not resume_path.is_dir():
        hint = ""
        if resume_path.suffix.lower() == ".json":
            hint = (
                "\n   → This looks like a session file. Use --load-session for a "
                "saved intake session, or pass the run *directory* (e.g. "
                "output/<run_id>/) for --resume-from."
            )
        print(
            f"❌ --resume-from must be a run directory, not a file: {resume_path}{hint}",
            file=sys.stderr,
        )
        sys.exit(1)

    syllabus_subdir = resume_path / "syllabus"
    if not syllabus_subdir.is_dir() or not list(syllabus_subdir.glob("*.md")):
        print(
            f"❌ No 'syllabus/' subdirectory with a .md file found in run directory: {resume_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    return resume_path


# ---------------------------------------------------------------------------
# Profile loading — delegated to Layer 1 (src.config_loader)
# ---------------------------------------------------------------------------
from src.config_loader import (
    _build_profile_context_string as _build_profile_context_impl,
    get_pre_populated_fields as _get_pre_populated_fields_impl,
    inject_profile as _inject_profile_impl,
    load_profile as _load_profile_impl,
)


def _load_profile(path: str) -> dict:
    """Load and validate a YAML cohort profile file (→ Layer 1)."""
    try:
        return _load_profile_impl(path)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"❌ {exc}", file=sys.stderr)
        sys.exit(1)


def _build_profile_context_string(profile: dict) -> str:
    """Serialize all profile sections (→ Layer 1)."""
    return _build_profile_context_impl(profile)


def _inject_profile(spec: CourseSpecification, profile: dict) -> CourseSpecification:
    """Inject pre-populated profile values (→ Layer 1)."""
    return _inject_profile_impl(spec, profile)


def _get_pre_populated_fields(spec: CourseSpecification) -> list[str]:
    """Return pre-populated field names (→ Layer 1)."""
    return _get_pre_populated_fields_impl(spec)


# ---------------------------------------------------------------------------
# Module chaining — resolve prerequisites from a previous course
# ---------------------------------------------------------------------------


def _resolve_prerequisites(builds_upon_slug: str) -> str:
    """Resolve prerequisite knowledge from a previous course's output.

    Searches ``output/`` for a run directory whose name contains
    *builds_upon_slug*, reads its ``course_graph.json``, and extracts
    the ``learning_objectives`` and ``key_concepts`` into a formatted
    string suitable for injection into the course context.

    Parameters
    ----------
    builds_upon_slug : str
        A slug fragment to match against run directory names (e.g.
        ``"ML_Basics"`` matches ``2026-08-22_153000_ML_Basics``).

    Returns
    -------
    str
        A formatted string describing what students have already mastered,
        ready to prepend to the course context.

    Raises
    ------
    SystemExit
        If no matching run directory or ``course_graph.json`` is found.
    """
    from src.models import CourseGraph

    output_dir = OUTPUT_ROOT
    if not output_dir.exists():
        print(
            f"❌ Output directory not found: {output_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Find run directories whose name contains the slug.
    matching_dirs = sorted(
        d for d in output_dir.iterdir() if d.is_dir() and builds_upon_slug in d.name
    )

    if not matching_dirs:
        print(
            f"❌ No run directory found matching slug '{builds_upon_slug}' in {output_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Use the first match (most recent when sorted alphabetically, which
    # works because run IDs start with YYYY-MM-DD_HHMMSS).
    run_dir = matching_dirs[0]
    graph_path = run_dir / "course_graph.json"

    if not graph_path.exists():
        print(
            f"❌ No course_graph.json found in {run_dir}.  "
            f"Ensure the previous course was fully generated (course_graph.json "
            f"is auto-generated by the Output Exporter).",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        raw_json = graph_path.read_text(encoding="utf-8")
        graph = CourseGraph.model_validate_json(raw_json)
    except Exception as exc:
        print(
            f"❌ Failed to parse course_graph.json from {graph_path}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build the prerequisite context string.
    parts: list[str] = [
        "The students have already mastered the following from a previous course:\n",
    ]

    if graph.learning_objectives:
        parts.append("**Learning Objectives:**")
        for obj in graph.learning_objectives:
            parts.append(f"- {obj}")
        parts.append("")

    if graph.key_concepts:
        parts.append("**Key Concepts:**")
        parts.append("- " + ", ".join(graph.key_concepts))
        parts.append("")

    if not graph.learning_objectives and not graph.key_concepts:
        parts.append(
            "(No learning objectives or key concepts were recorded in the previous course graph.)\n"
        )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Intake Specialist — interactive interview loop
# ---------------------------------------------------------------------------


def _run_intake(
    course_name: str,
    *,
    verbose: bool = False,
    pre_populated: CourseSpecification | None = None,
    prerequisites: str | None = None,
) -> tuple[str, str, str, str, str | None, str | None, int | None, str | None]:
    """Run the Intake Specialist to gather rich course context.

    Steps:
      1. Send the course name to the Intake Specialist, who replies with
         2-3 targeted clarifying questions.
      2. Display the questions and capture the user's multi-line response.
      3. Send the course name + user answers back to the Intake Specialist
         for synthesis into a ``CourseSpecification`` structured output.

    Parameters
    ----------
    course_name : str
        The initial course name / topic from the user.
    verbose : bool
        Enable detailed agent logging.
    pre_populated : CourseSpecification or None
        Pre-populated constraints from a cohort profile.  When provided,
        structured fields (grading_scale, student_pathway, year_level,
        hardware_constraints) are used as fallback values if the LLM
        synthesis does not populate them.
    prerequisites : str or None
        Optional prerequisite context from a previous course (via
        ``--builds-upon``).  When provided, injected into the question
        prompt so the Intake Specialist knows what students have already
        mastered and can tailor questions accordingly.

    Returns
    -------
    tuple[str, str, str, str, str | None, str | None, int | None, str | None]
        A ``(course_context, primary_language, questions, answers,
        grading_scale, student_pathway, year_level, hardware_constraints)``
        tuple.  *questions* and *answers* are empty strings when the
        intake failed (e.g. LLM error, no user input).
    """
    intake_agent = get_intake_specialist(verbose=verbose)

    # ── Build prerequisite context for the question prompt ─────────────
    prereq_section = ""
    if prerequisites:
        prereq_section = (
            f"\n**📎  Prerequisite Knowledge (from a previous course):**\n"
            f"{prerequisites}\n"
            f"Take this into account when formulating your questions — "
            f"do NOT ask about topics the students have already mastered.\n"
        )

    # ── Build profile context for the question prompt ──────────────────
    skip_section = ""
    if pre_populated is not None:
        filled = _get_pre_populated_fields(pre_populated)
        if filled:
            skip_notes: list[str] = []
            labels = {
                "grading_scale": "grading scale",
                "student_pathway": "student pathway (BOL/BBL)",
                "year_level": "year level",
                "hardware_constraints": "hardware constraints",
            }
            for field in filled:
                label = labels.get(field, field)
                val = getattr(pre_populated, field)
                skip_notes.append(f"     • {label}: **{val}** (pre-populated — skip)")
            if skip_notes:
                skip_section = (
                    "\n**⚠️  The following fields are ALREADY KNOWN from the "
                    "cohort profile — DO NOT ask about them:**\n" + "\n".join(skip_notes)
                )
        # Also note tech stack if primary_language is pre-populated
        if pre_populated.primary_language and pre_populated.primary_language not in ("", "Python"):
            skip_section += (
                f"\n     • primary language: **{pre_populated.primary_language}** "
                f"(pre-populated — skip)\n"
            )

    # ── Step 1: Ask clarifying questions ────────────────────────────────
    question_task = Task(
        description=(
            f"The user wants a syllabus for the following course:\n\n"
            f"**Course Name:** {course_name}\n"
            f"{prereq_section}\n"
            f"Your task: Ask 3-4 concise, targeted clarifying questions "
            f"about:\n"
            f"1. The tech stack and tooling (languages, frameworks, "
            f"platforms, version control, deployment tools)\n"
            f"2. Which kerntaken to emphasise: planning (P1-K1), designing "
            f"(P2-K1), building (P3-K1), and/or testing (P4-K1) software\n"
            f"3. The student profile: BOL or BBL pathway, year level "
            f"(1, 2, or 3), and BPV (internship) readiness\n"
            f"4. The specific schedule and time budget (number of weeks, "
            f"contact hours per week, self-study hours, and any known "
            f"disruptions like holidays or school trips)\n\n"
            f"{skip_section}\n"
            f"Output ONLY the questions — no preamble, no commentary, "
            f"no markdown formatting.  Number them 1, 2, 3, 4.  Keep each "
            f"question to one or two sentences."
        ),
        expected_output=(
            "2-3 numbered clarifying questions about tech stack, "
            "kerntaken focus, and student profile.  No preamble or "
            "commentary — just the questions."
        ),
        agent=intake_agent,
        async_execution=False,
    )

    print("\n🐝  Consulting the Intake Specialist...\n")

    try:
        question_crew = Crew(
            agents=[intake_agent],
            tasks=[question_task],
            process=Process.sequential,
            verbose=verbose,
        )
        question_result = question_crew.kickoff()
        questions = (
            question_result.raw if hasattr(question_result, "raw") else str(question_result)
        ).strip()
    except Exception as exc:
        print(f"\n⚠️  Intake Specialist failed to generate questions: {exc}")
        print("   Proceeding with bare course name as context.\n")
        return f"Course Name: {course_name}", "Python", "", "", None, None, None, None

    # ── Step 2: Display questions and capture answers ────────────────────
    if pre_populated is not None and _get_pre_populated_fields(pre_populated):
        print("📋  Pre-populated fields from profile:")
        for field in _get_pre_populated_fields(pre_populated):
            print(f"    ✓ {field}: {getattr(pre_populated, field)}")
        print()

    print(f"{'─' * 60}")
    print(questions)
    print(f"{'─' * 60}")
    print()
    print("📝  Please type your answers below.")
    print("    Press Enter twice (blank line) when finished.\n")

    user_answers_lines: list[str] = []
    try:
        while True:
            line = input()
            if line.strip() == "":
                if user_answers_lines and user_answers_lines[-1] == "":
                    # Two blank lines in a row → done
                    user_answers_lines.pop()  # remove the trailing blank
                    break
            user_answers_lines.append(line)
    except (EOFError, KeyboardInterrupt):
        print("\n⚠️  Input interrupted.", file=sys.stderr)

    user_answers = "\n".join(user_answers_lines).strip()

    if not user_answers:
        print("\n⚠️  No answers provided. Proceeding with bare course name.\n")
        return f"Course Name: {course_name}", "Python", "", "", None, None, None, None

    # ── Step 3: Synthesise course context ────────────────────────────────
    # Include pre-populated fields AND the full profile context in the
    # synthesis prompt so the LLM incorporates schedule, assessment,
    # kerntaken emphasis, and tech stack into the course_context.
    pre_pop_bonus = ""
    if pre_populated is not None:
        bonus_parts: list[str] = []

        # Direct fields (grading_scale, student_pathway, etc.)
        filled = _get_pre_populated_fields(pre_populated)
        if filled:
            field_lines: list[str] = []
            for field in filled:
                field_lines.append(f"   - {field}: {getattr(pre_populated, field)}")
            bonus_parts.append(
                "**Pre-populated constraints from cohort profile "
                "(already known — incorporate into course_context):**\n" + "\n".join(field_lines)
            )

        # Full profile context (schedule, assessment, kerntaken, tech stack)
        if pre_populated.course_context:
            bonus_parts.append(
                "**Additional profile data (schedule, assessment, "
                "kerntaken emphasis, tech stack) — incorporate ALL of "
                "this into the course_context:**\n" + pre_populated.course_context
            )

        if bonus_parts:
            pre_pop_bonus = "\n\n" + "\n\n".join(bonus_parts)

    synthesis_task = Task(
        description=(
            f"You interviewed the user about their course and received "
            f"the following information.\n\n"
            f"**Course Name:** {course_name}\n\n"
            f"**Your Questions:**\n{questions}\n\n"
            f"**User's Answers:**\n{user_answers}"
            f"{pre_pop_bonus}\n\n"
            f"Your task: Synthesise the course name, your questions, and "
            f"the user's answers into a structured ``CourseSpecification`` "
            f"object with two fields:\n\n"
            f"1. ``course_context`` — A rich, structured text block "
            f"(plain text, NOT Markdown) containing:\n"
            f"   - The course name and a one-sentence summary\n"
            f"   - Tech stack and tooling details\n"
            f"   - Kerntaken emphasis (which of P1-K1 through P4-K1 to "
            f"focus on)\n"
            f"   - Student profile (BOL/BBL, year level, BPV readiness)\n"
            f"   - Specific schedule and time budget (weeks, contact hours, "
            f"self-study hours, known disruptions like holidays or trips)\n"
            f"   - Any additional pedagogical notes or constraints\n\n"
            f"2. ``primary_language`` — The EXACT programming language "
            f"that will be used for coding labs (e.g., 'JavaScript', "
            f"'Python', 'TypeScript', 'Java', 'Go', 'Rust').  This MUST "
            f"be a single language name, not a list or description.\n\n"
            f"The course_context will be passed directly to the "
            f"Curriculum Architect agent who will design the syllabus.  "
            f"The primary_language will determine the file extensions, "
            f"linters, and tooling used in the coding labs."
        ),
        expected_output=(
            "A CourseSpecification object with two fields: "
            "course_context (rich pedagogical context as plain text) and "
            "primary_language (the exact programming language for labs, "
            "e.g. 'JavaScript', 'Python')."
        ),
        output_pydantic=CourseSpecification,
        agent=intake_agent,
        async_execution=False,
    )

    print("\n🐝  Synthesising course context...\n")

    try:
        synthesis_crew = Crew(
            agents=[intake_agent],
            tasks=[synthesis_task],
            process=Process.sequential,
            verbose=verbose,
        )
        synthesis_result = synthesis_crew.kickoff()

        # Extract the typed Pydantic model from the result.
        if hasattr(synthesis_result, "pydantic") and synthesis_result.pydantic is not None:
            spec: CourseSpecification = synthesis_result.pydantic
            course_context = spec.course_context.strip()
            primary_language = spec.primary_language.strip()
            grading_scale = spec.grading_scale
            student_pathway = spec.student_pathway
            year_level = spec.year_level
            hardware_constraints = spec.hardware_constraints
        else:
            # Fallback: parse from raw text if pydantic output unavailable.
            course_context = (
                synthesis_result.raw if hasattr(synthesis_result, "raw") else str(synthesis_result)
            ).strip()
            primary_language = "Python"
            grading_scale = None
            student_pathway = None
            year_level = None
            hardware_constraints = None

        # Fall back to pre-populated values when the LLM didn't populate them.
        if pre_populated is not None:
            if grading_scale is None:
                grading_scale = pre_populated.grading_scale
            if student_pathway is None:
                student_pathway = pre_populated.student_pathway
            if year_level is None:
                year_level = pre_populated.year_level
            if hardware_constraints is None:
                hardware_constraints = pre_populated.hardware_constraints

        if not course_context:
            print("⚠️  Synthesis produced no output. Using bare course name.\n")
            return f"Course Name: {course_name}", "Python", "", "", None, None, None, None

        print(f"{'─' * 60}")
        print("📋  Course Context (sent to Curriculum Architect):")
        print(f"{'─' * 60}")
        print(course_context)
        print(f"   Primary Language: {primary_language}")
        print(f"{'─' * 60}\n")

        return (
            course_context,
            primary_language,
            questions,
            user_answers,
            grading_scale,
            student_pathway,
            year_level,
            hardware_constraints,
        )

    except Exception as exc:
        print(f"\n⚠️  Synthesis failed: {exc}")
        print("   Proceeding with bare course name as context.\n")
        return f"Course Name: {course_name}", "Python", "", "", None, None, None, None


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def _print_summary(result: CrewResult, course_name: str) -> None:
    """Print a clear success/failure summary for both agents."""
    print(f"\n{'=' * 60}")
    print("  🐝  Syllabus Swarm — Results Summary")
    print(f"  Course: {course_name}")
    print(f"{'=' * 60}")

    # ── Syllabus Agent ──────────────────────
    print()
    if result.syllabus_ok:
        print("  ✅  Curriculum Architect  —  SUCCESS")
        print(f"      📄  Syllabus: {result.syllabus_path}")
        try:
            size = result.syllabus_path.stat().st_size
            print(f"      📏  Size:     {size:,} bytes")
        except OSError:
            pass
    else:
        print("  ❌  Curriculum Architect  —  FAILED")
        if result.syllabus_error:
            print(f"      ↳ {result.syllabus_error}")

    # ── Labs Agent ──────────────────────────
    if result.labs_ok:
        print("  ✅  Lab & Project Developer  —  SUCCESS")
        print(f"      📁  Labs:  {result.labs_base_path}/")
        _print_lab_tree(result.labs_base_path)
    else:
        print("  ❌  Lab & Project Developer  —  FAILED")
        if result.labs_error:
            print(f"      ↳ {result.labs_error}")

    # ── Manifest ────────────────────────────
    print()
    if result.manifest_path and result.manifest_path.exists():
        print(f"  📋  Output Manifest: {result.manifest_path}")
    else:
        print("  ⚠️   Output Manifest not generated.")

    # ── Export summary ──────────────────────
    _print_export_summary(result)

    # ── Overall verdict ─────────────────────
    print()
    if result.all_succeeded:
        print("  🎉  All agents completed successfully!")
    elif result.syllabus_ok:
        print("  ⚠️   Syllabus generated but labs failed.")
    else:
        print("  💥  Both agents failed.")
        print("      Check OPENROUTER_API_KEY and network connectivity.")
    print(f"{'=' * 60}\n")


def _print_lab_tree(base: Path, indent: int = 6) -> None:
    """Print a compact tree view of the lab directory structure."""
    prefix = " " * indent
    try:
        entries = sorted(base.iterdir())
    except OSError:
        return

    for entry in entries:
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            print(f"{prefix}▸ {entry.name}/")
            _print_lab_tree(entry, indent + 3)
        else:
            print(f"{prefix}  {entry.name}")


def _print_export_summary(result: CrewResult) -> None:
    """Print a summary of what was exported and where."""
    print()
    print("  ── What Was Exported ──")

    # Syllabus
    if result.syllabus_path.exists():
        try:
            sz = result.syllabus_path.stat().st_size
            print(f"  📄  Syllabus → {_fmt_size(sz):>10}  {result.syllabus_path}")
        except OSError:
            print(f"  📄  Syllabus →              {result.syllabus_path}")

    # Labs
    if result.labs_base_path.exists():
        file_count = 0
        total_size = 0
        for f in result.labs_base_path.rglob("*"):
            if f.is_file() and not f.name.startswith("."):
                file_count += 1
                try:
                    total_size += f.stat().st_size
                except OSError:
                    pass
        print(
            f"  🧪  Labs     → {_fmt_size(total_size):>10}  "
            f"{result.labs_base_path}/  ({file_count} files)"
        )

    # Manifest
    if result.manifest_path and result.manifest_path.exists():
        try:
            sz = result.manifest_path.stat().st_size
            print(f"  📋  Manifest → {_fmt_size(sz):>10}  {result.manifest_path}")
        except OSError:
            print(f"  📋  Manifest →              {result.manifest_path}")


def _fmt_size(num_bytes: int) -> str:
    """Format bytes as human-readable string."""
    if num_bytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    if idx == 0:
        return f"{int(size)} B"
    return f"{size:.1f} {units[idx]}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Entry point — orchestrate the intake + syllabus + labs pipeline.

    Parameters
    ----------
    argv : list[str] or None
        Command-line arguments.  When ``None``, reads from ``sys.argv``.
        Useful for testing.
    """

    # --- 0. Parse CLI arguments -------------------------------------------
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    skip_labs: bool = args.skip_labs
    resume_dir: str | None = args.resume_from
    profile_path: str | None = args.profile
    load_session_path: str | None = args.load_session
    builds_upon: str | None = args.builds_upon

    # Collapse the positional course_name list into a single string.
    parsed_course_name: str | None = (
        " ".join(args.course_name).strip() if args.course_name else None
    )

    # --- 1. Validate flag combinations ------------------------------------
    if skip_labs and resume_dir:
        print(
            "❌ --skip-labs and --resume-from cannot be used together (this would be a no-op).",
            file=sys.stderr,
        )
        sys.exit(1)

    if load_session_path and resume_dir:
        print(
            "❌ --load-session and --resume-from cannot be used together "
            "(--load-session already provides all context).",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- 1b. Validate --resume-from before any heavy work ------------------
    if resume_dir:
        resume_dir = str(_validate_resume_dir(resume_dir))

    # --- 2. Load profile if provided --------------------------------------
    profile_data: dict | None = None
    pre_populated: CourseSpecification | None = None
    if profile_path:
        profile_data = _load_profile(profile_path)
        profile_name = profile_data.get("profile", {}).get("name", profile_path)
        pre_populated = CourseSpecification(
            course_context="",
            primary_language="",
        )
        pre_populated = _inject_profile(pre_populated, profile_data)
        print(f"📋  Loaded cohort profile: {profile_name}")
        filled = _get_pre_populated_fields(pre_populated)
        if filled:
            print(f"    Pre-populated: {', '.join(filled)}")
        if pre_populated.primary_language and pre_populated.primary_language not in ("", "Python"):
            print(f"    Primary language: {pre_populated.primary_language}")

    # --- 1a. --load-session path (bypass interactive intake) -------------

    if load_session_path:
        session_file = Path(load_session_path)
        if not session_file.exists():
            print(
                f"❌ Session file not found: {session_file}",
                file=sys.stderr,
            )
            sys.exit(1)

        try:
            raw_json = session_file.read_text(encoding="utf-8")
            session = IntakeSession.model_validate_json(raw_json)
        except Exception as exc:
            print(
                f"❌ Failed to load session from {session_file}: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

        course_name = session.course_name
        course_context = session.course_specification.course_context
        primary_language = session.course_specification.primary_language

        print(f"\n{'=' * 60}")
        print("  🐝  Syllabus Swarm")
        print(f"  Course:     {course_name}")
        print("  Model:      Per-agent via OpenRouter (see .env.example)")
        print(f"  Labs:       {'Skip' if skip_labs else 'Generate'}")
        print(f"{'=' * 60}\n")

        print("📋  Loaded intake session:")
        print(f"   Course:     {course_name}")
        print(f"   Run ID:     {session.run_id}")
        print(f"   When:       {session.timestamp}")
        print(f"   Language:   {primary_language}")
        print()

        print("📋  --load-session mode — skipping interactive intake.\n")

        # --- Inject prerequisites from a previous course (--builds-upon) --
        if builds_upon:
            prereq_context = _resolve_prerequisites(builds_upon)
            course_context = prereq_context + "\n\n" + course_context
            print("📎  Injected prerequisites from previous course.\n")

        # Generate a fresh run_id for this pipeline run.
        safe_name = _sanitize_filename(course_name)
        run_id = generate_run_id(safe_name)

        # --- 2. Run the crew ---------------------------------------------
        try:
            result = run_syllabus_crew(
                course_context,
                course_name=course_name,
                primary_language=primary_language,
                verbose=True,
                skip_labs=skip_labs,
                run_id=run_id,
            )
        except RuntimeError as exc:
            print(f"\n❌  Fatal Runtime Error: {exc}", file=sys.stderr)
            print(
                "   → Check that OPENROUTER_API_KEY is set in .env.",
                file=sys.stderr,
            )
            sys.exit(2)
        except Exception as exc:
            print(f"\n❌  Fatal Unexpected Error: {exc}", file=sys.stderr)
            sys.exit(3)

        # --- 3. Print summary --------------------------------------------
        _print_summary(result, course_name)
        return

    # --- 1b. Normal intake flow ------------------------------------------

    course_name = _gather_course_name(parsed_course_name)
    course_name = _validate_name(course_name)

    # Compute run_id *before* the intake so we can save the session
    # inside the same run directory the crew will use later.
    safe_name = _sanitize_filename(course_name)
    run_id = generate_run_id(safe_name)
    run_dir = OUTPUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("  🐝  Syllabus Swarm")
    print(f"  Course:     {course_name}")
    print("  Model:      Per-agent via OpenRouter (see .env.example)")
    if resume_dir:
        print(f"  Resume:     {resume_dir}")
    print(f"  Labs:       {'Skip' if skip_labs else 'Generate'}")
    print(f"{'=' * 60}\n")

    # --- Resolve prerequisites before intake (--builds-upon) -------------
    prereq_context: str | None = None
    if builds_upon:
        prereq_context = _resolve_prerequisites(builds_upon)
        print("📎  Resolved prerequisites from previous course.\n")

    # --- 2. Run Intake Specialist (skip if resuming) ---------------------
    if resume_dir:
        # When resuming, we already have a syllabus — no need for intake.
        course_context = f"Course Name: {course_name}"
        primary_language = "Python"
        questions = ""
        answers = ""

        # Try to extract primary_language from the saved intake session
        # so that lab generation uses the correct file extensions/linters.
        resume_path = Path(resume_dir)
        session_file = resume_path / "intake_session.json"
        if session_file.exists():
            try:
                raw = session_file.read_text(encoding="utf-8")
                session_data = json.loads(raw)
                lang = (
                    session_data.get("course_specification", {}).get("primary_language", "").strip()
                )
                if lang:
                    primary_language = lang
                    course_context = session_data.get("course_specification", {}).get(
                        "course_context", course_context
                    )
            except (json.JSONDecodeError, KeyError, OSError):
                pass

        print("📋  Resume mode — skipping Intake Specialist.\n")
    else:
        (
            course_context,
            primary_language,
            questions,
            answers,
            grading_scale,
            student_pathway,
            year_level,
            hardware_constraints,
        ) = _run_intake(
            course_name,
            verbose=True,
            pre_populated=pre_populated,
            prerequisites=prereq_context,
        )

    # --- 2a. Auto-save intake session (Issue #9) -------------------------
    if questions and answers:
        # Only persist when the intake was fully successful (real
        # questions generated + user provided answers).
        session = IntakeSession(
            course_name=course_name,
            questions=questions,
            answers=answers,
            course_specification=CourseSpecification(
                course_context=course_context,
                primary_language=primary_language,
                grading_scale=grading_scale,
                student_pathway=student_pathway,
                year_level=year_level,
                hardware_constraints=hardware_constraints,
            ),
            timestamp=datetime.now(UTC).isoformat(),
            run_id=run_id,
        )
        session_path = run_dir / "intake_session.json"
        try:
            session_path.write_text(session.model_dump_json(indent=2), encoding="utf-8")
            print(f"💾  Intake session saved to: {session_path}\n")
        except OSError as exc:
            print(f"⚠️  Could not save intake session: {exc}\n")

    # --- 3. Run the crew -------------------------------------------------
    try:
        result = run_syllabus_crew(
            course_context,
            course_name=course_name,
            primary_language=primary_language,
            verbose=True,
            skip_labs=skip_labs,
            resume_dir=resume_dir,
            run_id=run_id,
        )
    except RuntimeError as exc:
        print(f"\n❌  Fatal Runtime Error: {exc}", file=sys.stderr)
        print(
            "   → Check that OPENROUTER_API_KEY is set in .env.",
            file=sys.stderr,
        )
        sys.exit(2)
    except Exception as exc:
        print(f"\n❌  Fatal Unexpected Error: {exc}", file=sys.stderr)
        sys.exit(3)

    # --- 4. Print summary ------------------------------------------------
    _print_summary(result, course_name)


if __name__ == "__main__":
    main()
