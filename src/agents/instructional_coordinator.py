"""
instructional_coordinator.py — The Instructional Coordinator (MBO4 Lesson Plan Specialist)
==========================================================================================

Defines a CrewAI Agent configured as an MBO4 instructional design specialist
who produces structured, teacher-ready lesson plans from syllabus and module
context.  Each lesson plan contains session-by-session breakdowns with timing,
differentiation, assessment checkpoints, and materials lists.

The agent obtains its LLM through
:func:`src.llm_factory.build_llm_for_agent` using the
``INSTRUCTIONAL_COORDINATOR`` role.
"""

from __future__ import annotations

from crewai import LLM, Agent

from src.exporters.tool import OutputExportTool
from src.llm_factory import (
    INSTRUCTIONAL_COORDINATOR,
    build_llm_for_agent,
    resolve_max_iter,
    resolve_max_rpm,
)

# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def create_instructional_coordinator(
    *,
    llm: LLM | None = None,
    verbose: bool = False,
    max_iter: int | None = None,
    max_rpm: int | None = None,
) -> Agent:
    """Create the Instructional Coordinator CrewAI agent."""
    if llm is None:
        llm = build_llm_for_agent(INSTRUCTIONAL_COORDINATOR)

    role = "MBO4 Instructional Coordinator and Lesson Plan Designer"

    goal = (
        "Transform a course syllabus and module context into a detailed, "
        "teacher-ready lesson plan.  For each module, produce a session-by-session "
        "breakdown that includes:\n\n"
        "- **Session title and duration** — realistic time budgets aligned with "
        "MBO4 contact hours (typically 45–90 minute blocks).\n"
        "- **Learning objectives** — 2–4 measurable objectives per session "
        "using action verbs (Bloom's taxonomy).\n"
        "- **Activities** — ordered list of teaching and learning activities "
        "with estimated time per activity.\n"
        "- **Resources** — all materials, tools, and equipment needed.\n"
        "- **Differentiation** — strategies for supporting struggling students "
        "and challenging advanced learners.\n"
        "- **Assessment** — formative checkpoints (exit tickets, quick quizzes, "
        "peer review) and summative milestones.\n\n"
        "The lesson plan must be practical, actionable, and immediately usable "
        "by an MBO4 teacher without additional interpretation."
    )

    backstory = (
        "You started your career as an MBO4 software development teacher at a "
        "large ROC in the Netherlands.  For seven years you taught BOL and BBL "
        "students everything from HTML/CSS basics to full-stack frameworks.  "
        "You experienced firsthand the daily reality of vocational education: "
        "diverse student backgrounds, varying language proficiency levels, and "
        "the constant pressure to prepare students for real internships (BPV).\n\n"
        "The frustration that drove you into instructional design was the "
        "complete mismatch between glossy curriculum documents and the actual "
        "classroom.  Syllabi described *what* to teach but never *how* to teach "
        "it.  You spent countless evenings translating one-page module "
        "descriptions into workable lesson plans — a process that should have "
        "been automated.\n\n"
        "You earned your master's in Educational Science (Onderwijskunde) from "
        "the University of Amsterdam, specialising in competency-based education "
        "(CGO) — the pedagogical foundation of the Dutch MBO system.  Your "
        "thesis examined how structured lesson planning with built-in "
        "differentiation reduces drop-out rates among MBO4 ICT students.\n\n"
        "Today you bring both perspectives — the practicing teacher and the "
        "instructional designer — to every lesson plan you produce.  You know "
        "that a good lesson plan is not just a timetable; it is a roadmap that "
        "anticipates where students will struggle and provides the teacher with "
        "concrete strategies to address those struggles before they derail the "
        "lesson.  You insist on practical language, realistic time estimates, "
        "and activities that work in a classroom with 25 students sharing "
        "Chromebooks."
    )

    export_tool = OutputExportTool(force=True)

    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
        max_iter=max_iter if max_iter is not None else resolve_max_iter(INSTRUCTIONAL_COORDINATOR, 25),
        max_rpm=max_rpm if max_rpm is not None else resolve_max_rpm(INSTRUCTIONAL_COORDINATOR, 20),
        tools=[export_tool],
    )


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_instructional_coordinator_instance: Agent | None = None


def get_instructional_coordinator(*, verbose: bool = False) -> Agent:
    """Return a shared, lazily-created Instructional Coordinator agent."""
    global _instructional_coordinator_instance
    if _instructional_coordinator_instance is None:
        _instructional_coordinator_instance = create_instructional_coordinator(verbose=verbose)
    return _instructional_coordinator_instance


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.llm_factory import get_effective_config

    agent = create_instructional_coordinator(verbose=True)
    config = get_effective_config(INSTRUCTIONAL_COORDINATOR)
    print("✅ Instructional Coordinator agent created successfully.\n")
    print(f"   Role:      {agent.role.split(chr(10))[0]}")
    print(f"   Model:     {config['model']}")
    print(f"   Base URL:  {config['base_url']}")
    print(f"   Temp:      {config['temperature']}")
    print(f"   Top-P:     {config['top_p']}")
    print(f"   Max Tokens:{config['max_tokens']}")
