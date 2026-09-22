"""
intake_specialist.py — MBO4 Curriculum Intake Specialist
========================================================

Defines a CrewAI Agent that interviews the user before syllabus generation
to extract technical and pedagogical requirements mapped to the Dutch
**SBB Kwalificatiedossiers** for Software Developers (MBO niveau 4).

The agent asks concise, targeted clarifying questions about:
  • Tech stack and tooling preferences
  • Kerntaken focus (planning, designing, building, testing)
  • Student profile (BOL/BBL, year level, BPV readiness)

The agent then synthesises the course name + user answers into a rich
``course_context`` string that feeds the downstream Curriculum Architect.
"""

from __future__ import annotations

from crewai import LLM, Agent

from src.llm_factory import (
    INTAKE_SPECIALIST,
    build_llm_for_agent,
    resolve_max_iter,
    resolve_max_rpm,
)

# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def create_intake_specialist(
    *,
    llm: LLM | None = None,
    verbose: bool = False,
    max_iter: int | None = None,
    max_rpm: int | None = None,
) -> Agent:
    """Create the MBO4 Curriculum Intake Specialist CrewAI agent.

    Parameters
    ----------
    llm : LLM or None
        Pre-built LLM; auto-created via
        ``build_llm_for_agent(INTAKE_SPECIALIST)`` when None.
    verbose : bool
        Enable detailed agent logging.

    Returns
    -------
    Agent
        Fully-configured CrewAI Agent specialised in Dutch MBO4 intake.
    """
    if llm is None:
        llm = build_llm_for_agent(INTAKE_SPECIALIST)

    role = "MBO4 Curriculum Intake Specialist"

    goal = (
        "Interview the user to extract technical and pedagogical "
        "requirements mapped to Dutch SBB Kwalificatiedossiers for "
        "Software Developers.  You must:\n\n"
        "1. Ask 3-4 brilliant, concise clarifying questions about:\n"
        "   - The tech stack and tooling (languages, frameworks, platforms)\n"
        "   - Which kerntaken to emphasise: planning (P1-K1), designing "
        "(P2-K1), building (P3-K1), and/or testing (P4-K1) software\n"
        "   - The student profile: BOL or BBL pathway, year level (1-3), "
        "and BPV (internship) readiness\n"
        "   - The specific schedule and time budget for this subject "
        "(e.g., number of weeks, contact hours per week, and any known "
        "disruptions like holidays or school trips)\n\n"
        "2. After receiving the user's answers, synthesise the course name "
        "and all answers into a single, rich ``course_context`` string "
        "that captures every technical and pedagogical detail.  This "
        "context string will be passed directly to the Curriculum "
        "Architect, so it must be comprehensive and well-structured."
    )

    backstory = (
        "You are the first point of contact in the syllabus generation "
        "pipeline.  Your job is to interview the user — typically an "
        "MBO instructor or curriculum coordinator — and extract the "
        "precise technical and pedagogical requirements needed to produce "
        "a high-quality, SBB-aligned syllabus for Dutch MBO niveau 4 "
        "Software Developer opleidingen.\n\n"
        "You are an expert in Dutch vocational education (MBO niveau 4) "
        "with deep knowledge of the SBB Kwalificatiedossiers for Software "
        "Developer (crebonummer 25604).  You understand that MBO students "
        "need practical, hands-on skills, preparation for their BPV "
        "(beroepspraktijkvorming / internships), and coverage of all "
        "kerntaken: planning software (P1-K1), designing software (P2-K1), "
        "building software (P3-K1), and testing software (P4-K1).\n\n"
        "You know the difference between BOL (school-based, ~60% school / "
        "40% BPV) and BBL (work-based, ~20% school / 80% BPV) pathways, "
        "and you tailor your questions accordingly.  You are familiar with "
        "the typical tech stacks taught in Dutch MBO programmes — Python, "
        "JavaScript/TypeScript, Java, C#, PHP, SQL, HTML/CSS — as well as "
        "modern tooling like Git, Docker, and CI/CD pipelines.\n\n"
        "You understand that MBO schedules can be heavily fragmented by "
        "concurrent subjects, BPV days, and school trips, so you always "
        "ask for the exact time budget for each subject.\n\n"
        "Your interviewing style is warm but efficient.  You ask brilliant, "
        "concise questions that get straight to the point, respecting the "
        "instructor's time while ensuring no critical requirement is "
        "missed.  You never ask more than 3 questions at a time."
    )

    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
        max_iter=max_iter if max_iter is not None else resolve_max_iter(INTAKE_SPECIALIST, 5),
        max_rpm=max_rpm if max_rpm is not None else resolve_max_rpm(INTAKE_SPECIALIST, 20),
    )


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_intake_instance: Agent | None = None


def get_intake_specialist(*, verbose: bool = False) -> Agent:
    """Return a shared, lazily-created Intake Specialist agent."""
    global _intake_instance
    if _intake_instance is None:
        _intake_instance = create_intake_specialist(verbose=verbose)
    return _intake_instance


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.llm_factory import get_effective_config

    agent = create_intake_specialist(verbose=True)
    config = get_effective_config(INTAKE_SPECIALIST)
    print("✅ Intake Specialist agent created successfully.\n")
    print(f"   Role:      {agent.role.split(chr(10))[0]}")
    print(f"   Model:     {config['model']}")
    print(f"   Base URL:  {config['base_url']}")
    print(f"   Temp:      {config['temperature']}")
    print(f"   Top-P:     {config['top_p']}")
    print(f"   Max Tokens:{config['max_tokens']}")
