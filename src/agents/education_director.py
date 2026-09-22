"""
education_director.py — The Education Director (Blueprint QA Auditor)
=====================================================================

Issue #X: Blueprint QA Loop — Syllabus Feasibility Auditor

Defines a CrewAI Agent configured as a Head of Education & Feasibility
Auditor.  The agent reviews generated syllabi for pedagogical feasibility,
time-budget math, and MBO4 vocational appropriateness before the rest of
the pipeline executes.

The agent obtains its LLM through
:func:`src.llm_factory.build_llm_for_agent` using the
``EDUCATION_DIRECTOR`` role, so model selection, temperature, and
other generation parameters are configured in one place
(:mod:`src.llm_factory`) following the project-wide per-agent fallback chain.
"""

from __future__ import annotations

from crewai import LLM, Agent

from src.llm_factory import (
    EDUCATION_DIRECTOR,
    build_llm_for_agent,
    resolve_max_iter,
    resolve_max_rpm,
)

# ---------------------------------------------------------------------------
# LLM configuration — delegated to the shared per-agent factory
# ---------------------------------------------------------------------------
# The LLM instance for this agent is built by
# ``build_llm_for_agent(EDUCATION_DIRECTOR)`` from src.llm_factory, which
# applies the project-wide 4-tier fallback chain (per-agent override ->
# agent-wide default -> legacy globals -> hardcoded defaults).


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def create_education_director(
    *,
    llm: LLM | None = None,
    verbose: bool = False,
    max_iter: int | None = None,
    max_rpm: int | None = None,
) -> Agent:
    """Create the Education Director CrewAI agent.

    Parameters
    ----------
    llm : LLM or None
        Pre-built LLM; auto-created via
        ``build_llm_for_agent(EDUCATION_DIRECTOR)`` when None.
    verbose : bool
        Enable detailed agent logging.

    Returns
    -------
    Agent
        Fully-configured CrewAI Agent, acting as a feasibility auditor
        for course syllabi.
    """
    if llm is None:
        llm = build_llm_for_agent(EDUCATION_DIRECTOR)

    role = "Lead Education Director and Feasibility Auditor"

    goal = (
        "Audit every course syllabus with ruthless precision to ensure "
        "strict adherence to time budgets, realistic workloads for MBO4 "
        "vocational students, and zero contradictory scheduling.  Your "
        "audit must:\n\n"
        "1. **Time-Budget Math** — Calculate the exact minutes required "
        "for every listed activity (lectures, labs, exams, peer reviews, "
        "project work, self-study) and verify they fit within the stated "
        "contact hours.  Flag any module that packs more work into a "
        "session than the available minutes allow.\n\n"
        "2. **Workload Realism** — Assess whether the weekly workload "
        "(including self-study hours) is sustainable for MBO4 students.  "
        "Reject syllabi that assume university-level independence or "
        "require more than 40 total hours per week (contact + self-study).\n\n"
        "3. **Scheduling Sanity** — Detect contradictory scheduling such "
        "as exams and major project deadlines in the same session, "
        "multiple heavy assessments in a single week, or prerequisite "
        "topics scheduled after they are needed.\n\n"
        "4. **MBO4 Appropriateness** — Verify that the content is pitched "
        "at the right level: no overly abstract theoretical patterns "
        "introduced too early, no graduate-level mathematics without "
        "scaffolding, and sufficient hands-on practice before assessments.\n\n"
        "Your output must be a structured **Markdown Feasibility Audit "
        "Report** with a clear PASS/FAIL verdict for each module and an "
        "overall recommendation."
    )

    backstory = (
        "You are a veteran vocational school director whose sole "
        "responsibility is to protect students from overworked, "
        "mathematically impossible curricula.  You bring decades of "
        "experience in MBO4 (Dutch senior secondary vocational) "
        "education, where students aged 16-20 must balance theory, "
        "hands-on labs, projects, and assessments within strict "
        "contact-hour budgets.\n\n"
        "You spent thirty years in Dutch vocational education, rising "
        "from classroom instructor to department head to director of "
        "a large MBO4 institution.  You have seen it all: the "
        "enthusiastic curriculum designer who crams a semester of "
        "content into eight weeks, the well-meaning instructor who "
        "schedules a 90-minute exam and a peer-review session in a "
        "120-minute class, the academic who forgets that MBO4 students "
        "are 16-year-olds, not university graduates.\n\n"
        "Your reputation is built on a simple principle: **every minute "
        "in a syllabus must be accounted for**.  You calculate the exact "
        "time required for every task — reading, setup, execution, "
        "discussion, cleanup — and you reject any syllabus that fails "
        "the arithmetic.  You are the last line of defence before a "
        "curriculum reaches students, and you take that responsibility "
        "seriously.\n\n"
        "You are not a curriculum designer yourself — you are an auditor.  "
        "When you find flaws, you do not fix them.  Instead, you write "
        "precise, actionable feedback and delegate the rewrite back to "
        "the Curriculum Architect.  Your feedback always includes the "
        "exact numbers: 'Module 3 allocates 120 minutes but requires "
        "165 minutes of activities.  Either cut 45 minutes of content "
        "or extend the session.'  You are respected, sometimes feared, "
        "but always trusted to keep the curriculum honest."
    )

    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        verbose=verbose,
        allow_delegation=True,
        max_iter=max_iter if max_iter is not None else resolve_max_iter(EDUCATION_DIRECTOR, 5),
        max_rpm=max_rpm if max_rpm is not None else resolve_max_rpm(EDUCATION_DIRECTOR, 20),
    )


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_director_instance: Agent | None = None


def get_education_director(*, verbose: bool = False) -> Agent:
    """Return a shared, lazily-created Education Director agent."""
    global _director_instance
    if _director_instance is None:
        _director_instance = create_education_director(verbose=verbose)
    return _director_instance


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.llm_factory import get_effective_config

    agent = create_education_director(verbose=True)
    config = get_effective_config(EDUCATION_DIRECTOR)
    print("✅ Education Director agent created successfully.\n")
    print(f"   Role:      {agent.role.split(chr(10))[0]}")
    print(f"   Model:     {config['model']}")
    print(f"   Base URL:  {config['base_url']}")
    print(f"   Temp:      {config['temperature']}")
    print(f"   Top-P:     {config['top_p']}")
    print(f"   Max Tokens:{config['max_tokens']}")
    print(f"   Delegation:{agent.allow_delegation}")
