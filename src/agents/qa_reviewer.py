"""
qa_reviewer.py — The QA Reviewer & MBO4 Didactic Expert
========================================================

Issue #7: AI-Driven Feedback Loop — QA Reviewer Agent

Defines a CrewAI Agent configured as a strict-but-fair senior QA engineer
who specialises in reviewing generated lab code for technical correctness
and didactic appropriateness for MBO4 vocational students.

The agent is equipped with ``DirectoryReadTool`` and ``FileReadTool`` from
``crewai_tools`` so it can inspect the files written by the Lab Developer.
Critically, ``allow_delegation`` is set to ``True``, enabling the QA Reviewer
to delegate fix tasks back to the Lab Developer when issues are found.

The agent obtains its LLM through
:func:`src.llm_factory.build_llm_for_agent` using the ``QA_REVIEWER``
role, so model selection, temperature, and other generation parameters are
configured in one place (:mod:`src.llm_factory`) following the project-wide
per-agent fallback chain.
"""

from __future__ import annotations

from crewai import LLM, Agent
from crewai_tools import DirectoryReadTool, FileReadTool

from src.exporters.tool import OutputExportTool
from src.llm_factory import (
    QA_REVIEWER,
    build_llm_for_agent,
    resolve_max_iter,
    resolve_max_rpm,
)

# ---------------------------------------------------------------------------
# LLM configuration — delegated to the shared per-agent factory
# ---------------------------------------------------------------------------
# The LLM instance for this agent is built by
# ``build_llm_for_agent(QA_REVIEWER)`` from src.llm_factory, which applies
# the project-wide 4-tier fallback chain (per-agent override -> agent-wide
# default -> legacy globals -> hardcoded defaults).
# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def create_qa_reviewer(
    *,
    llm: LLM | None = None,
    verbose: bool = False,
    max_iter: int | None = None,
    max_rpm: int | None = None,
) -> Agent:
    """Create the QA Reviewer CrewAI agent.

    Parameters
    ----------
    llm : LLM or None
        Pre-built LLM; auto-created via
        ``build_llm_for_agent(QA_REVIEWER)`` when None.
    verbose : bool
        Enable detailed agent logging.

    Returns
    -------
    Agent
        Fully-configured CrewAI Agent that reviews generated labs for
        technical correctness and MBO4 didactic appropriateness.
    """
    if llm is None:
        llm = build_llm_for_agent(QA_REVIEWER)

    role = "Strict MBO4 QA Reviewer and Didactic Expert"

    goal = (
        "Verify that all generated lab code is 100% bug-free and that the "
        "instructions are didactically correct for MBO4 vocational students.\n\n"
        "1. **Technical Correctness Check** — Ensure zero syntax errors, no "
        "missing imports, no hallucinated variables or functions, and "
        "completely self-contained execution.  Every file must be runnable "
        "as-is after following the README instructions.\n\n"
        "2. **Didactic & Clarity Check** — Ensure the README instructions "
        "and code comments use clear, accessible language suited for "
        "vocational students.  Avoid overly academic jargon.  TODO markers "
        "MUST be actionable and unambiguous — a student should know exactly "
        "what to do and how to verify their work.\n\n"
        "3. **Delegation** — If ANY lab fails either check, you MUST use "
        "CrewAI's delegation to assign a fix task back to the Lab & Project "
        "Developer with specific, actionable feedback on what needs to be "
        "rewritten and why."
    )

    backstory = (
        "You are a strict but fair senior developer who has pivoted into "
        "quality assurance for educational content.  You understand the "
        "MBO4 target audience perfectly — vocational students who learn by "
        "doing, not by reading academic papers.  You refuse to let broken "
        "code or vague instructions reach students.\n\n"
        "You spent twelve years as a senior software engineer before "
        "pivoting into QA for educational technology.  You've seen too many "
        "students struggle — not because they lacked ability, but because "
        "the starter code they were given was broken, confusing, or assumed "
        "knowledge they didn't have yet.\n\n"
        "You know the MBO4 student profile inside and out: practical, "
        "hands-on learners who thrive on clear instructions and working "
        "examples.  They don't need academic theory — they need code they "
        "can run, modify, and learn from.  Every error message they "
        "encounter should be intentional and educational, not a bug in the "
        "starter code.\n\n"
        "You are meticulous in your reviews.  You check every import, every "
        "function call, every file path.  You read every README as if you "
        "were a student seeing it for the first time.  You are not afraid "
        "to send work back — in fact, you consider it your duty.  Better to "
        "catch issues now than to have a classroom of frustrated students "
        "later.\n\n"
        "Your reviews are constructive, specific, and always include the "
        "*why* behind every requested change.  You don't just say 'fix this' "
        "— you explain what's wrong, why it matters for MBO4 students, and "
        "how to fix it."
    )

    # Equip the agent with file-reading tools so it can inspect the
    # generated lab files written by the Lab Developer, plus the
    # OutputExportTool for writing QA reports and fix instructions.
    dir_tool = DirectoryReadTool()
    file_tool = FileReadTool()
    export_tool = OutputExportTool(force=True)

    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        verbose=verbose,
        allow_delegation=True,
        max_iter=max_iter if max_iter is not None else resolve_max_iter(QA_REVIEWER, 30),
        max_rpm=max_rpm if max_rpm is not None else resolve_max_rpm(QA_REVIEWER, 20),
        tools=[dir_tool, file_tool, export_tool],
    )


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_qa_reviewer_instance: Agent | None = None


def get_qa_reviewer(*, verbose: bool = False) -> Agent:
    """Return a shared, lazily-created QA Reviewer agent."""
    global _qa_reviewer_instance
    if _qa_reviewer_instance is None:
        _qa_reviewer_instance = create_qa_reviewer(verbose=verbose)
    return _qa_reviewer_instance


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.llm_factory import get_effective_config

    agent = create_qa_reviewer(verbose=True)
    config = get_effective_config(QA_REVIEWER)
    print("✅ QA Reviewer agent created successfully.\n")
    print(f"   Role:             {agent.role.split(chr(10))[0]}")
    print(f"   Model:            {config['model']}")
    print(f"   Base URL:         {config['base_url']}")
    print(f"   Temp:             {config['temperature']}")
    print(f"   Top-P:            {config['top_p']}")
    print(f"   Max Tokens:       {config['max_tokens']}")
    print(f"   Allow Delegation: {agent.allow_delegation}")
    print(f"   Tools:            {[t.name for t in agent.tools]}")
