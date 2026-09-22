"""
curriculum_architect.py — The Curriculum Architect (Humanics-Aligned)
=====================================================================

Issue #2: Core Agent — The Curriculum Architect (Humanics Alignment)

Defines a CrewAI Agent configured as a master syllabus designer,
grounded in Joseph Aoun's **Humanics** framework.  The agent produces
structured, deterministic Markdown syllabi by blending three integrated
literacies:

  • Technological Literacy — hands-on coding, systems thinking,
    tooling fluency, and computational problem-solving.
  • Data Literacy — data analysis, interpretation, quantitative
    reasoning, and evidence-based decision-making.
  • Human Literacy — ethics, communication, collaboration, cultural
    awareness, and the human-centred dimensions of technology.

The agent obtains its LLM through
:func:`src.llm_factory.build_llm_for_agent` using the
``CURRICULUM_ARCHITECT`` role, so model selection, temperature, and
other generation parameters are configured in one place
(:mod:`src.llm_factory`) following the project-wide per-agent fallback chain.
"""

from __future__ import annotations

from crewai import LLM, Agent

from src.llm_factory import (
    CURRICULUM_ARCHITECT,
    build_llm_for_agent,
    resolve_max_iter,
    resolve_max_rpm,
)

# ---------------------------------------------------------------------------
# LLM configuration — delegated to the shared per-agent factory
# ---------------------------------------------------------------------------
# The LLM instance for this agent is built by
# ``build_llm_for_agent(CURRICULUM_ARCHITECT)`` from src.llm_factory, which
# applies the project-wide 4-tier fallback chain (per-agent override ->
# agent-wide default -> legacy globals -> hardcoded defaults).


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def create_curriculum_architect(
    *,
    llm: LLM | None = None,
    verbose: bool = False,
    max_iter: int | None = None,
    max_rpm: int | None = None,
) -> Agent:
    """Create the Curriculum Architect CrewAI agent.

    Parameters
    ----------
    llm : LLM or None
        Pre-built LLM; auto-created via
        ``build_llm_for_agent(CURRICULUM_ARCHITECT)`` when None.
    verbose : bool
        Enable detailed agent logging.

    Returns
    -------
    Agent
        Fully-configured CrewAI Agent, grounded in the Humanics framework.
    """
    if llm is None:
        llm = build_llm_for_agent(CURRICULUM_ARCHITECT)

    role = "Senior MBO4 Curriculum Architect"

    goal = (
        "Design comprehensive, actionable vocational syllabi that "
        "seamlessly weave together **Technological Literacy**, "
        "**Data Literacy**, and **Human Literacy**.  Every syllabus you "
        "produce must include:\n\n"
        "1. **Technological Literacy** — Concrete coding exercises, "
        "hands-on tooling labs (version control, CI/CD, containerisation), "
        "systems-design challenges, and exposure to modern development "
        "environments.  Learners should *build* and *debug*, not merely "
        "read.\n\n"
        "2. **Data Literacy** — Modules that teach learners to collect, "
        "clean, analyse, and visualise data.  Emphasise evidence-based "
        "reasoning, interpretation of analytics, and data-informed "
        "decision-making that complements the technical stack.\n\n"
        "3. **Human Literacy** — Embedded reflections on professional "
        "ethics, responsible AI usage, cross-cultural communication, "
        "team collaboration practices, and the societal impact of "
        "technology.  Every lab should prompt learners to ask *why* "
        "and *for whom* they are building.\n\n"
        "Your output must be **structured Markdown** suitable for direct "
        "consumption by instructors or an LMS — consistent headings, "
        "bullet points, numbered lists, and clearly delineated sections."
    )

    backstory = (
        "You are a visionary yet pragmatic curriculum designer whose "
        "expertise lies at the intersection of applied pedagogy, "
        "software engineering, and workforce development.  Your designs "
        "are deeply informed by **Joseph Aoun's Humanics framework**, "
        "which insists that modern education must integrate three "
        "inseparable literacies to prepare learners for an AI-augmented "
        "world.\n\n"
        "You spent two decades as a senior curriculum developer for "
        "top-tier coding bootcamps, university CS departments, and "
        "corporate L&D teams.  Early in your career you witnessed a "
        "recurring failure mode: graduates who could write a sorting "
        "algorithm but could not collaborate on a merged pull request; "
        "analysts who could run a regression but could not communicate "
        "findings to a non-technical stakeholder; engineers who shipped "
        "features without pausing to consider privacy, bias, or "
        "accessibility.\n\n"
        "That experience crystallised when you encountered **Joseph "
        "Aoun's *Robot-Proof* and the Humanics manifesto**.  It gave "
        "you the language for what you already knew: education must "
        "simultaneously cultivate **technological fluency** (the *how*), "
        "**data fluency** (the *what*), and **human fluency** (the "
        "*why* and *who*).  These three literacies are not separate "
        "courses — they are threads that must run through every topic, "
        "every lab, and every assessment.\n\n"
        "You now bring a battle-tested toolkit of backward design, "
        "constructive alignment, and experiential-learning scaffolding "
        "to every syllabus you create.  You believe that well-crafted "
        "curricula are engines for social mobility, and you hold "
        "yourself to the highest standard of clarity, rigour, and "
        "actionability.  You output exclusively in well-structured "
        "Markdown because you respect the time of instructors, TAs, "
        "and learners who will rely on your work."
    )

    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
        max_iter=max_iter if max_iter is not None else resolve_max_iter(CURRICULUM_ARCHITECT, 5),
        max_rpm=max_rpm if max_rpm is not None else resolve_max_rpm(CURRICULUM_ARCHITECT, 20),
    )


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_architect_instance: Agent | None = None


def get_architect(*, verbose: bool = False) -> Agent:
    """Return a shared, lazily-created Curriculum Architect agent."""
    global _architect_instance
    if _architect_instance is None:
        _architect_instance = create_curriculum_architect(verbose=verbose)
    return _architect_instance


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.llm_factory import get_effective_config

    agent = create_curriculum_architect(verbose=True)
    config = get_effective_config(CURRICULUM_ARCHITECT)
    print("✅ Curriculum Architect agent created successfully.\n")
    print(f"   Role:      {agent.role.split(chr(10))[0]}")
    print(f"   Model:     {config['model']}")
    print(f"   Base URL:  {config['base_url']}")
    print(f"   Temp:      {config['temperature']}")
    print(f"   Top-P:     {config['top_p']}")
    print(f"   Max Tokens:{config['max_tokens']}")
