"""
lab_developer.py — The Lab & Project Developer (Humanics-Aligned)
=================================================================

Issue #3: Core Agent — The Lab & Project Developer (Tiered Coding Challenges)

Defines a CrewAI Agent configured as a master coding-lab designer who
produces tiered, self-contained coding exercises derived from a course
syllabus.  The agent is grounded in Joseph Aoun's **Humanics** framework
and generates labs across three progressive tiers:

  • Tier 1 — Foundations: Syntax drills, basic patterns, single-file
    exercises with TODO markers and fully-commented solutions.
  • Tier 2 — Application: Multi-file mini-projects involving APIs,
    data pipelines, CLI tools, and testing.
  • Tier 3 — Architecture: Service-oriented systems with Docker,
    docker-compose, CI/CD, and observability.

Every lab includes starter/ scaffolding and solution/ reference
implementations, plus a README.md with learning objectives, key concepts,
and Humanics reflection prompts.

The agent obtains its LLM through
:func:`src.llm_factory.build_llm_for_agent` using the ``LAB_DEVELOPER``
role, so model selection, temperature, and other generation parameters are
configured in one place (:mod:`src.llm_factory`) following the project-wide
per-agent fallback chain.
"""

from __future__ import annotations

from crewai import LLM, Agent

from src.exporters.tool import OutputExportTool
from src.llm_factory import (
    LAB_DEVELOPER,
    build_llm_for_agent,
    resolve_max_iter,
    resolve_max_rpm,
)

# ---------------------------------------------------------------------------
# LLM configuration — delegated to the shared per-agent factory
# ---------------------------------------------------------------------------
# The LLM instance for this agent is built by
# ``build_llm_for_agent(LAB_DEVELOPER)`` from src.llm_factory, which applies
# the project-wide 4-tier fallback chain (per-agent override -> agent-wide
# default -> legacy globals -> hardcoded defaults).
# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def create_lab_developer(
    *,
    llm: LLM | None = None,
    verbose: bool = False,
    max_iter: int | None = None,
    max_rpm: int | None = None,
) -> Agent:
    """Create the Lab & Project Developer CrewAI agent.

    Parameters
    ----------
    llm : LLM or None
        Pre-built LLM; auto-created via
        ``build_llm_for_agent(LAB_DEVELOPER)`` when None.
    verbose : bool
        Enable detailed agent logging.

    Returns
    -------
    Agent
        Fully-configured CrewAI Agent that designs tiered coding labs
        grounded in the Humanics framework.
    """
    if llm is None:
        llm = build_llm_for_agent(LAB_DEVELOPER)

    role = "Senior Lab and Project Developer"

    goal = (
        "Generate comprehensive, self-contained tiered coding labs that "
        "transform a syllabus into actionable, runnable exercises.\n\n"
        "1. **Tier 1 — Foundations** — Single-file exercises covering "
        "syntax, control flow, data structures, file I/O, error handling, "
        "and basic OOP.  Each with starter (TODO markers) and solution.\n\n"
        "2. **Tier 2 — Application** — Multi-file mini-projects (3-5 files) "
        "covering REST APIs, data pipelines (ETL), CLI tools, web scraping, "
        "testing with pytest, and packaging.  Each with requirements.txt.\n\n"
        "3. **Tier 3 — Architecture** — Service-oriented systems (5-10 files) "
        "covering Docker, docker-compose, CI/CD pipelines, cloud deployment, "
        "and observability.  Each with Dockerfile, docker-compose.yml, Makefile.\n\n"
        "Every lab must integrate **Technological [T]**, **Data [D]**, and "
        "**Human [H]** literacies per Joseph Aoun's Humanics framework."
    )

    backstory = (
        "You are a seasoned software engineer and educator who specialises "
        "in translating curriculum blueprints into concrete, hands-on "
        "coding laboratories.  Your designs span three progressive tiers "
        "of difficulty, each building on the previous to create a coherent "
        "learning arc from syntax fundamentals to distributed systems "
        "architecture.\n\n"
        "You spent fifteen years as a senior software engineer across "
        "startups and enterprise cloud providers before pivoting into "
        "technical education.  You witnessed the gap between what graduates "
        "could *explain* in theory and what they could *build* on day one.\n\n"
        "That frustration drove you to design coding labs the way you wish "
        "you had learned: scaffolded, self-contained, and relentlessly "
        "practical.  Every TODO marker is a precise, actionable instruction.  "
        "Every solution file teaches through its comments.\n\n"
        "You are a polyglot developer comfortable across Python, JavaScript, "
        "Go, and Rust.  You embrace Docker as a first-class learning objective "
        "in Tier 3.  You treat CI/CD pipelines as teachable moments.\n\n"
        "Above all, you are a Humanics practitioner: you embed ethics "
        "reflections, accessibility checklists, and collaboration prompts "
        "into every lab because you know the best engineers are not just "
        "technically brilliant — they are humane, communicative, and aware "
        "of the systems they build and the people those systems affect."
    )

    export_tool = OutputExportTool(force=True)

    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
        max_iter=max_iter if max_iter is not None else resolve_max_iter(LAB_DEVELOPER, 60),
        max_rpm=max_rpm if max_rpm is not None else resolve_max_rpm(LAB_DEVELOPER, 30),
        tools=[export_tool],
    )


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_lab_dev_instance: Agent | None = None


def get_lab_developer(*, verbose: bool = False) -> Agent:
    """Return a shared, lazily-created Lab & Project Developer agent."""
    global _lab_dev_instance
    if _lab_dev_instance is None:
        _lab_dev_instance = create_lab_developer(verbose=verbose)
    return _lab_dev_instance


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.llm_factory import get_effective_config

    agent = create_lab_developer(verbose=True)
    config = get_effective_config(LAB_DEVELOPER)
    print("✅ Lab & Project Developer agent created successfully.\n")
    print(f"   Role:      {agent.role.split(chr(10))[0]}")
    print(f"   Model:     {config['model']}")
    print(f"   Base URL:  {config['base_url']}")
    print(f"   Temp:      {config['temperature']}")
    print(f"   Top-P:     {config['top_p']}")
    print(f"   Max Tokens:{config['max_tokens']}")
