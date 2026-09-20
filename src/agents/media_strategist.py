"""Media Strategist agent — pedagogical modality routing specialist.

Issue #9: Layer 3 Interface Adapter.

Analyzes curriculum module complexity and outputs a strict
``ModalityDecision`` that the swarm state machine reads to route
generation to the appropriate content generator.

CRITICAL: This module MUST NOT hard-import from ``src.llm_factory``
at module level (Clean Architecture Layer 3 → Layer 4 constraint).
The LLM is resolved via lazy import inside ``create_media_strategist``
only when no explicit LLM argument is provided.
"""

from __future__ import annotations

from crewai import LLM, Agent


def create_media_strategist(
    *,
    llm: LLM | None = None,
    verbose: bool = False,
) -> Agent:
    """Create the Media Strategist CrewAI agent.

    Parameters
    ----------
    llm : LLM or None
        Pre-built LLM; auto-created via lazy import of
        ``build_llm_for_agent(MEDIA_STRATEGIST)`` when None.
    verbose : bool
        Enable detailed agent logging.

    Returns
    -------
    Agent
        Fully-configured CrewAI Agent that analyzes curriculum modules
        and produces ``ModalityDecision`` outputs for routing.
    """
    if llm is None:
        # Lazy import — no module-level Layer 4 dependency.
        from src.llm_factory import MEDIA_STRATEGIST, build_llm_for_agent  # noqa: PLC0415

        llm = build_llm_for_agent(MEDIA_STRATEGIST)

    role = "MBO4 Media Strategist and Modality Router"

    goal = (
        "Analyze each curriculum module's pedagogical complexity and "
        "determine the optimal instructional modality.  For every module "
        "you receive, output a strict JSON ``ModalityDecision`` choosing "
        "from: CLASSIC_READER (text/Markdown), INTERACTIVE_WEB (browser-based "
        "HTML/JS), INTERACTIVE_CLI (terminal scripts), or VIDEO_AS_CODE "
        "(React/Remotion JSX).  Your decision drives the entire downstream "
        "generation pipeline — choose wisely based on concept complexity, "
        "required visualisation depth, interactivity needs, and the MBO4 "
        "vocational learning context."
    )

    backstory = (
        "You are a veteran instructional designer who spent 15 years "
        "in Dutch MBO4 vocational education.  You have seen every type "
        "of learner struggle and every type of concept fail because it "
        "was delivered through the wrong medium.  You developed a scientific "
        "modality taxonomy — CLASSIC_READER for syntax and terminology, "
        "INTERACTIVE_WEB for visual algorithms and state machines, "
        "INTERACTIVE_CLI for terminal-driven workflows, and VIDEO_AS_CODE "
        "for concepts that demand temporal, animated explanation (recursion, "
        "network protocols, distributed systems).  Your decisions are never "
        "arbitrary — every ``ModalityDecision`` is backed by a clear pedagogical "
        "rationale and a calibrated complexity score from 0.0 (trivial) to "
        "1.0 (extremely complex).  You are the routing brain of the swarm."
    )

    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
        max_iter=15,
        max_rpm=20,
    )


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_media_strategist_instance: Agent | None = None


def get_media_strategist(*, verbose: bool = False) -> Agent:
    """Return a shared, lazily-created Media Strategist agent."""
    global _media_strategist_instance
    if _media_strategist_instance is None:
        _media_strategist_instance = create_media_strategist(verbose=verbose)
    return _media_strategist_instance
