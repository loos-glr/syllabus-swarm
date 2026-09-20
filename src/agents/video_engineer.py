"""Video Engineer agent — deterministic Video-as-Code (VaC) specialist.

Issue #9: Layer 3 Interface Adapter.

Generates deterministic temporal code (React/Remotion JSX) for
educational video compositions.  Receives a ``ModalityDecision`` and
produces a ``RemotionManifest`` with structured scene/sequence data.

CRITICAL: This module MUST NOT hard-import from ``src.llm_factory``
at module level (Clean Architecture Layer 3 → Layer 4 constraint).
The LLM is resolved via lazy import inside ``create_video_engineer``
only when no explicit LLM argument is provided.
"""

from __future__ import annotations

from crewai import LLM, Agent


def create_video_engineer(
    *,
    llm: LLM | None = None,
    verbose: bool = False,
) -> Agent:
    """Create the Video Engineer CrewAI agent.

    Parameters
    ----------
    llm : LLM or None
        Pre-built LLM; auto-created via lazy import of
        ``build_llm_for_agent(VIDEO_ENGINEER)`` when None.
    verbose : bool
        Enable detailed agent logging.

    Returns
    -------
    Agent
        Fully-configured CrewAI Agent that generates React/Remotion
        JSX compositions and ``RemotionManifest`` descriptors.
    """
    if llm is None:
        # Lazy import — no module-level Layer 4 dependency.
        from src.llm_factory import VIDEO_ENGINEER, build_llm_for_agent  # noqa: PLC0415

        llm = build_llm_for_agent(VIDEO_ENGINEER)

    role = "Remotion Video Engineer and React/JSX Specialist"

    goal = (
        "Generate deterministic temporal code using React and Remotion.  "
        "For every curriculum module routed to you (via ``VIDEO_AS_CODE`` "
        "modality), produce a structured ``RemotionManifest`` containing "
        "composition metadata (fps, resolution, duration_in_frames) and an "
        "ordered list of scene/sequence component descriptors.  Each "
        "component is a dict with type, props, and optional children — "
        "mirroring the React component tree.  You strictly avoid narrative "
        "prose or pixel-based media; your output is pure structural data "
        "that can be deterministically rendered by a Remotion runtime."
    )

    backstory = (
        "You are a senior creative technologist who transitioned from "
        "front-end engineering at a major streaming platform into educational "
        "technology.  You mastered React, Remotion, and the art of "
        "deterministic video rendering — where every frame is a pure function "
        "of its index and data, not a recorded pixel stream.  You believe "
        "that the best way to explain recursion, network protocols, sorting "
        "algorithms, and system architectures is through temporal, animated "
        "code — not static diagrams or walls of text.  Your compositions are "
        "version-controllable, reproducible, and mathematically precise.  "
        "You generate ``RemotionManifest`` descriptors that a rendering "
        "pipeline can transform into actual ``.tsx`` files without ambiguity.  "
        "You are the Video-as-Code engine of the swarm."
    )

    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
        max_iter=20,
        max_rpm=15,
    )


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_video_engineer_instance: Agent | None = None


def get_video_engineer(*, verbose: bool = False) -> Agent:
    """Return a shared, lazily-created Video Engineer agent."""
    global _video_engineer_instance
    if _video_engineer_instance is None:
        _video_engineer_instance = create_video_engineer(verbose=verbose)
    return _video_engineer_instance
