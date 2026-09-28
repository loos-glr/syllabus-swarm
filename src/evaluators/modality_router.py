"""
modality_router.py — Deterministic Modality Routing (System One)
================================================================

.. rubric:: Replaces the retired ``Media Strategist`` agent

Routes a curriculum module to the appropriate content generator using a
**single** System One round-trip containing two questions:

1. a ``choice`` over the four :class:`~src.models.ModalityType` options, and
2. a ``score`` for pedagogical complexity.

No prose is generated: the returned :class:`~src.models.ModalityDecision` is
built deterministically from the typed answers, so the orchestrator never has
to parse free text to make a control-flow decision.

Public API
----------
* ``ROUTING_MAP`` — ``ModalityType`` -> generator/agent name.
* ``route_module(state, client=...)`` — produce a typed decision.
* ``routing_target(modality)`` — look up the downstream generator.
* ``get_modality_router()`` — lazy singleton router.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.models import ModalityDecision, ModalityType
from src.system_one import SystemOneError, choice_question, score_question

if TYPE_CHECKING:
    from src.system_one import SystemOneClient

# ---------------------------------------------------------------------------
# Decision configuration
# ---------------------------------------------------------------------------

#: Human-readable option descriptions handed to the System One ``choice``.
MODALITY_CRITERIA: dict[ModalityType, str] = {
    ModalityType.CLASSIC_READER: (
        "Plain text/Markdown theory. Best for syntax, terminology, definitions "
        "and reference material where prose is sufficient and no visualisation "
        "or interactivity is required."
    ),
    ModalityType.INTERACTIVE_WEB: (
        "Browser-based interactive HTML/JS artifact. Best for algorithms, state "
        "machines, data structures and anything that benefits from visual "
        "exploration or user-driven experimentation."
    ),
    ModalityType.INTERACTIVE_CLI: (
        "Terminal-based interactive script. Best for command-line workflows, "
        "build tooling, git, package managers and shell-driven processes that "
        "belong in a terminal."
    ),
    ModalityType.VIDEO_AS_CODE: (
        "Deterministic React/Remotion video composition. Best for concepts that "
        "demand temporal, animated explanation over time — recursion, network "
        "protocols, distributed systems, request lifecycles."
    ),
}

#: Ordered complexity rubric (lowest level first) handed to the ``score``
#: primitive.  ``len(COMPLEXITY_LEVELS)`` levels map linearly onto 0.0–1.0.
COMPLEXITY_LEVELS: tuple[str, ...] = (
    "Trivial: single-concept, no prerequisites, memorisation-level.",
    "Simple: one concept with a trivial worked example.",
    "Moderate: two or three interacting concepts, some mental simulation needed.",
    "Complex: multiple interacting concepts, non-obvious control flow.",
    "Very complex: abstract, multi-step, hard to hold in working memory.",
)

#: ``ModalityType`` -> name of the downstream generator / agent.
ROUTING_MAP: dict[ModalityType, str] = {
    ModalityType.CLASSIC_READER: "theory_instructor",
    ModalityType.INTERACTIVE_WEB: "theory_instructor",
    ModalityType.INTERACTIVE_CLI: "theory_instructor",
    ModalityType.VIDEO_AS_CODE: "video_engineer",
}

_DEFAULT_MIN_CONFIDENCE: float = 0.5


def routing_confidence_threshold() -> float:
    """Return the minimum System One confidence for an auto-accepted route.

    Read from ``SYSTEM_ONE_ROUTING_MIN_CONFIDENCE`` (default ``0.5``).  A
    decision below the threshold is still returned, but flagged with
    ``needs_review=True`` so callers can escalate instead of acting.
    """
    raw = os.getenv("SYSTEM_ONE_ROUTING_MIN_CONFIDENCE")
    if raw is None:
        return _DEFAULT_MIN_CONFIDENCE
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_MIN_CONFIDENCE
    return min(max(value, 0.0), 1.0)


def routing_target(modality: ModalityType) -> str:
    """Return the downstream generator name for *modality*.

    Raises
    ------
    KeyError
        When *modality* has no registered route (a programmer error — every
        ``ModalityType`` must be routable).
    """
    try:
        return ROUTING_MAP[modality]
    except KeyError as exc:  # pragma: no cover - guards future enum additions
        raise KeyError(f"No routing target registered for modality {modality!r}") from exc


# ---------------------------------------------------------------------------
# Routing state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutingState:
    """Everything the router needs to know about one curriculum module."""

    module_name: str
    module_summary: str
    learning_objectives: tuple[str, ...] = ()
    key_concepts: tuple[str, ...] = ()
    year_level: int | None = None
    primary_language: str | None = None

    def as_state(self) -> dict[str, object]:
        """Render the state payload sent to the System One model."""
        payload: dict[str, object] = {
            "module_name": self.module_name,
            "module_summary": self.module_summary,
        }
        if self.learning_objectives:
            payload["learning_objectives"] = list(self.learning_objectives)
        if self.key_concepts:
            payload["key_concepts"] = list(self.key_concepts)
        if self.year_level is not None:
            payload["year_level"] = self.year_level
        if self.primary_language:
            payload["primary_language"] = self.primary_language
        return payload


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class ModalityRouting:
    """Deterministic modality router backed by a System One client."""

    def __init__(self, client: SystemOneClient) -> None:
        self._client = client

    def route(self, state: RoutingState) -> ModalityDecision:
        """Return the typed :class:`~src.models.ModalityDecision` for *state*."""
        questions = {
            "modality": choice_question(
                instructions=(
                    "Choose the single best instructional modality for teaching "
                    "this module to 16–20 year-old MBO4 vocational students. "
                    f"Module: {state.module_name}. Summary: {state.module_summary}"
                ),
                criteria={m.value: desc for m, desc in MODALITY_CRITERIA.items()},
            ),
            "complexity": score_question(
                instructions=(
                    "Rate the pedagogical complexity of this module for an MBO4 "
                    "student, independent of the chosen modality."
                ),
                criteria=COMPLEXITY_LEVELS,
            ),
        }

        result = self._client.system_one(state=state.as_state(), questions=questions)

        choice = result.choice("modality")
        complexity = result.score("complexity")

        try:
            modality = ModalityType(choice.choice)
        except ValueError as exc:
            raise SystemOneError(
                f"System One returned an unknown modality {choice.choice!r}; "
                f"expected one of {[m.value for m in ModalityType]}"
            ) from exc

        span = max(len(COMPLEXITY_LEVELS) - 1, 1)
        complexity_score = min(max(complexity.score / span, 0.0), 1.0)

        target = routing_target(modality)
        return ModalityDecision(
            module_name=state.module_name,
            modality=modality,
            rationale=(
                f"System One selected '{modality.value}' (confidence "
                f"{choice.confidence:.2f}) for '{state.module_name}' and routed it "
                f"to the {target}."
            ),
            complexity_score=complexity_score,
            confidence=choice.confidence,
            probabilities=dict(choice.probabilities),
            model_version=result.model,
            needs_review=choice.confidence < routing_confidence_threshold(),
        )


def route_module(state: RoutingState, *, client: SystemOneClient) -> ModalityDecision:
    """Route a single module using *client* (convenience wrapper)."""
    return ModalityRouting(client).route(state)


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_modality_router: ModalityRouting | None = None


def get_modality_router() -> ModalityRouting:
    """Return a shared, lazily-created :class:`ModalityRouting` router.

    Raises
    ------
    SystemOneError
        When no System One client is configured (missing API key).
    """
    global _modality_router
    if _modality_router is None:
        from src.llm_factory import MODALITY_ROUTER, build_system_one_client  # noqa: PLC0415

        client = build_system_one_client(task=MODALITY_ROUTER)
        if client is None:
            raise SystemOneError(
                "Modality routing requires a System One client. Set "
                "SYSTEM_ONE_API_KEY (or TYPESAFE_API_KEY), or set "
                "SYSTEM_ONE_ENABLED=0 for degraded offline mode."
            )
        _modality_router = ModalityRouting(client)
    return _modality_router
