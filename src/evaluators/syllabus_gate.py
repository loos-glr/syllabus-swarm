"""
syllabus_gate.py — Deterministic Syllabus Completeness Gate (System One)
=======================================================================

.. rubric:: Makes the Education Director handoff decision definitive

The orchestrator advances from the syllabus stage to content generation only
when this gate returns ``complete=True``.  The judgment is deterministic:

* **Structural checks** (required sections, extracted time budgets) run in
  plain Python — the System One model is never asked to do arithmetic or
  counting.
* **Semantic checks** (is the syllabus coherent? is it pitched correctly for
  MBO4?) are asked as a ``noul`` and a ``score``.

The returned :class:`~src.models.SyllabusGateDecision` carries stable machine
``blockers`` (never prose) that the orchestrator uses to schedule a targeted
rewrite with the Curriculum Architect.

Public API
----------
* ``REQUIRED_SECTIONS`` — structural expectations (aliases per section).
* ``check_required_sections(text)`` — code-layer structural check.
* ``assess_syllabus(...)`` — produce the typed gate decision.
* ``get_syllabus_gate()`` — lazy singleton.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.models import SyllabusGateDecision
from src.system_one import noul_question, score_question

if TYPE_CHECKING:
    from src.system_one import SystemOneClient

#: Required syllabus sections mapped to the heading aliases that satisfy them.
REQUIRED_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("course_overview", ("course overview", "cursusoverzicht")),
    ("learning_objectives", ("learning objectives", "leerdoelen")),
    ("humanics_literacies", ("humanics", "literacies", "geletterdheid")),
    ("module_breakdown", ("module", "modules")),
    ("experiential_learning", ("experiential", "capstone", "industry", "bpv")),
    ("assessment_strategy", ("assessment", "beoordeling", "toetsing")),
    ("tools_and_resources", ("tools", "resources", "middelen")),
    ("schedule", ("schedule", "planning", "rooster")),
)

#: Ordered quality rubric handed to the System One ``score`` primitive.
QUALITY_LEVELS: tuple[str, ...] = (
    "Unusable: incoherent, contradictory or missing whole sections.",
    "Weak: present but unrealistic for MBO4 students or internally inconsistent.",
    "Adequate: broadly sound with fixable gaps.",
    "Good: coherent, realistic and MBO4-appropriate.",
    "Excellent: exemplary, immediately actionable for teachers.",
)

_DURATION_RE: re.Pattern[str] = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:min(?:ute)?s?|uur|hours?|weken|weeks)\b",
    re.IGNORECASE,
)

_DEFAULT_VERDICT_THRESHOLD: float = 0.5
_DEFAULT_MIN_CONFIDENCE: float = 0.5


def _threshold(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return min(max(float(raw), 0.0), 1.0)
    except (TypeError, ValueError):
        return default


def gate_thresholds() -> tuple[float, float]:
    """Resolve ``(verdict_threshold, min_confidence)`` from env."""
    return (
        _threshold("SYSTEM_ONE_GATE_VERDICT_THRESHOLD", _DEFAULT_VERDICT_THRESHOLD),
        _threshold("SYSTEM_ONE_GATE_MIN_CONFIDENCE", _DEFAULT_MIN_CONFIDENCE),
    )


def check_required_sections(text: str) -> list[str]:
    """Return the keys of the required sections missing from *text*.

    This is a pure-code structural check: a section counts as present when any
    of its aliases appears (case-insensitively) in the document.  It performs
    no semantic judgment — that is the model's job.
    """
    lowered = text.lower()
    return [
        key for key, aliases in REQUIRED_SECTIONS if not any(alias in lowered for alias in aliases)
    ]


def extract_duration_mentions(text: str) -> list[str]:
    """Return the raw duration mentions found in *text* (code-layer only).

    Used purely as additional *context* for the gate — the model never
    performs the arithmetic.
    """
    return [match.group(0).strip() for match in _DURATION_RE.finditer(text)]


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _StructuralReport:
    """Code-layer structural findings fed to the gate as state."""

    missing_sections: tuple[str, ...]
    duration_mentions: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "missing_sections": list(self.missing_sections),
            "duration_mentions": list(self.duration_mentions),
        }


class SyllabusGate:
    """Deterministic syllabus completeness gate backed by a System One client."""

    def __init__(self, client: SystemOneClient, *, max_chars: int = 20_000) -> None:
        self._client = client
        self._max_chars = max_chars

    def assess(
        self,
        syllabus_text: str,
        *,
        course_name: str = "",
        profile_context: str | None = None,
    ) -> SyllabusGateDecision:
        """Return the typed :class:`~src.models.SyllabusGateDecision`."""
        missing = tuple(check_required_sections(syllabus_text))
        durations = tuple(extract_duration_mentions(syllabus_text))
        structural = _StructuralReport(missing_sections=missing, duration_mentions=durations)

        questions: dict[str, dict[str, Any]] = {
            "complete": noul_question(
                instructions=(
                    "Is this syllabus complete and coherent enough to proceed to "
                    "theory and lab generation, or does it have gaps that would "
                    "derail the downstream content?"
                ),
                true="The syllabus covers every expected area and is internally coherent.",
                false="At least one expected area is missing, or the content is incoherent.",
            ),
            "coherent": noul_question(
                instructions=(
                    "Is the proposed schedule realistic and free of contradictions "
                    "(no impossible weekly workloads, no clashing assessments)?"
                ),
                true="The schedule is realistic and free of contradictions.",
                false="The schedule contains unrealistic or contradictory elements.",
            ),
            "quality": score_question(
                instructions=(
                    "Rate the overall pedagogical quality and MBO4 appropriateness of "
                    "this syllabus."
                ),
                criteria=QUALITY_LEVELS,
            ),
        }

        state: dict[str, Any] = {
            "course_name": course_name,
            "structural_checks": structural.as_dict(),
            "syllabus": syllabus_text[: self._max_chars],
        }
        if profile_context:
            state["profile_context"] = profile_context

        result = self._client.system_one(state=state, questions=questions)

        complete_answer = result.noul("complete")
        coherent_answer = result.noul("coherent")
        quality_answer = result.score("quality")

        verdict_threshold, min_confidence = gate_thresholds()
        span = max(len(QUALITY_LEVELS) - 1, 1)
        quality_score = min(max(quality_answer.score / span, 0.0), 1.0)

        blockers: list[str] = [f"missing_section:{key}" for key in missing]
        if complete_answer.probability < verdict_threshold:
            blockers.append("incomplete")
        if coherent_answer.probability < verdict_threshold:
            blockers.append("incoherent_schedule")
        if quality_score < min_confidence:
            blockers.append("low_quality")

        complete = not blockers

        return SyllabusGateDecision(
            complete=complete,
            probability=complete_answer.probability,
            confidence=quality_answer.confidence,
            quality_score=quality_score,
            blockers=blockers,
            model_version=result.model,
            needs_review=quality_answer.confidence < min_confidence,
        )


def render_gate_report(decision: SyllabusGateDecision, *, course_name: str) -> str:
    """Render a deterministic Markdown feasibility report (no LLM involved)."""
    verdict = "APPROVED" if decision.complete else "NEEDS REVISION"
    quality = "—" if decision.quality_score is None else f"{decision.quality_score:.2f}"
    confidence = "—" if decision.confidence is None else f"{decision.confidence:.2f}"

    lines: list[str] = [
        f"# Syllabus Feasibility Audit — {course_name}",
        "",
        "## Summary",
        f"- Completeness probability: {decision.probability:.2f}",
        f"- Quality score: {quality}",
        f"- Confidence: {confidence}",
        f"- Flagged for human review: {'yes' if decision.needs_review else 'no'}",
        f"- Verdict: {verdict}",
        "",
    ]

    if decision.blockers:
        lines.append("## Blockers")
        lines.extend(f"- `{blocker}`" for blocker in decision.blockers)
        lines.append("")
        lines.append(
            "Delegate these blockers back to the Curriculum Architect for a targeted rewrite."
        )
    else:
        lines.append(
            "✅ **FEASIBILITY SIGN-OFF: This syllabus is complete, coherent and "
            "suitable for MBO4 students. Approved to proceed to Theory and Lab "
            "generation.**"
        )

    return "\n".join(lines) + "\n"


def assess_syllabus(
    syllabus_text: str,
    *,
    client: SystemOneClient,
    course_name: str = "",
    profile_context: str | None = None,
) -> SyllabusGateDecision:
    """Assess a syllabus using *client* (convenience wrapper)."""
    return SyllabusGate(client).assess(
        syllabus_text,
        course_name=course_name,
        profile_context=profile_context,
    )


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_syllabus_gate: SyllabusGate | None = None


def get_syllabus_gate() -> SyllabusGate:
    """Return a shared, lazily-created :class:`SyllabusGate`.

    Raises
    ------
    SystemOneError
        When no System One client is configured (missing API key).
    """
    global _syllabus_gate
    if _syllabus_gate is None:
        from src.llm_factory import SYLLABUS_GATE, build_system_one_client  # noqa: PLC0415
        from src.system_one import SystemOneError  # noqa: PLC0415

        client = build_system_one_client(task=SYLLABUS_GATE)
        if client is None:
            raise SystemOneError(
                "The syllabus gate requires a System One client. Set "
                "SYSTEM_ONE_API_KEY (or TYPESAFE_API_KEY), or set "
                "SYSTEM_ONE_ENABLED=0 for degraded offline mode."
            )
        _syllabus_gate = SyllabusGate(client)
    return _syllabus_gate
