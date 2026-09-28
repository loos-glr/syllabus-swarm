"""
qa_scorer.py — Deterministic QA Scoring (System One)
=====================================================

.. rubric:: Replaces LLM-as-judge with typed, calibrated scores

Scores generated content against a **typed rubric** derived from the project's
configuration profiles (``config/profiles/*.yaml`` and ``config/school_defaults.yaml``)
using the System One ``score`` primitive for each criterion and a single
``noul`` question for the overall verdict.

The generative LLM is never asked to judge: it may still *format* the QA
report, but the authoritative pass/fail signal consumed by the orchestrator is
:attr:`~src.models.QAScore.verdict`.

Public API
----------
* ``QARubric`` — typed rubric + thresholds.
* ``default_qa_rubric(kind)`` / ``rubric_from_profile(profile, kind=...)``.
* ``score_artifact(...)`` — score one artifact.
* ``score_artifacts(...)`` — score many artifacts.
* ``qa_passed(scores)`` — authoritative aggregate verdict.
* ``render_qa_report(...)`` — deterministic Markdown report.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.models import QAScore, RubricCriterion
from src.system_one import SystemOneError, noul_question, score_question

if TYPE_CHECKING:
    from src.system_one import SystemOneClient

_DEFAULT_PASS_THRESHOLD: float = 0.6
_DEFAULT_MIN_CONFIDENCE: float = 0.5
_DEFAULT_VERDICT_THRESHOLD: float = 0.5


def _five_levels(subject: str) -> list[str]:
    """Build a standard 5-level rubric scale for *subject* (lowest first)."""
    return [
        f"Broken: {subject} is missing or wrong.",
        f"Weak: {subject} is present but clearly substandard.",
        f"Adequate: {subject} meets the minimum bar with minor issues.",
        f"Good: {subject} is solid and needs no significant change.",
        f"Excellent: {subject} is exemplary and student-ready.",
    ]


def _criterion(key: str, label: str, subject: str, *, weight: float = 1.0) -> RubricCriterion:
    """Convenience builder for a criterion with the standard 5-level scale."""
    return RubricCriterion(
        key=key,
        label=label,
        description=f"Measures {subject}.",
        levels=_five_levels(subject),
        weight=weight,
    )


#: Rubric applied to generated lab artifacts (starter + solution files).
LAB_CRITERIA: tuple[RubricCriterion, ...] = (
    _criterion(
        "technical_correctness",
        "Technical correctness",
        "whether the code runs as-is with no syntax errors, missing imports or hallucinated APIs",
        weight=2.0,
    ),
    _criterion(
        "didactic_clarity",
        "Didactic clarity",
        "whether the instructions are unambiguous and appropriate for MBO4 students",
        weight=2.0,
    ),
    _criterion(
        "self_containment",
        "Self-containment",
        "whether every file is runnable without external, unexplained context",
    ),
    _criterion(
        "profile_alignment",
        "Profile alignment",
        "whether the content matches the cohort profile's tech stack, year level and "
        "kerntaken emphasis",
    ),
    _criterion(
        "language_quality",
        "Language quality",
        "whether the material language is correct, consistent and free of errors",
    ),
)

#: Rubric applied to generated theory artifacts (HTML / CLI / Mermaid).
THEORY_CRITERIA: tuple[RubricCriterion, ...] = (
    _criterion(
        "technical_correctness",
        "Technical correctness",
        "whether the artifact loads and runs without JavaScript or shell errors",
        weight=2.0,
    ),
    _criterion(
        "didactic_clarity",
        "Didactic clarity",
        "whether the artifact explains the concept accessibly for MBO4 students",
        weight=2.0,
    ),
    _criterion(
        "interactivity",
        "Interactivity & engagement",
        "whether the artifact invites hands-on exploration rather than passive reading",
    ),
    _criterion(
        "self_containment",
        "Self-containment",
        "whether the artifact works offline with no broken external references",
    ),
    _criterion(
        "language_quality",
        "Language quality",
        "whether the material language is correct, consistent and free of errors",
    ),
)


@dataclass(frozen=True)
class QARubric:
    """A typed QA rubric plus the thresholds used to derive a verdict."""

    criteria: tuple[RubricCriterion, ...]
    pass_threshold: float = _DEFAULT_PASS_THRESHOLD
    min_confidence: float = _DEFAULT_MIN_CONFIDENCE
    verdict_threshold: float = _DEFAULT_VERDICT_THRESHOLD

    @property
    def keys(self) -> tuple[str, ...]:
        """The ordered criterion keys."""
        return tuple(c.key for c in self.criteria)


def default_qa_rubric(kind: str = "lab") -> QARubric:
    """Return the default rubric for *kind* (``"lab"`` or ``"theory"``)."""
    criteria = THEORY_CRITERIA if kind == "theory" else LAB_CRITERIA
    return QARubric(criteria=criteria)


def qa_thresholds() -> tuple[float, float, float]:
    """Resolve ``(pass_threshold, min_confidence, verdict_threshold)`` from env."""

    def _num(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return min(max(float(raw), 0.0), 1.0)
        except (TypeError, ValueError):
            return default

    return (
        _num("SYSTEM_ONE_QA_PASS_THRESHOLD", _DEFAULT_PASS_THRESHOLD),
        _num("SYSTEM_ONE_QA_MIN_CONFIDENCE", _DEFAULT_MIN_CONFIDENCE),
        _num("SYSTEM_ONE_QA_VERDICT_THRESHOLD", _DEFAULT_VERDICT_THRESHOLD),
    )


def rubric_from_profile(profile: Mapping[str, Any] | None, *, kind: str = "lab") -> QARubric:
    """Build a rubric from a cohort profile (or the defaults when ``None``)."""
    pass_threshold, min_confidence, verdict_threshold = qa_thresholds()
    base = default_qa_rubric(kind)

    if not profile:
        return QARubric(
            criteria=base.criteria,
            pass_threshold=pass_threshold,
            min_confidence=min_confidence,
            verdict_threshold=verdict_threshold,
        )

    tech = profile.get("tech_stack") or {}
    language = tech.get("primary_language") or "the primary language"
    year = profile.get("year_level")
    alignment = _criterion(
        "profile_alignment",
        "Profile alignment",
        f"whether the content matches the cohort's {language} stack"
        + (f", year {year} level" if year else "")
        + " and kerntaken emphasis",
    )
    criteria = tuple(alignment if c.key == "profile_alignment" else c for c in base.criteria)
    if not any(c.key == "profile_alignment" for c in criteria):
        criteria = (*criteria, alignment)

    return QARubric(
        criteria=criteria,
        pass_threshold=pass_threshold,
        min_confidence=min_confidence,
        verdict_threshold=verdict_threshold,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

#: Default character budget for the artifact text sent to System One.
DEFAULT_MAX_CHARS: int = 12_000


class QAScoring:
    """Deterministic QA scorer backed by a System One client."""

    def __init__(self, client: SystemOneClient, *, max_chars: int = DEFAULT_MAX_CHARS) -> None:
        self._client = client
        self._max_chars = max_chars

    def score(
        self,
        content: str,
        *,
        content_ref: str,
        kind: str = "lab",
        rubric: QARubric | None = None,
        extra_context: str | None = None,
    ) -> QAScore:
        """Score one artifact and return a typed :class:`~src.models.QAScore`."""
        rubric = rubric or default_qa_rubric(kind)

        questions: dict[str, dict[str, Any]] = {
            f"criterion_{c.key}": score_question(
                instructions=f"Rate {c.label}: {c.description}",
                criteria=c.levels,
            )
            for c in rubric.criteria
        }
        questions["ready"] = noul_question(
            instructions=(
                "Is this artifact ready to hand to MBO4 students without any further fixes?"
            ),
            true="The artifact is correct, clear and student-ready as-is.",
            false="The artifact needs at least one fix before students may use it.",
        )

        state: dict[str, Any] = {
            "content_ref": content_ref,
            "kind": kind,
            "content": content[: self._max_chars],
        }
        if extra_context:
            state["profile_context"] = extra_context

        result = self._client.system_one(state=state, questions=questions)

        per_criterion: dict[str, float] = {}
        confidences: list[float] = []
        below_threshold: list[str] = []
        weighted_sum = 0.0
        weight_total = 0.0

        for criterion in rubric.criteria:
            answer = result.score(f"criterion_{criterion.key}")
            span = max(len(criterion.levels) - 1, 1)
            normalized = min(max(answer.score / span, 0.0), 1.0)
            per_criterion[criterion.key] = normalized
            confidences.append(answer.confidence)
            weighted_sum += normalized * criterion.weight
            weight_total += criterion.weight
            if normalized < rubric.pass_threshold:
                below_threshold.append(f"{criterion.key}_below_threshold")

        aggregate = weighted_sum / weight_total if weight_total else 0.0
        ready = result.noul("ready")
        confidence = min(confidences) if confidences else None

        flags = list(below_threshold)
        if ready.probability < rubric.verdict_threshold:
            flags.append("not_ready")

        passed = not below_threshold and ready.probability >= rubric.verdict_threshold

        return QAScore(
            content_ref=content_ref,
            score=aggregate,
            confidence=confidence,
            per_criterion=per_criterion,
            verdict="pass" if passed else "needs_fixes",
            flags=flags,
            model_version=result.model,
            needs_review=confidence is not None and confidence < rubric.min_confidence,
        )


def score_artifact(
    content: str,
    *,
    content_ref: str,
    client: SystemOneClient,
    kind: str = "lab",
    rubric: QARubric | None = None,
    extra_context: str | None = None,
) -> QAScore:
    """Score a single artifact using *client* (convenience wrapper)."""
    return QAScoring(client).score(
        content,
        content_ref=content_ref,
        kind=kind,
        rubric=rubric,
        extra_context=extra_context,
    )


def score_artifacts(
    items: Iterable[tuple[str, str]],
    *,
    client: SystemOneClient,
    kind: str = "lab",
    rubric: QARubric | None = None,
    extra_context: str | None = None,
) -> list[QAScore]:
    """Score many ``(content_ref, content)`` pairs, preserving order."""
    scorer = QAScoring(client)
    return [
        scorer.score(
            content,
            content_ref=ref,
            kind=kind,
            rubric=rubric,
            extra_context=extra_context,
        )
        for ref, content in items
    ]


def qa_passed(scores: Sequence[QAScore]) -> bool:
    """Return True when *every* artifact passed the rubric."""
    return all(score.verdict == "pass" for score in scores)


def qa_needs_review(scores: Sequence[QAScore]) -> bool:
    """Return True when any artifact scored below the confidence floor."""
    return any(score.needs_review for score in scores)


def render_qa_report(scores: Sequence[QAScore], *, course_name: str) -> str:
    """Render a deterministic Markdown QA report (no LLM involved)."""
    if not scores:
        return f"# QA Review Report — {course_name}\n\n*No artifacts were scored.*\n"

    passed = sum(1 for s in scores if s.verdict == "pass")
    lines: list[str] = [
        f"# QA Review Report — {course_name}",
        "",
        "## Summary",
        f"- Artifacts scored: {len(scores)}",
        f"- Passed: {passed}",
        f"- Need fixes: {len(scores) - passed}",
        f"- Flagged for human review: {sum(1 for s in scores if s.needs_review)}",
        f"- Final verdict: {'PASSED' if qa_passed(scores) else 'NEEDS FIXES'}",
        "",
        "## Per-artifact scores",
        "",
        "| Artifact | Score | Confidence | Verdict | Flags |",
        "|---|---|---|---|---|",
    ]
    for s in scores:
        conf = "—" if s.confidence is None else f"{s.confidence:.2f}"
        flags = ", ".join(s.flags) if s.flags else "—"
        lines.append(
            f"| `{s.content_ref}` | {s.score:.2f} | {conf} | "
            f"{'✅ pass' if s.verdict == 'pass' else '❌ needs fixes'} | {flags} |"
        )

    lines.extend(["", "## Per-criterion detail", ""])
    for s in scores:
        lines.append(f"### `{s.content_ref}`")
        for key, value in s.per_criterion.items():
            lines.append(f"- {key}: {value:.2f}")
        lines.append("")

    if qa_passed(scores):
        lines.append(
            "✅ **QA SIGN-OFF: All artifacts are technically correct and "
            "didactically appropriate for MBO4 students. Ready for classroom use.**"
        )
    else:
        lines.append(
            "❌ **Fixes required before classroom use.** Delegating the flagged "
            "artifacts back to their responsible agents."
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_qa_scorer: QAScoring | None = None


def get_qa_scorer() -> QAScoring:
    """Return a shared, lazily-created :class:`QAScoring` scorer.

    Raises
    ------
    SystemOneError
        When no System One client is configured (missing API key).
    """
    global _qa_scorer
    if _qa_scorer is None:
        from src.llm_factory import QA_SCORER, build_system_one_client  # noqa: PLC0415

        client = build_system_one_client(task=QA_SCORER)
        if client is None:
            raise SystemOneError(
                "QA scoring requires a System One client. Set SYSTEM_ONE_API_KEY "
                "(or TYPESAFE_API_KEY), or set SYSTEM_ONE_ENABLED=0 for degraded "
                "offline mode."
            )
        _qa_scorer = QAScoring(client)
    return _qa_scorer
