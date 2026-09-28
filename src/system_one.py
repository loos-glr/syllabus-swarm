"""
system_one.py — System One (Jev) Decision Client
================================================

.. rubric:: Deterministic evaluation layer (non-generative)

The syllabus-swarm pipeline historically used generative LLMs for both
*content creation* and *control flow*.  This module provides the second,
**non-generative** inference client that offloads control-flow decisions
(modality routing, QA scoring, syllabus gating) to a fast, schema-driven
System One model — TypeSafe AI's ``Jev``.

Design constraints
------------------
* **No generative parameters.**  System One exposes no ``temperature``,
  ``top_p`` or ``max_tokens`` — only ``choice`` / ``score`` / ``noul``
  questions and typed answers.
* **Decoupled from the SDK.**  The public surface (question builders and
  answer value objects) is plain Python.  ``typesafe-sdk`` is imported
  *lazily* inside :class:`TypeSafeSystemOneClient`, so the rest of the
  codebase — and the whole test suite — works without the SDK installed.
* **Network-free testing.**  :class:`FakeSystemOneClient` implements the same
  protocol, supports scripted answers for tests, and offers a conservative
  *heuristic* mode for offline / degraded operation where every answer is
  returned with ``confidence == 0.0`` so callers escalate instead of silently
  acting.

Public API
----------
* ``SystemOneClient`` — protocol implemented by all clients.
* ``SystemOneError`` — raised for any decision-layer failure.
* ``NoulAnswer`` / ``ChoiceAnswer`` / ``ScoreAnswer`` / ``SystemOneResult``.
* ``noul_question`` / ``choice_question`` / ``score_question`` — typed builders.
* ``TypeSafeSystemOneClient`` — real client wrapping ``typesafe_sdk``.
* ``FakeSystemOneClient`` — deterministic test/offline double.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SystemOneError(RuntimeError):
    """Raised when a System One decision cannot be produced.

    Covers a missing SDK, a missing API key, a transport failure, or a
    response that does not contain the requested question keys.
    """


# ---------------------------------------------------------------------------
# Answer value objects (SDK-independent)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoulAnswer:
    """A calibrated yes/no answer expressed as a probability (0.0–1.0).

    ``noul`` answers intentionally carry **no** confidence field — the
    probability *is* the calibrated signal.
    """

    probability: float


@dataclass(frozen=True)
class ChoiceAnswer:
    """A selection between named options, with probabilities and confidence."""

    choice: str
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass(frozen=True)
class ScoreAnswer:
    """A rating against an ordered rubric, with probabilities and confidence."""

    score: float
    confidence: float = 0.0
    probabilities: dict[int, float] = field(default_factory=dict)
    legend: dict[int, str] = field(default_factory=dict)


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


@dataclass(frozen=True)
class SystemOneResult:
    """Normalized result of a single System One round-trip."""

    model: str
    answers: dict[str, Answer] = field(default_factory=dict)
    usage: dict[str, int] | None = None

    def noul(self, name: str) -> NoulAnswer:
        """Return the :class:`NoulAnswer` registered under *name*."""
        answer = self.answers.get(name)
        if not isinstance(answer, NoulAnswer):
            raise SystemOneError(
                f"Expected a noul answer for {name!r}, got {type(answer).__name__}"
            )
        return answer

    def choice(self, name: str) -> ChoiceAnswer:
        """Return the :class:`ChoiceAnswer` registered under *name*."""
        answer = self.answers.get(name)
        if not isinstance(answer, ChoiceAnswer):
            raise SystemOneError(
                f"Expected a choice answer for {name!r}, got {type(answer).__name__}"
            )
        return answer

    def score(self, name: str) -> ScoreAnswer:
        """Return the :class:`ScoreAnswer` registered under *name*."""
        answer = self.answers.get(name)
        if not isinstance(answer, ScoreAnswer):
            raise SystemOneError(
                f"Expected a score answer for {name!r}, got {type(answer).__name__}"
            )
        return answer


# ---------------------------------------------------------------------------
# Typed question builders
# ---------------------------------------------------------------------------
# These return plain dictionaries in the wire format accepted by the System
# One HTTP API and the ``typesafe-sdk`` client, keeping callers independent of
# the SDK's Pydantic question classes.


def noul_question(*, instructions: str, true: str, false: str) -> dict[str, Any]:
    """Build a ``noul`` (yes/no) question with explicit true/false criteria."""
    return {
        "type": "noul",
        "instructions": instructions,
        "criteria": {"true": true, "false": false},
    }


def choice_question(*, instructions: str, criteria: Mapping[str, str]) -> dict[str, Any]:
    """Build a ``choice`` question from an ordered mapping of option -> description."""
    if not criteria:
        raise SystemOneError("A choice question requires at least one option.")
    return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}


def score_question(*, instructions: str, criteria: Sequence[str]) -> dict[str, Any]:
    """Build a ``score`` question from an ordered list of level descriptions.

    The list is the rubric, lowest level first.  System One requires between
    2 and 10 levels.
    """
    levels = list(criteria)
    if not 2 <= len(levels) <= 10:
        raise SystemOneError(f"A score question requires 2–10 levels, got {len(levels)}.")
    return {"type": "score", "instructions": instructions, "criteria": levels}


# ---------------------------------------------------------------------------
# Client protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SystemOneClient(Protocol):
    """Protocol implemented by every System One decision client."""

    model: str

    def system_one(
        self,
        *,
        state: Any,
        questions: Mapping[str, dict[str, Any]],
        model: str | None = None,
    ) -> SystemOneResult:
        """Answer *questions* about *state* and return typed answers."""
        ...


# ---------------------------------------------------------------------------
# Real client — lazy ``typesafe_sdk`` adapter
# ---------------------------------------------------------------------------


class TypeSafeSystemOneClient:
    """System One client backed by the official ``typesafe-sdk``.

    The SDK is imported lazily so that the decision layer (and the entire test
    suite) remains usable when the optional dependency is not installed.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        if not api_key:
            raise SystemOneError(
                "No System One API key configured. Set SYSTEM_ONE_API_KEY (or TYPESAFE_API_KEY)."
            )
        try:
            from typesafe_sdk import TypeSafeClient  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - only without the SDK
            raise SystemOneError(
                "The 'typesafe-sdk' package is required for live System One "
                "decisions. Install it with `pip install typesafe-sdk`, or set "
                "SYSTEM_ONE_ENABLED=0 to run in degraded offline mode."
            ) from exc

        self.model = model
        self._client = TypeSafeClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
        )

    # -- context manager -------------------------------------------------
    def __enter__(self) -> TypeSafeSystemOneClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the underlying HTTP session."""
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    # -- decision call ---------------------------------------------------
    def system_one(
        self,
        *,
        state: Any,
        questions: Mapping[str, dict[str, Any]],
        model: str | None = None,
    ) -> SystemOneResult:
        """Run one System One round-trip and normalize the SDK response."""
        try:
            response = self._client.system_one(
                state=state,
                questions=dict(questions),
                model=model,
            )
        except SystemOneError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize every SDK failure
            raise SystemOneError(f"System One request failed: {exc}") from exc

        answers: dict[str, Answer] = {}
        for name, raw in dict(getattr(response, "answers", {}) or {}).items():
            answers[name] = _normalize_answer(name, raw)

        missing = set(questions) - set(answers)
        if missing:
            raise SystemOneError(f"System One response omitted answers for: {sorted(missing)}")

        return SystemOneResult(model=str(getattr(response, "model", self.model)), answers=answers)


def _normalize_answer(name: str, raw: Any) -> Answer:
    """Convert an SDK answer object into one of the local value objects."""
    kind = getattr(raw, "type", None)
    if kind == "noul":
        return NoulAnswer(probability=float(raw.noul))
    if kind == "choice":
        return ChoiceAnswer(
            choice=str(raw.choice),
            probabilities={str(k): float(v) for k, v in (raw.probabilities or {}).items()},
            confidence=float(raw.confidence),
        )
    if kind == "score":
        return ScoreAnswer(
            score=float(raw.score),
            confidence=float(raw.confidence),
            probabilities={int(k): float(v) for k, v in (raw.probabilities or {}).items()},
            legend={int(k): str(v) for k, v in (getattr(raw, "legend", {}) or {}).items()},
        )
    raise SystemOneError(f"Unrecognized System One answer type for {name!r}: {kind!r}")


# ---------------------------------------------------------------------------
# Fake client — tests, dry-runs and degraded offline operation
# ---------------------------------------------------------------------------


class FakeSystemOneClient:
    """Deterministic, network-free :class:`SystemOneClient` implementation.

    Parameters
    ----------
    answers : Mapping[str, Answer] or None
        Scripted answers keyed by question name.  Used by the test suite.
    model : str
        Model identifier reported on every result.
    heuristic : bool
        When True (default), un-scripted questions receive a conservative
        synthesized answer with ``confidence == 0.0`` so callers escalate to
        review instead of acting on a guess.  This is the degraded offline mode
        selected by ``SYSTEM_ONE_ENABLED=0``.
    """

    def __init__(
        self,
        answers: Mapping[str, Answer] | None = None,
        *,
        model: str = "fake-system-one",
        heuristic: bool = True,
    ) -> None:
        self.answers: dict[str, Answer] = dict(answers or {})
        self.model = model
        self.heuristic = heuristic
        self.calls: list[dict[str, Any]] = []

    def system_one(
        self,
        *,
        state: Any,
        questions: Mapping[str, dict[str, Any]],
        model: str | None = None,
    ) -> SystemOneResult:
        """Return the scripted (or heuristic) answer for every requested question."""
        self.calls.append({"state": state, "questions": dict(questions), "model": model})

        resolved: dict[str, Answer] = {}
        for name, question in questions.items():
            if name in self.answers:
                resolved[name] = self.answers[name]
            elif self.heuristic:
                resolved[name] = _heuristic_answer(question)
            else:
                raise SystemOneError(
                    f"FakeSystemOneClient has no scripted answer for {name!r} "
                    "and heuristic mode is disabled."
                )

        return SystemOneResult(model=model or self.model, answers=resolved)


def _heuristic_answer(question: Mapping[str, Any]) -> Answer:
    """Synthesize a deliberately *uncertain* answer for degraded operation."""
    kind = question.get("type")
    if kind == "noul":
        # 0.5 is maximally uncertain -> never auto-approve.
        return NoulAnswer(probability=0.5)
    if kind == "choice":
        options = list((question.get("criteria") or {}).keys())
        if not options:
            raise SystemOneError("Heuristic choice answer needs at least one option.")
        share = 1.0 / len(options)
        return ChoiceAnswer(
            choice=options[0],
            probabilities=dict.fromkeys(options, share),
            confidence=0.0,
        )
    if kind == "score":
        levels = list(question.get("criteria") or [])
        if len(levels) < 2:
            raise SystemOneError("Heuristic score answer needs at least two levels.")
        return ScoreAnswer(score=(len(levels) - 1) / 2, confidence=0.0)
    raise SystemOneError(f"Cannot build a heuristic answer for question type {kind!r}.")
