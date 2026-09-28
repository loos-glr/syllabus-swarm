"""Tests for the LLM resilience helpers in ``src.crews.syllabus_crew``.

Covers:

* the three-way LLM-error classifier (retryable / futile / deterministic),
* the empty-LLM-response error annotation,
* :func:`_kickoff_with_retry`, which retries genuinely transient failures but
  aborts immediately on *futile* ones so credits are not burned, and
* the run-level circuit breaker (:class:`_FatalAbortGuard`) that skips every
  remaining pipeline stage once a futile failure is detected.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from src.crews.syllabus_crew import (
    _DEFAULT_FUTILE_RETRY_ATTEMPTS,
    _FUTILE_RETRY_ATTEMPTS_ENV,
    FatalLLMError,
    _annotate_empty_llm_response,
    _classify_llm_error,
    _FatalAbortGuard,
    _is_futile_llm_error,
    _is_transient_llm_error,
    _kickoff_with_retry,
    _resolve_futile_attempts,
)

_EMPTY_RESPONSE_MSG: str = "Invalid response from LLM call - None or empty."

# ---------------------------------------------------------------------------
# _classify_llm_error / _is_futile_llm_error / _is_transient_llm_error
# ---------------------------------------------------------------------------


class TestClassifyLlmError:
    """The classifier drives retry-vs-abort, so its buckets are load-bearing."""

    def test_empty_response_is_futile(self) -> None:
        assert _classify_llm_error(ValueError(_EMPTY_RESPONSE_MSG)) == "futile"
        assert (
            _classify_llm_error(
                ValueError("Received None or empty response from LLM call.")
            )
            == "futile"
        )

    def test_futile_predicate_matches_empty_response(self) -> None:
        assert _is_futile_llm_error(ValueError(_EMPTY_RESPONSE_MSG))
        assert not _is_futile_llm_error(Exception("429 Too Many Requests"))

    def test_rate_limit_is_retryable_not_futile(self) -> None:
        assert _classify_llm_error(Exception("429 Too Many Requests")) == "retryable"
        assert _classify_llm_error(Exception("rate limit exceeded")) == "retryable"
        assert _classify_llm_error(Exception("Request timed out")) == "retryable"
        assert not _is_futile_llm_error(Exception("rate limit exceeded"))

    def test_litellm_module_errors_are_retryable(self) -> None:
        cls = type("LitellmError", (Exception,), {"__module__": "litellm.exceptions"})
        assert _classify_llm_error(cls("boom")) == "retryable"

    def test_unrelated_error_is_deterministic(self) -> None:
        assert _classify_llm_error(ValueError("some unrelated error")) == "deterministic"

    def test_transient_predicate_is_union_of_retryable_and_futile(self) -> None:
        """Backward-compatible helper still reports empty responses as transient."""
        assert _is_transient_llm_error(ValueError(_EMPTY_RESPONSE_MSG))
        assert _is_transient_llm_error(
            ValueError("Received None or empty response from LLM call.")
        )
        assert _is_transient_llm_error(Exception("429 Too Many Requests"))
        assert not _is_transient_llm_error(ValueError("some unrelated error"))


# ---------------------------------------------------------------------------
# _resolve_futile_attempts
# ---------------------------------------------------------------------------


class TestResolveFutileAttempts:
    """The futile-retry budget defaults to 1 (abort on first indication)."""

    def test_defaults_to_one_when_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert _resolve_futile_attempts() == _DEFAULT_FUTILE_RETRY_ATTEMPTS == 1

    def test_env_override_is_honoured(self) -> None:
        with patch.dict(os.environ, {_FUTILE_RETRY_ATTEMPTS_ENV: "4"}, clear=True):
            assert _resolve_futile_attempts() == 4

    def test_invalid_env_value_falls_back_to_default(self) -> None:
        with patch.dict(
            os.environ, {_FUTILE_RETRY_ATTEMPTS_ENV: "not-a-number"}, clear=True
        ):
            assert _resolve_futile_attempts() == _DEFAULT_FUTILE_RETRY_ATTEMPTS

    def test_zero_env_value_is_clamped_to_one(self) -> None:
        with patch.dict(os.environ, {_FUTILE_RETRY_ATTEMPTS_ENV: "0"}, clear=True):
            assert _resolve_futile_attempts() == 1


# ---------------------------------------------------------------------------
# _annotate_empty_llm_response
# ---------------------------------------------------------------------------


class TestAnnotateEmptyLlmResponse:
    def test_appends_actionable_hint(self) -> None:
        out = _annotate_empty_llm_response(_EMPTY_RESPONSE_MSG, "QA_REVIEWER")
        assert "EMPTY LLM RESPONSE DETECTED" in out
        assert "AGENT_QA_REVIEWER_MAX_TOKENS=16384" in out
        assert "AGENT_QA_REVIEWER_MODEL" in out

    def test_appends_hint_for_theory_instructor(self) -> None:
        out = _annotate_empty_llm_response(_EMPTY_RESPONSE_MSG, "THEORY_INSTRUCTOR")
        assert "AGENT_THEORY_INSTRUCTOR_MAX_TOKENS=16384" in out

    def test_leaves_unrelated_errors_unchanged(self) -> None:
        msg = "some other error"
        assert _annotate_empty_llm_response(msg, "QA_REVIEWER") == msg


# ---------------------------------------------------------------------------
# _kickoff_with_retry — retry semantics per error bucket
# ---------------------------------------------------------------------------


class TestKickoffWithRetry:
    def test_succeeds_on_first_attempt(self) -> None:
        crew = MagicMock()
        crew.kickoff.return_value = "result"
        built: list[int] = []

        def build() -> MagicMock:
            built.append(1)
            return crew

        result = _kickoff_with_retry(build)
        assert result == "result"
        assert len(built) == 1
        assert crew.kickoff.call_count == 1

    def test_retries_retryable_failure_then_succeeds(self) -> None:
        """A rate-limit blip is absorbed by a bounded retry."""
        crew = MagicMock()
        crew.kickoff.side_effect = [
            Exception("429 Too Many Requests"),
            "success",
        ]

        with patch("src.crews.syllabus_crew.time.sleep"):
            result = _kickoff_with_retry(lambda: crew, backoff_seconds=1.0)

        assert result == "success"
        assert crew.kickoff.call_count == 2

    def test_exhausts_attempts_on_persistent_retryable_error(self) -> None:
        crew = MagicMock()
        crew.kickoff.side_effect = Exception("rate limit exceeded")

        with patch("src.crews.syllabus_crew.time.sleep"):
            with pytest.raises(Exception, match="rate limit exceeded"):
                _kickoff_with_retry(lambda: crew, attempts=3)

        assert crew.kickoff.call_count == 3

    def test_raises_immediately_on_non_transient_error(self) -> None:
        crew = MagicMock()
        crew.kickoff.side_effect = ValueError("deterministic failure")

        with patch("src.crews.syllabus_crew.time.sleep") as mock_sleep:
            with pytest.raises(ValueError):
                _kickoff_with_retry(lambda: crew)

        mock_sleep.assert_not_called()
        assert crew.kickoff.call_count == 1

    # ── Fail-fast behaviour (the credit-saving path) ──────────────────

    def test_futile_failure_aborts_on_first_indication(self) -> None:
        """An empty response must NOT be retried by default."""
        crew = MagicMock()
        crew.kickoff.side_effect = ValueError(_EMPTY_RESPONSE_MSG)

        with patch("src.crews.syllabus_crew.time.sleep") as mock_sleep:
            with pytest.raises(FatalLLMError):
                _kickoff_with_retry(lambda: crew, agent_role="THEORY_INSTRUCTOR")

        mock_sleep.assert_not_called()
        assert crew.kickoff.call_count == 1

    def test_fatal_error_carries_agent_role_and_hint(self) -> None:
        crew = MagicMock()
        crew.kickoff.side_effect = ValueError(_EMPTY_RESPONSE_MSG)

        with patch("src.crews.syllabus_crew.time.sleep"):
            with pytest.raises(FatalLLMError) as excinfo:
                _kickoff_with_retry(lambda: crew, agent_role="THEORY_INSTRUCTOR")

        assert excinfo.value.agent_role == "THEORY_INSTRUCTOR"
        assert "EMPTY LLM RESPONSE DETECTED" in str(excinfo.value)
        assert "AGENT_THEORY_INSTRUCTOR_MAX_TOKENS=16384" in str(excinfo.value)

    def test_futile_retry_budget_can_be_raised(self) -> None:
        """futile_attempts=2 allows exactly one extra attempt, then aborts."""
        crew = MagicMock()
        crew.kickoff.side_effect = ValueError(_EMPTY_RESPONSE_MSG)

        with patch("src.crews.syllabus_crew.time.sleep"):
            with pytest.raises(FatalLLMError):
                _kickoff_with_retry(lambda: crew, futile_attempts=2)

        assert crew.kickoff.call_count == 2

    def test_futile_budget_from_env_is_respected(self) -> None:
        crew = MagicMock()
        crew.kickoff.side_effect = ValueError(_EMPTY_RESPONSE_MSG)

        with patch.dict(os.environ, {_FUTILE_RETRY_ATTEMPTS_ENV: "3"}, clear=True):
            with patch("src.crews.syllabus_crew.time.sleep"):
                with pytest.raises(FatalLLMError):
                    _kickoff_with_retry(lambda: crew)

        assert crew.kickoff.call_count == 3

    def test_futile_then_success_returns_result(self) -> None:
        """With a raised budget, a recovered provider blip still succeeds."""
        crew = MagicMock()
        crew.kickoff.side_effect = [ValueError(_EMPTY_RESPONSE_MSG), "recovered"]

        with patch("src.crews.syllabus_crew.time.sleep"):
            result = _kickoff_with_retry(lambda: crew, futile_attempts=2)

        assert result == "recovered"
        assert crew.kickoff.call_count == 2



# ---------------------------------------------------------------------------
# _FatalAbortGuard — run-level circuit breaker
# ---------------------------------------------------------------------------


class TestFatalAbortGuard:
    """Once tripped, the guard must skip every remaining generation stage."""

    def test_starts_untripped(self) -> None:
        guard = _FatalAbortGuard()
        assert not guard.tripped
        assert guard.reason is None
        assert guard.stage is None
        assert guard.agent_role is None
        assert guard.should_skip("Anything") is False

    def test_trip_marks_guard_and_returns_reason(self, capsys: pytest.CaptureFixture) -> None:
        guard = _FatalAbortGuard()
        reason = guard.trip(
            stage="Theory artifacts (tier1_foundations)",
            agent_role="THEORY_INSTRUCTOR",
            exc=ValueError(_EMPTY_RESPONSE_MSG),
        )

        assert guard.tripped
        assert guard.stage == "Theory artifacts (tier1_foundations)"
        assert guard.agent_role == "THEORY_INSTRUCTOR"
        assert "THEORY_INSTRUCTOR" in reason
        assert "avoid burning credits" in reason

        # The loud banner must reach stderr so the operator cannot miss it.
        err = capsys.readouterr().err
        assert "RUN ABORTED" in err
        assert "THEORY_INSTRUCTOR" in err

    def test_trip_is_idempotent(self, capsys: pytest.CaptureFixture) -> None:
        """The first futile failure wins; later ones do not overwrite it."""
        guard = _FatalAbortGuard()
        guard.trip(
            stage="Theory artifacts",
            agent_role="THEORY_INSTRUCTOR",
            exc=ValueError(_EMPTY_RESPONSE_MSG),
        )
        capsys.readouterr()  # discard the banner from the first trip

        guard.trip(
            stage="Labs",
            agent_role="LAB_DEVELOPER",
            exc=ValueError(_EMPTY_RESPONSE_MSG),
        )

        assert guard.stage == "Theory artifacts"
        assert guard.agent_role == "THEORY_INSTRUCTOR"
        assert capsys.readouterr().err == ""

    def test_should_skip_true_after_trip(self, capsys: pytest.CaptureFixture) -> None:
        guard = _FatalAbortGuard(verbose=True)
        guard.trip(
            stage="Theory artifacts",
            agent_role="THEORY_INSTRUCTOR",
            exc=ValueError(_EMPTY_RESPONSE_MSG),
        )
        capsys.readouterr()

        assert guard.should_skip("Labs") is True
        assert "Labs: skipped" in capsys.readouterr().out

    def test_detail_includes_actionable_env_hint(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        guard = _FatalAbortGuard()
        guard.trip(
            stage="Theory artifacts",
            agent_role="THEORY_INSTRUCTOR",
            exc=ValueError(_EMPTY_RESPONSE_MSG),
        )
        capsys.readouterr()

        assert guard.detail is not None
        assert "AGENT_THEORY_INSTRUCTOR_MAX_TOKENS=16384" in guard.detail

