"""Tests for the QA-review resilience helpers in ``src.crews.syllabus_crew``.

Covers the transient-LLM-error classifier, the empty-LLM-response error
annotation, and the whole-crew retry wrapper that prevents a single
"Invalid response from LLM call - None or empty" failure from aborting the
QA review step.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.crews.syllabus_crew import (
    _annotate_empty_llm_response,
    _is_transient_llm_error,
    _kickoff_with_retry,
)

# ---------------------------------------------------------------------------
# _is_transient_llm_error
# ---------------------------------------------------------------------------


class TestIsTransientLlmError:
    def test_empty_response_marker_is_transient(self) -> None:
        assert _is_transient_llm_error(
            ValueError("Invalid response from LLM call - None or empty.")
        )

    def test_received_none_marker_is_transient(self) -> None:
        assert _is_transient_llm_error(
            ValueError("Received None or empty response from LLM call.")
        )

    def test_litellm_module_errors_are_transient(self) -> None:
        cls = type("LitellmError", (Exception,), {"__module__": "litellm.exceptions"})
        assert _is_transient_llm_error(cls("boom"))

    def test_rate_limit_text_is_transient(self) -> None:
        assert _is_transient_llm_error(Exception("429 Too Many Requests"))
        assert _is_transient_llm_error(Exception("rate limit exceeded"))

    def test_deterministic_error_is_not_transient(self) -> None:
        assert not _is_transient_llm_error(ValueError("some unrelated error"))


# ---------------------------------------------------------------------------
# _annotate_empty_llm_response
# ---------------------------------------------------------------------------


class TestAnnotateEmptyLlmResponse:
    def test_appends_actionable_hint(self) -> None:
        out = _annotate_empty_llm_response(
            "Invalid response from LLM call - None or empty.", "QA_REVIEWER"
        )
        assert "EMPTY LLM RESPONSE DETECTED" in out
        assert "AGENT_QA_REVIEWER_MAX_TOKENS=16384" in out
        assert "AGENT_QA_REVIEWER_MODEL" in out

    def test_leaves_unrelated_errors_unchanged(self) -> None:
        msg = "some other error"
        assert _annotate_empty_llm_response(msg, "QA_REVIEWER") == msg


# ---------------------------------------------------------------------------
# _kickoff_with_retry
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

    def test_retries_transient_failure_then_succeeds(self) -> None:
        crew = MagicMock()
        crew.kickoff.side_effect = [
            ValueError("Invalid response from LLM call - None or empty."),
            "success",
        ]

        with patch("src.crews.syllabus_crew.time.sleep"):
            result = _kickoff_with_retry(lambda: crew, backoff_seconds=1.0)

        assert result == "success"
        assert crew.kickoff.call_count == 2

    def test_raises_immediately_on_non_transient_error(self) -> None:
        crew = MagicMock()
        crew.kickoff.side_effect = ValueError("deterministic failure")

        with patch("src.crews.syllabus_crew.time.sleep") as mock_sleep:
            with pytest.raises(ValueError):
                _kickoff_with_retry(lambda: crew)

        mock_sleep.assert_not_called()
        assert crew.kickoff.call_count == 1

    def test_exhausts_attempts_on_persistent_transient_error(self) -> None:
        crew = MagicMock()
        crew.kickoff.side_effect = ValueError(
            "Invalid response from LLM call - None or empty."
        )

        with patch("src.crews.syllabus_crew.time.sleep"):
            with pytest.raises(ValueError):
                _kickoff_with_retry(lambda: crew, attempts=3)

        assert crew.kickoff.call_count == 3
