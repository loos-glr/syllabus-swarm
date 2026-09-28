"""test_fail_fast.py — Fail-fast / circuit-breaker behaviour of the pipeline.

Regression tests for the credit-burning failure mode where a model returns
``None``/empty content (CrewAI: ``"Invalid response from LLM call - None or
empty"``).  Instead of continuing through every remaining tier and stage —
each of which would fail identically and spend more API credits — the pipeline
must:

1. detect the *futile* failure immediately,
2. trip a run-level circuit breaker, and
3. skip every remaining generation stage (extra theory tiers, lesson plans,
   presentations, labs, QA).

A static pre-flight audit is also covered: a reasoning model configured with
an undersized ``max_tokens`` budget aborts the run *before* any agent starts.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import src.crews.syllabus_crew as crew_module
from src.crews.syllabus_crew import FatalLLMError, run_syllabus_crew
from src.llm_factory import ConfigIssue

_EMPTY_RESPONSE_MSG: str = "Invalid response from LLM call - None or empty."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent(role: str) -> MagicMock:
    """Build a MagicMock agent whose ``role`` is inspectable."""
    agent = MagicMock()
    agent.role = role
    return agent


class _FakeCrew:
    """Stand-in for ``crewai.Crew`` whose kickoff fails for the theory agent.

    Every constructed instance is recorded so tests can assert which stages
    were reached and how many times each crew was kicked off.
    """

    instances: list[_FakeCrew] = []

    def __init__(self, *, agents, tasks, process=None, verbose=False) -> None:
        self.agents = list(agents)
        self.tasks = list(tasks)
        self.kickoff_count = 0
        self.roles = " ".join(str(getattr(a, "role", "")) for a in self.agents).lower()
        _FakeCrew.instances.append(self)

    def kickoff(self):
        self.kickoff_count += 1
        if "theory" in self.roles:
            raise ValueError(_EMPTY_RESPONSE_MSG)
        result = MagicMock()
        result.raw = "# Feasibility audit\n\nAll good."
        return result


def _make_run_dir(root: Path, name: str) -> Path:
    """Create a run directory containing a loadable syllabus."""
    run_dir = root / name
    syllabus_dir = run_dir / "syllabus"
    syllabus_dir.mkdir(parents=True, exist_ok=True)
    (syllabus_dir / f"{name}.md").write_text(
        "# WebXR Introductie\n\n## Tier 1\nFoundations.\n", encoding="utf-8"
    )
    return run_dir


# ---------------------------------------------------------------------------
# Futile failure aborts the run and skips remaining stages
# ---------------------------------------------------------------------------


class TestFutileFailureAbortsRun:
    @pytest.fixture
    def aborted_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Run the crew with a theory crew that always returns empty content."""
        _FakeCrew.instances.clear()
        output_root = tmp_path / "output"
        output_root.mkdir()
        monkeypatch.setattr(crew_module, "OUTPUT_ROOT", output_root)

        run_dir = _make_run_dir(output_root, "2026-01-01_000000_Test_Course")

        theory_task_mock = MagicMock()
        create_theory_task = MagicMock(return_value=theory_task_mock)
        # Each agent getter must RETURN the mock agent — the crew code calls the
        # getter, so a bare MagicMock would yield an unconfigured return_value.
        getters = {
            "get_architect": MagicMock(return_value=_agent("Curriculum Architect")),
            "get_education_director": MagicMock(
                return_value=_agent("Education Director")
            ),
            "get_theory_instructor": MagicMock(
                return_value=_agent("MBO4 Theory Instructor and Technical Writer")
            ),
            "get_instructional_coordinator": MagicMock(),
            "get_presentation_designer": MagicMock(),
            "get_lab_developer": MagicMock(),
            "get_qa_reviewer": MagicMock(),
        }

        with (
            patch.multiple(crew_module, Crew=_FakeCrew, **getters),
            patch.object(
                crew_module, "create_theory_task", create_theory_task
            ),
            patch.object(
                crew_module, "create_syllabus_review_task", return_value=MagicMock()
            ),
            patch.object(crew_module, "_run_preflight_audit", return_value=[]),
            patch.object(crew_module, "update_output_manifest", return_value=None),
        ):
            result = run_syllabus_crew(
                "Course Name: Test Course",
                course_name="Test Course",
                run_id="2026-01-01_000000_Test_Course",
                resume_dir=str(run_dir),
                verbose=False,
            )

        return result, getters, create_theory_task

    def test_result_is_marked_aborted(self, aborted_run) -> None:
        result, _, _ = aborted_run
        assert result.aborted is True
        assert result.abort_reason is not None
        assert "burning credits" in result.abort_reason

    def test_theory_stage_fails_with_abort_reason(self, aborted_run) -> None:
        result, _, _ = aborted_run
        assert result.theory_ok is False
        assert result.theory_error == result.abort_reason

    def test_only_the_first_tier_is_attempted(self, aborted_run) -> None:
        """The remaining tiers must not be attempted after a futile failure."""
        _, _, create_theory_task = aborted_run
        assert create_theory_task.call_count == 1

        theory_crews = [c for c in _FakeCrew.instances if "theory" in c.roles]
        assert len(theory_crews) == 1
        # One kickoff only — no retries were burned on the empty response.
        assert theory_crews[0].kickoff_count == 1

    def test_downstream_agents_are_never_instantiated(self, aborted_run) -> None:
        _, getters, _ = aborted_run
        for name in (
            "get_instructional_coordinator",
            "get_presentation_designer",
            "get_lab_developer",
            "get_qa_reviewer",
        ):
            getters[name].assert_not_called()

    def test_downstream_stages_report_the_abort_reason(self, aborted_run) -> None:
        result, _, _ = aborted_run
        assert result.lesson_plan_ok is False
        assert result.lesson_plan_error == result.abort_reason
        assert result.presentation_ok is False
        assert result.presentation_error == result.abort_reason
        assert result.labs_ok is False
        assert result.labs_error == result.abort_reason

    def test_overall_verdict_is_not_success(self, aborted_run) -> None:
        result, _, _ = aborted_run
        assert result.all_succeeded is False



# ---------------------------------------------------------------------------
# Pre-flight configuration gate
# ---------------------------------------------------------------------------


class TestPreflightGate:
    def test_fatal_config_issue_aborts_before_any_agent_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output_root = tmp_path / "output"
        output_root.mkdir()
        monkeypatch.setattr(crew_module, "OUTPUT_ROOT", output_root)

        fatal = [
            ConfigIssue(
                role="THEORY_INSTRUCTOR",
                severity="fatal",
                message="reasoning model with max_tokens=8192",
                fix="raise AGENT_THEORY_INSTRUCTOR_MAX_TOKENS",
            )
        ]

        with patch.object(crew_module, "_run_preflight_audit", return_value=fatal):
            with pytest.raises(FatalLLMError) as excinfo:
                run_syllabus_crew(
                    "Course Name: Test Course",
                    course_name="Test Course",
                    verbose=False,
                )

        assert excinfo.value.stage == "pre-flight"
        assert "reasoning model with max_tokens=8192" in str(excinfo.value)

    def test_preflight_can_be_disabled_via_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(crew_module._SKIP_PREFLIGHT_ENV, "1")
        assert crew_module._preflight_enabled() is False

    def test_preflight_enabled_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(crew_module._SKIP_PREFLIGHT_ENV, raising=False)
        assert crew_module._preflight_enabled() is True



# ---------------------------------------------------------------------------
# CLI exit codes for the fail-fast paths
# ---------------------------------------------------------------------------


class TestMainCliExitCodes:
    """src/main.py must exit non-zero and never prompt for feedback on abort."""

    def test_fatal_llm_error_exits_with_code_4(self, capsys: pytest.CaptureFixture) -> None:
        from src.main import _exit_on_fatal_llm_error

        exc = FatalLLMError("boom", agent_role="THEORY_INSTRUCTOR", stage="pre-flight")

        with pytest.raises(SystemExit) as excinfo:
            _exit_on_fatal_llm_error(exc)

        assert excinfo.value.code == 4
        err = capsys.readouterr().err
        assert "Fatal LLM Error" in err
        assert "before any further API credits were spent" in err

    def test_aborted_run_exits_with_code_4(self, capsys: pytest.CaptureFixture) -> None:
        from src.main import _exit_on_aborted_run

        result = MagicMock()
        result.aborted = True
        result.abort_reason = "Aborted during Theory artifacts (THEORY_INSTRUCTOR)"

        with pytest.raises(SystemExit) as excinfo:
            _exit_on_aborted_run(result)

        assert excinfo.value.code == 4
        assert "Run aborted early" in capsys.readouterr().err


class TestMaybePrintIterHint:
    """The summary hint must cover empty responses, not just iter exhaustion."""

    def test_empty_response_hint(self, capsys: pytest.CaptureFixture) -> None:
        from src.main import _maybe_print_iter_hint

        _maybe_print_iter_hint(
            "Invalid response from LLM call - None or empty.\n"
            "AGENT_THEORY_INSTRUCTOR_MAX_TOKENS=16384"
        )
        out = capsys.readouterr().out
        assert "AGENT_THEORY_INSTRUCTOR_MAX_TOKENS" in out

    def test_empty_response_hint_without_env_var(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        from src.main import _maybe_print_iter_hint

        _maybe_print_iter_hint("the model returned None/empty content")
        assert "Empty LLM response" in capsys.readouterr().out

    def test_iter_exhaustion_hint_still_works(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        from src.main import _maybe_print_iter_hint

        _maybe_print_iter_hint("Maximum iterations reached\nAGENT_QA_REVIEWER_MAX_ITER")
        assert "MAX_ITER" in capsys.readouterr().out

    def test_unrelated_error_prints_nothing(self, capsys: pytest.CaptureFixture) -> None:
        from src.main import _maybe_print_iter_hint

        _maybe_print_iter_hint("some other failure")
        assert capsys.readouterr().out == ""

