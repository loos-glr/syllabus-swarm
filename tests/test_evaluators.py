"""Tests for the deterministic System One evaluation layer (``src/evaluators``).

Covers:

* the typed question builders and answer value objects in ``src/system_one``,
* :mod:`src.evaluators.modality_router` (replaces the retired Media Strategist),
* :mod:`src.evaluators.qa_scorer` (typed rubric scoring + verdicts),
* :mod:`src.evaluators.syllabus_gate` (structural checks + typed gate).

Every test uses :class:`src.system_one.FakeSystemOneClient`, so the suite is
network-free and needs no System One API key.
"""

from __future__ import annotations

import pytest

from src.evaluators.modality_router import (
    COMPLEXITY_LEVELS,
    MODALITY_CRITERIA,
    ROUTING_MAP,
    ModalityRouting,
    RoutingState,
    route_module,
    routing_target,
)
from src.evaluators.qa_scorer import (
    LAB_CRITERIA,
    THEORY_CRITERIA,
    QAScoring,
    default_qa_rubric,
    qa_needs_review,
    qa_passed,
    render_qa_report,
    rubric_from_profile,
    score_artifact,
    score_artifacts,
)
from src.evaluators.syllabus_gate import (
    QUALITY_LEVELS,
    REQUIRED_SECTIONS,
    assess_syllabus,
    check_required_sections,
    extract_duration_mentions,
    render_gate_report,
)
from src.models import ModalityType, QAScore, RubricCriterion
from src.system_one import (
    ChoiceAnswer,
    FakeSystemOneClient,
    NoulAnswer,
    ScoreAnswer,
    SystemOneError,
    choice_question,
    noul_question,
    score_question,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _routing_client(
    modality: ModalityType,
    complexity: float,
    *,
    confidence: float = 0.9,
) -> FakeSystemOneClient:
    """Scripted System One client for a single routing round-trip."""
    return FakeSystemOneClient(
        {
            "modality": ChoiceAnswer(
                modality.value,
                {modality.value: 1.0},
                confidence,
            ),
            "complexity": ScoreAnswer(complexity, 0.8),
        }
    )


def _qa_client(
    *,
    ready: float = 0.9,
    criterion_score: float = 4.0,
    confidence: float = 0.9,
    rubric_criteria: tuple[RubricCriterion, ...] = LAB_CRITERIA,
) -> FakeSystemOneClient:
    """Scripted System One client returning one score per rubric criterion."""
    answers: dict[str, object] = {
        f"criterion_{c.key}": ScoreAnswer(criterion_score, confidence) for c in rubric_criteria
    }
    answers["ready"] = NoulAnswer(ready)
    return FakeSystemOneClient(answers)


def _gate_client(
    *,
    complete: float = 0.9,
    coherent: float = 0.9,
    quality: float = 4.0,
    confidence: float = 0.9,
) -> FakeSystemOneClient:
    return FakeSystemOneClient(
        {
            "complete": NoulAnswer(complete),
            "coherent": NoulAnswer(coherent),
            "quality": ScoreAnswer(quality, confidence),
        }
    )


FULL_SYLLABUS: str = (
    "# Course Syllabus\n\n"
    "## Course Overview\nOverview text.\n\n"
    "## Learning Objectives\nObjectives text.\n\n"
    "## Humanics Literacies\nLiteracies text.\n\n"
    "## Module Breakdown\nModule 1 lasts 120 minutes.\n\n"
    "## Experiential Learning\nCapstone text.\n\n"
    "## Assessment Strategy\nAssessment text.\n\n"
    "## Required Tools & Resources\nTools text.\n\n"
    "## Schedule at a Glance\nSchedule text.\n"
)


# ---------------------------------------------------------------------------
# src.system_one — question builders, answers, fake client
# ---------------------------------------------------------------------------


class TestQuestionBuilders:
    """The builders emit the wire format the System One API expects."""

    def test_noul_question_shape(self) -> None:
        q = noul_question(instructions="Q?", true="yes-ish", false="no-ish")
        assert q["type"] == "noul"
        assert q["criteria"] == {"true": "yes-ish", "false": "no-ish"}

    def test_choice_question_shape(self) -> None:
        q = choice_question(instructions="Pick", criteria={"a": "first", "b": "second"})
        assert q["type"] == "choice"
        assert list(q["criteria"]) == ["a", "b"]

    def test_score_question_shape(self) -> None:
        q = score_question(instructions="Rate", criteria=["low", "mid", "high"])
        assert q["type"] == "score"
        assert q["criteria"] == ["low", "mid", "high"]

    def test_choice_question_requires_options(self) -> None:
        with pytest.raises(SystemOneError):
            choice_question(instructions="Pick", criteria={})

    def test_score_question_rejects_too_few_levels(self) -> None:
        with pytest.raises(SystemOneError):
            score_question(instructions="Rate", criteria=["only-one"])

    def test_score_question_rejects_too_many_levels(self) -> None:
        with pytest.raises(SystemOneError):
            score_question(instructions="Rate", criteria=[str(i) for i in range(11)])


class TestSystemOneResultAccessors:
    """Accessors enforce the declared answer type (type correctness is the point)."""

    def test_wrong_answer_type_raises(self) -> None:
        client = FakeSystemOneClient({"x": NoulAnswer(0.7)})
        result = client.system_one(
            state={}, questions={"x": noul_question(instructions="?", true="y", false="n")}
        )
        assert result.noul("x").probability == 0.7
        with pytest.raises(SystemOneError):
            result.choice("x")

    def test_missing_question_raises(self) -> None:
        client = FakeSystemOneClient({"a": NoulAnswer(0.5)}, heuristic=False)
        with pytest.raises(SystemOneError):
            client.system_one(
                state={}, questions={"b": noul_question(instructions="?", true="y", false="n")}
            )


class TestFakeSystemOneClient:
    """The fake client is the safety net for offline/degraded operation."""

    def test_heuristic_answers_are_never_confident(self) -> None:
        client = FakeSystemOneClient()
        result = client.system_one(
            state={},
            questions={
                "n": noul_question(instructions="?", true="y", false="n"),
                "c": choice_question(instructions="?", criteria={"a": "A", "b": "B"}),
                "s": score_question(instructions="?", criteria=["lo", "hi"]),
            },
        )
        assert result.noul("n").probability == 0.5
        assert result.choice("c").confidence == 0.0
        assert result.score("s").confidence == 0.0

    def test_heuristic_choice_picks_first_option(self) -> None:
        client = FakeSystemOneClient()
        result = client.system_one(
            state={},
            questions={"c": choice_question(instructions="?", criteria={"a": "A", "b": "B"})},
        )
        assert result.choice("c").choice == "a"

    def test_records_calls_for_assertions(self) -> None:
        client = FakeSystemOneClient()
        client.system_one(
            state={"k": "v"},
            questions={"n": noul_question(instructions="?", true="y", false="n")},
        )
        assert len(client.calls) == 1
        assert client.calls[0]["state"] == {"k": "v"}


# ---------------------------------------------------------------------------
# Modality routing (replaces the retired Media Strategist agent)
# ---------------------------------------------------------------------------


class TestModalityRouting:
    """Routing must be exhaustive, typed, and free of generated prose."""

    def test_routing_map_covers_every_modality(self) -> None:
        for modality in ModalityType:
            assert modality in ROUTING_MAP, f"No route for {modality}"

    def test_modality_criteria_covers_every_modality(self) -> None:
        for modality in ModalityType:
            assert modality in MODALITY_CRITERIA, f"No criteria for {modality}"

    def test_routing_targets(self) -> None:
        assert routing_target(ModalityType.VIDEO_AS_CODE) == "video_engineer"
        assert routing_target(ModalityType.CLASSIC_READER) == "theory_instructor"
        assert routing_target(ModalityType.INTERACTIVE_WEB) == "theory_instructor"
        assert routing_target(ModalityType.INTERACTIVE_CLI) == "theory_instructor"

    def test_route_module_returns_typed_decision(self) -> None:
        client = _routing_client(ModalityType.VIDEO_AS_CODE, 4.0)
        decision = route_module(
            RoutingState("Recursion", "Recursive functions"),
            client=client,
        )
        assert decision.modality is ModalityType.VIDEO_AS_CODE
        assert decision.complexity_score == 1.0
        assert decision.confidence == 0.9
        assert decision.probabilities == {"video_as_code": 1.0}
        assert decision.needs_review is False
        assert decision.rationale  # deterministically populated, never empty

    def test_complexity_is_normalised_to_unit_interval(self) -> None:
        state = RoutingState("Syntax", "Basics")
        low = route_module(state, client=_routing_client(ModalityType.CLASSIC_READER, 0.0))
        mid = route_module(state, client=_routing_client(ModalityType.CLASSIC_READER, 2.0))
        high = route_module(state, client=_routing_client(ModalityType.CLASSIC_READER, 4.0))
        assert low.complexity_score == 0.0
        assert mid.complexity_score == 0.5
        assert high.complexity_score == 1.0

    def test_low_confidence_flags_for_review(self) -> None:
        client = _routing_client(ModalityType.CLASSIC_READER, 1.0, confidence=0.1)
        decision = route_module(RoutingState("Syntax", "Basics"), client=client)
        assert decision.needs_review is True

    def test_routing_asks_exactly_two_questions(self) -> None:
        client = _routing_client(ModalityType.INTERACTIVE_WEB, 2.0)
        route_module(RoutingState("DOM", "The DOM"), client=client)
        assert set(client.calls[0]["questions"]) == {"modality", "complexity"}

    def test_unknown_modality_raises(self) -> None:
        client = FakeSystemOneClient(
            {
                "modality": ChoiceAnswer("not_a_modality", {}, 0.9),
                "complexity": ScoreAnswer(1.0, 0.9),
            }
        )
        with pytest.raises(SystemOneError):
            ModalityRouting(client).route(RoutingState("X", "Y"))

    def test_routing_state_payload_omits_empty_optional_fields(self) -> None:
        payload = RoutingState("M", "S").as_state()
        assert payload == {"module_name": "M", "module_summary": "S"}

    def test_complexity_levels_within_system_one_bounds(self) -> None:
        assert 2 <= len(COMPLEXITY_LEVELS) <= 10


# ---------------------------------------------------------------------------
# QA scoring
# ---------------------------------------------------------------------------


class TestQAScoring:
    """The verdict must be derivable purely from typed scores."""

    def test_full_marks_pass(self) -> None:
        score = score_artifact("content", content_ref="lab/a.py", client=_qa_client())
        assert score.verdict == "pass"
        assert score.score == 1.0
        assert score.flags == []
        assert score.needs_review is False

    def test_low_ready_noul_fails(self) -> None:
        score = score_artifact("content", content_ref="x", client=_qa_client(ready=0.1))
        assert score.verdict == "needs_fixes"
        assert "not_ready" in score.flags

    def test_below_threshold_criterion_fails(self) -> None:
        score = score_artifact(
            "content",
            content_ref="x",
            client=_qa_client(criterion_score=2.0),
        )
        assert score.verdict == "needs_fixes"
        assert len(score.flags) == len(LAB_CRITERIA)

    def test_low_confidence_flags_for_review(self) -> None:
        score = score_artifact(
            "content",
            content_ref="x",
            client=_qa_client(confidence=0.1),
        )
        assert score.needs_review is True
        assert score.confidence == 0.1

    def test_per_criterion_keys_match_rubric(self) -> None:
        score = score_artifact("content", content_ref="x", client=_qa_client())
        assert set(score.per_criterion) == {c.key for c in LAB_CRITERIA}

    def test_theory_rubric_is_used_for_theory_kind(self) -> None:
        client = _qa_client(rubric_criteria=THEORY_CRITERIA)
        score = score_artifact("content", content_ref="t.html", kind="theory", client=client)
        assert set(score.per_criterion) == {c.key for c in THEORY_CRITERIA}

    def test_default_rubrics_are_distinct(self) -> None:
        assert default_qa_rubric("lab").criteria == LAB_CRITERIA
        assert default_qa_rubric("theory").criteria == THEORY_CRITERIA

    def test_rubric_covers_all_criteria_in_request(self) -> None:
        client = _qa_client()
        score_artifact("c", content_ref="x", client=client)
        asked = set(client.calls[0]["questions"])
        assert asked == {f"criterion_{c.key}" for c in LAB_CRITERIA} | {"ready"}

    def test_rubric_from_profile_mentions_tech_stack(self) -> None:
        profile = {"tech_stack": {"primary_language": "PHP"}, "year_level": 2}
        rubric = rubric_from_profile(profile)
        alignment = next(c for c in rubric.criteria if c.key == "profile_alignment")
        assert "PHP" in alignment.description

    def test_rubric_from_none_profile_uses_defaults(self) -> None:
        assert rubric_from_profile(None).criteria == LAB_CRITERIA

    def test_qa_passed_aggregates_all_scores(self) -> None:
        passing = score_artifact("c", content_ref="a", client=_qa_client())
        failing = score_artifact("c", content_ref="b", client=_qa_client(ready=0.0))
        assert qa_passed([passing]) is True
        assert qa_passed([passing, failing]) is False
        assert qa_needs_review([failing]) is False
        assert (
            qa_needs_review(
                [score_artifact("c", content_ref="c", client=_qa_client(confidence=0.0))]
            )
            is True
        )

    def test_score_artifacts_preserves_order(self) -> None:
        scores = score_artifacts(
            [("a.py", "a"), ("b.py", "b")],
            client=_qa_client(),
        )
        assert [s.content_ref for s in scores] == ["a.py", "b.py"]

    def test_render_qa_report_includes_sign_off_when_passing(self) -> None:
        report = render_qa_report(
            [score_artifact("c", content_ref="a.py", client=_qa_client())],
            course_name="Demo",
        )
        assert "QA SIGN-OFF" in report
        assert "PASSED" in report

    def test_render_qa_report_requests_fixes_when_failing(self) -> None:
        report = render_qa_report(
            [score_artifact("c", content_ref="a.py", client=_qa_client(ready=0.0))],
            course_name="Demo",
        )
        assert "NEEDS FIXES" in report
        assert "not_ready" in report

    def test_render_qa_report_handles_no_scores(self) -> None:
        assert "No artifacts were scored" in render_qa_report([], course_name="Demo")


# ---------------------------------------------------------------------------
# Syllabus gate
# ---------------------------------------------------------------------------


class TestSyllabusGateStructuralChecks:
    """Structural checks run in code — never in the model."""

    def test_complete_syllabus_has_no_missing_sections(self) -> None:
        assert check_required_sections(FULL_SYLLABUS) == []

    def test_missing_sections_are_detected(self) -> None:
        missing = check_required_sections("# Syllabus\n\n## Learning Objectives\nOnly this.")
        assert "assessment_strategy" in missing
        assert "module_breakdown" in missing
        assert "learning_objectives" not in missing

    def test_every_required_section_has_aliases(self) -> None:
        for key, aliases in REQUIRED_SECTIONS:
            assert aliases, f"{key} has no heading aliases"

    def test_duration_mentions_are_extracted(self) -> None:
        mentions = extract_duration_mentions("Session 1: 120 minutes. Homework: 2 uur.")
        assert any("120" in m for m in mentions)
        assert len(mentions) == 2


class TestSyllabusGate:
    """The gate verdict must be fully derivable from typed answers + code checks."""

    def test_complete_syllabus_passes(self) -> None:
        decision = assess_syllabus(
            FULL_SYLLABUS,
            client=_gate_client(),
            course_name="Demo",
        )
        assert decision.complete is True
        assert decision.blockers == []
        assert decision.quality_score == 1.0
        assert decision.needs_review is False

    def test_missing_sections_become_blockers(self) -> None:
        decision = assess_syllabus("# Minimal\n", client=_gate_client(), course_name="Demo")
        assert decision.complete is False
        assert any(b.startswith("missing_section:") for b in decision.blockers)

    def test_incomplete_noul_becomes_blocker(self) -> None:
        decision = assess_syllabus(
            FULL_SYLLABUS,
            client=_gate_client(complete=0.1),
            course_name="Demo",
        )
        assert "incomplete" in decision.blockers

    def test_incoherent_schedule_becomes_blocker(self) -> None:
        decision = assess_syllabus(
            FULL_SYLLABUS,
            client=_gate_client(coherent=0.1),
            course_name="Demo",
        )
        assert "incoherent_schedule" in decision.blockers

    def test_low_quality_becomes_blocker(self) -> None:
        decision = assess_syllabus(
            FULL_SYLLABUS,
            client=_gate_client(quality=1.0),
            course_name="Demo",
        )
        assert "low_quality" in decision.blockers

    def test_low_confidence_flags_for_review(self) -> None:
        decision = assess_syllabus(
            FULL_SYLLABUS,
            client=_gate_client(confidence=0.1),
            course_name="Demo",
        )
        assert decision.needs_review is True

    def test_quality_levels_within_system_one_bounds(self) -> None:
        assert 2 <= len(QUALITY_LEVELS) <= 10

    def test_render_gate_report_approves_when_complete(self) -> None:
        decision = assess_syllabus(FULL_SYLLABUS, client=_gate_client(), course_name="Demo")
        report = render_gate_report(decision, course_name="Demo")
        assert "FEASIBILITY SIGN-OFF" in report
        assert "APPROVED" in report

    def test_render_gate_report_lists_blockers(self) -> None:
        decision = assess_syllabus("# Minimal\n", client=_gate_client(), course_name="Demo")
        report = render_gate_report(decision, course_name="Demo")
        assert "NEEDS REVISION" in report
        assert "Blockers" in report


# ---------------------------------------------------------------------------
# Decision models — typed contract enforcement (Layer 1)
# ---------------------------------------------------------------------------


class TestDecisionModels:
    """The typed schemas must reject malformed decision data."""

    def test_qascore_rejects_unknown_verdict(self) -> None:
        from pydantic import ValidationError  # noqa: PLC0415

        with pytest.raises(ValidationError):
            QAScore(content_ref="x", score=0.5, verdict="maybe")

    def test_rubric_criterion_requires_two_to_ten_levels(self) -> None:
        from pydantic import ValidationError  # noqa: PLC0415

        with pytest.raises(ValidationError):
            RubricCriterion(key="k", label="L", description="d", levels=["only-one"])
        with pytest.raises(ValidationError):
            RubricCriterion(
                key="k",
                label="L",
                description="d",
                levels=[str(i) for i in range(11)],
            )

    def test_modality_decision_provenance_defaults(self) -> None:
        from src.models import ModalityDecision  # noqa: PLC0415

        decision = ModalityDecision(
            module_name="M",
            modality=ModalityType.CLASSIC_READER,
            rationale="because",
            complexity_score=0.5,
        )
        assert decision.confidence is None
        assert decision.probabilities == {}
        assert decision.model_version is None
        assert decision.needs_review is False


# ---------------------------------------------------------------------------
# Orchestrator decision helpers (src.crews.syllabus_crew)
# ---------------------------------------------------------------------------


class TestOrchestratorDecisionHelpers:
    """The crew's deterministic helpers must be side-effect free and testable."""

    def test_collect_qa_artifacts_reads_text_files(self, tmp_path) -> None:
        from src.crews.syllabus_crew import _collect_qa_artifacts  # noqa: PLC0415

        tier = tmp_path / "labs" / "tier1_foundations"
        (tier / "starter").mkdir(parents=True)
        (tier / "theory").mkdir()
        (tier / "starter" / "lab1.py").write_text("print(1)\n")
        (tier / "starter" / "lab2.js").write_text("console.log(1)\n")
        (tier / "theory" / "demo.html").write_text("<html></html>\n")
        (tier / "starter" / ".hidden").write_text("secret\n")
        (tier / "starter" / "image.png").write_bytes(b"\x89PNG")

        refs = [ref for ref, _ in _collect_qa_artifacts(tmp_path)]

        assert "labs/tier1_foundations/starter/lab1.py" in refs
        assert "labs/tier1_foundations/starter/lab2.js" in refs
        assert "labs/tier1_foundations/theory/demo.html" in refs
        assert not any(ref.endswith(".hidden") for ref in refs)
        assert not any(ref.endswith(".png") for ref in refs)

    def test_collect_qa_artifacts_on_missing_dir_returns_empty(self, tmp_path) -> None:
        from src.crews.syllabus_crew import _collect_qa_artifacts  # noqa: PLC0415

        assert _collect_qa_artifacts(tmp_path) == []

    def test_collect_qa_artifacts_respects_limit(self, tmp_path) -> None:
        from src.crews.syllabus_crew import _collect_qa_artifacts  # noqa: PLC0415

        tier = tmp_path / "labs" / "tier1_foundations" / "starter"
        tier.mkdir(parents=True)
        for i in range(5):
            (tier / f"lab{i}.py").write_text("x = 1\n")

        assert len(_collect_qa_artifacts(tmp_path, limit=3)) == 3

    def test_collect_qa_artifacts_skips_undecodable_files(self, tmp_path) -> None:
        from src.crews.syllabus_crew import _collect_qa_artifacts  # noqa: PLC0415

        tier = tmp_path / "labs" / "tier1" / "starter"
        tier.mkdir(parents=True)
        (tier / "broken.py").write_bytes(b"\xff\xfe\x00")
        (tier / "ok.py").write_text("x = 1\n")

        refs = [ref for ref, _ in _collect_qa_artifacts(tmp_path)]
        assert refs == ["labs/tier1/starter/ok.py"]

    def test_build_qa_helpers_use_the_injected_system_one_client(self, patch_system_one) -> None:
        from src.crews.syllabus_crew import (  # noqa: PLC0415
            _build_qa_scorer,
            _build_syllabus_gate,
        )

        scorer = _build_qa_scorer()
        gate = _build_syllabus_gate()
        assert isinstance(scorer, QAScoring)
        assert gate is not None
        assert patch_system_one.called

    def test_build_helpers_return_none_without_a_client(self) -> None:
        from unittest.mock import patch  # noqa: PLC0415

        from src.crews.syllabus_crew import (  # noqa: PLC0415
            _build_qa_scorer,
            _build_syllabus_gate,
        )

        with patch(
            "src.crews.syllabus_crew.build_system_one_client",
            return_value=None,
        ):
            assert _build_qa_scorer() is None
            assert _build_syllabus_gate() is None
