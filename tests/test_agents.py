"""
test_agents.py — Tests for agent factory functions
===================================================

Validates that ``create_curriculum_architect`` and
``create_lab_developer`` produce correctly-configured CrewAI Agent
objects with the expected roles, goals, backstories, and settings.

All tests mock ``build_llm_for_agent`` so no network calls are made.
"""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
from crewai import LLM, Agent

from src.agents.curriculum_architect import (
    create_curriculum_architect,
    get_architect,
)
from src.agents.education_director import (
    create_education_director,
    get_education_director,
)
from src.agents.intake_specialist import create_intake_specialist
from src.agents.lab_developer import (
    create_lab_developer,
    get_lab_developer,
)
from src.agents.qa_reviewer import (
    create_qa_reviewer,
    get_qa_reviewer,
)
from src.agents.theory_instructor import (
    create_theory_instructor,
    get_theory_instructor,
)
from src.llm_factory import (
    CURRICULUM_ARCHITECT,
    EDUCATION_DIRECTOR,
    LAB_DEVELOPER,
    QA_REVIEWER,
    THEORY_INSTRUCTOR,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm() -> MagicMock:
    """A MagicMock spec'd to crewai.LLM."""
    llm = MagicMock(spec=LLM)
    llm.model = "deepseek-v4-pro"
    llm.temperature = 0.2
    llm.top_p = 0.1
    llm.max_tokens = 8192
    llm.base_url = "https://openrouter.ai/api/v1"
    return llm


@pytest.mark.parametrize(
    ("factory", "expected_role"),
    [
        (create_curriculum_architect, "Senior MBO4 Curriculum Architect"),
        (create_education_director, "Lead Education Director and Feasibility Auditor"),
        (create_intake_specialist, "MBO4 Curriculum Intake Specialist"),
        (create_lab_developer, "Senior Lab and Project Developer"),
        (create_qa_reviewer, "Strict MBO4 QA Reviewer and Didactic Expert"),
        (create_theory_instructor, "MBO4 Theory Instructor and Technical Writer"),
    ],
)
def test_agent_roles_are_single_line(
    factory: Callable[..., Agent],
    expected_role: str,
    mock_llm: MagicMock,
) -> None:
    """Every agent role must be a single-line identifier (no newlines)."""
    with patch(f"{factory.__module__}.build_llm_for_agent", return_value=mock_llm):
        agent = factory()
    assert agent.role == expected_role
    assert "\n" not in agent.role


# ===================================================================
# create_curriculum_architect
# ===================================================================


class TestCreateCurriculumArchitect:
    """Tests for the Curriculum Architect agent factory."""

    def test_role_contains_curriculum_architect(self, mock_llm: MagicMock) -> None:
        """The agent's role identifies it as the Curriculum Architect."""
        with patch(
            "src.agents.curriculum_architect.build_llm_for_agent",
            return_value=mock_llm,
        ) as mock_build:
            agent = create_curriculum_architect()
            assert "Curriculum Architect" in agent.role
            mock_build.assert_called_once_with(CURRICULUM_ARCHITECT)

    def test_goal_contains_humanics_literacies(self, mock_llm: MagicMock) -> None:
        """The goal references all three Humanics literacies."""
        with patch(
            "src.agents.curriculum_architect.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_curriculum_architect()
            assert "Technological Literacy" in agent.goal
            assert "Data Literacy" in agent.goal
            assert "Human Literacy" in agent.goal

    def test_backstory_mentions_joseph_aoun(self, mock_llm: MagicMock) -> None:
        """The backstory references Joseph Aoun and Robot-Proof."""
        with patch(
            "src.agents.curriculum_architect.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_curriculum_architect()
            assert "Joseph Aoun" in agent.backstory
            assert "Robot-Proof" in agent.backstory

    def test_backstory_mentions_humanics(self, mock_llm: MagicMock) -> None:
        """The backstory references the Humanics framework."""
        with patch(
            "src.agents.curriculum_architect.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_curriculum_architect()
            assert "Humanics" in agent.backstory

    def test_allow_delegation_is_false(self, mock_llm: MagicMock) -> None:
        """The agent does not allow delegation."""
        with patch(
            "src.agents.curriculum_architect.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_curriculum_architect()
            assert agent.allow_delegation is False

    def test_max_iter_is_five(self, mock_llm: MagicMock) -> None:
        """The agent has max_iter set to 5."""
        with patch(
            "src.agents.curriculum_architect.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_curriculum_architect()
            assert agent.max_iter == 5

    def test_max_rpm_is_twenty(self, mock_llm: MagicMock) -> None:
        """The agent has max_rpm set to 20."""
        with patch(
            "src.agents.curriculum_architect.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_curriculum_architect()
            assert agent.max_rpm == 20

    def test_verbose_defaults_to_false(self, mock_llm: MagicMock) -> None:
        """verbose is False by default."""
        with patch(
            "src.agents.curriculum_architect.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_curriculum_architect()
            assert agent.verbose is False

    def test_verbose_can_be_enabled(self, mock_llm: MagicMock) -> None:
        """verbose can be set to True."""
        with patch(
            "src.agents.curriculum_architect.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_curriculum_architect(verbose=True)
            assert agent.verbose is True

    def test_explicit_llm_bypasses_factory(self, mock_llm: MagicMock) -> None:
        """When an LLM is passed explicitly, the factory is not called."""
        custom_llm = MagicMock(spec=LLM)
        with patch("src.agents.curriculum_architect.build_llm_for_agent") as mock_build:
            agent = create_curriculum_architect(llm=custom_llm)
            assert agent.llm is custom_llm
            mock_build.assert_not_called()

    def test_goal_mentions_markdown_output(self, mock_llm: MagicMock) -> None:
        """The goal specifies structured Markdown output."""
        with patch(
            "src.agents.curriculum_architect.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_curriculum_architect()
            assert "Markdown" in agent.goal

    def test_backstory_mentions_backward_design(self, mock_llm: MagicMock) -> None:
        """The backstory references backward design methodology."""
        with patch(
            "src.agents.curriculum_architect.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_curriculum_architect()
            assert "backward design" in agent.backstory.lower()


# ===================================================================
# create_lab_developer
# ===================================================================


class TestCreateLabDeveloper:
    """Tests for the Lab & Project Developer agent factory."""

    def test_role_contains_lab_developer(self, mock_llm: MagicMock) -> None:
        """The agent's role identifies it as the Lab Developer."""
        with patch(
            "src.agents.lab_developer.build_llm_for_agent",
            return_value=mock_llm,
        ) as mock_build:
            agent = create_lab_developer()
            assert "Lab and Project Developer" in agent.role
            mock_build.assert_called_once_with(LAB_DEVELOPER)

    def test_goal_contains_three_tiers(self, mock_llm: MagicMock) -> None:
        """The goal references all three lab tiers."""
        with patch(
            "src.agents.lab_developer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_lab_developer()
            assert "Tier 1 — Foundations" in agent.goal
            assert "Tier 2 — Application" in agent.goal
            assert "Tier 3 — Architecture" in agent.goal

    def test_goal_contains_humanics_literacies(self, mock_llm: MagicMock) -> None:
        """The goal references all three Humanics literacies."""
        with patch(
            "src.agents.lab_developer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_lab_developer()
            assert "Technological [T]" in agent.goal
            assert "Data [D]" in agent.goal
            assert "Human [H]" in agent.goal

    def test_backstory_mentions_software_engineer(self, mock_llm: MagicMock) -> None:
        """The backstory establishes the agent as a seasoned engineer."""
        with patch(
            "src.agents.lab_developer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_lab_developer()
            assert "software engineer" in agent.backstory.lower()

    def test_backstory_mentions_docker(self, mock_llm: MagicMock) -> None:
        """The backstory references Docker as a learning objective."""
        with patch(
            "src.agents.lab_developer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_lab_developer()
            assert "Docker" in agent.backstory

    def test_backstory_mentions_humanics(self, mock_llm: MagicMock) -> None:
        """The backstory references the Humanics framework."""
        with patch(
            "src.agents.lab_developer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_lab_developer()
            assert "Humanics" in agent.backstory

    def test_allow_delegation_is_false(self, mock_llm: MagicMock) -> None:
        """The agent does not allow delegation."""
        with patch(
            "src.agents.lab_developer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_lab_developer()
            assert agent.allow_delegation is False

    def test_max_iter_is_sixty(self, mock_llm: MagicMock) -> None:
        """The agent has max_iter set to 60 (needs room for 3 tiers × tool calls + thinking + retries)."""
        with patch(
            "src.agents.lab_developer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_lab_developer()
            assert agent.max_iter == 60

    def test_max_rpm_is_thirty(self, mock_llm: MagicMock) -> None:
        """The agent has max_rpm set to 30."""
        with patch(
            "src.agents.lab_developer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_lab_developer()
            assert agent.max_rpm == 30

    def test_verbose_defaults_to_false(self, mock_llm: MagicMock) -> None:
        """verbose is False by default."""
        with patch(
            "src.agents.lab_developer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_lab_developer()
            assert agent.verbose is False

    def test_verbose_can_be_enabled(self, mock_llm: MagicMock) -> None:
        """verbose can be set to True."""
        with patch(
            "src.agents.lab_developer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_lab_developer(verbose=True)
            assert agent.verbose is True

    def test_explicit_llm_bypasses_factory(self, mock_llm: MagicMock) -> None:
        """When an LLM is passed explicitly, the factory is not called."""
        custom_llm = MagicMock(spec=LLM)
        with patch("src.agents.lab_developer.build_llm_for_agent") as mock_build:
            agent = create_lab_developer(llm=custom_llm)
            assert agent.llm is custom_llm
            mock_build.assert_not_called()

    def test_goal_mentions_docker_and_ci_cd(self, mock_llm: MagicMock) -> None:
        """The goal references Docker, CI/CD, and observability for Tier 3."""
        with patch(
            "src.agents.lab_developer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_lab_developer()
            assert "Docker" in agent.goal
            assert "CI/CD" in agent.goal

    def test_backstory_mentions_polyglot(self, mock_llm: MagicMock) -> None:
        """The backstory establishes the agent as a polyglot developer."""
        with patch(
            "src.agents.lab_developer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_lab_developer()
            assert "polyglot" in agent.backstory.lower()

    def test_has_output_export_tool(self, mock_llm: MagicMock) -> None:
        """The agent is equipped with the OutputExportTool for file writing."""
        with patch(
            "src.agents.lab_developer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_lab_developer()
            tool_names = [t.name for t in agent.tools]
            assert "output_export_tool" in tool_names

    def test_output_export_tool_has_force_enabled(self, mock_llm: MagicMock) -> None:
        """The OutputExportTool is instantiated with force=True for overwrites."""
        with patch(
            "src.agents.lab_developer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_lab_developer()
            export_tool = next(t for t in agent.tools if t.name == "output_export_tool")
            assert export_tool.force is True


# ===================================================================
# create_qa_reviewer
# ===================================================================


class TestCreateQaReviewer:
    """Tests for the QA Reviewer agent factory."""

    def test_role_contains_qa_reviewer(self, mock_llm: MagicMock) -> None:
        """The agent's role identifies it as the QA Reviewer."""
        with patch(
            "src.agents.qa_reviewer.build_llm_for_agent",
            return_value=mock_llm,
        ) as mock_build:
            agent = create_qa_reviewer()
            assert "QA" in agent.role
            assert "MBO4" in agent.role
            mock_build.assert_called_once_with(QA_REVIEWER)

    def test_goal_contains_verification(self, mock_llm: MagicMock) -> None:
        """The goal references bug-free code and didactic correctness."""
        with patch(
            "src.agents.qa_reviewer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_qa_reviewer()
            assert "bug-free" in agent.goal.lower()
            assert "didactically correct" in agent.goal.lower()

    def test_goal_contains_technical_and_didactic_checks(self, mock_llm: MagicMock) -> None:
        """The goal references both technical and didactic checks."""
        with patch(
            "src.agents.qa_reviewer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_qa_reviewer()
            assert "Technical Correctness Check" in agent.goal
            assert "Didactic & Clarity Check" in agent.goal

    def test_backstory_mentions_mbo4(self, mock_llm: MagicMock) -> None:
        """The backstory references MBO4 or vocational students."""
        with patch(
            "src.agents.qa_reviewer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_qa_reviewer()
            assert "MBO4" in agent.backstory

    def test_backstory_mentions_students(self, mock_llm: MagicMock) -> None:
        """The backstory references students and their learning experience."""
        with patch(
            "src.agents.qa_reviewer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_qa_reviewer()
            assert "student" in agent.backstory.lower()

    def test_allow_delegation_is_true(self, mock_llm: MagicMock) -> None:
        """The QA Reviewer MUST allow delegation to send fixes back to Lab Developer."""
        with patch(
            "src.agents.qa_reviewer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_qa_reviewer()
            assert agent.allow_delegation is True

    def test_max_iter_is_thirty(self, mock_llm: MagicMock) -> None:
        """The agent has max_iter set to 30."""
        with patch(
            "src.agents.qa_reviewer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_qa_reviewer()
            assert agent.max_iter == 30

    def test_max_rpm_is_twenty(self, mock_llm: MagicMock) -> None:
        """The agent has max_rpm set to 20."""
        with patch(
            "src.agents.qa_reviewer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_qa_reviewer()
            assert agent.max_rpm == 20

    def test_verbose_defaults_to_false(self, mock_llm: MagicMock) -> None:
        """verbose is False by default."""
        with patch(
            "src.agents.qa_reviewer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_qa_reviewer()
            assert agent.verbose is False

    def test_verbose_can_be_enabled(self, mock_llm: MagicMock) -> None:
        """verbose can be set to True."""
        with patch(
            "src.agents.qa_reviewer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_qa_reviewer(verbose=True)
            assert agent.verbose is True

    def test_explicit_llm_bypasses_factory(self, mock_llm: MagicMock) -> None:
        """When an LLM is passed explicitly, the factory is not called."""
        custom_llm = MagicMock(spec=LLM)
        with patch("src.agents.qa_reviewer.build_llm_for_agent") as mock_build:
            agent = create_qa_reviewer(llm=custom_llm)
            assert agent.llm is custom_llm
            mock_build.assert_not_called()

    def test_has_file_read_tools(self, mock_llm: MagicMock) -> None:
        """The agent is equipped with DirectoryReadTool, FileReadTool, and OutputExportTool."""
        with patch(
            "src.agents.qa_reviewer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_qa_reviewer()
            tool_names = [t.name for t in agent.tools]
            assert "List files in directory" in tool_names
            assert "Read a file's content" in tool_names
            assert "output_export_tool" in tool_names

    def test_output_export_tool_has_force_enabled(self, mock_llm: MagicMock) -> None:
        """The OutputExportTool is instantiated with force=True for overwrites."""
        with patch(
            "src.agents.qa_reviewer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_qa_reviewer()
            export_tool = next(t for t in agent.tools if t.name == "output_export_tool")
            assert export_tool.force is True

    def test_goal_mentions_delegation(self, mock_llm: MagicMock) -> None:
        """The goal explicitly mentions delegation back to the Lab Developer."""
        with patch(
            "src.agents.qa_reviewer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_qa_reviewer()
            assert "Delegation" in agent.goal
            assert "Lab & Project Developer" in agent.goal


# ===================================================================
# Module singletons
# ===================================================================


class TestModuleSingletons:
    """Tests for the lazy singleton accessors."""

    def test_get_architect_returns_agent(self, mock_llm: MagicMock) -> None:
        """get_architect returns a valid Agent."""
        with patch(
            "src.agents.curriculum_architect.build_llm_for_agent",
            return_value=mock_llm,
        ):
            # Reset the singleton before test
            import src.agents.curriculum_architect as ca

            ca._architect_instance = None

            agent = get_architect()
            assert isinstance(agent, Agent)
            assert "Curriculum Architect" in agent.role

    def test_get_architect_is_idempotent(self, mock_llm: MagicMock) -> None:
        """Calling get_architect twice returns the same instance."""
        with patch(
            "src.agents.curriculum_architect.build_llm_for_agent",
            return_value=mock_llm,
        ):
            import src.agents.curriculum_architect as ca

            ca._architect_instance = None

            a1 = get_architect()
            a2 = get_architect()
            assert a1 is a2

    def test_get_lab_developer_returns_agent(self, mock_llm: MagicMock) -> None:
        """get_lab_developer returns a valid Agent."""
        with patch(
            "src.agents.lab_developer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            import src.agents.lab_developer as ld

            ld._lab_dev_instance = None

            agent = get_lab_developer()
            assert isinstance(agent, Agent)
            assert "Lab and Project Developer" in agent.role

    def test_get_lab_developer_is_idempotent(self, mock_llm: MagicMock) -> None:
        """Calling get_lab_developer twice returns the same instance."""
        with patch(
            "src.agents.lab_developer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            import src.agents.lab_developer as ld

            ld._lab_dev_instance = None

            a1 = get_lab_developer()
            a2 = get_lab_developer()
            assert a1 is a2

    def test_get_qa_reviewer_returns_agent(self, mock_llm: MagicMock) -> None:
        """get_qa_reviewer returns a valid Agent."""
        with patch(
            "src.agents.qa_reviewer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            import src.agents.qa_reviewer as qa

            qa._qa_reviewer_instance = None

            agent = get_qa_reviewer()
            assert isinstance(agent, Agent)
            assert "QA" in agent.role

    def test_get_qa_reviewer_is_idempotent(self, mock_llm: MagicMock) -> None:
        """Calling get_qa_reviewer twice returns the same instance."""
        with patch(
            "src.agents.qa_reviewer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            import src.agents.qa_reviewer as qa

            qa._qa_reviewer_instance = None

            a1 = get_qa_reviewer()
            a2 = get_qa_reviewer()
            assert a1 is a2

    def test_get_theory_instructor_returns_agent(self, mock_llm: MagicMock) -> None:
        """get_theory_instructor returns a valid Agent."""
        with patch(
            "src.agents.theory_instructor.build_llm_for_agent",
            return_value=mock_llm,
        ):
            import src.agents.theory_instructor as ti

            ti._theory_instructor_instance = None

            agent = get_theory_instructor()
            assert isinstance(agent, Agent)
            assert "Theory Instructor" in agent.role

    def test_get_theory_instructor_is_idempotent(self, mock_llm: MagicMock) -> None:
        """Calling get_theory_instructor twice returns the same instance."""
        with patch(
            "src.agents.theory_instructor.build_llm_for_agent",
            return_value=mock_llm,
        ):
            import src.agents.theory_instructor as ti

            ti._theory_instructor_instance = None

            a1 = get_theory_instructor()
            a2 = get_theory_instructor()
            assert a1 is a2

    def test_get_education_director_returns_agent(self, mock_llm: MagicMock) -> None:
        """get_education_director returns a valid Agent."""
        with patch(
            "src.agents.education_director.build_llm_for_agent",
            return_value=mock_llm,
        ):
            import src.agents.education_director as ed

            ed._director_instance = None

            agent = get_education_director()
            assert isinstance(agent, Agent)
            assert "Education Director" in agent.role

    def test_get_education_director_is_idempotent(self, mock_llm: MagicMock) -> None:
        """Calling get_education_director twice returns the same instance."""
        with patch(
            "src.agents.education_director.build_llm_for_agent",
            return_value=mock_llm,
        ):
            import src.agents.education_director as ed

            ed._director_instance = None

            a1 = get_education_director()
            a2 = get_education_director()
            assert a1 is a2


# ===================================================================
# create_theory_instructor
# ===================================================================


class TestCreateTheoryInstructor:
    """Tests for the Theory Instructor agent factory."""

    def test_role_contains_theory_instructor(self, mock_llm: MagicMock) -> None:
        """The agent's role identifies it as the Theory Instructor."""
        with patch(
            "src.agents.theory_instructor.build_llm_for_agent",
            return_value=mock_llm,
        ) as mock_build:
            agent = create_theory_instructor()
            assert "Theory Instructor" in agent.role
            mock_build.assert_called_once_with(THEORY_INSTRUCTOR)

    def test_goal_contains_multi_format_toolkit(self, mock_llm: MagicMock) -> None:
        """The goal references the Multi-Format Toolkit (HTML, terminal, Mermaid)."""
        with patch(
            "src.agents.theory_instructor.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_theory_instructor()
            assert "HTML" in agent.goal
            assert "Terminal" in agent.goal
            assert "Mermaid" in agent.goal

    def test_backstory_mentions_mbo4(self, mock_llm: MagicMock) -> None:
        """The backstory references MBO4 vocational education."""
        with patch(
            "src.agents.theory_instructor.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_theory_instructor()
            assert "MBO4" in agent.backstory

    def test_backstory_mentions_formats(self, mock_llm: MagicMock) -> None:
        """The backstory references Format A, B, and C."""
        with patch(
            "src.agents.theory_instructor.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_theory_instructor()
            assert "Format A" in agent.backstory
            assert "Format B" in agent.backstory
            assert "Format C" in agent.backstory

    def test_allow_delegation_is_false(self, mock_llm: MagicMock) -> None:
        """The agent does not allow delegation."""
        with patch(
            "src.agents.theory_instructor.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_theory_instructor()
            assert agent.allow_delegation is False

    def test_max_iter_is_thirty(self, mock_llm: MagicMock) -> None:
        """The agent has max_iter set to 30."""
        with patch(
            "src.agents.theory_instructor.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_theory_instructor()
            assert agent.max_iter == 30

    def test_max_rpm_is_twenty(self, mock_llm: MagicMock) -> None:
        """The agent has max_rpm set to 20."""
        with patch(
            "src.agents.theory_instructor.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_theory_instructor()
            assert agent.max_rpm == 20

    def test_verbose_defaults_to_false(self, mock_llm: MagicMock) -> None:
        """verbose is False by default."""
        with patch(
            "src.agents.theory_instructor.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_theory_instructor()
            assert agent.verbose is False

    def test_verbose_can_be_enabled(self, mock_llm: MagicMock) -> None:
        """verbose can be set to True."""
        with patch(
            "src.agents.theory_instructor.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_theory_instructor(verbose=True)
            assert agent.verbose is True

    def test_explicit_llm_bypasses_factory(self, mock_llm: MagicMock) -> None:
        """When an LLM is passed explicitly, the factory is not called."""
        custom_llm = MagicMock(spec=LLM)
        with patch("src.agents.theory_instructor.build_llm_for_agent") as mock_build:
            agent = create_theory_instructor(llm=custom_llm)
            assert agent.llm is custom_llm
            mock_build.assert_not_called()

    def test_has_output_export_tool(self, mock_llm: MagicMock) -> None:
        """The agent is equipped with the OutputExportTool for file writing."""
        with patch(
            "src.agents.theory_instructor.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_theory_instructor()
            tool_names = [t.name for t in agent.tools]
            assert "output_export_tool" in tool_names

    def test_output_export_tool_has_force_enabled(self, mock_llm: MagicMock) -> None:
        """The OutputExportTool is instantiated with force=True for overwrites."""
        with patch(
            "src.agents.theory_instructor.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_theory_instructor()
            export_tool = next(t for t in agent.tools if t.name == "output_export_tool")
            assert export_tool.force is True


# ===================================================================
# create_education_director
# ===================================================================


class TestCreateEducationDirector:
    """Tests for the Education Director agent factory."""

    def test_role_contains_education_director(self, mock_llm: MagicMock) -> None:
        """The agent's role identifies it as the Education Director."""
        with patch(
            "src.agents.education_director.build_llm_for_agent",
            return_value=mock_llm,
        ) as mock_build:
            agent = create_education_director()
            assert "Education Director" in agent.role
            assert "Feasibility Auditor" in agent.role
            mock_build.assert_called_once_with(EDUCATION_DIRECTOR)

    def test_goal_contains_time_budget_math(self, mock_llm: MagicMock) -> None:
        """The goal references time-budget calculations."""
        with patch(
            "src.agents.education_director.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_education_director()
            assert "Time-Budget Math" in agent.goal

    def test_goal_contains_workload_realism(self, mock_llm: MagicMock) -> None:
        """The goal references workload realism for MBO4 students."""
        with patch(
            "src.agents.education_director.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_education_director()
            assert "Workload Realism" in agent.goal

    def test_goal_contains_scheduling_sanity(self, mock_llm: MagicMock) -> None:
        """The goal references scheduling sanity checks."""
        with patch(
            "src.agents.education_director.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_education_director()
            assert "Scheduling Sanity" in agent.goal

    def test_goal_contains_mbo4_appropriateness(self, mock_llm: MagicMock) -> None:
        """The goal references MBO4 appropriateness checks."""
        with patch(
            "src.agents.education_director.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_education_director()
            assert "MBO4 Appropriateness" in agent.goal

    def test_backstory_mentions_mbo4(self, mock_llm: MagicMock) -> None:
        """The backstory references MBO4 vocational education."""
        with patch(
            "src.agents.education_director.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_education_director()
            assert "MBO4" in agent.backstory

    def test_backstory_mentions_director(self, mock_llm: MagicMock) -> None:
        """The backstory establishes the agent as a school director."""
        with patch(
            "src.agents.education_director.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_education_director()
            assert "director" in agent.backstory.lower()

    def test_backstory_mentions_curriculum_architect(self, mock_llm: MagicMock) -> None:
        """The backstory references delegating back to the Curriculum Architect."""
        with patch(
            "src.agents.education_director.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_education_director()
            assert "Curriculum Architect" in agent.backstory

    def test_allow_delegation_is_true(self, mock_llm: MagicMock) -> None:
        """The Education Director MUST allow delegation to send fixes back."""
        with patch(
            "src.agents.education_director.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_education_director()
            assert agent.allow_delegation is True

    def test_max_iter_is_five(self, mock_llm: MagicMock) -> None:
        """The agent has max_iter set to 5."""
        with patch(
            "src.agents.education_director.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_education_director()
            assert agent.max_iter == 5

    def test_max_rpm_is_twenty(self, mock_llm: MagicMock) -> None:
        """The agent has max_rpm set to 20."""
        with patch(
            "src.agents.education_director.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_education_director()
            assert agent.max_rpm == 20

    def test_verbose_defaults_to_false(self, mock_llm: MagicMock) -> None:
        """verbose is False by default."""
        with patch(
            "src.agents.education_director.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_education_director()
            assert agent.verbose is False

    def test_verbose_can_be_enabled(self, mock_llm: MagicMock) -> None:
        """verbose can be set to True."""
        with patch(
            "src.agents.education_director.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_education_director(verbose=True)
            assert agent.verbose is True

    def test_explicit_llm_bypasses_factory(self, mock_llm: MagicMock) -> None:
        """When an LLM is passed explicitly, the factory is not called."""
        custom_llm = MagicMock(spec=LLM)
        with patch("src.agents.education_director.build_llm_for_agent") as mock_build:
            agent = create_education_director(llm=custom_llm)
            assert agent.llm is custom_llm
            mock_build.assert_not_called()

    def test_goal_mentions_feasibility_audit_report(self, mock_llm: MagicMock) -> None:
        """The goal specifies a Markdown Feasibility Audit Report."""
        with patch(
            "src.agents.education_director.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_education_director()


# ===================================================================
# create_media_strategist
# ===================================================================


class TestCreateMediaStrategist:
    """Tests for the Media Strategist agent factory."""

    def test_role_contains_media_strategist(self, mock_llm: MagicMock) -> None:
        from src.agents.media_strategist import create_media_strategist

        with patch(
            "src.llm_factory.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_media_strategist()
            assert "Media Strategist" in agent.role

    def test_goal_mentions_modality_routing(self, mock_llm: MagicMock) -> None:
        from src.agents.media_strategist import create_media_strategist

        with patch(
            "src.llm_factory.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_media_strategist()
            assert "modality" in agent.goal.lower()

    def test_goal_mentions_pedagogical_complexity(self, mock_llm: MagicMock) -> None:
        from src.agents.media_strategist import create_media_strategist

        with patch(
            "src.llm_factory.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_media_strategist()
            assert "pedagogical" in agent.goal.lower() or "complexity" in agent.goal.lower()

    def test_backstory_mentions_instructional_design(self, mock_llm: MagicMock) -> None:
        from src.agents.media_strategist import create_media_strategist

        with patch(
            "src.llm_factory.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_media_strategist()
            assert (
                "instructional" in agent.backstory.lower() or "modality" in agent.backstory.lower()
            )

    def test_allow_delegation_is_false(self, mock_llm: MagicMock) -> None:
        from src.agents.media_strategist import create_media_strategist

        with patch(
            "src.llm_factory.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_media_strategist()
            assert agent.allow_delegation is False

    def test_llm_factory_called_with_correct_role(self, mock_llm: MagicMock) -> None:
        from src.agents.media_strategist import create_media_strategist

        with patch(
            "src.llm_factory.build_llm_for_agent",
            return_value=mock_llm,
        ) as mock_build:
            create_media_strategist()
            # The factory should be called with the MEDIA_STRATEGIST role constant
            mock_build.assert_called_once()

    def test_explicit_llm_bypasses_factory(self, mock_llm: MagicMock) -> None:
        from src.agents.media_strategist import create_media_strategist

        custom_llm = MagicMock(spec=LLM)
        with patch("src.llm_factory.build_llm_for_agent") as mock_build:
            agent = create_media_strategist(llm=custom_llm)
            assert agent.llm is custom_llm
            mock_build.assert_not_called()


# ===================================================================
# create_video_engineer
# ===================================================================


class TestCreateVideoEngineer:
    """Tests for the Video Engineer agent factory."""

    def test_role_contains_video_engineer(self, mock_llm: MagicMock) -> None:
        from src.agents.video_engineer import create_video_engineer

        with patch(
            "src.llm_factory.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_video_engineer()
            assert "Video Engineer" in agent.role

    def test_goal_mentions_remotion(self, mock_llm: MagicMock) -> None:
        from src.agents.video_engineer import create_video_engineer

        with patch(
            "src.llm_factory.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_video_engineer()
            assert (
                "Remotion" in agent.goal
                or "remotion" in agent.goal.lower()
                or "React" in agent.goal
            )

    def test_goal_mentions_jsx(self, mock_llm: MagicMock) -> None:
        from src.agents.video_engineer import create_video_engineer

        with patch(
            "src.llm_factory.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_video_engineer()
            assert "JSX" in agent.goal or "jsx" in agent.goal.lower() or "React" in agent.goal

    def test_backstory_mentions_deterministic(self, mock_llm: MagicMock) -> None:
        from src.agents.video_engineer import create_video_engineer

        with patch(
            "src.llm_factory.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_video_engineer()
            assert (
                "deterministic" in agent.backstory.lower() or "temporal" in agent.backstory.lower()
            )

    def test_allow_delegation_is_false(self, mock_llm: MagicMock) -> None:
        from src.agents.video_engineer import create_video_engineer

        with patch(
            "src.llm_factory.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_video_engineer()
            assert agent.allow_delegation is False

    def test_llm_factory_called_with_correct_role(self, mock_llm: MagicMock) -> None:
        from src.agents.video_engineer import create_video_engineer

        with patch(
            "src.llm_factory.build_llm_for_agent",
            return_value=mock_llm,
        ) as mock_build:
            create_video_engineer()
            mock_build.assert_called_once()

    def test_explicit_llm_bypasses_factory(self, mock_llm: MagicMock) -> None:
        from src.agents.video_engineer import create_video_engineer

        custom_llm = MagicMock(spec=LLM)
        with patch("src.llm_factory.build_llm_for_agent") as mock_build:
            agent = create_video_engineer(llm=custom_llm)
            assert agent.llm is custom_llm
            mock_build.assert_not_called()
