"""
conftest.py — Shared pytest fixtures for the syllabus-swarm test suite.

All fixtures in this module are automatically available to every test
file in the ``tests/`` directory without explicit imports.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from crewai import LLM

from src.system_one import FakeSystemOneClient

# ---------------------------------------------------------------------------
# Mock LLM — a MagicMock spec'd to crewai.LLM so tests never hit the network
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm() -> MagicMock:
    """Return a MagicMock that looks like a crewai.LLM instance.

    The mock carries the same attribute names that production code
    inspects (``model``, ``temperature``, ``top_p``, ``max_tokens``,
    ``base_url``) so assertions can verify they are set correctly.
    """
    llm = MagicMock(spec=LLM)
    llm.model = "deepseek-v4-pro"
    llm.temperature = 0.2
    llm.top_p = 0.1
    llm.max_tokens = 8192
    llm.base_url = "https://openrouter.ai/api/v1"
    return llm


# ---------------------------------------------------------------------------
# Mock LLM factory — patches build_llm_for_agent globally so no agent or
# task test ever constructs a real LLM.
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm_factory(mock_llm: MagicMock) -> MagicMock:
    """Patch ``src.llm_factory.build_llm_for_agent`` to return *mock_llm*.

    The patched function is returned so callers can assert it was invoked
    with the expected role constant.
    """
    with patch("src.llm_factory.build_llm_for_agent", return_value=mock_llm) as mock_build:
        yield mock_build


# ---------------------------------------------------------------------------
# System One (Jev) — deterministic decision-layer fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_system_one() -> FakeSystemOneClient:
    """Return a network-free System One client.

    In heuristic mode every un-scripted question is answered with
    ``confidence == 0.0`` (and ``noul`` at 0.5), so evaluators escalate rather
    than silently approving.  Script specific answers via ``client.answers``
    or by constructing the client directly in the test.
    """
    return FakeSystemOneClient()


@pytest.fixture
def patch_system_one(fake_system_one: FakeSystemOneClient) -> MagicMock:
    """Patch every ``build_system_one_client`` entry point to return *fake_system_one*.

    Both the factory (used by the lazy evaluator singletons) and the module-level
    binding inside ``src.crews.syllabus_crew`` are patched so tests can exercise
    the orchestrator's deterministic paths without a live API key.  The yielded
    mock is the orchestrator-facing patch, which is the one callers can assert on.
    """
    import src.crews.syllabus_crew as syllabus_crew  # noqa: PLC0415

    with (
        patch("src.llm_factory.build_system_one_client", return_value=fake_system_one),
        patch.object(
            syllabus_crew,
            "build_system_one_client",
            return_value=fake_system_one,
        ) as orchestrator_patch,
    ):
        yield orchestrator_patch


# ---------------------------------------------------------------------------
# Reusable test values
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_course_name() -> str:
    """A representative course name used across multiple test modules."""
    return "Data Science with Python"


@pytest.fixture
def sample_course_name_safe() -> str:
    """The sanitised version of *sample_course_name*."""
    return "Data_Science_with_Python"


# ---------------------------------------------------------------------------
# Temporary project root — mimics the real project layout inside tmp_path
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_project_root(tmp_path: Path) -> Path:
    """Create a disposable project root with ``src/`` and ``output/`` dirs.

    Tests that need to write files under a fake project root can use this
    fixture and then monkeypatch ``_PROJECT_ROOT`` in the module under test.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "output").mkdir()
    return tmp_path
