"""
test_output_enforcement.py — Output-layout enforcement tests
============================================================

Locks in the corrective behaviour added to stop misplaced, boilerplate and
wrong-run-id artefacts from polluting ``output/``:

* ``_create_lab_scaffolding`` creates the directory skeleton but never a
  boilerplate ``README.md``.
* ``_bind_run_id`` propagates the active run id onto an agent's
  ``output_export_tool`` so wrong-run-id writes are rejected.

All filesystem interaction happens under pytest's ``tmp_path``.
"""

from __future__ import annotations

import types
from pathlib import Path

from src.crews.syllabus_crew import _bind_run_id, _create_lab_scaffolding
from src.exporters.tool import OutputExportTool

# ===================================================================
# _create_lab_scaffolding
# ===================================================================


class TestCreateLabScaffolding:
    """Scaffolding creates directories only — no boilerplate files."""

    def test_creates_tier_directories(self, tmp_path: Path) -> None:
        labs = tmp_path / "labs"
        _create_lab_scaffolding(labs)
        for tier in ("tier1_foundations", "tier2_application", "tier3_architecture"):
            assert (labs / tier / "starter").is_dir()
            assert (labs / tier / "solution").is_dir()

    def test_writes_no_boilerplate_readme(self, tmp_path: Path) -> None:
        """No 'Labs for this tier will be generated here.' stub is written."""
        labs = tmp_path / "labs"
        _create_lab_scaffolding(labs)
        for tier in ("tier1_foundations", "tier2_application", "tier3_architecture"):
            assert not (labs / tier / "README.md").exists()

    def test_gitkeep_placeholders_created(self, tmp_path: Path) -> None:
        labs = tmp_path / "labs"
        _create_lab_scaffolding(labs)
        assert (labs / "tier1_foundations" / "starter" / ".gitkeep").exists()
        assert (labs / "tier1_foundations" / "solution" / ".gitkeep").exists()


# ===================================================================
# _bind_run_id
# ===================================================================


class TestBindRunId:
    """The active run id is propagated onto an agent's export tool."""

    def _agent_with_tool(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(tools=[OutputExportTool(force=True)])

    def test_binds_run_id_to_export_tool(self) -> None:
        agent = self._agent_with_tool()
        _bind_run_id(agent, "2026-09-29_053658_WebXR")
        export_tool = agent.tools[0]
        assert export_tool.run_id == "2026-09-29_053658_WebXR"

    def test_none_agent_is_noop(self) -> None:
        # Must not raise.
        _bind_run_id(None, "2026-09-29_053658_WebXR")

    def test_agent_without_export_tool_is_noop(self) -> None:
        agent = types.SimpleNamespace(tools=[])
        _bind_run_id(agent, "2026-09-29_053658_WebXR")  # must not raise

    def test_rebinding_overwrites_previous_run_id(self) -> None:
        agent = self._agent_with_tool()
        _bind_run_id(agent, "run-A")
        _bind_run_id(agent, "run-B")
        assert agent.tools[0].run_id == "run-B"
