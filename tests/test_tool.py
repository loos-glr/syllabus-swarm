"""
test_tool.py — Tests for src/exporters/tool.py (OutputExportTool)
==================================================================

Tests cover the CrewAI tool wrapper, focusing on JSON auto-parse
behaviour in ``_handle_write_directory_tree`` and ``_handle_write_file``
— the bug that caused the Theory Instructor to fail writing artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.exporters.tool import OutputExportTool

# ===================================================================
# _handle_write_directory_tree — JSON auto-parse
# ===================================================================


class TestHandleWriteDirectoryTree:
    """Tests for the write-directory-tree command handler."""

    @pytest.fixture(autouse=True)
    def _patch_root(self, tmp_path: Path) -> None:
        """Redirect _PROJECT_ROOT to the test's temporary directory."""
        import src.exporters.file_writer as fw
        import src.exporters.tool as t

        self._original_fw_root = fw._PROJECT_ROOT
        self._original_tool_root = t._PROJECT_ROOT
        fw._PROJECT_ROOT = tmp_path.resolve()
        t._PROJECT_ROOT = tmp_path.resolve()
        self._out = tmp_path
        self._out = tmp_path / "output"
        self._out.mkdir()
        fw.OUTPUT_PATHS = fw.OutputPathConfig(root=tmp_path.resolve())
        yield
        fw.OUTPUT_PATHS = fw.OutputPathConfig(root=self._original_fw_root)
        fw._PROJECT_ROOT = self._original_fw_root
        t._PROJECT_ROOT = self._original_tool_root

    def _make_tool(self) -> OutputExportTool:
        return OutputExportTool(force=True)

    # -- JSON string files (the bug scenario) ----------------------------

    def test_accepts_json_string_files(self) -> None:
        """A JSON-encoded string for 'files' is auto-parsed and written."""
        tool = self._make_tool()
        base = str(self._out / "theory_json")
        files_dict = {"artifact.html": "<h1>Hello</h1>", "walkthrough.sh": "#!/bin/bash\necho hi"}
        result = tool._handle_write_directory_tree(
            {
                "base_path": base,
                "files": json.dumps(files_dict),
            }
        )
        parsed = json.loads(result)
        assert parsed["status"] == "ok", f"Expected ok, got: {parsed}"
        assert (self._out / "theory_json" / "artifact.html").read_text() == "<h1>Hello</h1>"
        assert (self._out / "theory_json" / "walkthrough.sh").read_text() == "#!/bin/bash\necho hi"

    def test_accepts_native_dict_files(self) -> None:
        """A native Python dict for 'files' still works (regression)."""
        tool = self._make_tool()
        base = str(self._out / "theory_native")
        result = tool._handle_write_directory_tree(
            {
                "base_path": base,
                "files": {"index.html": "<p>Native</p>"},
            }
        )
        parsed = json.loads(result)
        assert parsed["status"] == "ok", f"Expected ok, got: {parsed}"
        assert (self._out / "theory_native" / "index.html").read_text() == "<p>Native</p>"

    def test_rejects_invalid_json_string(self) -> None:
        """An unparseable JSON string returns an error."""
        tool = self._make_tool()
        result = tool._handle_write_directory_tree(
            {
                "base_path": str(self._out / "bad_json"),
                "files": "not valid json {{{",
            }
        )
        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert "could not parse json" in parsed["message"].lower()

    def test_rejects_missing_files_param(self) -> None:
        """Missing 'files' parameter returns an error."""
        tool = self._make_tool()
        result = tool._handle_write_directory_tree(
            {
                "base_path": str(self._out / "no_files"),
            }
        )
        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert "missing or invalid" in parsed["message"].lower()

    def test_rejects_missing_base_path(self) -> None:
        """Missing 'base_path' parameter returns an error."""
        tool = self._make_tool()
        result = tool._handle_write_directory_tree(
            {
                "files": {"a.txt": "content"},
            }
        )
        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert "base_path" in parsed["message"].lower()

    def test_json_string_with_special_characters(self) -> None:
        """JSON-encoded files with special characters (newlines, quotes) are preserved."""
        tool = self._make_tool()
        base = str(self._out / "special_chars")
        content_with_specials = "const x = \"hello\";\nconst y = 'world';\n// comment"
        result = tool._handle_write_directory_tree(
            {
                "base_path": base,
                "files": json.dumps({"code.js": content_with_specials}),
            }
        )
        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        written = (self._out / "special_chars" / "code.js").read_text()
        assert written == content_with_specials


# ===================================================================
# _handle_write_file — JSON auto-parse
# ===================================================================


class TestHandleWriteFile:
    """Tests for the write-file command handler."""

    @pytest.fixture(autouse=True)
    def _patch_root(self, tmp_path: Path) -> None:
        """Redirect _PROJECT_ROOT to the test's temporary directory."""
        import src.exporters.file_writer as fw
        import src.exporters.tool as t

        self._original_fw_root = fw._PROJECT_ROOT
        self._original_tool_root = t._PROJECT_ROOT
        fw._PROJECT_ROOT = tmp_path.resolve()
        t._PROJECT_ROOT = tmp_path.resolve()
        self._out = tmp_path / "output"
        self._out.mkdir()
        fw.OUTPUT_PATHS = fw.OutputPathConfig(root=tmp_path.resolve())
        yield
        fw.OUTPUT_PATHS = fw.OutputPathConfig(root=self._original_fw_root)
        fw._PROJECT_ROOT = self._original_fw_root
        t._PROJECT_ROOT = self._original_tool_root

    def _make_tool(self) -> OutputExportTool:
        return OutputExportTool(force=True)

    def test_unwraps_json_encoded_content_string(self) -> None:
        """A JSON-encoded string for 'content' is unwrapped to the inner string."""
        tool = self._make_tool()
        dest = str(self._out / "unwrapped.txt")
        result = tool._handle_write_file(
            {
                "path": dest,
                "content": json.dumps("Hello from JSON string"),
            }
        )
        parsed = json.loads(result)
        assert parsed["status"] == "ok", f"Expected ok, got: {parsed}"
        assert (self._out / "unwrapped.txt").read_text() == "Hello from JSON string"

    def test_preserves_raw_string_content(self) -> None:
        """A plain (non-JSON) string is written as-is (regression)."""
        tool = self._make_tool()
        dest = str(self._out / "raw.txt")
        result = tool._handle_write_file(
            {
                "path": dest,
                "content": "Just a plain string, not JSON",
            }
        )
        parsed = json.loads(result)
        assert parsed["status"] == "ok", f"Expected ok, got: {parsed}"
        assert (self._out / "raw.txt").read_text() == "Just a plain string, not JSON"

    def test_preserves_json_object_as_string(self) -> None:
        """A JSON object (dict) passed as content is NOT unwrapped — it stays as a JSON string."""
        tool = self._make_tool()
        dest = str(self._out / "obj.txt")
        content = json.dumps({"key": "value"})
        result = tool._handle_write_file(
            {
                "path": dest,
                "content": content,
            }
        )
        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        # The content is a JSON string representing an object, so it should
        # NOT be unwrapped (only plain strings get unwrapped).
        assert (self._out / "obj.txt").read_text() == content

    def test_rejects_missing_path(self) -> None:
        """Missing 'path' parameter returns an error."""
        tool = self._make_tool()
        result = tool._handle_write_file(
            {
                "content": "some content",
            }
        )
        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert "path" in parsed["message"].lower()

    def test_rejects_missing_content(self) -> None:
        """Missing 'content' parameter returns an error."""
        tool = self._make_tool()
        result = tool._handle_write_file(
            {
                "path": str(self._out / "missing.txt"),
            }
        )
        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert "content" in parsed["message"].lower()


# ===================================================================
# _run — command dispatch
# ===================================================================


class TestRunDispatch:
    """Tests for the _run entry point that dispatches commands."""

    @pytest.fixture(autouse=True)
    def _patch_root(self, tmp_path: Path) -> None:
        """Redirect _PROJECT_ROOT to the test's temporary directory."""
        import src.exporters.file_writer as fw
        import src.exporters.tool as t

        self._original_fw_root = fw._PROJECT_ROOT
        self._original_tool_root = t._PROJECT_ROOT
        fw._PROJECT_ROOT = tmp_path.resolve()
        t._PROJECT_ROOT = tmp_path.resolve()
        self._out = tmp_path / "output"
        self._out.mkdir()
        fw.OUTPUT_PATHS = fw.OutputPathConfig(root=tmp_path.resolve())
        yield
        fw.OUTPUT_PATHS = fw.OutputPathConfig(root=self._original_fw_root)
        fw._PROJECT_ROOT = self._original_fw_root
        t._PROJECT_ROOT = self._original_tool_root

    def _make_tool(self) -> OutputExportTool:
        return OutputExportTool(force=True)

    def test_dispatches_write_directory_tree_with_json_files(self) -> None:
        """_run dispatches 'write-directory-tree' and handles JSON string files."""
        tool = self._make_tool()
        base = str(self._out / "dispatched")
        result = tool._run(
            command="write-directory-tree",
            base_path=base,
            files=json.dumps({"page.html": "<h1>Dispatched</h1>"}),
        )
        parsed = json.loads(result)
        assert parsed["status"] == "ok", f"Expected ok, got: {parsed}"
        assert (self._out / "dispatched" / "page.html").read_text() == "<h1>Dispatched</h1>"

    def test_dispatches_write_file_with_json_content(self) -> None:
        """_run dispatches 'write-file' and handles JSON-encoded content."""
        tool = self._make_tool()
        dest = str(self._out / "dispatched_file.txt")
        result = tool._run(
            command="write-file",
            path=dest,
            content=json.dumps("Dispatched content"),
        )
        parsed = json.loads(result)
        assert parsed["status"] == "ok", f"Expected ok, got: {parsed}"
        assert (self._out / "dispatched_file.txt").read_text() == "Dispatched content"

    def test_unknown_command_returns_error(self) -> None:
        """An unknown command returns an error."""
        tool = self._make_tool()
        result = tool._run(command="nonexistent-command")
        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert "unknown command" in parsed["message"].lower()
