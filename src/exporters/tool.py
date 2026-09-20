"""
tool.py — CrewAI Tool Wrapper for the Output Exporter System
=============================================================

Issue #4: Automated Output Exporter & Markdown Packager

Wraps the :mod:`src.exporters.file_writer` and :mod:`src.exporters.manifest`
functionality into a single **CrewAI Tool** that agents can invoke during
crew execution, plus a **standalone CLI** for direct use by operators.

CrewAI Integration
------------------
Agents use this tool by calling :meth:`OutputExportTool._run` with a
``command`` string and keyword arguments::

    tool = OutputExportTool(force=True)
    tool._run(command="write-syllabus",
              course_name="Data Science", content="# Syllabus ...")

CLI Usage
---------
.. code-block:: bash

    # Write a syllabus directly:
    python -m src.exporters.tool write-syllabus \\
        --course "Data Science with Python" \\
        --content-file ./syllabus.md

    # Write a batch of lab files:
    python -m src.exporters.tool write-labs \\
        --course "Data Science with Python" \\
        --tier tier1_foundations \\
        --dir ./labs_output

    # Generate / refresh the output manifest:
    python -m src.exporters.tool generate-manifest \\
        --course "Data Science with Python"

    # With force overwrite:
    python -m src.exporters.tool write-syllabus \\
        --course "Data Science" --content "..." --force

Public API
----------
* ``OutputExportTool`` — the CrewAI ``BaseTool`` subclass.
* ``build_cli_parser()`` — builds the ``argparse`` argument parser.
* ``main()`` — CLI entry point (also ``if __name__ == "__main__"``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from src.exporters.file_writer import (
    OUTPUT_PATHS,
    FileWriteError,
    write_directory_tree,
    write_file,
    write_remotion_manifest,
    write_syllabus,
)
from src.exporters.manifest import (
    update_output_manifest,
)

# ---------------------------------------------------------------------------
# Project-root resolution (mirrors file_writer.py)
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Tool result helpers
# ---------------------------------------------------------------------------


def _ok(message: str, path: str | Path | None = None) -> str:
    """Format a successful tool result string."""
    if path:
        return json.dumps({"status": "ok", "message": message, "path": str(path)})
    return json.dumps({"status": "ok", "message": message})


def _err(message: str) -> str:
    """Format an error tool result string."""
    return json.dumps({"status": "error", "message": message})


# ---------------------------------------------------------------------------
# CrewAI Tool — OutputExportTool
# ---------------------------------------------------------------------------


class OutputExportToolArgs(BaseModel):
    """Argument schema for :class:`OutputExportTool`.

    ``command`` is required so the LLM always selects an operation.  The
    remaining fields are optional and only relevant to specific commands —
    they are ignored by commands that do not use them.
    """

    command: str = Field(
        ...,
        description=(
            "The operation to perform. One of: write-syllabus, write-labs, "
            "generate-manifest, export-course-graph, write-file, "
            "write-directory-tree."
        ),
    )
    course_name: str = Field(default="", description="Human-facing course name.")
    content: str = Field(default="", description="Full file content (Markdown or source code).")
    tier: str = Field(
        default="",
        description="Lab tier directory name, e.g. 'tier1_foundations'.",
    )
    run_id: str = Field(
        default="",
        description=("Per-run output directory id, e.g. '2026-08-24_071602_Course_Name'."),
    )
    files: Any = Field(
        default=None,
        description=(
            "Mapping of relative file paths to file contents. May be a dict or a JSON string."
        ),
    )
    path: str = Field(default="", description="Single file path (write-file command).")
    base_path: str = Field(
        default="",
        description="Base directory (write-directory-tree command).",
    )
    course_slug: str = Field(default="", description="URL-safe course slug (export-course-graph).")
    specification: Any = Field(
        default=None, description="Course specification dict (export-course-graph)."
    )
    learning_objectives: Any = Field(
        default=None, description="Learning objectives list (export-course-graph)."
    )
    key_concepts: Any = Field(default=None, description="Key concepts list (export-course-graph).")
    prerequisites: Any = Field(
        default=None, description="Prerequisites list (export-course-graph)."
    )
    modules: Any = Field(default=None, description="Modules list (export-course-graph).")


class OutputExportTool(BaseTool):
    """CrewAI Tool that wraps syllabus-swarm file-writer and manifest operations.

    Agents use this tool to persist generated syllabi, tiered labs, and
    rubrics to disk, and to refresh the ``output/README.md`` manifest.

    The tool accepts a ``command`` argument to select the operation,
    plus keyword parameters specific to each command.

    Commands
    --------
    ``write-syllabus``
        Write a single syllabus Markdown file.
        Required kwargs: ``course_name``, ``content``.

    ``write-labs``
        Write a batch of lab files from a directory-tree mapping.
        Required kwargs: ``course_name``, ``tier``, ``files``.

    ``generate-manifest``
        Scan ``output/`` and regenerate ``output/README.md``.
        Optional kwargs: ``course_name``.

    ``write-file``
        Low-level: write arbitrary content to a single file.
        Required kwargs: ``path``, ``content``.

    ``write-directory-tree``
        Low-level: write a batch of files from a ``{rel_path: content}`` dict.
        Required kwargs: ``base_path``, ``files``.

    Parameters
    ----------
    force : bool
        When ``True``, existing files are silently overwritten.  When
        ``False`` (the default), a ``FileWriteError`` is raised for
        pre-existing files.
    """

    name: str = "output_export_tool"
    description: str = (
        "Writes syllabus, labs, and manifest files to the output/ directory "
        "tree.  Call it with a required 'command' keyword argument plus "
        "command-specific keyword arguments.  Supported commands: "
        "write-syllabus, write-labs, generate-manifest, export-course-graph, "
        "write-file, write-directory-tree, write-remotion-manifest.  "
        "Example: command='write-syllabus', course_name='ML 101', "
        "content='# Syllabus\\n...'"
    )

    args_schema: type[BaseModel] = OutputExportToolArgs

    force: bool = False

    # ------------------------------------------------------------------
    # CrewAI entry point
    # ------------------------------------------------------------------

    @staticmethod
    def _usage_hint(reason: str) -> str:
        """Build a constructive usage error that tells the agent how to retry."""
        return (
            f"{reason}  Call output_export_tool with a 'command' keyword "
            "argument, e.g. command='write-labs', course_name='...', "
            "tier='tier1_foundations', run_id='...', files={...}.  "
            "Supported commands: write-syllabus, write-labs, "
            "generate-manifest, export-course-graph, write-file, "
            "write-directory-tree, write-remotion-manifest."
        )

    def _run(self, **kwargs: Any) -> str:
        """Execute the tool command selected by *kwargs*.

        The first positional/keyword argument is interpreted as the command
        name.  All other keyword arguments are forwarded to the matching
        internal handler.

        Returns
        -------
        str
            JSON-encoded result: ``{"status": "ok", ...}`` or
            ``{"status": "error", "message": "..."}``.
        """
        # ── Normalise: accept a plain string or JSON-encoded dict ──────
        parsed: dict[str, Any]
        if "command" not in kwargs:
            # CrewAI agents may pass a single JSON string as the first arg.
            raw = kwargs.get("input", kwargs.get("request_id", ""))
            if isinstance(raw, str) and raw.strip():
                try:
                    parsed = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    return _err(self._usage_hint(f"Invalid JSON input: {raw!r}"))
                if not isinstance(parsed, dict) or "command" not in parsed:
                    return _err(self._usage_hint("Missing 'command' key in parsed input."))
            else:
                return _err(self._usage_hint("No arguments were provided."))
        else:
            parsed = dict(kwargs)

        command = str(parsed.get("command", "")).strip()

        # ── Dispatch ─────────────────────────────────────────────────
        try:
            if command == "write-syllabus":
                return self._handle_write_syllabus(parsed)
            elif command == "write-labs":
                return self._handle_write_labs(parsed)
            elif command == "generate-manifest":
                return self._handle_generate_manifest(parsed)
            elif command == "export-course-graph":
                return self._handle_export_course_graph(parsed)
            elif command == "write-file":
                return self._handle_write_file(parsed)
            elif command == "write-directory-tree":
                return self._handle_write_directory_tree(parsed)
            else:
                return _err(
                    f"Unknown command: '{command}'.  Supported commands: "
                    "write-syllabus, write-labs, generate-manifest, "
                    "export-course-graph, write-file, write-directory-tree."
                )
        except FileWriteError as exc:
            return _err(str(exc))
        except ValueError as exc:
            return _err(str(exc))
        except Exception as exc:
            return _err(f"Unexpected error: {exc}")

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    def _handle_write_syllabus(self, params: dict[str, Any]) -> str:
        """Write a syllabus Markdown file.

        When a ``run_id`` is provided, the syllabus is written to
        ``output/<run_id>/syllabus/<course>.md`` (per-run isolation).
        Without a ``run_id``, it falls back to the global
        ``output/syllabus/<course>.md`` path.
        """
        course_name = str(params.get("course_name", ""))
        if not course_name:
            return _err("Missing required parameter: 'course_name'.")

        content = params.get("content", "")
        if not content:
            return _err("Missing required parameter: 'content'.")

        run_id = str(params.get("run_id", "") or "")
        path = write_syllabus(course_name, content, force=self.force, run_id=run_id or None)
        return _ok(f"Syllabus written for '{course_name}'.", path)

    def _handle_write_labs(self, params: dict[str, Any]) -> str:
        """Write a batch of lab files from a files-dict mapping.

        Files are written to one of two locations depending on whether
        a ``run_id`` is provided:

        * **With run_id**: ``output/<run_id>/labs/<tier>/``
          (per-run isolation, the default for pipeline runs).
        * **Without run_id**: ``output/labs/<tier>/``
          (shared global path, used by the CLI or standalone tool calls).

        The ``course_name`` is sanitised into a safe directory name.
        """
        course_name = str(params.get("course_name", ""))
        if not course_name:
            return _err("Missing required parameter: 'course_name'.")

        tier = str(params.get("tier", "tier1_foundations"))
        run_id = str(params.get("run_id", "") or "")

        files_raw = params.get("files")
        if not files_raw:
            return _err(
                "Missing or invalid 'files' parameter.  "
                "Expected a dict of {relative_path: content}."
            )

        # Auto-parse JSON strings — CrewAI agents often pass the files
        # mapping as a JSON-encoded string rather than a native dict.
        if isinstance(files_raw, str):
            try:
                files_raw = json.loads(files_raw)
            except (json.JSONDecodeError, TypeError):
                return _err(
                    "Invalid 'files' parameter: could not parse JSON string.  "
                    "Expected a JSON object mapping relative paths to content."
                )

        if not isinstance(files_raw, dict):
            return _err(
                "Missing or invalid 'files' parameter.  "
                "Expected a dict of {relative_path: content}."
            )

        files_dict: dict[str, Any] = files_raw

        if run_id:
            base = _PROJECT_ROOT / "output" / run_id / "labs" / tier
        else:
            base = OUTPUT_PATHS.labs_dir / tier

        written = write_directory_tree(base, files_dict, force=self.force)
        return _ok(
            f"Wrote {len(written)} lab file(s) for '{course_name}' under tier '{tier}'.",
            str(base),
        )

    def _handle_generate_manifest(self, params: dict[str, Any] | None = None) -> str:
        """Scan output/ and regenerate output/README.md."""
        if params is None:
            params = {}
        course_name = str(params.get("course_name", ""))
        path = update_output_manifest(course_name=course_name)
        return _ok("Manifest regenerated.", path)

    def _handle_export_course_graph(self, params: dict[str, Any]) -> str:
        """Export a machine-readable course graph as JSON.

        Required kwargs: ``course_name``, ``course_slug``,
        ``specification`` (dict with course_context and primary_language).

        Optional kwargs: ``learning_objectives``, ``key_concepts``,
        ``prerequisites``, ``modules``, ``run_id``.
        """
        from src.models import CourseGraph, CourseSpecification, ModuleSummary

        course_name = str(params.get("course_name", ""))
        if not course_name:
            return _err("Missing required parameter: 'course_name'.")

        course_slug = str(params.get("course_slug", ""))
        if not course_slug:
            return _err("Missing required parameter: 'course_slug'.")

        spec_dict = params.get("specification")
        if not spec_dict or not isinstance(spec_dict, dict):
            return _err(
                "Missing or invalid 'specification' parameter.  "
                "Expected a dict with 'course_context' and 'primary_language'."
            )

        try:
            specification = CourseSpecification.model_validate(spec_dict)
        except Exception as exc:
            return _err(f"Invalid specification: {exc}")

        # ── Build ModuleSummary list ────────────────────────────────
        modules_raw = params.get("modules")
        modules: list[ModuleSummary] = []
        if modules_raw and isinstance(modules_raw, list):
            for m in modules_raw:
                if isinstance(m, dict):
                    try:
                        modules.append(ModuleSummary.model_validate(m))
                    except Exception as exc:
                        return _err(f"Invalid module entry: {exc}")
                elif isinstance(m, ModuleSummary):
                    modules.append(m)

        # ── Build CourseGraph ───────────────────────────────────────
        graph = CourseGraph(
            specification=specification,
            course_slug=course_slug,
            learning_objectives=list(params.get("learning_objectives", []) or []),
            key_concepts=list(params.get("key_concepts", []) or []),
            prerequisites=list(params.get("prerequisites", []) or []),
            modules=modules,
        )

        # ── Write JSON ──────────────────────────────────────────────
        run_id = str(params.get("run_id", "") or "")
        if run_id:
            output_path = _PROJECT_ROOT / "output" / run_id / "course_graph.json"
        else:
            # Fall back to top-level output dir
            output_path = _PROJECT_ROOT / "output" / "course_graph.json"

        json_content = graph.model_dump_json(indent=2)
        path = write_file(output_path, json_content, force=self.force)

        return _ok(f"Course graph exported for '{course_name}'.", path)

    def _handle_write_file(self, params: dict[str, Any]) -> str:
        """Low-level: write arbitrary content to a single file."""
        file_path = str(params.get("path", ""))
        if not file_path:
            return _err("Missing required parameter: 'path'.")

        content = params.get("content", "")
        if not content:
            return _err("Missing required parameter: 'content'.")

        # Auto-parse JSON strings — CrewAI agents often pass content
        # as a JSON-encoded string rather than a native string.
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, str):
                    content = parsed
            except (json.JSONDecodeError, TypeError):
                pass  # Not JSON; use the raw string as-is.

        path = write_file(file_path, content, force=self.force)
        return _ok("File written.", path)

    def _handle_write_directory_tree(self, params: dict[str, Any]) -> str:
        """Low-level: write a batch of files from a directory-tree mapping."""
        base_path = str(params.get("base_path", ""))
        if not base_path:
            return _err("Missing required parameter: 'base_path'.")

        files_raw = params.get("files")
        if not files_raw:
            return _err(
                "Missing or invalid 'files' parameter.  "
                "Expected a dict of {relative_path: content}."
            )

        # Auto-parse JSON strings — CrewAI agents often pass the files
        # mapping as a JSON-encoded string rather than a native dict.
        if isinstance(files_raw, str):
            try:
                files_raw = json.loads(files_raw)
            except (json.JSONDecodeError, TypeError):
                return _err(
                    "Invalid 'files' parameter: could not parse JSON string.  "
                    "Expected a JSON object mapping relative paths to content."
                )

        if not isinstance(files_raw, dict):
            return _err(
                "Missing or invalid 'files' parameter.  "
                "Expected a dict of {relative_path: content}."
            )

        files_dict: dict[str, Any] = files_raw
        written = write_directory_tree(base_path, files_dict, force=self.force)
        return _ok(
            f"Wrote {len(written)} file(s) to '{base_path}'.",
            str(written[0]) if written else base_path,
        )

    def _handle_write_remotion_manifest(self, params: dict[str, Any]) -> str:
        """Write a RemotionManifest as a .tsx file to src/export/vac/.

        Required kwargs: ``manifest`` (a ``RemotionManifest``-compatible dict).

        The *manifest* dict must contain at minimum a ``composition_id``
        (``str``) and ``components`` (``list[dict]``) field.  Optional fields
        include ``duration_in_frames``, ``fps``, ``width``, ``height``, and
        ``module_name``.
        """
        from src.models import RemotionManifest

        manifest_raw = params.get("manifest")
        if not manifest_raw:
            return _err(
                "Missing required parameter: 'manifest'.  Expected a dict with "
                "at least 'composition_id' and 'components'."
            )

        # Auto-parse JSON strings.
        if isinstance(manifest_raw, str):
            try:
                manifest_raw = json.loads(manifest_raw)
            except (json.JSONDecodeError, TypeError):
                return _err("Invalid 'manifest' parameter: could not parse JSON string.")

        if not isinstance(manifest_raw, dict):
            return _err(
                f"Invalid 'manifest' parameter: expected a dict, got {type(manifest_raw).__name__}."
            )

        try:
            manifest = RemotionManifest.model_validate(manifest_raw)
        except Exception as exc:
            return _err(f"Invalid RemotionManifest: {exc}")

        try:
            path = write_remotion_manifest(manifest, force=self.force)
        except FileWriteError as exc:
            return _err(str(exc))

        return _ok(f"Remotion manifest written for '{manifest.composition_id}'.", path)


# ---------------------------------------------------------------------------
# CLI — Argument Parser
# ---------------------------------------------------------------------------


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``output_export_tool`` CLI.

    Returns
    -------
    argparse.ArgumentParser
        A fully-configured parser with subcommands for all operations.
    """
    parser = argparse.ArgumentParser(
        prog="output-export-tool",
        description=(
            "CrewAI Output Export Tool — write syllabi, labs, and "
            "manifests to the output/ directory from the command line."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  %(prog)s write-syllabus --course "ML 101" '
            '--content "# Syllabus"\n'
            '  %(prog)s write-syllabus --course "ML 101" '
            "--content-file ./syllabus.md\n"
            '  %(prog)s write-labs --course "ML 101" '
            '--tier tier1_foundations --files \'{"lab1.py": "..."}\'\n'
            '  %(prog)s generate-manifest --course "ML 101"\n'
            '  %(prog)s export-course-graph --course "ML 101" '
            '--slug ml-101 --spec \'{"course_context":"...",'
            '"primary_language":"Python"}\'\n'
            "  %(prog)s write-file --path output/test.md "
            '--content "# Hello"\n'
            "  %(prog)s write-directory-tree --base-path output/labs "
            '--files \'{"a.py": "# a"}\'\n'
            "  %(prog)s write-remotion-manifest "
            '--manifest \'{"composition_id":"demo","components":[]}\'\n'
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing files without raising an error.",
    )

    sub = parser.add_subparsers(dest="command", required=True, help="Operation to perform.")

    # ── write-syllabus ──────────────────────────────────────────────
    ws = sub.add_parser(
        "write-syllabus",
        help="Write a syllabus .md file to output/syllabus/<course>.md.",
        description=(
            "Write a Humanics-aligned syllabus as a Markdown file under "
            "``output/syllabus/``.  The course name is sanitised into a "
            "safe filename automatically."
        ),
    )
    ws.add_argument(
        "--course",
        "-c",
        required=True,
        dest="course_name",
        help="Human-readable course title (e.g. 'Data Science with Python').",
    )
    ws.add_argument(
        "--content",
        default="",
        help="Syllabus content as a raw string.",
    )
    ws.add_argument(
        "--content-file",
        default=None,
        dest="content_file",
        help="Path to a file containing the syllabus content.",
    )

    # ── write-labs ──────────────────────────────────────────────────
    wl = sub.add_parser(
        "write-labs",
        help="Write a batch of lab files into output/labs/<course>/.",
        description=(
            "Populate a tiered lab directory under "
            "``output/labs/<course>/``.  "
            "Files are specified as a JSON mapping from relative paths "
            "to their content."
        ),
    )
    wl.add_argument(
        "--course",
        "-c",
        required=True,
        dest="course_name",
        help="Human-readable course title.",
    )
    wl.add_argument(
        "--tier",
        "-t",
        default="tier1_foundations",
        help="Tier directory name (default: 'tier1_foundations').",
    )
    wl.add_argument(
        "--files",
        default="{}",
        help=(
            "JSON object mapping relative file paths to their content.  "
            'Example: \'{"starter/lab1.py": "# TODO"}\''
        ),
    )
    wl.add_argument(
        "--files-dir",
        default=None,
        dest="files_dir",
        help=(
            "Directory containing lab files to write.  The directory "
            "tree is mirrored under the target path."
        ),
    )

    # ── generate-manifest ───────────────────────────────────────────
    gm = sub.add_parser(
        "generate-manifest",
        help="Scan output/ and regenerate output/README.md.",
        description=(
            "Walk the entire ``output/`` directory, collect statistics "
            "for every artifact, and rewrite ``output/README.md`` with "
            "a summary table and directory tree."
        ),
    )
    gm.add_argument(
        "--course",
        "-c",
        default="",
        dest="course_name",
        help="Optional course name included in the manifest header.",
    )

    # ── export-course-graph ─────────────────────────────────────────
    ecg = sub.add_parser(
        "export-course-graph",
        help="Export a machine-readable course graph as JSON.",
        description=(
            "Construct a ``CourseGraph`` model from the provided data "
            "and write it as ``output/<run_id>/course_graph.json``."
        ),
    )
    ecg.add_argument(
        "--course",
        "-c",
        required=True,
        dest="course_name",
        help="Human-readable course title.",
    )
    ecg.add_argument(
        "--slug",
        required=True,
        dest="course_slug",
        help="URL- / filesystem-safe identifier for the course.",
    )
    ecg.add_argument(
        "--spec",
        required=True,
        dest="specification",
        help=(
            "JSON object with 'course_context' and 'primary_language' "
            "fields (i.e. a CourseSpecification)."
        ),
    )
    ecg.add_argument(
        "--objectives",
        default="[]",
        dest="learning_objectives",
        help="JSON array of learning objective strings.",
    )
    ecg.add_argument(
        "--concepts",
        default="[]",
        dest="key_concepts",
        help="JSON array of key concept strings.",
    )
    ecg.add_argument(
        "--prereqs",
        default="[]",
        dest="prerequisites",
        help="JSON array of prerequisite strings.",
    )
    ecg.add_argument(
        "--modules",
        default="[]",
        help=(
            "JSON array of module objects, each with 'title', "
            "'duration_weeks', and 'topics' fields."
        ),
    )
    ecg.add_argument(
        "--run-id",
        default="",
        dest="run_id",
        help="Optional run ID directory (e.g. '2026-08-23_120000_course').",
    )

    # ── write-file (low-level) ──────────────────────────────────────
    wf = sub.add_parser(
        "write-file",
        help="Low-level: write arbitrary content to a single file.",
        description="Write a single file anywhere under the project root.",
    )
    wf.add_argument(
        "--path",
        required=True,
        help="Destination path (relative to project root, or absolute).",
    )
    wf.add_argument(
        "--content",
        default="",
        help="Text content to write.",
    )
    wf.add_argument(
        "--content-file",
        default=None,
        dest="content_file",
        help="Path to a file whose contents will be written.",
    )

    # ── write-directory-tree (low-level) ────────────────────────────
    wdt = sub.add_parser(
        "write-directory-tree",
        help="Low-level: write a batch of files from a path→content map.",
        description=(
            "Write multiple files under a base directory.  Each entry is "
            "a relative path and its content."
        ),
    )
    wdt.add_argument(
        "--base-path",
        required=True,
        dest="base_path",
        help="Root directory under which all files will be written.",
    )
    wdt.add_argument(
        "--files",
        default="{}",
        help=(
            "JSON object mapping relative file paths to their content.  "
            'Example: \'{"a/b.txt": "hello", "a/c.py": "# code"}\''
        ),
    )

    # ── write-remotion-manifest ─────────────────────────────────────
    wrm = sub.add_parser(
        "write-remotion-manifest",
        help="Write a RemotionManifest as a .tsx file to src/export/vac/.",
        description=(
            "Write a deterministic Video-as-Code React/Remotion .tsx "
            "composition from a RemotionManifest descriptor."
        ),
    )
    wrm.add_argument(
        "--manifest",
        required=True,
        help=(
            "JSON object representing a RemotionManifest.  "
            "Must contain 'composition_id' (str) and "
            "'components' (list[dict]).  "
            'Example: \'{"composition_id":"demo","components":[]}\''
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _read_content(content: str, content_file: str | None) -> str:
    """Resolve content: use *content* if non-empty, else read *content_file*."""
    if content.strip():
        return content
    if content_file:
        p = Path(content_file)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        return p.read_text(encoding="utf-8")
    return ""


def _collect_files_from_dir(dir_path: str) -> dict[str, str]:
    """Walk *dir_path* and return ``{relative_path: content}`` mapping."""
    base = Path(dir_path)
    if not base.is_absolute():
        base = _PROJECT_ROOT / base
    result: dict[str, str] = {}
    if not base.is_dir():
        print(f"Warning: Not a directory: {base}", file=sys.stderr)
        return result
    for f in sorted(base.rglob("*")):
        if f.is_file() and not f.name.startswith("."):
            rel = str(f.relative_to(base))
            result[rel] = f.read_text(encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point — parse arguments, build tool, execute command.

    Parameters
    ----------
    argv : list[str] or None
        Command-line arguments.  When ``None``, reads from ``sys.argv``.
    """
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    # ── Build the tool ────────────────────────────────────────────────
    tool = OutputExportTool(force=args.force)
    command = args.command

    # ── Dispatch ──────────────────────────────────────────────────────
    result_str: str

    if command == "write-syllabus":
        content = _read_content(args.content, getattr(args, "content_file", None))
        result_str = tool._handle_write_syllabus(
            {"course_name": args.course_name, "content": content}
        )

    elif command == "write-labs":
        files_dict = json.loads(args.files)
        if args.files_dir:
            files_dict.update(_collect_files_from_dir(args.files_dir))
        if not files_dict:
            result_str = _err(
                "No files provided.  Use --files or --files-dir to specify lab content."
            )
        else:
            result_str = tool._handle_write_labs(
                {
                    "course_name": args.course_name,
                    "tier": args.tier,
                    "files": files_dict,
                }
            )

    elif command == "generate-manifest":
        result_str = tool._handle_generate_manifest({"course_name": args.course_name})

    elif command == "export-course-graph":
        spec = json.loads(args.specification)
        objectives = json.loads(args.learning_objectives)
        concepts = json.loads(args.key_concepts)
        prereqs = json.loads(args.prerequisites)
        modules_list = json.loads(args.modules)
        result_str = tool._handle_export_course_graph(
            {
                "course_name": args.course_name,
                "course_slug": args.course_slug,
                "specification": spec,
                "learning_objectives": objectives,
                "key_concepts": concepts,
                "prerequisites": prereqs,
                "modules": modules_list,
                "run_id": args.run_id,
            }
        )

    elif command == "write-file":
        content = _read_content(args.content, getattr(args, "content_file", None))
        result_str = tool._handle_write_file({"path": args.path, "content": content})

    elif command == "write-directory-tree":
        files_dict = json.loads(args.files)
        result_str = tool._handle_write_directory_tree(
            {"base_path": args.base_path, "files": files_dict}
        )

    elif command == "write-remotion-manifest":
        manifest_raw = args.manifest
        result_str = tool._handle_write_remotion_manifest({"manifest": manifest_raw})

    else:
        result_str = _err(f"Unknown command: '{command}'.")

    # ── Print result ──────────────────────────────────────────────────
    try:
        parsed = json.loads(result_str)
        status = parsed.get("status", "unknown")
        message = parsed.get("message", "")
    except json.JSONDecodeError:
        status = "unknown"
        message = result_str

    if status == "ok":
        path = json.loads(result_str).get("path", "")
        if path:
            print(f"OK: {message}")
            print(f"     -> {path}")
        else:
            print(f"OK: {message}")
    else:
        print(f"ERROR: {message}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
