"""
manifest.py — Output Manifest Generator for the Exporter System
================================================================

Issue #4: Automated Output Exporter & Markdown Packager

Scans the ``output/`` directory after every crew run and produces (or
updates) ``output/README.md`` — a single Markdown manifest that lists
every exported artifact with paths, sizes, and a directory tree so that
a human (or downstream tool) can understand exactly what was produced.

Public API
----------
* ``update_output_manifest(course_name, syllabus_path, labs_base_path)``
  — the primary entry point.  Called by ``syllabus_crew`` after all agents
  complete.
* ``ArtifactSummary`` — a lightweight dataclass holding per-artifact stats.

Design rules
------------
* The manifest is **always** written with ``force=True`` so it is
  updated on every run rather than accumulating stale entries.
* If the ``output/`` directory does not exist yet the function returns
  a minimal manifest noting "No artifacts exported yet."
* All paths in the manifest are relative to the project root so the
  file is portable across machines.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from src.exporters.file_writer import (
    OUTPUT_PATHS,
    _sanitize_filename,
    write_file,
)

if TYPE_CHECKING:
    from src.models import CourseGraph

# ---------------------------------------------------------------------------
# Project-root resolution
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Path = OUTPUT_PATHS.root


# ---------------------------------------------------------------------------
# Helper — build a relative path string for display
# ---------------------------------------------------------------------------


def _rel(path: Path) -> str:
    """Return *path* relative to the project root, or its absolute form."""
    try:
        return str(path.relative_to(_PROJECT_ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ArtifactSummary:
    """Lightweight stats for a single exported file or directory."""

    path: str  # relative path from project root
    size_bytes: int = 0
    is_dir: bool = False
    file_count: int = 0  # recursive file count when is_dir=True


@dataclass
class ManifestData:
    """Aggregated summary of everything under ``output/``."""

    generated_at: str = field(
        default_factory=lambda: datetime.datetime.now(tz=datetime.UTC).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    )
    course_name: str = ""
    syllabus: ArtifactSummary | None = None
    lab_tiers: list[ArtifactSummary] = field(default_factory=list)
    total_files: int = 0
    total_size_bytes: int = 0


# ---------------------------------------------------------------------------
# Directory scanner
# ---------------------------------------------------------------------------


def _scan_directory(base: Path) -> list[ArtifactSummary]:
    """Recursively walk *base* and return one ``ArtifactSummary`` per entry.

    Directories get a ``file_count`` and cumulative ``size_bytes`` for the
    whole subtree.  Files report only their own size.
    """
    result: list[ArtifactSummary] = []
    if not base.exists():
        return result

    for entry in sorted(base.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            count = 0
            total = 0
            for f in entry.rglob("*"):
                if f.is_file() and not f.name.startswith("."):
                    try:
                        total += f.stat().st_size
                    except OSError:
                        pass
                    count += 1
            result.append(
                ArtifactSummary(
                    path=_rel(entry),
                    size_bytes=total,
                    is_dir=True,
                    file_count=count,
                )
            )
        else:
            try:
                sz = entry.stat().st_size
            except OSError:
                sz = 0
            result.append(
                ArtifactSummary(
                    path=_rel(entry),
                    size_bytes=sz,
                    is_dir=False,
                    file_count=1,
                )
            )
    return result


# ---------------------------------------------------------------------------
# Tree builder — renders an ASCII directory tree
# ---------------------------------------------------------------------------


def _build_tree(
    path: Path,
    prefix: str = "",
    is_last: bool = True,
) -> list[str]:
    """Recursively build an ASCII tree view of *path*."""
    lines: list[str] = []
    if not path.exists():
        return lines

    name = path.name or str(path)
    connector = "└── " if is_last else "├── "
    lines.append(f"{prefix}{connector}{name}/" if path.is_dir() else f"{prefix}{connector}{name}")

    if path.is_dir():
        entries = sorted(
            [e for e in path.iterdir() if not e.name.startswith(".")],
            key=lambda e: (not e.is_dir(), e.name),
        )
        for i, entry in enumerate(entries):
            child_prefix = prefix + ("    " if is_last else "│   ")
            child_is_last = i == len(entries) - 1
            lines.extend(_build_tree(entry, child_prefix, child_is_last))
    return lines


def _build_tree_children(path: Path, prefix: str = "") -> list[str]:
    """Build tree lines for the *children* of *path* (not the path itself).

    Uses ``_render_subtree`` to recursively walk the full tree without
    duplicating the root node.
    """
    lines: list[str] = []
    if not path.exists() or not path.is_dir():
        return lines

    entries = sorted(
        [e for e in path.iterdir() if not e.name.startswith(".")],
        key=lambda e: (not e.is_dir(), e.name),
    )
    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        lines.extend(_render_subtree(entry, prefix, is_last))
    return lines


def _render_subtree(node: Path, prefix: str, is_last: bool) -> list[str]:
    """Render a single node and its descendants."""
    lines: list[str] = []
    connector = "└── " if is_last else "├── "
    child_prefix = prefix + ("    " if is_last else "│   ")

    if node.is_dir():
        lines.append(f"{prefix}{connector}{node.name}/")
        children = sorted(
            [e for e in node.iterdir() if not e.name.startswith(".")],
            key=lambda e: (not e.is_dir(), e.name),
        )
        for j, child in enumerate(children):
            child_last = j == len(children) - 1
            lines.extend(_render_subtree(child, child_prefix, child_last))
    else:
        lines.append(f"{prefix}{connector}{node.name}")
    return lines


# ---------------------------------------------------------------------------
# Size formatter
# ---------------------------------------------------------------------------


def _format_size(num_bytes: int) -> str:
    """Return a human-readable size string (e.g. ``"12.3 KB"``)."""
    if num_bytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    unit_idx = 0
    while size >= 1024 and unit_idx < len(units) - 1:
        size /= 1024
        unit_idx += 1
    if unit_idx == 0:
        return f"{int(size)} B"
    return f"{size:.1f} {units[unit_idx]}"


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------


def update_output_manifest(
    course_name: str = "",
    *,
    syllabus_path: Path | None = None,
    labs_base_path: Path | None = None,
    rubrics_path: Path | None = None,
    course_graph: CourseGraph | None = None,
) -> Path:
    """Scan the ``output/`` directory and write ``output/README.md``.

    If *course_graph* is provided, ``output/course_graph.json`` is also
    written alongside the README manifest.

    Always recomputes from disk; writes with ``force=True``.
    """
    output_dir = _PROJECT_ROOT / "output"

    # Scan the entire output directory recursively to pick up all run
    # subdirectories (each run is scoped under output/<run_id>/).
    syllabus_entries: list[ArtifactSummary] = []
    labs_entries: list[ArtifactSummary] = []
    rubrics_entries: list[ArtifactSummary] = []
    lesson_plan_entries: list[ArtifactSummary] = []
    presentation_entries: list[ArtifactSummary] = []

    if output_dir.exists():
        for run_dir in sorted(output_dir.iterdir()):
            if not run_dir.is_dir() or run_dir.name.startswith("."):
                continue
            if run_dir.name == "README.md":
                continue
            syl_dir = run_dir / "syllabus"
            if syl_dir.exists():
                syllabus_entries.extend(_scan_directory(syl_dir))
            lab_dir = run_dir / "labs"
            if lab_dir.exists():
                labs_entries.extend(_scan_directory(lab_dir))
            rub_dir = run_dir / "rubrics"
            if rub_dir.exists():
                rubrics_entries.extend(_scan_directory(rub_dir))
            lp_dir = run_dir / "lesson_plans"
            if lp_dir.exists():
                lesson_plan_entries.extend(_scan_directory(lp_dir))
            pres_dir = run_dir / "presentations"
            if pres_dir.exists():
                presentation_entries.extend(_scan_directory(pres_dir))

    all_entries = (
        syllabus_entries + labs_entries + rubrics_entries
        + lesson_plan_entries + presentation_entries
    )
    total_files = sum(e.file_count for e in all_entries)
    total_bytes = sum(e.size_bytes for e in all_entries)

    ts = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: list[str] = []
    lines.append("# 📦  Syllabus Swarm — Output Manifest\n")
    lines.append(f"> Auto-generated on **{ts}**\n")
    if course_name:
        safe = _sanitize_filename(course_name)
        lines.append(f"> Latest course: **{course_name}**  \n")
        lines.append(f"> Course slug:  `{safe}`\n")

    lines.append("\n---\n\n")
    lines.append("## 📊  Summary\n\n")
    lines.append("| Category       | Count | Total Size |\n")
    lines.append("|----------------|-------|------------|\n")

    syl_bytes = sum(e.size_bytes for e in syllabus_entries)
    lab_bytes = sum(e.size_bytes for e in labs_entries)
    rub_bytes = sum(e.size_bytes for e in rubrics_entries)
    lp_bytes = sum(e.size_bytes for e in lesson_plan_entries)
    pres_bytes = sum(e.size_bytes for e in presentation_entries)

    lines.append(
        f"| Syllabi        | {len(syllabus_entries):>5} | {_format_size(syl_bytes):>10} |\n"
    )
    lines.append(f"| Lab courses    | {len(labs_entries):>5} | {_format_size(lab_bytes):>10} |\n")
    lines.append(
        f"| Rubrics        | {len(rubrics_entries):>5} | {_format_size(rub_bytes):>10} |\n"
    )
    lines.append(
        f"| Lesson Plans   | {len(lesson_plan_entries):>5} | {_format_size(lp_bytes):>10} |\n"
    )
    lines.append(
        f"| Presentations  | {len(presentation_entries):>5} | {_format_size(pres_bytes):>10} |\n"
    )
    lines.append(
        f"| **Total**      | **{total_files:>3}** | **{_format_size(total_bytes):>8}** |\n"
    )
    lines.append("\n")

    if syllabus_entries:
        lines.append("## 📄  Syllabi\n\n")
        for entry in syllabus_entries:
            lines.append(f"- [`{entry.path}`]({entry.path}) ({_format_size(entry.size_bytes)})\n")
        lines.append("\n")

    if labs_entries:
        lines.append("## 🧪  Lab Directories\n\n")
        for entry in labs_entries:
            if entry.is_dir:
                lines.append(
                    f"- 📁 **{entry.path}/** — "
                    f"{entry.file_count} file(s), "
                    f"{_format_size(entry.size_bytes)}\n"
                )
                _append_tier_breakdown(lines, _PROJECT_ROOT / entry.path)
            else:
                lines.append(
                    f"- [`{entry.path}`]({entry.path}) ({_format_size(entry.size_bytes)})\n"
                )
        lines.append("\n")

    if rubrics_entries:
        lines.append("## 📋  Rubrics\n\n")
        for entry in rubrics_entries:
            lines.append(f"- [`{entry.path}`]({entry.path}) ({_format_size(entry.size_bytes)})\n")
        lines.append("\n")

    if lesson_plan_entries:
        lines.append("## 📝  Lesson Plans\n\n")
        for entry in lesson_plan_entries:
            lines.append(f"- [`{entry.path}`]({entry.path}) ({_format_size(entry.size_bytes)})\n")
        lines.append("\n")

    if presentation_entries:
        lines.append("## 🎬  Presentations\n\n")
        for entry in presentation_entries:
            lines.append(f"- [`{entry.path}`]({entry.path}) ({_format_size(entry.size_bytes)})\n")
        lines.append("\n")

    if output_dir.exists():
        lines.append("## 🌲  Output Directory Tree\n\n")
        lines.append("```\n")
        # Build tree of immediate children to avoid duplicating "output/"
        tree_lines = _build_tree_children(output_dir, prefix="")
        lines.append("output/\n")
        for line in tree_lines:
            lines.append(f"{line}\n")
        lines.append("```\n\n")

    lines.append("---\n\n")
    lines.append(
        "*This manifest is auto-generated by the Syllabus Swarm "
        "Output Exporter.  It is updated on every pipeline run.*\n"
    )
    lines.append("<!-- generated by src/exporters/manifest.py -->\n")

    manifest_content = "".join(lines)
    manifest_path = output_dir / "README.md"

    # ── Optionally write course_graph.json alongside README ───────────
    if course_graph is not None:
        graph_path = output_dir / "course_graph.json"
        write_file(graph_path, course_graph.model_dump_json(indent=2), force=True)

    return write_file(manifest_path, manifest_content, force=True)


def _append_tier_breakdown(lines: list[str], dir_path: Path) -> None:
    """Append per-tier file counts and sizes for a lab course directory."""
    if not dir_path.exists():
        return
    for tier in sorted(dir_path.iterdir()):
        if not tier.is_dir() or tier.name.startswith("."):
            continue
        tier_files = sum(1 for _ in tier.rglob("*") if _.is_file() and not _.name.startswith("."))
        tier_size = 0
        for f in tier.rglob("*"):
            if f.is_file() and not f.name.startswith("."):
                try:
                    tier_size += f.stat().st_size
                except OSError:
                    pass
        lines.append(f"  └─ `{tier.name}/` — {tier_files} file(s), {_format_size(tier_size)}\n")
    lines.append("\n")
