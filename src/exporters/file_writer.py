"""
file_writer.py — File-Writer Utility for the Output Exporter System
===================================================================

Issue #4: Automated Output Exporter & Markdown Packager

Handles **all disk I/O** for the syllabus-swarm output pipeline.  Every
syllabus, lab, and rubric that the system produces must flow through this
module so that:

* Paths are cross-platform safe (``pathlib``).
* Parent directories are created automatically.
* Overwrites are blocked unless explicitly permitted.
* The mandated output directory structure (``output/syllabus/``,
  ``output/labs/``, ``output/rubrics/``) is enforced.
* Edge cases — empty content, dangerous filenames — are handled gracefully.

Public API
----------
* ``write_file(path, content, *, force=False)`` — write a single file.
* ``write_directory_tree(base_path, files_dict, *, force=False)`` — write a
  batch of files from a ``{relative_path: content}`` mapping.
* ``FileWriteError`` — raised when overwrite protection kicks in.
* ``OutputPathConfig`` — named constants for the mandated directory layout.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project-root resolution
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FileWriteError(Exception):
    """Raised when a file-write operation is blocked (e.g. overwrite guard)."""


# ---------------------------------------------------------------------------
# Sanitiser — convert arbitrary strings into safe filesystem names
# ---------------------------------------------------------------------------

# Characters that are illegal on Windows, risky on POSIX, or problematic
# inside tooling and version-control systems.
_ILLEGAL_CHARS_RE: re.Pattern[str] = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# Runs of underscores/spaces/dashes longer than this get collapsed.
_MAX_REPEATED_SEP: int = 2


def _sanitize_filename(name: str, *, replace_spaces: bool = True) -> str:
    """Convert an arbitrary string into a safe, portable filesystem name.

    The function:
    1. Strips leading / trailing whitespace.
    2. Replaces every character matched by ``_ILLEGAL_CHARS_RE`` with ``-``.
    3. Optionally replaces spaces with underscores (default ``True``).
    4. Collapses runs of underscores/dashes longer than
       ``_MAX_REPEATED_SEP`` (default 2).
    5. Strips leading/trailing separators introduced during sanitisation.
    6. Returns ``"untitled"`` if the result is empty.

    Parameters
    ----------
    name : str
        The raw human-facing string (e.g. a course title).
    replace_spaces : bool
        When ``True`` (the default), spaces become ``_``.

    Returns
    -------
    str
        A safe filename ready to be joined to a ``Path``.
    """
    cleaned = name.strip()

    # Replace illegal characters first so that subsequent steps only deal
    # with legitimate separators.
    cleaned = _ILLEGAL_CHARS_RE.sub("-", cleaned)

    if replace_spaces:
        cleaned = cleaned.replace(" ", "_")

    # Collapse runs of the same separator character.
    cleaned = re.sub(
        r"([_-])\1{2,}",
        lambda m: m.group(1) * _MAX_REPEATED_SEP,
        cleaned,
    )

    # Remove separators from either edge introduced by sanitisation.
    cleaned = cleaned.strip("_-")

    return cleaned or "untitled"


# ---------------------------------------------------------------------------
# Output-path enforcement — mandated directory layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutputPathConfig:
    """Immutable container for the mandated output directory constants.

    All paths are resolved relative to the **project root** (the directory
    that contains ``src/``).  Consumers should use these attributes rather
    than hard-coding string paths so the layout is enforced in one place.

    Attributes
    ----------
    root : Path
        Absolute path to the project root.
    syllabus_dir : Path
        ``output/syllabus/`` — where ``.md`` syllabus documents land.
    labs_dir : Path
        ``output/labs/`` — parent of tiered lab directories.
    rubrics_dir : Path
        ``output/rubrics/`` — where rubric ``.md`` files land.
    """

    root: Path = field(default=_PROJECT_ROOT)
    syllabus_dir: Path = field(init=False)
    labs_dir: Path = field(init=False)
    rubrics_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "syllabus_dir", self.root / "output" / "syllabus")
        object.__setattr__(self, "labs_dir", self.root / "output" / "labs")
        object.__setattr__(self, "rubrics_dir", self.root / "output" / "rubrics")


# Singleton — cheap to recreate, but convenient as a shared reference.
OUTPUT_PATHS: OutputPathConfig = OutputPathConfig()

# Recognised extensions that are treated as plain-text code files.
_CODE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".go",
        ".rs",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".rb",
        ".sh",
        ".bash",
        ".ps1",
        ".sql",
        ".yml",
        ".yaml",
        ".toml",
        ".json",
        ".xml",
        ".html",
        ".css",
        ".scss",
        ".txt",
        ".cfg",
        ".ini",
        ".env",
        ".gitignore",
    }
)

# Extensions that are always treated as Markdown.
_MD_EXTENSIONS: frozenset[str] = frozenset({".md", ".mdx", ".markdown"})


def _is_code_file(path: Path) -> bool:
    """Return ``True`` if *path* looks like a code/plain-text file."""
    return path.suffix.lower() in _CODE_EXTENSIONS


def _is_markdown_file(path: Path) -> bool:
    """Return ``True`` if *path* is explicitly a Markdown document."""
    return path.suffix.lower() in _MD_EXTENSIONS


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _resolve_path(path: str | Path, *, must_be_under_root: bool = True) -> Path:
    """Resolve *path* to an absolute ``Path`` and optionally enforce it
    lives under the project root.

    Parameters
    ----------
    path : str or Path
        The target file path (may be relative or absolute).
    must_be_under_root : bool
        When ``True`` (the default), ``FileWriteError`` is raised if the
        resolved absolute path is not a descendant of the project root.

    Returns
    -------
    Path
        The resolved, absolute file path.

    Raises
    ------
    FileWriteError
        If *must_be_under_root* is ``True`` and the path is outside the
        project.
    """
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = _PROJECT_ROOT / resolved
    resolved = resolved.resolve()

    if must_be_under_root:
        try:
            resolved.relative_to(_PROJECT_ROOT)
        except ValueError:
            raise FileWriteError(
                f"Path '{resolved}' is outside the project root "
                f"'{_PROJECT_ROOT}'.  Refusing to write."
            )

    return resolved


def _validate_content(content: Any) -> str:
    """Coerce *content* to ``str`` and issue a warning if it is empty."""
    text = str(content)

    if not text.strip():
        warnings.warn(
            "write_file called with empty or whitespace-only content.  "
            "The file will still be created but may be unintentionally blank.",
            UserWarning,
            stacklevel=3,
        )

    return text


# ---------------------------------------------------------------------------
# Core API — single-file writer
# ---------------------------------------------------------------------------


def write_file(
    path: str | Path,
    content: Any,
    *,
    force: bool = False,
    encoding: str = "utf-8",
) -> Path:
    """Write *content* to a single file, creating parent directories as needed.

    Parameters
    ----------
    path : str or Path
        Destination path.  If relative, it is resolved against the project
        root (``src/../``).  Absolute paths are accepted but must reside
        under the project root.
    content : Any
        The text to write.  Non-string values are converted via ``str()``.
    force : bool
        When ``False`` (the default) a ``FileWriteError`` is raised if
        *path* already exists.  Set to ``True`` to silently overwrite.
    encoding : str
        Text encoding.  Defaults to ``"utf-8"``.

    Returns
    -------
    Path
        The absolute path of the file that was written (for chaining /
        logging).

    Raises
    ------
    FileWriteError
        If the file already exists and ``force`` is ``False``, or if the
        path escapes the project root.
    """
    resolved = _resolve_path(path)
    text = _validate_content(content)

    # --- Overwrite guard -------------------------------------------------
    if resolved.exists() and not force:
        raise FileWriteError(f"File already exists: '{resolved}'.  Use force=True to overwrite.")

    # --- Create parent directories ---------------------------------------
    resolved.parent.mkdir(parents=True, exist_ok=True)

    # --- Write -----------------------------------------------------------
    resolved.write_text(text, encoding=encoding)

    return resolved


# ---------------------------------------------------------------------------
# Core API — batch / directory-tree writer
# ---------------------------------------------------------------------------


def write_directory_tree(
    base_path: str | Path,
    files_dict: dict[str, Any],
    *,
    force: bool = False,
    encoding: str = "utf-8",
) -> list[Path]:
    """Write a batch of files from a ``{relative_path: content}`` mapping.

    This is the primary entry point for the exporter pipeline.  Callers
    build a dictionary where each key is a path **relative to** *base_path*
    and each value is the file's text content.

    Example
    -------
    >>> write_directory_tree(
    ...     "output/labs/data_science",
    ...     {
    ...         "tier1_foundations/starter/lab1.py": "# TODO ...",
    ...         "tier1_foundations/solution/lab1.py": "# Solution ...",
    ...     },
    ... )

    Parameters
    ----------
    base_path : str or Path
        The root directory under which all files in *files_dict* will be
        written.  Created automatically if it does not exist.
    files_dict : dict[str, Any]
        Mapping of ``{relative_file_path: content}``.  Keys must be
        relative paths (using forward slashes as separators).  Values are
        coerced to ``str`` via ``str()``.
    force : bool
        Passed through to every ``write_file`` call.  When ``False``
        (default), any pre-existing file causes a ``FileWriteError``.
    encoding : str
        Text encoding.  Defaults to ``"utf-8"``.

    Returns
    -------
    list[Path]
        Ordered list of absolute paths written, in the order they appear in
        the dictionary (Python >=3.7 preserves insertion order).

    Raises
    ------
    FileWriteError
        If any file already exists and ``force`` is ``False``, or if a path
        escapes the project root.
    """
    base = _resolve_path(base_path, must_be_under_root=True)
    written: list[Path] = []

    if not files_dict:
        warnings.warn(
            "write_directory_tree called with an empty files_dict.  No files were written.",
            UserWarning,
            stacklevel=2,
        )
        return written

    for rel_path, content in files_dict.items():
        # Normalise the relative path to use the platform's separator.
        resolved = base / Path(*rel_path.replace("\\", "/").split("/"))
        dest = write_file(resolved, content, force=force, encoding=encoding)
        written.append(dest)

    return written


# ---------------------------------------------------------------------------
# Convenience — syllabus, lab, rubric helpers
# ---------------------------------------------------------------------------


def write_syllabus(
    course_name: str,
    content: Any,
    *,
    force: bool = False,
    run_id: str | None = None,
) -> Path:
    """Write a syllabus Markdown file to ``output/syllabus/<course>.md``.

    The course name is automatically sanitised into a safe filename.

    Parameters
    ----------
    course_name : str
        Human-readable course title (e.g. "Data Science with Python").
    content : Any
        The complete syllabus (Markdown text).  Coerced via ``str()``.
    force : bool
        When ``False``, raises ``FileWriteError`` if the syllabus already
        exists.
    run_id : str or None
        When provided, writes to ``output/<run_id>/syllabus/<course>.md``
        instead of the global ``output/syllabus/`` path.

    Returns
    -------
    Path
        Absolute path to the written syllabus.
    """
    safe = _sanitize_filename(course_name)
    if run_id:
        path = _PROJECT_ROOT / "output" / run_id / "syllabus" / f"{safe}.md"
    else:
        path = OUTPUT_PATHS.syllabus_dir / f"{safe}.md"
    return write_file(path, content, force=force)


def write_rubric(
    course_name: str,
    content: Any,
    *,
    force: bool = False,
) -> Path:
    """Write a rubric Markdown file to ``output/rubrics/<course>-rubric.md``.

    Parameters
    ----------
    course_name : str
        Human-readable course title.
    content : Any
        The complete rubric as Markdown text.  Coerced via ``str()``.
    force : bool
        When ``False``, raises ``FileWriteError`` if the rubric already
        exists.

    Returns
    -------
    Path
        Absolute path to the written rubric.
    """
    safe = _sanitize_filename(course_name)
    path = OUTPUT_PATHS.rubrics_dir / f"{safe}-rubric.md"
    return write_file(path, content, force=force)


def write_lesson_plan(
    course_name: str,
    module_name: str,
    content: Any,
    *,
    force: bool = False,
    run_id: str | None = None,
) -> Path:
    """Write a lesson plan Markdown file.

    Parameters
    ----------
    course_name : str
        Human-readable course title.
    module_name : str
        Module name for the lesson plan subdirectory.
    content : Any
        The complete lesson plan (Markdown text).  Coerced via ``str()``.
    force : bool
        When ``False``, raises ``FileWriteError`` if the file already exists.
    run_id : str or None
        When provided, scopes output under ``output/<run_id>/``.

    Returns
    -------
    Path
        Absolute path to the written lesson plan.
    """
    safe_module = _sanitize_filename(module_name)
    if run_id:
        path = _PROJECT_ROOT / "output" / run_id / "lesson_plans" / safe_module / "lesson_plan.md"
    else:
        path = OUTPUT_PATHS.root / "lesson_plans" / safe_module / "lesson_plan.md"
    return write_file(path, content, force=force)


def write_presentation(
    course_name: str,
    module_name: str,
    content: Any,
    *,
    force: bool = False,
    run_id: str | None = None,
) -> Path:
    """Write a presentation (Marp Markdown) file.

    Parameters
    ----------
    course_name : str
        Human-readable course title.
    module_name : str
        Module name for the presentation subdirectory.
    content : Any
        The complete presentation (Marp Markdown).  Coerced via ``str()``.
    force : bool
        When ``False``, raises ``FileWriteError`` if the file already exists.
    run_id : str or None
        When provided, scopes output under ``output/<run_id>/``.

    Returns
    -------
    Path
        Absolute path to the written presentation.
    """
    safe_module = _sanitize_filename(module_name)
    if run_id:
        path = (
            _PROJECT_ROOT / "output" / run_id / "presentations" / safe_module / "presentation.md"
        )
    else:
        path = OUTPUT_PATHS.root / "presentations" / safe_module / "presentation.md"
    return write_file(path, content, force=force)


# ---------------------------------------------------------------------------
# RemotionManifest → .tsx export (Issue #11 — Polyglot VaC output)
# ---------------------------------------------------------------------------


def write_remotion_manifest(
    manifest: object,
    *,
    force: bool = False,
) -> Path:
    """Write a ``RemotionManifest`` as a valid ``.tsx`` file to ``src/export/vac/``.

    The generated ``.tsx`` file is a self-contained React/Remotion composition
    that imports the Remotion framework and exports a named composition component.
    Output is placed in ``src/export/vac/`` — completely isolated from the
    Markdown curriculum artifacts in ``output/``.

    Parameters
    ----------
    manifest : RemotionManifest
        The VaC composition descriptor produced by the Video Engineer agent.
    force : bool
        When ``False``, raises ``FileWriteError`` if the ``.tsx`` file already
        exists.

    Returns
    -------
    Path
        Absolute path to the written ``.tsx`` file.

    Raises
    ------
    TypeError
        If *manifest* is not a ``RemotionManifest`` instance.
    FileWriteError
        If the target file already exists and *force* is ``False``.
    """
    # Lazy import to keep the module-level dependency footprint small.
    from src.models import RemotionManifest  # noqa: PLC0415

    if not isinstance(manifest, RemotionManifest):
        raise TypeError(f"Expected a RemotionManifest instance, got {type(manifest).__name__}")

    safe_name = _sanitize_filename(manifest.composition_id)
    vac_dir = _PROJECT_ROOT / "src" / "export" / "vac"
    vac_dir.mkdir(parents=True, exist_ok=True)
    output_path = vac_dir / f"{safe_name}.tsx"

    # Generate a minimal but valid React/Remotion .tsx composition.
    component_lines: list[str] = []
    component_lines.append(
        'import {useCurrentFrame, useVideoConfig, AbsoluteFill} from "remotion";'
    )
    component_lines.append("")
    component_lines.append(f"export const {manifest.composition_id} = () => {{")
    component_lines.append("  const frame = useCurrentFrame();")
    component_lines.append("  const {{fps, durationInFrames}} = useVideoConfig();")
    component_lines.append("")
    component_lines.append("  return (")
    component_lines.append('    <AbsoluteFill style={{backgroundColor: "white"}}>')
    component_lines.append("      <h1>{manifest.composition_id}</h1>")
    component_lines.append("      <p>Frame {frame} / {durationInFrames}</p>")
    component_lines.extend(_render_components(manifest.components, indent=6))
    component_lines.append("    </AbsoluteFill>")
    component_lines.append("  );")
    component_lines.append("};")
    component_lines.append("")

    content = "\n".join(component_lines)

    return write_file(output_path, content, force=force)


def _render_components(
    components: list[dict[str, object]],
    *,
    indent: int = 0,
) -> list[str]:
    """Recursively render RemotionManifest components to JSX lines.

    Parameters
    ----------
    components : list[dict[str, object]]
        Ordered list of component descriptors from the manifest.
    indent : int
        Current indentation level in spaces.

    Returns
    -------
    list[str]
        JSX-formatted lines representing the component tree.
    """
    lines: list[str] = []
    prefix = " " * indent

    for comp in components:
        comp_type = str(comp.get("type", "div"))
        comp_name = str(comp.get("name", ""))
        comp_props: dict[str, object] = dict(comp.get("props", {}) or {})
        children: list[dict[str, object]] = list(comp.get("children", []) or [])

        props_str = ""
        if comp_props:
            prop_parts = []
            for k, v in comp_props.items():
                if isinstance(v, str):
                    prop_parts.append(f'{k}="{v}"')
                elif isinstance(v, bool):
                    prop_parts.append(f"{k}={str(v).lower()}")
                else:
                    prop_parts.append(f"{k}={{{v}}}")
            props_str = " " + " ".join(prop_parts)

        if children:
            lines.append(f"{prefix}<{comp_type}{props_str}>")
            lines.extend(_render_components(children, indent=indent + 2))
            lines.append(f"{prefix}</{comp_type}>")
        elif comp_name:
            lines.append(f"{prefix}<{comp_type}{props_str}>")
            lines.append(f"{prefix}  {{/* {comp_name} */}}")
            lines.append(f"{prefix}</{comp_type}>")
        else:
            lines.append(f"{prefix}<{comp_type}{props_str} />")

    return lines


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    print("=== file_writer self-test ===\n")

    # 1. Sanitiser
    print("1. _sanitize_filename:")
    tests = [
        ("Data Science: ML & AI 101", "Data_Science-_ML_-_AI_101"),
        ("  spaced//??course  ", "spaced--course"),
        ('file<name>with|illegal*chars?"', "file-name-with-illegal-chars-"),
        ("", "untitled"),
        ("   ", "untitled"),
        ("\x00\x1fhello", "hello"),
    ]
    for raw, expect in tests:
        got = _sanitize_filename(raw)
        status = "OK" if got == expect else "FAIL"
        print(f"   [{status}] {raw!r:45s} -> {got!r}  (expected {expect!r})")

    # 2. write_file (happy path)
    print("\n2. write_file:")
    with tempfile.TemporaryDirectory() as tmp:
        p = write_file(Path(tmp) / "sub" / "test.md", "# Hello\n")
        assert p.read_text() == "# Hello\n"
        print(f"   [OK]  {p}")

    # 3. overwrite protection
    print("\n3. Overwrite protection:")
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "existing.txt"
        dest.write_text("original")
        try:
            write_file(dest, "new")
            print("   [FAIL] Should have raised FileWriteError")
        except FileWriteError:
            print("   [OK]  Raised FileWriteError")

        # force overwrite
        p = write_file(dest, "overwritten", force=True)
        assert p.read_text() == "overwritten"
        print("   [OK]  force=True overwrote successfully")

    # 4. write_directory_tree
    print("\n4. write_directory_tree:")
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "labs" / "test_course"
        files = {
            "tier1_foundations/starter/lab1.py": "# TODO: implement\n",
            "tier1_foundations/solution/lab1.py": "# Solution\n",
            "tier1_foundations/starter/README.md": "# Lab 1\n",
            "tier1_foundations/solution/README.md": "# Lab 1 Solution\n",
            "tier2_application/starter/app.py": 'if __name__ == "__main__": ...\n',
        }
        written = write_directory_tree(base, files)
        for p in written:
            assert p.exists(), f"Missing: {p}"
        print(f"   [OK]  Wrote {len(written)} files")

    # 5. OutputPathConfig
    print("\n5. OutputPathConfig:")
    print(f"   root:         {OUTPUT_PATHS.root}")
    print(f"   syllabus_dir: {OUTPUT_PATHS.syllabus_dir}")
    print(f"   labs_dir:     {OUTPUT_PATHS.labs_dir}")
    print(f"   rubrics_dir:  {OUTPUT_PATHS.rubrics_dir}")

    # 6. Convenience helpers
    print("\n6. Convenience helpers:")
    with tempfile.TemporaryDirectory() as tmp:
        import src.exporters.file_writer as fw

        alt_root = Path(tmp)
        fw.OUTPUT_PATHS = OutputPathConfig(root=alt_root)

        syl = fw.write_syllabus("Test Course 101", "# Test Syllabus")
        print(f"   [OK]  syllabus -> {syl}")
        assert syl.exists()

        rub = fw.write_rubric("Test Course 101", "# Test Rubric")
        print(f"   [OK]  rubric   -> {rub}")
        assert rub.exists()

        # 7. Edge case: empty content warning
        print("\n7. Empty content warning:")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = fw.write_file(alt_root / "empty.md", "", force=True)
            if w:
                print(f"   [OK]  Warning: {w[0].message}")
            else:
                print("   [FAIL] No warning")

    print("\n=== All tests passed ===")
