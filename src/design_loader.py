"""
design_loader.py — GLR Media Creative Design System Loader
==========================================================

Parses the YAML frontmatter of ``DESIGN.md`` into structured
:class:`DesignSystem` Pydantic models, then exposes format-specific
helpers that inject the GLR brand into generated materials.

Public API
----------
* ``load_design_system() -> DesignSystem`` — parse DESIGN.md tokens.
* ``get_cached_design_system() -> DesignSystem`` — lazy singleton.
* ``get_design_system_path() -> Path`` — resolve the DESIGN.md path.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from src.models import (
    DesignColorTokens,
    DesignSpacingTokens,
    DesignSystem,
    DesignTypographyScale,
    TypographyToken,
)

# ---------------------------------------------------------------------------
# Project-root resolution
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def get_design_system_path() -> Path:
    """Return the resolved path to DESIGN.md.

    Respects the ``DESIGN_SYSTEM_PATH`` environment variable; falls back
    to ``<project_root>/DESIGN.md``.
    """
    env_path = os.getenv("DESIGN_SYSTEM_PATH")
    if env_path:
        candidate = Path(env_path)
        if candidate.is_absolute():
            return candidate
        return _PROJECT_ROOT / candidate
    return _PROJECT_ROOT / "DESIGN.md"


# ---------------------------------------------------------------------------
# YAML frontmatter extraction
# ---------------------------------------------------------------------------

_FRONTMATTER_RE: re.Pattern[str] = re.compile(
    r"^---\s*\n(.*?)\n---", re.DOTALL
)


def _extract_frontmatter(raw: str) -> dict:
    """Extract and parse the YAML frontmatter block from a Markdown document."""
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        raise ValueError(
            "DESIGN.md does not contain valid YAML frontmatter "
            "(expected a leading ``---`` block)."
        )
    data = yaml.safe_load(m.group(1))
    if not isinstance(data, dict):
        raise ValueError("DESIGN.md frontmatter is not a YAML mapping.")
    return data


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_colors(raw_colors: dict) -> DesignColorTokens:
    """Build a ``DesignColorTokens`` from the raw YAML ``colors`` mapping."""
    mapping: dict[str, str] = {}
    for key, value in raw_colors.items():
        python_key = key.replace("-", "_")
        mapping[python_key] = str(value)
    return DesignColorTokens(**mapping)


def _parse_typography(raw_typography: dict) -> DesignTypographyScale:
    """Build a ``DesignTypographyScale`` from the raw YAML ``typography`` mapping."""
    tokens: dict[str, TypographyToken] = {}
    for key, value in raw_typography.items():
        python_key = key.replace("-", "_")
        tokens[python_key] = TypographyToken(
            fontFamily=str(value.get("fontFamily", "Space Grotesk")),
            fontSize=str(value.get("fontSize", "16px")),
            fontWeight=str(value.get("fontWeight", "400")),
            lineHeight=str(value.get("lineHeight", "24px")),
            letterSpacing=str(value.get("letterSpacing", "0em")),
        )
    return DesignTypographyScale(**tokens)


def _parse_spacing(raw_spacing: dict) -> DesignSpacingTokens:
    """Build DesignSpacingTokens from raw YAML spacing mapping."""
    return DesignSpacingTokens(
        gutter=str(raw_spacing.get("gutter", "1.5rem")),
        gutter_mobile=str(raw_spacing.get("gutter-mobile", "1rem")),
        margin=str(raw_spacing.get("margin", "3rem")),
        margin_mobile=str(raw_spacing.get("margin-mobile", "1.25rem")),
        space_xs=str(raw_spacing.get("space-xs", "0.25rem")),
        space_sm=str(raw_spacing.get("space-sm", "0.5rem")),
        space_md=str(raw_spacing.get("space-md", "1rem")),
        space_lg=str(raw_spacing.get("space-lg", "1.5rem")),
        space_xl=str(raw_spacing.get("space-xl", "2.5rem")),
    )


def _extract_narrative(raw: str) -> tuple[str, str]:
    """Return ``(brand_statement, design_movement)`` from the narrative body."""
    body = _FRONTMATTER_RE.sub("", raw, count=1).strip()

    movement = ""
    movement_match = re.search(
        r"The design movement is\s+\*\*(.*?)\*\*",
        body,
    )
    if movement_match:
        movement = movement_match.group(1).strip()

    brand_match = re.search(
        r"## Brand & Style\s*\n+(.*?)(?:\n##|\Z)",
        body,
        re.DOTALL,
    )
    brand = ""
    if brand_match:
        brand = brand_match.group(1).strip()
        brand = re.sub(
            r"\nThe design movement is\s+\*\*.*?\*\*\s*fused.*?containment\.",
            "",
            brand,
        )
        brand = brand.strip()

    return brand, movement


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_design_system(*, path: str | Path | None = None) -> DesignSystem:
    """Parse DESIGN.md and return a fully-populated ``DesignSystem``.

    Parameters
    ----------
    path : str or Path or None
        Explicit path to DESIGN.md.  When ``None``, resolves via
        ``get_design_system_path()``.

    Returns
    -------
    DesignSystem
        All design tokens ready for consumption by agents and tasks.
    """
    resolved = Path(path) if path else get_design_system_path()
    if not resolved.exists():
        raise FileNotFoundError(f"Design system file not found: {resolved}")

    raw = resolved.read_text(encoding="utf-8")
    frontmatter = _extract_frontmatter(raw)

    name = str(frontmatter.get("name", "GLR Media Creative"))
    colors = _parse_colors(frontmatter.get("colors", {}))
    typography = _parse_typography(frontmatter.get("typography", {}))
    spacing = _parse_spacing(frontmatter.get("spacing", {}))
    brand, movement = _extract_narrative(raw)

    return DesignSystem(
        name=name,
        brand_statement=brand,
        design_movement=movement,
        colors=colors,
        typography=typography,
        spacing=spacing,
    )


# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_design_system_cache: DesignSystem | None = None


def get_cached_design_system() -> DesignSystem:
    """Return a lazily-loaded, cached ``DesignSystem`` singleton."""
    global _design_system_cache
    if _design_system_cache is None:
        _design_system_cache = load_design_system()
    return _design_system_cache


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ds = load_design_system()
    print("✅ Design system loaded successfully.\n")
    print(f"   Name:            {ds.name}")
    print(f"   Movement:        {ds.design_movement}")
    print(f"   Primary:         {ds.colors.primary}")
    print(f"   Electric Lime:   {ds.colors.electric_lime}")
    print(f"   Pure Black:      {ds.colors.pure_black}")
    print(f"   Pure White:      {ds.colors.pure_white}")
    print(f"   Accent Blue:     {ds.colors.accent_ultramarine}")
    print(f"   Headline Font:   {ds.typography.headline_md.fontFamily}")
    print(f"   Body Font:       {ds.typography.body_md.fontFamily}")
    print(f"   Border Radius:   {ds.shapes_border_radius}")
    print(f"   Depth Style:     {ds.depth_style}")