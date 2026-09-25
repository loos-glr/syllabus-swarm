"""
glr_marp_theme.py — GLR Media Creative Marp CSS Theme
======================================================

Generates a complete Marp-compatible CSS ``style: |`` block from the
GLR design tokens loaded via ``src.design_loader``.

Public API
----------
* ``generate_marp_css(design=None) -> str`` — produce a complete ``style: |``
  CSS block for Marp frontmatter.
* ``get_marp_frontmatter(design=None) -> dict`` — return Marp frontmatter
  dict with theme config.
"""

from __future__ import annotations

from src.design_loader import get_cached_design_system
from src.models import DesignSystem


def generate_marp_css(design: DesignSystem | None = None) -> str:
    """Generate a complete Marp ``style: |`` CSS block from design tokens."""
    if design is None:
        design = get_cached_design_system()

    c = design.colors
    t = design.typography
    s = design.spacing

    primary = c.primary_container  # #76b800 — signature GLR green
    electric_lime = c.electric_lime  # #A6E22E
    black = c.pure_black  # #000000
    white = c.pure_white  # #FFFFFF
    ultra = c.accent_ultramarine  # #002BFF
    surface_subtle = c.surface_subtle  # #F4F4F6
    border_color = c.border_structural  # #E5E7EB
    background = c.background  # #faf9fd

    heading_font = t.headline_md.fontFamily  # Space Grotesk
    body_font = t.body_md.fontFamily  # Hanken Grotesk

    css = f"""  /* === GLR Media Creative — High-Contrast Brutalist Modernism === */
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Hanken+Grotesk:wght@400;500;600;700&display=swap');

  /* ---- Base ---- */
  section {{
    font-family: '{body_font}', -apple-system, sans-serif;
    font-size: {t.body_md.fontSize};
    line-height: {t.body_md.lineHeight};
    color: {black};
    background-color: {background};
  }}

  /* ---- Headings (Space Grotesk) ---- */
  h1 {{
    font-family: '{heading_font}', sans-serif;
    font-size: {t.headline_lg.fontSize};
    font-weight: {t.headline_lg.fontWeight};
    line-height: {t.headline_lg.lineHeight};
    letter-spacing: {t.headline_lg.letterSpacing};
    color: {black};
    margin-bottom: {s.space_md};
  }}
  h2 {{
    font-family: '{heading_font}', sans-serif;
    font-size: {t.headline_md.fontSize};
    font-weight: {t.headline_md.fontWeight};
    line-height: {t.headline_md.lineHeight};
    letter-spacing: {t.headline_md.letterSpacing};
    color: {black};
  }}
  h3 {{
    font-family: '{heading_font}', sans-serif;
    font-size: {t.headline_sm.fontSize};
    font-weight: {t.headline_sm.fontWeight};
    line-height: {t.headline_sm.lineHeight};
    letter-spacing: {t.headline_sm.letterSpacing};
    color: {black};
  }}
  h4, h5, h6 {{
    font-family: '{heading_font}', sans-serif;
    font-size: {t.title_md.fontSize};
    font-weight: {t.title_md.fontWeight};
    line-height: {t.title_md.lineHeight};
  }}

  /* ---- Code ---- */
  code {{
    font-family: '{heading_font}', monospace;
    background: {black};
    color: {electric_lime};
    padding: 2px 6px;
    border-radius: 0;
    font-size: {t.label_code.fontSize};
    letter-spacing: {t.label_code.letterSpacing};
  }}
  pre {{
    background: {black};
    color: {electric_lime};
    border-radius: 0;
    padding: {s.space_md};
    border: 1px solid {border_color};
  }}
  pre code {{
    background: transparent;
    color: inherit;
  }}

  /* ---- Lists ---- */
  ul, ol {{
    padding-left: {s.space_lg};
  }}
  li {{
    font-family: '{body_font}', sans-serif;
    line-height: {t.body_md.lineHeight};
    margin-bottom: {s.space_xs};
  }}

  /* ---- Tables ---- */
  table {{
    border-collapse: collapse;
    width: 100%;
    border: 1px solid {black};
  }}
  th {{
    font-family: '{heading_font}', sans-serif;
    font-weight: 600;
    font-size: {t.label_md.fontSize};
    letter-spacing: {t.label_md.letterSpacing};
    text-transform: uppercase;
    background: {black};
    color: {electric_lime};
    padding: {s.space_sm} {s.space_md};
    border: 1px solid {black};
  }}
  td {{
    padding: {s.space_sm} {s.space_md};
    border: 1px solid {border_color};
  }}

  /* ---- Blockquotes ---- */
  blockquote {{
    border-left: 4px solid {primary};
    margin-left: 0;
    padding: {s.space_sm} {s.space_md};
    background: {surface_subtle};
  }}"""

# ── GLR Component Styles ──────────────────────────────────────────
    css += f"""

  /* ---- GLR Components ---- */

  /* Primary Button */
  .glr-btn-primary {{
    display: inline-block;
    background: {primary};
    color: {black};
    font-family: '{heading_font}', sans-serif;
    font-size: {t.label_lg.fontSize};
    font-weight: 600;
    letter-spacing: {t.label_lg.letterSpacing};
    text-transform: uppercase;
    padding: {s.space_sm} {s.space_md};
    border: 1px solid {black};
    border-radius: 0;
    text-decoration: none;
  }}

  /* Card */
  .glr-card {{
    background: {white};
    border: 1px solid {black};
    border-radius: 0;
    padding: {s.space_lg};
  }}

  /* Dark Spotlight Card */
  .glr-card-dark {{
    background: {black};
    color: {white};
    border: 1px solid {primary};
    border-radius: 0;
    padding: {s.space_lg};
  }}
  .glr-card-dark h2, .glr-card-dark h3 {{
    color: {electric_lime};
  }}

  /* Badge / Chip */
  .glr-badge {{
    display: inline-block;
    background: {black};
    color: {electric_lime};
    font-family: '{heading_font}', sans-serif;
    font-size: {t.label_code.fontSize};
    font-weight: 700;
    letter-spacing: {t.label_code.letterSpacing};
    text-transform: uppercase;
    padding: {s.space_xs} {s.space_sm};
    border-radius: 0;
  }}
  .glr-badge-outline {{
    background: transparent;
    color: {black};
    border: 1px solid {border_color};
  }}
  .glr-badge-active {{
    background: {primary};
    color: {black};
  }}

  /* Columns */
  .columns {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: {s.space_lg};
  }}
  .columns-3 {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: {s.space_md};
  }}

  /* Marquee Ticker */
  .glr-marquee {{
    background: {primary};
    color: {black};
    font-family: '{heading_font}', sans-serif;
    font-size: {t.label_lg.fontSize};
    font-weight: 600;
    letter-spacing: {t.label_lg.letterSpacing};
    text-transform: uppercase;
    padding: {s.space_sm} {s.space_md};
  }}

  /* Media Frame */
  .glr-media-frame {{
    position: relative;
    border: 1px solid {black};
    border-radius: 0;
  }}
  .glr-media-frame::after {{
    content: 'GLR // 16:9';
    position: absolute;
    bottom: 4px;
    right: 4px;
    font-family: '{heading_font}', sans-serif;
    font-size: {t.label_code.fontSize};
    letter-spacing: {t.label_code.letterSpacing};
    color: {electric_lime};
    background: {black};
    padding: 2px 4px;
  }}

  /* Humanics Tags — mapped to GLR palette */
  .humanics-t {{
    color: {primary};
    font-weight: bold;
  }}
  .humanics-d {{
    color: {ultra};
    font-weight: bold;
  }}
  .humanics-h {{
    color: {black};
    font-weight: bold;
  }}
"""

    return css


def get_marp_frontmatter(design: DesignSystem | None = None) -> dict[str, str]:
    """Return Marp frontmatter dict for a GLR-styled presentation."""
    return {
        "marp": "true",
        "paginate": "true",
        "size": "16:9",
        "style": "|\n" + generate_marp_css(design),
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    css = generate_marp_css()
    print("✅ GLR Marp CSS generated successfully.\n")
    print(css[:500])
    print(f"\n... ({len(css)} characters total)")