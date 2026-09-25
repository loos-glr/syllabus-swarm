"""
presentation_generation.py — Presentation Generation Task
=========================================================

Defines a CrewAI **Task** that, when executed by the Presentation Designer
agent, produces a Marp Markdown slide deck from syllabus and lesson plan context.

The task embeds constraints from ``skills/educational-presentation-nl/``:
Gagné's 9 instructional events, Mayer's 12 multimedia principles, C.R.A.P.
design rules, and WCAG 2.1 AA accessibility.
"""

from __future__ import annotations

from crewai import Agent, Task

from src.exporters.glr_marp_theme import generate_marp_css

_PRESENTATION_STRUCTURE: str = (
    "## 📐  Presentation Structure Requirements\n\n"
    "Generate Marp Markdown following **Gagné's 9 instructional events** "
    "as mandatory macro-structure:\n\n"
    "| # | Event | Slide type |\n"
    "|---|-------|-----------|\n"
    "| 1 | **Attention** | Compelling hook — question, stat, case study |\n"
    "| 2 | **Objectives** | \"At the end you can…\" + measurable action verbs (Bloom) |\n"
    "| 3 | **Prior Knowledge** | \"What do you already know about…?\" activation |\n"
    "| 4 | **Content** | Chunked concepts: 1 idea per slide, visual + keywords, "
    "max 3-5 min per chunk |\n"
    "| 5 | **Guidance** | Worked example, analogy, or non-example for scaffolding |\n"
    "| 6 | **Practice** | \"Now you!\" — hands-on exercise, individual or group |\n"
    "| 7 | **Feedback** | Solution + explanation of the reasoning |\n"
    "| 8 | **Assessment** | Quiz question or mini-deliverable |\n"
    "| 9 | **Transfer** | \"How will you use this…?\" — real-world application |\n\n"
    "**Every slide MUST respect Mayer's principles:**\n"
    "- **Redundancy** (MOST CRITICAL): ZERO paragraph text on slides. Move all "
    "paragraphs to speaker notes. Slides contain ONLY: keywords, graphs, or "
    "diagrams.\n"
    "- **Coherence** (MOST CRITICAL): No decorative clipart, busy backgrounds, "
    "or irrelevant details. Every element must serve a clear instructional "
    "purpose.\n"
    "- **Segmentatie**: Progressive disclosure for lists >=3 items. Never dump "
    "all content at once - build up step by step.\n"
    "- **Modality**: Visual + narration (good). Visual + text wall + narration "
    "(cognitive overload - bad).\n"
    "- **Personalisatie**: Use conversational 'je/jij' register (Dutch)/'you' "
    "(English).\n\n"
    "**Slide types required:**\n"
    "- **Title slide** — course name, module title, lesson overview.\n"
    "- **Concept slides** — 1 idea, visual + keywords + speaker notes.\n"
    "- **Code slides** — syntax-highlighted code with annotations.\n"
    "- **Diagram slides** — Mermaid.js diagrams for architecture/workflows.\n"
    "- **Activity slides** — exercise instructions with timebox.\n"
    "- **Summary slide** — recap of learning objectives + next steps.\n"
    "- **Thank-you slide** — contact info, slide availability, QR if applicable.\n\n"
    "**Non-negotiable design constraints (WCAG 2.1 AA + C.R.A.P.):**\n"
    "- Font >= 24pt body text / >= 36pt titles (sans-serif: Arial, Calibri).\n"
    "- Contrast >= 4.5:1 on all text (verify against WebAIM checker).\n"
    "- Max 2 fonts across entire deck; 60-30-10 colour rule.\n"
    "- Strong alignment (invisible grid); related items grouped together.\n"
    "- Alt-text for every image; no colour-only meaning (add shape/text).\n"
    "- Each slide <= 10 words visible; all detail in speaker notes.\n\n"
    "**Speaker notes:** Every slide MUST have detailed speaker notes "
    "(HTML comment `<!-- ... -->`) containing:\n"
    "- Full explanation of the concept (what the teacher says aloud).\n"
    "- Estimated timing for that slide.\n"
    "- Transition cues to the next slide.\n"
    "- Answers/keys for exercises.\n"
)

_TOOL_USAGE_MANDATE: str = (
    "## 🔴  CRITICAL: Tool Usage Requirement\n\n"
    'You MUST use the `output_export_tool` with `command="write-directory-tree"` '
    "to write the presentation to disk:\n"
    "- `output/<run_id>/presentations/<module>/presentation.md`\n\n"
    "After writing the file, produce a brief Markdown summary.\n"
)


def create_presentation_task(
    agent: Agent,
    *,
    course_name: str,
    syllabus_context: str,
    module_name: str = "",
    run_id: str | None = None,
    material_language: str = "Dutch",
    human_feedback: str | None = None,
    verbose: bool = False,
) -> Task:
    """Create a CrewAI Task for generating a Marp presentation deck."""
    module_label = module_name or course_name

    language_directive = ""
    if material_language == "Dutch":
        language_directive = (
            "\n\n## 🇳🇱  TAALRICHTLIJN\n\n"
            "**Alle presentatiematerialen MOETEN in het Nederlands.**\n"
            "- Dias, titels, opsommingen en sprekersnotities in het Nederlands.\n"
            "- Codevoorbeelden en technische termen mogen in het Engels.\n"
            "- **Gebruik 'je'/'jij'** (niet 'u') — dit volgt Mayers "
            "Personalisatieprincipe: een conversationele stijl verbetert "
            "het leerresultaat bij MBO4-studenten.\n"
        )
    elif material_language == "English":
        language_directive = (
            "\n\n## 🇬🇧  LANGUAGE DIRECTIVE\n\n"
            "**All presentation materials MUST be in English.**\n"
            "- Slides, titles, bullets, and speaker notes in English.\n"
        )

    # ── GLR Design System Directive ─────────────────────────────────
    marp_css = generate_marp_css()
    glr_design_directive = (
        "\n\n## 🎨  GLR MEDIA CREATIVE — BRAND DESIGN SYSTEM (NON-NEGOTIABLE)\n\n"
        "**CRITICAL: Replace `theme: uncover` with the EXACT frontmatter below.**\n\n"
        "Every presentation deck MUST use this Marp frontmatter:\n\n"
        "```yaml\n"
        "---\n"
        "marp: true\n"
        "paginate: true\n"
        "size: 16:9\n"
        "style: |\n"
        f"{marp_css}"
        "---\n"
        "```\n\n"
        "### GLR Colour Rules (CRITICAL)\n"
        "- **Primary signal green:** `#A6E22E` / `#76B800` — use for buttons, "
        "badges, key highlights, ``.glr-badge``, ``.glr-marquee``\n"
        "- **Primary black:** `#000000` — headings, borders, dark cards (``.glr-card-dark``)\n"
        "- **Primary white:** `#FFFFFF` — light cards (``.glr-card``), text on dark\n"
        "- **Accent blue:** `#002BFF` — ``.humanics-d`` tags, tertiary links\n"
        "- **NEVER wash out green** — always pair against solid black or white\n"
        "- **0px border-radius on EVERYTHING** — buttons, cards, badges, code blocks\n"
        "- **Hard shadows only** — no blur/glow; use ``box-shadow: 4px 4px 0px #000``\n\n"
        "### GLR Typography\n"
        "- **Headings:** ``Space Grotesk`` — geometric, angular, bold. H1 48px/700, "
        "H2 32px/600, H3 24px/600\n"
        "- **Body:** ``Hanken Grotesk`` — clean, neutral, readable. 16px/400\n"
        "- **Labels:** ``Space Grotesk`` uppercase, wide letter-spacing (``label-lg``, ``label-md``)\n"
        "- **Code:** ``Space Grotesk`` 11px/700, black background with lime text\n\n"
        "### GLR Component Classes (USE THESE)\n"
        "- ``.glr-btn-primary`` — lime button with black border, 0px radius\n"
        "- ``.glr-card`` — white card, 1px black border, 0px radius\n"
        "- ``.glr-card-dark`` — black card with lime accents\n"
        "- ``.glr-badge`` / ``.glr-badge-active`` / ``.glr-badge-outline``\n"
        "- ``.glr-marquee`` — lime banner with black text\n"
        "- ``.glr-media-frame`` — image container with GLR dimension marks\n"
        "- ``.columns`` / ``.columns-3`` — grid layouts\n\n"
        "### Humanics Tags (GLR Palette)\n"
        "- ``[T]`` Technological → green (``.humanics-t``)\n"
        "- ``[D]`` Data → ultramarine blue (``.humanics-d``)\n"
        "- ``[H]`` Human → black (``.humanics-h``)\n"
    )

    description = (
        f"# Presentation Generation Task\n\n"
        f"**Course:** {course_name}\n"
        f"**Module:** {module_label}\n"
        f"**Methodology:** `skills/educational-presentation-nl/SKILL.md` "
        f"(Cognitive Load Theory + Mayer + Gagné + C.R.A.P. + WCAG 2.1 AA)\n\n"
        f"## 📖  Syllabus Context\n\n"
        f"```\n{syllabus_context}\n```\n\n"
        f"{_PRESENTATION_STRUCTURE}\n\n"
        f"{glr_design_directive}\n\n"
        f"{_TOOL_USAGE_MANDATE}"
        f"{language_directive}"
    )

    # ── Inject human feedback (HITL loop) ───────────────────────────
    if human_feedback:
        description += (
            f"\n\n## ⚠️ Human Feedback (Instructor Review)\n\n"
            f"The following feedback was provided by a human reviewer "
            f"and MUST be addressed in this iteration:\n\n"
            f"{human_feedback}\n\n"
            f"Please revise the presentation to incorporate this feedback "
            f"while maintaining all other requirements.\n"
        )

    safe_module = module_name.replace(" ", "_").replace("-", "_").lower() if module_name else "module"
    out_prefix = (
        f"output/{run_id}/presentations/{safe_module}"
        if run_id else f"output/presentations/{safe_module}"
    )

    expected_output = (
        "## 🔴 CRITICAL: You MUST use the `output_export_tool`\n\n"
        'Use the `output_export_tool` with `command="write-directory-tree"` '
        "to write the presentation to disk:\n\n"
        f"- `{out_prefix}/presentation.md`\n\n"
        "**Once the file is written**, produce a Markdown summary listing "
        "the module name, number of slides, slide types per Gagné event, "
        "and a self-audit confirming all 6 items of the 30-second checklist "
        "pass (1 idea, <10 words, >=24pt font, >=4.5:1 contrast, aligned, "
        "speaker notes present) on every slide.\n"
    )

    output_file = f"{out_prefix}/README.md"

    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
        output_file=output_file,
        async_execution=False,
    )


create_presentation_generation_task = create_presentation_task
