"""
presentation_generation.py — Presentation Generation Task
=========================================================

Defines a CrewAI **Task** that, when executed by the Presentation Designer
agent, produces a Marp Markdown slide deck from syllabus and lesson plan context.
"""

from __future__ import annotations

from crewai import Agent, Task

_PRESENTATION_STRUCTURE: str = (
    "## 📐  Presentation Structure Requirements\n\n"
    "Generate Marp Markdown with these slide types:\n\n"
    "- **Title slide** — course name, module title, lesson overview.\n"
    "- **Bullet slides** — key concepts with concise text.\n"
    "- **Code slides** — syntax-highlighted code examples.\n"
    "- **Diagram slides** — Mermaid.js diagrams for architecture.\n"
    "- **Activity slides** — instructions for in-class exercises.\n"
    "- **Summary slide** — recap of learning objectives.\n\n"
    "Each slide MUST include speaker notes (HTML comment `<!-- ... -->`).\n"
    "Include Marp frontmatter (theme, paginate, size).\n"
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
        )
    elif material_language == "English":
        language_directive = (
            "\n\n## 🇬🇧  LANGUAGE DIRECTIVE\n\n"
            "**All presentation materials MUST be in English.**\n"
            "- Slides, titles, bullets, and speaker notes in English.\n"
        )

    description = (
        f"# Presentation Generation Task\n\n"
        f"**Course:** {course_name}\n"
        f"**Module:** {module_label}\n\n"
        f"## 📖  Syllabus Context\n\n"
        f"```\n{syllabus_context}\n```\n\n"
        f"{_PRESENTATION_STRUCTURE}\n\n"
        f"{_TOOL_USAGE_MANDATE}"
        f"{language_directive}"
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
        "the module name, number of slides, and slide types.\n"
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
