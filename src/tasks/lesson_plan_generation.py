"""
lesson_plan_generation.py — Lesson Plan Generation Task
=======================================================

Defines a CrewAI **Task** that, when executed by the Instructional Coordinator
agent, produces a structured lesson plan from syllabus context.
"""

from __future__ import annotations

from crewai import Agent, Task

_LESSON_PLAN_STRUCTURE: str = (
    "## 📐  Lesson Plan Structure Requirements\n\n"
    "Each lesson plan must follow this exact structure:\n\n"
    "### Frontmatter\n"
    "- Module name, total duration, prerequisites, materials needed.\n\n"
    "### Session Breakdown\n"
    "For each session, include:\n"
    "- **Session number and title**.\n"
    "- **Duration** — in minutes (typically 45, 60, or 90).\n"
    "- **Learning objectives** — 2–4 measurable objectives.\n"
    "- **Activities** — ordered list with estimated time per activity.\n"
    "- **Resources** — tools, files, equipment needed.\n"
    "- **Differentiation** — strategies for struggling and advanced students.\n"
    "- **Assessment** — formative checkpoints for this session.\n\n"
    "### Global Sections\n"
    "- **Differentiation strategies** — overarching strategies for the module.\n"
    "- **Assessment checkpoints** — key milestones across sessions.\n"
    "- **Materials required** — complete list for the entire module.\n\n"
    "### Language\n"
    "Use clear, direct language suitable for MBO4 teachers.  Avoid academic "
    "jargon.\n"
)

_TOOL_USAGE_MANDATE: str = (
    "## 🔴  CRITICAL: Tool Usage Requirement\n\n"
    'You MUST use the `output_export_tool` with `command="write-directory-tree"` '
    "to write the lesson plan to disk.\n\n"
    "Write exactly ONE Markdown file:\n"
    "- `output/<run_id>/lesson_plans/<module>/lesson_plan.md`\n\n"
    "After writing the file, produce a brief Markdown summary.\n"
)
def create_lesson_plan_task(
    agent: Agent,
    *,
    course_name: str,
    syllabus_context: str,
    module_name: str = "",
    run_id: str | None = None,
    material_language: str = "Dutch",
    verbose: bool = False,
) -> Task:
    """Create a CrewAI Task that generates a lesson plan for a module."""
    module_label = module_name or course_name

    language_directive = ""
    if material_language == "Dutch":
        language_directive = (
            "\n\n## 🇳🇱  TAALRICHTLIJN (LANGUAGE DIRECTIVE)\n\n"
            "**Alle lesmaterialen MOETEN in het Nederlands worden geschreven.**\n"
            "- Lesplan titels, sessiebeschrijvingen, leerdoelen, activiteiten, "
            "differentiatie en assessment moeten in het Nederlands zijn.\n"
            "- Codevoorbeelden en technische termen mogen in het Engels blijven.\n"
        )
    elif material_language == "English":
        language_directive = (
            "\n\n## 🇬🇧  LANGUAGE DIRECTIVE\n\n"
            "**All lesson plan materials MUST be written in English.**\n"
            "- Session titles, descriptions, learning objectives, activities, "
            "differentiation, and assessment must be in English.\n"
            "- Code examples and technical terms should remain in English.\n"
        )

    description = (
        f"# Lesson Plan Generation Task\n\n"
        f"**Course:** {course_name}\n"
        f"**Module:** {module_label}\n\n"
        f"## 📖  Syllabus Context\n\n"
        f"Use the following syllabus as the source of truth for generating "
        f"the lesson plan:\n\n"
        f"```\n{syllabus_context}\n```\n\n"
        f"{_LESSON_PLAN_STRUCTURE}\n\n"
        f"{_TOOL_USAGE_MANDATE}"
        f"{language_directive}"
    )

    safe_module = module_name.replace(" ", "_").replace("-", "_").lower() if module_name else "module"
    out_prefix = (
        f"output/{run_id}/lesson_plans/{safe_module}"
        if run_id else f"output/lesson_plans/{safe_module}"
    )

    expected_output = (
        "## 🔴 CRITICAL: You MUST use the `output_export_tool`\n\n"
        'Use the `output_export_tool` with `command="write-directory-tree"` '
        "to write the lesson plan to disk:\n\n"
        f"- `{out_prefix}/lesson_plan.md`\n\n"
        "**Once the file is written**, produce a Markdown summary listing "
        "the module name, number of sessions, and total duration.\n"
    )

    output_file = f"{out_prefix}/README.md"

    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
        output_file=output_file,
        async_execution=False,
    )


# Convenience alias
create_lesson_plan_generation_task = create_lesson_plan_task
