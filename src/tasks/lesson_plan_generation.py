"""
lesson_plan_generation.py — Lesson Plan Generation Task
=======================================================

Defines a CrewAI **Task** that, when executed by the Instructional Coordinator
agent, produces a structured lesson plan from syllabus context.
"""

from __future__ import annotations

from crewai import Agent, Task

from src.exporters.file_writer import canonical_tier

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
    'You MUST use the `output_export_tool` with `command="write-lesson-plan"` '
    "to write the lesson plan.  The tool decides the destination — you supply "
    "the `course_name`, the `module_name`, the `run_id`, and the complete "
    "lesson plan Markdown as `content`.\n\n"
    "Do NOT construct file paths yourself; the tool writes to the canonical "
    "`output/<run_id>/lesson_plans/<module>/lesson_plan.md` location and will "
    "reject anything else.\n\n"
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
    human_feedback: str | None = None,
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

    # ── Inject human feedback (HITL loop) ───────────────────────────
    if human_feedback:
        description += (
            f"\n\n## ⚠️ Human Feedback (Instructor Review)\n\n"
            f"The following feedback was provided by a human reviewer "
            f"and MUST be addressed in this iteration:\n\n"
            f"{human_feedback}\n\n"
            f"Please revise the lesson plan to incorporate this feedback "
            f"while maintaining all other requirements.\n"
        )

    # Normalise the module to a canonical tier directory name when it
    # identifies a tier (e.g. "Tier 1 — Foundations" -> "tier1_foundations");
    # fall back to a sanitised slug for genuinely non-tier modules.
    if module_name:
        safe_module = canonical_tier(module_name) or (
            module_name.replace(" ", "_").replace("-", "_").lower()
        )
    else:
        safe_module = "module"

    out_prefix = (
        f"output/{run_id}/lesson_plans/{safe_module}"
        if run_id else f"output/lesson_plans/{safe_module}"
    )

    run_id_hint = run_id or "<run_id-from-context>"
    expected_output = (
        "## 🔴 CRITICAL: You MUST use the `output_export_tool`\n\n"
        'Call the `output_export_tool` with `command="write-lesson-plan"` '
        "to write the lesson plan:\n\n"
        f'- `command="write-lesson-plan"`, `course_name="{course_name}"`, '
        f'`module_name="{safe_module}"`, `run_id="{run_id_hint}"`, '
        "and `content` = the complete lesson plan Markdown.\n\n"
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
