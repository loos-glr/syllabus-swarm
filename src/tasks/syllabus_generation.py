"""
syllabus_generation.py — Syllabus Generation Task (Humanics-Aligned)
===================================================================

Issue #2: Core Agent — The Curriculum Architect (Humanics Alignment)

Defines a CrewAI **Task** that, when executed by the Curriculum Architect
agent, produces a complete, structured vocational course syllabus in
Markdown.  The task specification:

  • Embeds all three **Humanics** literacies — Technological, Data, and
    Human Literacy — as non-negotiable threads woven through every module.
  • Mandates a dedicated **Experiential Learning** section covering
    real-world co-ops, capstone projects, and/or industry partnerships.
  • Enforces consistent Markdown headings and formatting so the output
    can be consumed directly by an LMS or human instructor.
  • Writes the final syllabus to ``output/syllabus/<course_name>.md``.
"""

from __future__ import annotations

from crewai import Agent, Task

# ---------------------------------------------------------------------------
# Markdown heading & formatting requirement (injected into both description
# and expected_output to keep the agent deterministic).
# ---------------------------------------------------------------------------
_MARKDOWN_STRUCTURE_REQUIREMENT: str = (
    "## Markdown Formatting Requirements\n\n"
    "Your entire output MUST be valid Markdown with the following structure:\n\n"
    "```\n"
    "# {Course Name} — Course Syllabus\n\n"
    "## 1. Course Overview\n"
    "   - Course description, target audience, prerequisites, duration,\n"
    "     and learning philosophy grounded in the Humanics framework.\n\n"
    "## 2. Learning Objectives\n"
    "   - Categorised under **Technological Literacy**, **Data Literacy**,\n"
    "     and **Human Literacy** sub-headings (H3).\n\n"
    "## 3. Humanics Literacies Integration\n"
    "   ### 3.1 Technological Literacy\n"
    "   ### 3.2 Data Literacy\n"
    "   ### 3.3 Human Literacy\n"
    "   - For each literacy: concrete skills, tools, and competencies\n"
    "     learners will develop.  Use bullet lists and inline examples.\n\n"
    "## 4. Module Breakdown\n"
    "   - One H2 per module.  Each module H2 contains:\n"
    "     * Duration (hours / weeks)\n"
    "     * Learning goals (numbered list)\n"
    "     * Key topics (bullet list)\n"
    "     * Hands-on lab / exercise description\n"
    "     * Humanics tags: `[T]` Technological, `[D]` Data, `[H]` Human\n\n"
    "## 5. Experiential Learning & Industry Integration\n"
    "   ### 5.1 Co-operative Education\n"
    "   ### 5.2 Capstone Project\n"
    "   ### 5.3 Industry Partnerships\n"
    "   - Detailed descriptions, timelines, deliverables, and evaluation\n"
    "     criteria for each experiential component.\n\n"
    "## 6. Assessment Strategy\n"
    "   - Weighting breakdown, rubric philosophy, formative vs. summative\n"
    "     assessments, and re-assessment policies.\n\n"
    "## 7. Required Tools & Resources\n"
    "   - Software, hardware, textbooks, online platforms, and APIs.\n\n"
    "## 8. Schedule-at-a-Glance\n"
    "   - Week-by-week table mapping modules, assessments, and milestones.\n\n"
    "## 9. Instructor Notes\n"
    "   - Facilitation tips, common learner pitfalls, differentiation\n"
    "     strategies, and accommodations.\n"
    "```\n\n"
    "**Rules:**\n"
    "- Use ATX-style headings only (`#`, `##`, `###`, `####`).\n"
    "- Use `-` for unordered lists and `1.` for ordered lists.\n"
    "- Enclose inline code, file paths, and terminal commands in backticks.\n"
    "- Use fenced code blocks with a language tag for multi-line snippets.\n"
    "- Separate sections with blank lines for readability.\n"
    "- **Never** use horizontal rules (`---`) inside the document body.\n"
    "- Tag every module with at least one Humanics label per literacy:\n"
    "  `[T]`, `[D]`, `[H]`.\n"
)
# ---------------------------------------------------------------------------
# Humanics embedding mandate — injected verbatim into the description.
# ---------------------------------------------------------------------------
_HUMANICS_EMBEDDING_MANDATE: str = (
    "## 🔴 MANDATORY — Humanics Literacies Embedding\n\n"
    "You are grounded in **Joseph Aoun's Humanics framework**.  Every "
    "module, every lab, and every assessment must visibly integrate "
    "**three inseparable literacies**:\n\n"
    "- **Technological Literacy [T]** — Hands-on coding, systems thinking, "
    "tooling fluency (git, CI/CD, containers), computational problem-solving, "
    "and exposure to modern development environments.\n"
    "- **Data Literacy [D]** — Data collection, cleaning, analysis, "
    "visualisation, quantitative reasoning, and evidence-based "
    "decision-making that complements the technical stack.\n"
    "- **Human Literacy [H]** — Professional ethics, responsible AI usage, "
    "cross-cultural communication, team collaboration, accessibility, "
    "and the societal impact of technology.\n\n"
    "These are **not** separate modules — they are threads that must run "
    "through every topic, lab, and assessment.  Tag each module component "
    "with `[T]`, `[D]`, or `[H]` so instructors can verify coverage at a "
    "glance.\n"
)

# ---------------------------------------------------------------------------
# Experiential learning mandate — injected verbatim into the description.
# ---------------------------------------------------------------------------
_EXPERIENTIAL_LEARNING_MANDATE: str = (
    "## 🔴 MANDATORY — Experiential Learning & Industry Integration\n\n"
    "Every syllabus you produce MUST include a dedicated **Experiential "
    "Learning & Industry Integration** section (H2 heading).  This section "
    "must contain at least three sub-sections:\n\n"
    "1. **Co-operative Education (H3)** — Describe a structured co-op or "
    "work-integrated-learning placement.  Specify the minimum hours, "
    "expected workplace competencies, employer evaluation criteria, and "
    "reflection artefacts (e.g. a learning journal, a portfolio entry, or "
    "a supervisor sign-off form).  Tie co-op objectives back to all three "
    "Humanics literacies.\n\n"
    "2. **Capstone Project (H3)** — Define a culminating project that "
    "integrates skills from the entire course.  Include: the project "
    "brief, team formation guidelines (if group-based), milestone "
    "schedule, deliverable list (code repo, presentation, written report), "
    "and a grading rubric.  The capstone must require learners to exercise "
    "**Technological** (build/deploy), **Data** (measure/analyse), and "
    "**Human** (present/collaborate/reflect-on-ethics) skills.\n\n"
    "3. **Industry Partnerships (H3)** — Outline how industry partners "
    "are engaged in the course: guest lectures, mentorship, live project "
    "briefs, mock interviews, site visits, or hiring-pipeline arrangements.  "
    "Include at least one concrete example of a partner-led activity.\n\n"
    "If the course level or subject makes a traditional co-op infeasible, "
    "replace it with a **simulated industry sprint** — a multi-day, "
    "time-boxed exercise that mirrors a real workplace scenario — and "
    "justify the substitution inline.  **Omitting this section entirely is "
    "not acceptable under any circumstances.**\n"
)

# ---------------------------------------------------------------------------
# Task factory
# ---------------------------------------------------------------------------


def create_syllabus_generation_task(
    *,
    course_name: str,
    agent: Agent,
    course_context: str | None = None,
    course_description: str | None = None,
    course_duration: str | None = None,
    target_audience: str | None = None,
    human_feedback: str | None = None,
    verbose: bool = False,
) -> Task:
    """Create a CrewAI Task that generates a Humanics-aligned syllabus.

    Parameters
    ----------
    course_name : str
        The name of the vocational course (e.g. "Full-Stack Web Development").
    agent : Agent
        The Curriculum Architect agent that will execute this task.
    course_context : str or None
        Rich course context string from the Intake Specialist containing
        tech stack, kerntaken emphasis, student profile, and pedagogical
        notes.  When provided, this is injected as the primary grounding
        context for the syllabus design.
    course_description : str or None
        Optional one-paragraph description of the course's scope and focus.
        When omitted the agent infers this from the course name.
    course_duration : str or None
        Optional duration string (e.g. "12 weeks" or "6 months").
    target_audience : str or None
        Optional description of the target learner profile.
    verbose : bool
        Enable detailed task execution logging.

    Returns
    -------
    Task
        Fully-configured CrewAI Task ready to be assigned to a Crew.
    """
    # ---- Build the description ------------------------------------------
    description_parts: list[str] = [
        f"Generate a detailed, vocational course syllabus for the "
        f"following course:\n\n"
        f"**Course Name:** {course_name}\n",
    ]

    if course_context:
        description_parts.append(
            f"\n**Course Context (from Intake Specialist):**\n{course_context}\n"
        )

    if course_description:
        description_parts.append(f"\n**Course Description:** {course_description}\n")

    if course_duration:
        description_parts.append(f"\n**Course Duration:** {course_duration}\n")

    if target_audience:
        description_parts.append(f"\n**Target Audience:** {target_audience}\n")

    description_parts.append(
        "\n\nThe syllabus must be designed according to **backward design** "
        "principles: start with the desired outcomes, then design "
        "assessments, and finally plan the learning activities.  Ground "
        "every decision in the conviction that vocational education must "
        "prepare learners not just for their first job, but for a career "
        "of continuous adaptation in an AI-augmented economy.\n\n"
        "_HUMANICS_EMBEDDING_MANDATE_\n\n"
        "_EXPERIENTIAL_LEARNING_MANDATE_\n\n"
        "_MARKDOWN_STRUCTURE_REQUIREMENT_\n"
    )

    # Inject the full mandate text (not references).
    description = (
        "".join(description_parts)
        .replace("_HUMANICS_EMBEDDING_MANDATE_", _HUMANICS_EMBEDDING_MANDATE)
        .replace("_EXPERIENTIAL_LEARNING_MANDATE_", _EXPERIENTIAL_LEARNING_MANDATE)
        .replace("_MARKDOWN_STRUCTURE_REQUIREMENT_", _MARKDOWN_STRUCTURE_REQUIREMENT)
    )

    # ── Inject human feedback (HITL loop) ───────────────────────────
    if human_feedback:
        description += (
            f"\n\n## ⚠️ Human Feedback (Instructor Review)\n\n"
            f"The following feedback was provided by a human reviewer "
            f"and MUST be addressed in this iteration:\n\n"
            f"{human_feedback}\n\n"
            f"Please revise the output to incorporate this feedback "
            f"while maintaining all other requirements.\n"
        )

    # ---- Build the expected_output --------------------------------------
    expected_output = (
        "A complete, self-contained Markdown document that **exactly** "
        "follows the structure specified in the Markdown Formatting "
        "Requirements above.  The output must:\n\n"
        "1. Open with an H1 heading matching the course name.\n"
        "2. Contain all nine required sections (Course Overview through "
        "Instructor Notes) as H2 headings in the prescribed order.\n"
        "3. Embed all three Humanics literacies — **Technological [T]**, "
        "**Data [D]**, and **Human [H]** — as visible tags in every "
        "module breakdown, with concrete skill descriptions under each "
        "literacy sub-heading (Section 3).\n"
        "4. Include the mandatory **Experiential Learning & Industry "
        "Integration** section (Section 5) with all three sub-sections: "
        "Co-operative Education (or a justified simulated industry sprint), "
        "Capstone Project, and Industry Partnerships.\n"
        "5. Use only ATX-style headings, `-` bullet lists, `1.` numbered "
        "lists, backtick-quoted inline code, and fenced code blocks with "
        "language tags.  No horizontal rules.\n"
        "6. Produce a week-by-week schedule table (Section 8) with columns "
        "for Week, Module, Topics, Assessments, and Humanics Tags.\n"
        "7. Be comprehensive enough that an instructor could teach the "
        "course with no additional materials.\n\n"
        f"The first line of the output MUST be:\n\n"
        f"`# {course_name} — Course Syllabus`\n"
    )

    # ---- Assemble and return the Task -----------------------------------
    # NOTE: The syllabus file is written explicitly by the orchestrator in
    # syllabus_crew.py using the canonical _sanitize_filename(), so there is
    # no output_file on the Task.  Setting one causes a duplicate file when
    # CrewAI persists the task output under a different name.
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
        async_execution=False,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.agents.curriculum_architect import create_curriculum_architect

    agent = create_curriculum_architect(verbose=True)
    task = create_syllabus_generation_task(
        course_name="Python for Data Engineering",
        agent=agent,
        course_duration="10 weeks",
        target_audience="Career changers with basic programming experience",
        verbose=True,
    )
    print("✅ Syllabus Generation Task created successfully.\n")
    print(f"   Agent role:   {agent.role.split(chr(10))[0]}")
    print(f"   Async:        {task.async_execution}")
    print(f"   Description length: {len(task.description)} chars")
    print(f"   Expected output length: {len(task.expected_output)} chars")
