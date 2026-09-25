"""
presentation_designer.py — The Presentation Designer (MBO4 Slide Deck Architect)
================================================================================

Defines a CrewAI Agent configured as an MBO4 slide-deck architect who generates
Marp Markdown presentations from syllabus and lesson plan context.  Each slide
deck contains typed slides (title, bullets, code, diagram, activity, summary)
with speaker notes for teacher-led classroom delivery.

The agent obtains its LLM through
:func:`src.llm_factory.build_llm_for_agent` using the
``PRESENTATION_DESIGNER`` role.
"""

from __future__ import annotations

from crewai import LLM, Agent

from src.exporters.tool import OutputExportTool
from src.llm_factory import (
    PRESENTATION_DESIGNER,
    build_llm_for_agent,
    resolve_max_iter,
    resolve_max_rpm,
)


def create_presentation_designer(
    *,
    llm: LLM | None = None,
    verbose: bool = False,
    max_iter: int | None = None,
    max_rpm: int | None = None,
) -> Agent:
    """Create the Presentation Designer CrewAI agent."""
    if llm is None:
        llm = build_llm_for_agent(PRESENTATION_DESIGNER)

    role = "MBO4 Presentation Designer and Slide Deck Architect"

    goal = (
        "Transform a course syllabus and lesson plan into a polished, "
        "teacher-ready Marp Markdown slide deck.  Each deck must include:\n\n"
        "- **Marp frontmatter** — theme, pagination, and sizing configuration.\n"
        "- **Title slide** — course name, module title, and lesson overview.\n"
        "- **Bullet slides** — key concepts with concise, scannable text.\n"
        "- **Code slides** — syntax-highlighted examples with annotations.\n"
        "- **Diagram slides** — Mermaid.js diagrams for architecture and workflows.\n"
        "- **Activity slides** — instructions for in-class exercises and discussions.\n"
        "- **Summary slide** — recap of learning objectives and next steps.\n"
        "- **Speaker notes** — guidance for the teacher on each slide.\n\n"
        "The presentation must be visually clean, pedagogically sound, and "
        "immediately usable in an MBO4 classroom."
    )

    backstory = (
        "You are a former graphic designer turned MBO4 ICT teacher who discovered "
        "that great slides can transform a confusing lesson into an 'aha!' moment.  "
        "For five years you taught web development at an ROC in Rotterdam, and "
        "every week you saw the same pattern: teachers with deep technical "
        "knowledge but slides that were walls of bullet points — dense, boring, "
        "and ignored by students.\n\n"
        "You started redesigning your colleagues' slides as a side project.  "
        "You applied principles from presentation design (Nancy Duarte, Garr "
        "Reynolds) to vocational education: one idea per slide, visual hierarchy, "
        "code snippets with syntax highlighting, diagrams that animate concepts, "
        "and speaker notes that actually help the teacher deliver the lesson.\n\n"
        "When your redesigned slides led to measurable improvements in student "
        "engagement and assessment results, the school management asked you to "
        "train the entire ICT department.  You developed a workshop called "
        "'Slides That Teach' which is now part of the onboarding programme for "
        "new MBO4 instructors.\n\n"
        "Your speciality is Marp Markdown — you love that it is plain text "
        "(version-controllable!), renders to beautiful HTML/PDF, and lets "
        "teachers focus on content instead of fiddling with PowerPoint.  You "
        "know every Marp theme, every directive, and every trick for making "
        "code blocks look great on a 16:9 projector.  You insist that every "
        "slide deck you produce must have speaker notes — because slides are "
        "for students, but speaker notes are for teachers."
    )

    export_tool = OutputExportTool(force=True)

    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
        max_iter=max_iter if max_iter is not None else resolve_max_iter(PRESENTATION_DESIGNER, 25),
        max_rpm=max_rpm if max_rpm is not None else resolve_max_rpm(PRESENTATION_DESIGNER, 20),
        tools=[export_tool],
    )


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_presentation_designer_instance: Agent | None = None


def get_presentation_designer(*, verbose: bool = False) -> Agent:
    """Return a shared, lazily-created Presentation Designer agent."""
    global _presentation_designer_instance
    if _presentation_designer_instance is None:
        _presentation_designer_instance = create_presentation_designer(verbose=verbose)
    return _presentation_designer_instance


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.llm_factory import get_effective_config

    agent = create_presentation_designer(verbose=True)
    config = get_effective_config(PRESENTATION_DESIGNER)
    print("✅ Presentation Designer agent created successfully.\n")
    print(f"   Role:      {agent.role.split(chr(10))[0]}")
    print(f"   Model:     {config['model']}")
    print(f"   Base URL:  {config['base_url']}")
    print(f"   Temp:      {config['temperature']}")
    print(f"   Top-P:     {config['top_p']}")
    print(f"   Max Tokens:{config['max_tokens']}")
