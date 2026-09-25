"""
models.py — Shared Pydantic Models for Syllabus Swarm
=====================================================

Issue #10: Machine-Readable Course Graph JSON Export (Phase 3 — Export Side)

Defines the **CourseSpecification**, **IntakeSession**, **CourseGraph**,
and **ModuleSummary** Pydantic models that are shared across the entire
syllabus-swarm codebase.

Design constraint
-----------------
* ``CourseGraph`` **composes** ``CourseSpecification`` via a
  ``specification`` field — it never duplicates ``course_context``
  or ``primary_language``.
* ``ModuleSummary`` is a lightweight model representing a single
  course module.

Public API
----------
* ``CourseSpecification`` — structured output from Intake Specialist.
* ``IntakeSession`` — serializable record of a completed intake interview.
* ``ModuleSummary`` — title, duration_weeks, topics.
* ``CourseGraph`` — specification, course_slug, learning_objectives,
  key_concepts, prerequisites, modules, generated_at.
"""

from __future__ import annotations

import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# CourseSpecification
# ---------------------------------------------------------------------------


class CourseSpecification(BaseModel):
    """Structured output from the Intake Specialist synthesis step.

    This model ensures the LLM returns both the rich pedagogical context
    AND the exact programming language, eliminating the need for brittle
    regex-based language detection downstream.

    The four Optional fields — ``grading_scale``, ``student_pathway``,
    ``year_level``, and ``hardware_constraints`` — can be pre-populated
    from a cohort profile (``--profile <path>``) to skip those intake
    questions.  When a field is ``None`` the Intake Specialist will still
    prompt for it.
    """

    course_context: str = Field(
        description="The rich pedagogical context and requirements "
        "synthesised from the user's answers."
    )
    primary_language: str = Field(
        description="The exact programming language to be used for labs "
        "(e.g., 'JavaScript', 'Python', 'TypeScript', 'Java', 'Go', 'Rust')."
    )
    grading_scale: str | None = Field(
        default=None,
        description="The grading scale to use (e.g., 'OVG' for Dutch MBO "
        "Onvoldoende/Voldoende/Goed, '1-10', 'A-F').  When pre-populated "
        "from a profile, the Intake Specialist skips this question.",
    )
    student_pathway: str | None = Field(
        default=None,
        description="The student pathway: 'BOL' (school-based) or 'BBL' "
        "(work-based).  When pre-populated from a profile, the Intake "
        "Specialist skips this question.",
    )
    year_level: int | None = Field(
        default=None,
        ge=1,
        le=3,
        description="The student year level (1, 2, or 3).  When pre-populated "
        "from a profile, the Intake Specialist skips this question.",
    )
    hardware_constraints: str | None = Field(
        default=None,
        description="Description of hardware/device constraints (e.g., BYOD, "
        "Chromebooks, thin clients).  When pre-populated from a profile, "
        "the Intake Specialist skips this question.",
    )
    material_language: Literal["Dutch", "English"] = Field(
        default="Dutch",
        description="The human language in which ALL instructional materials "
        "(syllabi, theory, labs, READMEs, code comments, assessments) "
        "will be written.  'Dutch' (Nederlands) or 'English'.  When "
        "pre-populated from a profile, the Intake Specialist skips this "
        "question.",
    )


# ---------------------------------------------------------------------------
# IntakeSession
# ---------------------------------------------------------------------------


class IntakeSession(BaseModel):
    """Serializable record of a completed intake interview (Issue #9).

    Captures the full intake conversation and its synthesised result,
    enabling the ``--load-session`` flag to restore a prior intake
    and bypass the interactive interview entirely.
    """

    course_name: str = Field(description="Original course name / topic from the user")
    questions: str = Field(description="Questions asked by the Intake Specialist agent")
    answers: str = Field(description="User's answers to the intake questions")
    course_specification: CourseSpecification = Field(
        description="Synthesised course specification (course_context + primary_language)"
    )
    timestamp: str = Field(description="ISO 8601 timestamp of when the intake interview completed")
    run_id: str = Field(description="Unique run identifier (YYYY-MM-DD_HHMMSS_course_slug)")


# ---------------------------------------------------------------------------
# ModuleSummary
# ---------------------------------------------------------------------------


class ModuleSummary(BaseModel):
    """Lightweight representation of a single course module.

    Used by ``CourseGraph.modules`` to describe the module-level
    structure without duplicating full syllabus content.
    """

    title: str = Field(
        description="Human-readable module title (e.g. 'Python Fundamentals').",
        min_length=1,
    )
    duration_weeks: float = Field(
        description="Module duration in weeks (supports fractional weeks).",
        ge=0.0,
    )
    hours_per_week: float = Field(
        description=(
            "Contact hours per week for this module (e.g. 3.0 for "
            "a 3-hour/week module).  Combined with duration_weeks this "
            "gives total module effort: duration_weeks × hours_per_week."
        ),
        ge=0.0,
    )
    topics: list[str] = Field(
        default_factory=list,
        description="Key topics covered in this module.",
    )


# ---------------------------------------------------------------------------
# CourseGraph
# ---------------------------------------------------------------------------


class CourseGraph(BaseModel):
    """Machine-readable course graph that composes the existing spec.

    This model is the primary export format for downstream tooling
    (curriculum memory engine, module chaining, LMS import).
    It **never** duplicates fields from ``CourseSpecification``;
    instead it holds a reference via ``specification``.

    Example
    -------
    >>> spec = CourseSpecification(
    ...     course_context="Intro to Python for DS.",
    ...     primary_language="Python",
    ... )
    >>> graph = CourseGraph(
    ...     specification=spec,
    ...     course_slug="intro-python-ds",
    ...     learning_objectives=["Write Python scripts"],
    ...     key_concepts=["variables", "loops"],
    ...     prerequisites=["Basic computer literacy"],
    ...     modules=[
    ...         ModuleSummary(
    ...             title="Getting Started",
    ...             duration_weeks=1.0,
    ...             topics=["installation", "first script"],
    ...         ),
    ...     ],
    ... )
    """

    specification: CourseSpecification = Field(
        description="The existing course specification — never duplicated.",
    )
    course_slug: str = Field(
        description="URL- / filesystem-safe identifier for the course.",
        min_length=1,
    )
    learning_objectives: list[str] = Field(
        default_factory=list,
        description="Top-level learning objectives for the entire course.",
    )
    key_concepts: list[str] = Field(
        default_factory=list,
        description="Core concepts / competencies learners will acquire.",
    )
    prerequisites: list[str] = Field(
        default_factory=list,
        description="Required prior knowledge or courses.",
    )
    modules: list[ModuleSummary] = Field(
        default_factory=list,
        description="Ordered list of course modules.",
    )
    generated_at: str = Field(
        default_factory=lambda: datetime.datetime.now(tz=datetime.UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        description="ISO 8601 UTC timestamp when the graph was generated.",
    )


# ---------------------------------------------------------------------------
# ModalityType — Pedagogical modality enum for modality routing
# ---------------------------------------------------------------------------


class ModalityType(str, Enum):
    """Pedagogical modality for content delivery.

    Used by the Media Strategist agent to route curriculum modules
    to the appropriate content generator.

    Values
    ------
    CLASSIC_READER
        Traditional text/Markdown-based theory delivery.
    INTERACTIVE_WEB
        Browser-based interactive HTML/JS learning artifacts.
    INTERACTIVE_CLI
        Terminal-based interactive learning scripts.
    VIDEO_AS_CODE
        Deterministic React/Remotion video generation (VaC).
    """

    CLASSIC_READER = "classic_reader"
    INTERACTIVE_WEB = "interactive_web"
    INTERACTIVE_CLI = "interactive_cli"
    VIDEO_AS_CODE = "video_as_code"


# ---------------------------------------------------------------------------
# ModalityDecision — Media Strategist routing output
# ---------------------------------------------------------------------------


class ModalityDecision(BaseModel):
    """Output of the Media Strategist agent — routes a module to a generator.

    Each curriculum module receives exactly one ModalityDecision that
    the swarm state machine reads to determine whether to invoke the
    Theory Instructor (CLASSIC_READER) or Video Engineer (VIDEO_AS_CODE).
    """

    module_name: str = Field(
        description="Human-readable module title the decision applies to.",
        min_length=1,
    )
    modality: ModalityType = Field(
        description="Selected pedagogical modality for this module.",
    )
    rationale: str = Field(
        description="Pedagogical justification for the modality choice.",
        min_length=1,
    )
    complexity_score: float = Field(
        description="Estimated complexity of the module (0.0 = trivial, 1.0 = very complex).",
        ge=0.0,
        le=1.0,
    )
    suggested_components: list[str] = Field(
        default_factory=list,
        description="Suggested visual/code components for the chosen modality.",
    )


# ---------------------------------------------------------------------------
# RemotionManifest — Deterministic VaC composition data
# ---------------------------------------------------------------------------


class RemotionManifest(BaseModel):
    """Deterministic Video-as-Code composition descriptor.

    Produced by the Video Engineer agent.  Contains all the structural
    metadata needed to render a React/Remotion composition — scene
    sequences, timing, and component hierarchy — without any narrative
    prose or pixel-based media.
    """

    composition_id: str = Field(
        description="Unique identifier for the Remotion composition (e.g. 'recursion_basics').",
        min_length=1,
    )
    duration_in_frames: int = Field(
        description="Total duration of the composition in frames.",
        gt=0,
    )
    fps: int = Field(
        default=30,
        description="Frames per second for the composition.",
        gt=0,
    )
    width: int = Field(
        default=1920,
        description="Canvas width in pixels.",
        gt=0,
    )
    height: int = Field(
        default=1080,
        description="Canvas height in pixels.",
        gt=0,
    )
    components: list[dict[str, object]] = Field(
        default_factory=list,
        description="Ordered list of scene/sequence descriptors.  Each dict "
        "represents a React component with type, props, and optional children.",
    )
    module_name: str = Field(
        description="The curriculum module this composition belongs to.",
        min_length=1,
    )


# ---------------------------------------------------------------------------
# GenerationState — tracks resume/restart progress
# ---------------------------------------------------------------------------


class TierState(BaseModel):
    """State of a single lab tier during generation.

    Used by :class:`GenerationState` to track which tiers have been
    successfully completed so the resume logic can skip them.
    """

    status: str = "incomplete"
    """One of ``"complete"``, ``"incomplete"``, or ``"failed"``."""

    files: int = 0
    """Number of non-gitkeep files written for this tier."""

    error: str | None = None
    """Error message if the tier failed."""


class GenerationState(BaseModel):
    """Serialisable snapshot of the generation pipeline progress.

    Written to ``_generation_state.json`` inside the run directory so
    that a ``--resume-from`` invocation can skip already-completed
    tiers and theory artifacts without re-running expensive LLM calls.

    When the state file is missing (e.g. a run produced before this
    feature existed), the resume logic auto-generates one by scanning
    the filesystem for existing output.
    """

    run_id: str = Field(description="The run identifier this state belongs to.")
    course_name: str = Field(description="Human-readable course name.")
    tiers: dict[str, TierState] = Field(
        default_factory=dict,
        description="Per-tier lab generation state keyed by tier directory name "
        "(e.g. 'tier1_foundations').",
    )
    theory: dict[str, str] = Field(
        default_factory=dict,
        description="Per-tier theory generation state keyed by tier directory name. "
        "Values are 'complete', 'incomplete', or 'failed'.",
    )
    syllabus_review: str = Field(
        default="incomplete",
        description="Syllabus review state: 'complete' or 'incomplete'.",
    )
    qa_review: str = Field(
        default="incomplete",
        description="QA review state: 'complete' or 'incomplete'.",
    )
    lesson_plan: dict[str, str] = Field(
        default_factory=dict,
        description="Per-module lesson plan generation state keyed by module name. "
        "Values are 'complete', 'incomplete', or 'failed'.",
    )
    presentation: dict[str, str] = Field(
        default_factory=dict,
        description="Per-module presentation generation state keyed by module name. "
        "Values are 'complete', 'incomplete', or 'failed'.",
    )


# ---------------------------------------------------------------------------
# Lesson Plan & Presentation Artifacts (Issue #12 — Pedagogical Asset Pipeline)
# ---------------------------------------------------------------------------


class SessionBlock(BaseModel):
    """A single teaching session within a lesson plan.

    Represents one block of instruction (typically 45–90 minutes) with
    learning objectives, activities, resources, differentiation, and
    assessment checkpoints tailored for MBO4 vocational education.
    """

    session_number: int = Field(description="Sequential session number within the module.", ge=1)
    title: str = Field(description="Descriptive title for this session.", min_length=1)
    duration_minutes: int = Field(
        description="Duration of this session in minutes.",
        gt=0,
    )
    learning_objectives: list[str] = Field(
        default_factory=list,
        description="Specific learning objectives for this session.",
    )
    activities: list[str] = Field(
        default_factory=list,
        description="Ordered list of teaching and learning activities.",
    )
    resources: list[str] = Field(
        default_factory=list,
        description="Materials, tools, and equipment needed for this session.",
    )
    differentiation: str = Field(
        default="",
        description="Strategies for differentiating instruction for diverse learners.",
    )
    assessment: str = Field(
        default="",
        description="Formative or summative assessment approach for this session.",
    )


class LessonPlanManifest(BaseModel):
    """A complete lesson plan for a single curriculum module.

    Produced by the Instructional Coordinator agent.  Contains a session-by-session
    breakdown with timing, differentiation, and assessment checkpoints suitable
    for MBO4 vocational teachers.
    """

    module_name: str = Field(
        description="The curriculum module this lesson plan belongs to.",
        min_length=1,
    )
    sessions: list[SessionBlock] = Field(
        default_factory=list,
        description="Ordered list of teaching sessions for this module.",
    )
    total_duration_minutes: int = Field(
        default=0,
        description="Sum of all session durations in minutes.",
        ge=0,
    )
    differentiation_strategies: list[str] = Field(
        default_factory=list,
        description="Global differentiation strategies applicable across all sessions.",
    )
    assessment_checkpoints: list[str] = Field(
        default_factory=list,
        description="Key assessment milestones across the module.",
    )
    materials_required: list[str] = Field(
        default_factory=list,
        description="All materials and equipment needed for the entire module.",
    )
    prerequisites: list[str] = Field(
        default_factory=list,
        description="Knowledge, skills, or resources required before starting this module.",
    )


class Slide(BaseModel):
    """A single slide in a teacher presentation deck.

    Used by the Presentation Designer to generate Marp Markdown slide decks
    with structured content and speaker notes.
    """

    slide_number: int = Field(description="Position of this slide in the deck (1-based).", ge=0)
    slide_type: str = Field(
        description="Type of slide: 'title', 'bullets', 'code', 'diagram', 'activity', or 'summary'.",
        min_length=1,
    )
    title: str = Field(description="Slide title / heading.", min_length=1)
    content: str = Field(
        default="",
        description="Slide body content (bullets, code blocks, or descriptive text).",
    )
    speaker_notes: str = Field(
        default="",
        description="Speaker notes for the teacher delivering this slide.",
    )
    transition: str = Field(
        default="",
        description="Optional slide transition hint (e.g. 'fade', 'none').",
    )


class PresentationManifest(BaseModel):
    """A teacher slide deck manifest for a single curriculum module.

    Produced by the Presentation Designer agent.  Contains structured slide data
    that can be rendered as Marp Markdown for teacher-led classroom presentations.
    """

    module_name: str = Field(
        description="The curriculum module this presentation belongs to.",
        min_length=1,
    )
    slides: list[Slide] = Field(
        default_factory=list,
        description="Ordered list of slides in the presentation deck.",
    )
    total_estimated_minutes: int = Field(
        default=0,
        description="Estimated total delivery time in minutes.",
        ge=0,
    )
    marp_frontmatter: dict[str, str] = Field(
        default_factory=dict,
        description="Marp Markdown frontmatter configuration "
        "(e.g. theme, paginate, size, backgroundColor).",
    )


# ---------------------------------------------------------------------------
# DesignSystem — GLR Media Creative Design Tokens
# ---------------------------------------------------------------------------
# Parsed from DESIGN.md YAML frontmatter.  Provides type-safe access to
# all color, typography, spacing, and component tokens that agents use
# to embed the GLR brand into generated materials.


class DesignColorTokens(BaseModel):
    """All color tokens defined in the DESIGN.md YAML frontmatter."""

    surface: str = "#faf9fd"
    surface_dim: str = "#dbd9dd"
    surface_bright: str = "#faf9fd"
    surface_container_lowest: str = "#ffffff"
    surface_container_low: str = "#f5f3f7"
    surface_container: str = "#efedf1"
    surface_container_high: str = "#e9e7ec"
    surface_container_highest: str = "#e3e2e6"
    on_surface: str = "#1b1b1f"
    on_surface_variant: str = "#424936"
    inverse_surface: str = "#2f3034"
    inverse_on_surface: str = "#f2f0f4"
    outline: str = "#727a64"
    outline_variant: str = "#c2cab1"
    surface_tint: str = "#416900"
    primary: str = "#416900"
    on_primary: str = "#ffffff"
    primary_container: str = "#76b800"
    on_primary_container: str = "#284300"
    inverse_primary: str = "#95da32"
    secondary: str = "#5f5e5e"
    on_secondary: str = "#ffffff"
    secondary_container: str = "#e5e2e1"
    on_secondary_container: str = "#656464"
    tertiary: str = "#2540ff"
    on_tertiary: str = "#ffffff"
    tertiary_container: str = "#939fff"
    on_tertiary_container: str = "#001dbb"
    error: str = "#ba1a1a"
    on_error: str = "#ffffff"
    error_container: str = "#ffdad6"
    on_error_container: str = "#93000a"
    primary_fixed: str = "#aff74e"
    primary_fixed_dim: str = "#95da32"
    on_primary_fixed: str = "#102000"
    on_primary_fixed_variant: str = "#304f00"
    secondary_fixed: str = "#e5e2e1"
    secondary_fixed_dim: str = "#c8c6c5"
    on_secondary_fixed: str = "#1c1b1b"
    on_secondary_fixed_variant: str = "#474646"
    tertiary_fixed: str = "#dfe0ff"
    tertiary_fixed_dim: str = "#bcc2ff"
    on_tertiary_fixed: str = "#000a63"
    on_tertiary_fixed_variant: str = "#0023d9"
    background: str = "#faf9fd"
    on_background: str = "#1b1b1f"
    surface_variant: str = "#e3e2e6"
    electric_lime: str = "#A6E22E"
    pure_black: str = "#000000"
    pure_white: str = "#FFFFFF"
    surface_subtle: str = "#F4F4F6"
    border_structural: str = "#E5E7EB"
    accent_ultramarine: str = "#002BFF"


class TypographyToken(BaseModel):
    """A single typography style definition."""

    fontFamily: str = "Space Grotesk"
    fontSize: str = "16px"
    fontWeight: str = "400"
    lineHeight: str = "24px"
    letterSpacing: str = "0em"


class DesignTypographyScale(BaseModel):
    """All typography styles defined in DESIGN.md."""

    display_xl: TypographyToken = Field(default_factory=TypographyToken)
    display_xl_mobile: TypographyToken = Field(default_factory=TypographyToken)
    headline_lg: TypographyToken = Field(default_factory=TypographyToken)
    headline_lg_mobile: TypographyToken = Field(default_factory=TypographyToken)
    headline_md: TypographyToken = Field(default_factory=TypographyToken)
    headline_sm: TypographyToken = Field(default_factory=TypographyToken)
    title_md: TypographyToken = Field(default_factory=TypographyToken)
    body_lg: TypographyToken = Field(default_factory=TypographyToken)
    body_md: TypographyToken = Field(default_factory=TypographyToken)
    body_sm: TypographyToken = Field(default_factory=TypographyToken)
    label_lg: TypographyToken = Field(default_factory=TypographyToken)
    label_md: TypographyToken = Field(default_factory=TypographyToken)
    label_code: TypographyToken = Field(default_factory=TypographyToken)


class DesignSpacingTokens(BaseModel):
    """Spacing / layout tokens from DESIGN.md."""

    gutter: str = "1.5rem"
    gutter_mobile: str = "1rem"
    margin: str = "3rem"
    margin_mobile: str = "1.25rem"
    space_xs: str = "0.25rem"
    space_sm: str = "0.5rem"
    space_md: str = "1rem"
    space_lg: str = "1.5rem"
    space_xl: str = "2.5rem"


class DesignSystem(BaseModel):
    """Top-level container for the complete GLR Media Creative design system.

    Parsed from ``DESIGN.md`` and used as the single source of truth for
    all design tokens referenced by agents during material generation.
    """

    name: str = Field(default="GLR Media Creative", description="Design system name.")
    brand_statement: str = Field(
        default="",
        description="Narrative description of the brand identity, movement, and tone.",
    )
    design_movement: str = Field(
        default="",
        description="The design movement label (e.g. 'High-Contrast Brutalist Modernism').",
    )
    colors: DesignColorTokens = Field(default_factory=DesignColorTokens)
    typography: DesignTypographyScale = Field(default_factory=DesignTypographyScale)
    spacing: DesignSpacingTokens = Field(default_factory=DesignSpacingTokens)
    shapes_border_radius: str = Field(
        default="0px",
        description="All UI surfaces use sharp 0px border-radius.",
    )
    depth_style: str = Field(
        default="crisp-architectural",
        description="Depth is achieved via structural borders and hard offset shadows, not blur.",
    )
