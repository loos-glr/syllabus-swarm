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
