"""
config_loader.py — Layer 1: Configuration Loading & Profile Validation
======================================================================

.. rubric:: Issue #16 — Define Configuration & Domain Entities

All YAML parsing, profile validation, and CourseSpecification injection
primitives live here in **Layer 1** (Enterprise Business Rules).  No
imports from Layer 2–4 are permitted.

Public API
----------
* ``CohortProfile`` — Pydantic model for validating a cohort profile YAML.
* ``load_profile(path)`` — Parse and validate a ``.yaml`` cohort profile.
* ``inject_profile(spec, profile_data)`` — Merge profile values into a
  ``CourseSpecification``.
* ``get_pre_populated_fields(spec)`` — List fields pre-populated by a profile.
* ``CourseSpecification`` — Re-exported from ``src.models`` for convenience.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from src.models import CourseSpecification


# ---------------------------------------------------------------------------
# Pydantic Models — Profile Validation
# ---------------------------------------------------------------------------


class CohortProfile(BaseModel):
    """Schema for a YAML cohort profile.

    Validates the top-level structure of a profile file.  Detailed
    sub-sections (tech_stack, kerntaken_emphasis, assessment) are kept
    as flexible ``dict`` fields because their shapes vary per cohort.
    """

    profile: dict[str, str] = Field(
        default_factory=dict,
        description="Profile metadata: name, description.",
    )
    year_level: int | None = Field(
        default=None, ge=1, le=3, description="Student year level (1–3).",
    )
    student_pathway: str | None = Field(
        default=None, description="Pathway: 'BOL' or 'BBL'.",
    )
    bpv_readiness: str | None = Field(
        default=None, description="BPV readiness: 'pre-bpv' or 'in-bpv'.",
    )
    grading_scale: str | dict | None = Field(
        default=None, description="Grading scale, e.g. 'OVG' or a dict.",
    )
    hardware_constraints: str | dict | None = Field(
        default=None, description="Hardware constraint description or dict.",
    )
    tech_stack: dict = Field(
        default_factory=dict, description="Technology stack configuration.",
    )
    kerntaken_emphasis: dict[str, str] = Field(
        default_factory=dict, description="Kerntaken emphasis: P1-K1 through P4-K1.",
    )
    assessment: dict = Field(
        default_factory=dict, description="Assessment configuration.",
    )


# ---------------------------------------------------------------------------
# YAML Loading
# ---------------------------------------------------------------------------


def load_profile(path: str | Path) -> dict:
    """Load and validate a YAML cohort profile file."""
    profile_path = Path(path)
    if not profile_path.is_absolute():
        profile_path = Path(__file__).resolve().parent.parent / profile_path

    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    if profile_path.suffix.lower() not in (".yaml", ".yml"):
        raise FileNotFoundError(
            f"Profile must be a .yaml or .yml file, got: {profile_path.suffix}"
        )

    try:
        with open(profile_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in profile: {exc}") from exc
    except OSError as exc:
        raise OSError(f"Cannot read profile: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Profile YAML must be a mapping (dict) at the root.")

    # Validate against Pydantic model
    _ = CohortProfile.model_validate(data)

    return data


# ---------------------------------------------------------------------------
# Profile Context Builder
# ---------------------------------------------------------------------------


def _build_profile_context_string(profile: dict) -> str:
    """Serialize all profile sections into a structured text block."""
    parts: list[str] = []

    prof = profile.get("profile", {})
    if prof:
        name = prof.get("name", "")
        desc = prof.get("description", "")
        if name:
            parts.append(f"Cohort: {name}")
        if desc:
            parts.append(f"Cohort Description: {desc.strip()}")

    bpv = profile.get("bpv_readiness")
    if bpv:
        parts.append(f"BPV Readiness: {bpv}")

    tech = profile.get("tech_stack", {})
    if tech:
        lines = ["Tech Stack:"]
        for key in (
            "primary_language", "framework", "frontend", "database",
            "version_control", "editor", "ci_cd", "containerisation", "deployment",
        ):
            val = tech.get(key)
            if val:
                lines.append(f"  - {key}: {val}")
        testing = tech.get("testing", {})
        if testing:
            tf = testing.get("framework", "")
            tc = testing.get("coverage_target", "")
            if tf:
                lines.append(f"  - testing_framework: {tf}")
            if tc:
                lines.append(f"  - testing_coverage_target: {tc}")
        parts.append("\n".join(lines))

    ke = profile.get("kerntaken_emphasis", {})
    if ke:
        lines = ["Kerntaken Emphasis:"]
        for kt in ("P1-K1", "P2-K1", "P3-K1", "P4-K1"):
            val = ke.get(kt)
            if val:
                lines.append(f"  - {kt}: {val}")
        parts.append("\n".join(lines))

    assess = profile.get("assessment", {})
    if assess:
        lines = ["Assessment:"]
        for key in (
            "practical_exams", "portfolio_items",
            "proeve_preparation", "code_reviews_per_semester",
        ):
            val = assess.get(key)
            if val is not None:
                lines.append(f"  - {key}: {val}")
        parts.append("\n".join(lines))

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Profile Injection
# ---------------------------------------------------------------------------


def inject_profile(
    spec: CourseSpecification,
    profile: dict,
) -> CourseSpecification:
    """Inject pre-populated profile values into a CourseSpecification."""
    for key in ("grading_scale", "student_pathway", "year_level", "hardware_constraints"):
        if key in profile and profile[key] is not None:
            current = getattr(spec, key)
            if current is None:
                setattr(spec, key, profile[key])

    if spec.primary_language == "" or spec.primary_language == "Python":
        tech = profile.get("tech_stack", {})
        pl = tech.get("primary_language")
        if pl:
            spec.primary_language = pl

    profile_context = _build_profile_context_string(profile)
    if profile_context:
        if spec.course_context:
            spec.course_context = profile_context + "\n\n" + spec.course_context
        else:
            spec.course_context = profile_context

    return spec


def get_pre_populated_fields(spec: CourseSpecification) -> list[str]:
    """Return a list of field names that are pre-populated (non-None)."""
    fields: list[str] = []
    for key in ("grading_scale", "student_pathway", "year_level", "hardware_constraints"):
        if getattr(spec, key) is not None:
            fields.append(key)
    return fields

    return data