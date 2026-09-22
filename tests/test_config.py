"""
test_config.py — Tests for the Config System & --profile Flag (Issue #8)
========================================================================

Validates YAML parsing, schema validation, profile injection into
CourseSpecification, CLI flag handling, and Intake Specialist skip
behaviour for pre-populated fields.

.. rubric:: Issue #16 — Layer 1 Architecture Enforcement

All config-loading functions MUST be importable from a Layer 1 module
(``src.config_loader``), **not** from ``src.main`` (Layer 4).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.main import (
    CourseSpecification,
    _get_pre_populated_fields,
    _inject_profile,
    _load_profile,
)

# ═══════════════════════════════════════════════════════════════════════
# Issue #16 RED PHASE — Architecture Enforcement Tests
# These tests MUST fail until src.config_loader exists and is Layer 1.
# ═══════════════════════════════════════════════════════════════════════


class TestLayer1ConfigLoaderExists:
    """Verify that config-loading primitives live in Layer 1."""

    def test_load_profile_importable_from_config_loader(self) -> None:
        """``load_profile`` MUST be importable from src.config_loader (Layer 1)."""
        from src.config_loader import load_profile

        assert callable(load_profile)

    def test_inject_profile_importable_from_config_loader(self) -> None:
        """``inject_profile`` MUST be importable from src.config_loader (Layer 1)."""
        from src.config_loader import inject_profile

        assert callable(inject_profile)

    def test_get_pre_populated_fields_importable_from_config_loader(self) -> None:
        """``get_pre_populated_fields`` MUST be importable from src.config_loader (Layer 1)."""
        from src.config_loader import get_pre_populated_fields

        assert callable(get_pre_populated_fields)

    def test_config_loader_has_no_layer4_imports(self) -> None:
        """src.config_loader MUST NOT import from src.main (Layer 4)."""
        import ast
        from pathlib import Path

        config_loader_path = Path(__file__).resolve().parent.parent / "src" / "config_loader.py"
        assert config_loader_path.exists(), "src/config_loader.py (Layer 1) must exist"
        source = config_loader_path.read_text()
        tree = ast.parse(source)
        layer4_imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "main" in alias.name.lower():
                        layer4_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and "src.main" in node.module:
                    layer4_imports.append(node.module)
        assert not layer4_imports, (
            f"src/config_loader.py (Layer 1) MUST NOT import from "
            f"Layer 4 (src.main). Found: {layer4_imports}"
        )

    def test_course_specification_importable_from_config_loader(self) -> None:
        """CourseSpecification model MUST be re-exportable from config_loader."""
        from src.config_loader import CourseSpecification

        spec = CourseSpecification(
            course_context="test",
            primary_language="Python",
            grading_scale="OVG",
        )
        assert spec.grading_scale == "OVG"

    def test_cohort_profile_pydantic_model_exists(self) -> None:
        """src.config_loader MUST expose a CohortProfile Pydantic model."""
        # Verify it's a Pydantic BaseModel
        from pydantic import BaseModel

        from src.config_loader import CohortProfile

        assert issubclass(CohortProfile, BaseModel), "CohortProfile must be a Pydantic BaseModel"


# ---------------------------------------------------------------------------
# Paths to the real config fixtures shipped with the project
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCHOOL_DEFAULTS = _PROJECT_ROOT / "config" / "school_defaults.yaml"
_PROGRAM1_PROFILE = _PROJECT_ROOT / "config" / "profiles" / "program1_profile.yaml"


# ===================================================================
# CourseSpecification — Optional fields
# ===================================================================


class TestCourseSpecificationOptionalFields:
    """Validate the new Optional fields on CourseSpecification."""

    def test_default_none_for_optional_fields(self) -> None:
        """All new Optional fields default to None."""
        spec = CourseSpecification(
            course_context="ctx",
            primary_language="Python",
        )
        assert spec.grading_scale is None
        assert spec.student_pathway is None
        assert spec.year_level is None
        assert spec.hardware_constraints is None

    def test_grading_scale_can_be_set(self) -> None:
        """grading_scale accepts 'OVG'."""
        spec = CourseSpecification(
            course_context="ctx",
            primary_language="Python",
            grading_scale="OVG",
        )
        assert spec.grading_scale == "OVG"

    def test_student_pathway_can_be_set(self) -> None:
        """student_pathway accepts 'BOL'."""
        spec = CourseSpecification(
            course_context="ctx",
            primary_language="Python",
            student_pathway="BOL",
        )
        assert spec.student_pathway == "BOL"

    def test_year_level_boundaries(self) -> None:
        """year_level enforces 1 <= value <= 3."""
        # Valid
        spec = CourseSpecification(
            course_context="ctx",
            primary_language="Python",
            year_level=2,
        )
        assert spec.year_level == 2

        # Invalid: too low
        with pytest.raises(ValidationError):
            CourseSpecification(
                course_context="ctx",
                primary_language="Python",
                year_level=0,
            )

        # Invalid: too high
        with pytest.raises(ValidationError):
            CourseSpecification(
                course_context="ctx",
                primary_language="Python",
                year_level=4,
            )

    def test_hardware_constraints_can_be_set(self) -> None:
        """hardware_constraints accepts a multi-line string."""
        hw = "BYOD laptop, 8GB RAM, Chromebook fallback"
        spec = CourseSpecification(
            course_context="ctx",
            primary_language="Python",
            hardware_constraints=hw,
        )
        assert spec.hardware_constraints == hw

    def test_all_optional_fields_together(self) -> None:
        """All four optional fields can be set simultaneously."""
        spec = CourseSpecification(
            course_context="Full context",
            primary_language="PHP",
            grading_scale="OVG",
            student_pathway="BOL",
            year_level=2,
            hardware_constraints="BYOD laptops",
        )
        assert spec.grading_scale == "OVG"
        assert spec.student_pathway == "BOL"
        assert spec.year_level == 2
        assert spec.hardware_constraints == "BYOD laptops"
        assert spec.primary_language == "PHP"

    def test_serialization_includes_optional_none_fields(self) -> None:
        """serialization includes Optional fields as None."""
        spec = CourseSpecification(
            course_context="ctx",
            primary_language="Python",
        )
        data = spec.model_dump()
        assert data["grading_scale"] is None


# ===================================================================
# YAML Parsing — school_defaults.yaml
# ===================================================================


class TestSchoolDefaultsYaml:
    """Validate the structure and values of config/school_defaults.yaml."""

    def test_file_exists(self) -> None:
        """school_defaults.yaml is present."""
        assert _SCHOOL_DEFAULTS.exists(), f"Missing: {_SCHOOL_DEFAULTS}"

    def test_is_valid_yaml(self) -> None:
        """The file parses as valid YAML."""
        with open(_SCHOOL_DEFAULTS, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert isinstance(data, dict)

    def test_has_organisation_section(self) -> None:
        """Contains an organisation section."""
        with open(_SCHOOL_DEFAULTS, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert "organisation" in data
        org = data["organisation"]
        assert "crebonummer" in org
        assert org["crebonummer"] == "25604"

    def test_has_grading_scale_ovg(self) -> None:
        """The grading scale is OVG with O, V, G levels."""
        with open(_SCHOOL_DEFAULTS, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        gs = data["grading_scale"]
        assert gs["type"] == "OVG"
        keys = [level["key"] for level in gs["levels"]]
        assert keys == ["O", "V", "G"]

    def test_has_hardware_constraints(self) -> None:
        """Contains hardware_constraints with BYOD info."""
        with open(_SCHOOL_DEFAULTS, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        hw = data["hardware_constraints"]
        assert "BYOD" in hw["student_device"]
        assert hw["min_ram_gb"] >= 8

    def test_has_four_kerntaken(self) -> None:
        """Contains exactly 4 kerntaken: P1-K1 through P4-K1."""
        with open(_SCHOOL_DEFAULTS, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        kt = data["kerntaken"]
        assert len(kt) == 4
        ids = [k["id"] for k in kt]
        assert ids == ["P1-K1", "P2-K1", "P3-K1", "P4-K1"]

    def test_has_student_pathways(self) -> None:
        """Contains BOL and BBL pathway definitions."""
        with open(_SCHOOL_DEFAULTS, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        sp = data["student_pathways"]
        assert "BOL" in sp
        assert "BBL" in sp
        assert sp["BOL"]["min_bpv_hours"] == 900
        assert sp["BBL"]["min_bpv_hours"] == 1350

    def test_has_year_levels(self) -> None:
        """Contains year levels 1, 2, 3."""
        with open(_SCHOOL_DEFAULTS, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert 1 in data["year_levels"]
        assert 2 in data["year_levels"]
        assert 3 in data["year_levels"]


# ===================================================================
# YAML Parsing — program1_profile.yaml
# ===================================================================


class TestProgram1ProfileYaml:
    """Validate the structure and values of config/profiles/program1_profile.yaml."""

    def test_file_exists(self) -> None:
        """program1_profile.yaml is present."""
        assert _PROGRAM1_PROFILE.exists(), f"Missing: {_PROGRAM1_PROFILE}"

    def test_is_valid_yaml(self) -> None:
        """The file parses as valid YAML."""
        with open(_PROGRAM1_PROFILE, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert isinstance(data, dict)

    def test_has_profile_metadata(self) -> None:
        """Contains a profile section with name and description."""
        with open(_PROGRAM1_PROFILE, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        p = data["profile"]
        assert "name" in p
        assert "PHP" in p["name"]

    def test_year_level_is_2(self) -> None:
        """Profile specifies year_level: 2."""
        with open(_PROGRAM1_PROFILE, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert data["year_level"] == 2

    def test_student_pathway_is_BOL(self) -> None:
        """Profile specifies student_pathway: BOL."""
        with open(_PROGRAM1_PROFILE, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert data["student_pathway"] == "BOL"

    def test_grading_scale_is_OVG(self) -> None:
        """Profile specifies grading_scale: OVG."""
        with open(_PROGRAM1_PROFILE, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert data["grading_scale"] == "OVG"

    def test_has_hardware_constraints(self) -> None:
        """Profile contains hardware_constraints string."""
        with open(_PROGRAM1_PROFILE, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert "BYOD" in data["hardware_constraints"]

    def test_tech_stack_has_php_laravel(self) -> None:
        """Tech stack specifies PHP and Laravel 11."""
        with open(_PROGRAM1_PROFILE, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        ts = data["tech_stack"]
        assert ts["primary_language"] == "PHP"
        assert "Laravel" in ts["framework"]

    def test_has_bpv_readiness(self) -> None:
        """Profile specifies bpv_readiness."""
        with open(_PROGRAM1_PROFILE, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert data["bpv_readiness"] == "bpv-ready"

    def test_has_kerntaken_emphasis(self) -> None:
        """Profile defines kerntaken emphasis for all four."""
        with open(_PROGRAM1_PROFILE, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        ke = data["kerntaken_emphasis"]
        assert "P1-K1" in ke
        assert "P2-K1" in ke
        assert "P3-K1" in ke
        assert "P4-K1" in ke

    def test_has_schedule_and_assessment(self) -> None:
        """Profile contains assessment section."""
        with open(_PROGRAM1_PROFILE, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert "assessment" in data


# ===================================================================
# _load_profile — YAML loading with validation
# ===================================================================


class TestLoadProfile:
    """Tests for the _load_profile helper function."""

    def test_loads_valid_profile(self) -> None:
        """_load_profile returns a dict for a valid YAML file."""
        data = _load_profile(str(_PROGRAM1_PROFILE))
        assert isinstance(data, dict)
        assert "year_level" in data

    def test_raises_system_exit_for_missing_file(self) -> None:
        """_load_profile exits with 1 for a non-existent file."""
        with pytest.raises(SystemExit) as exc:
            _load_profile("config/profiles/nonexistent.yaml")
        assert exc.value.code == 1

    def test_raises_system_exit_for_non_yaml_extension(self) -> None:
        """_load_profile exits with 1 for .txt files."""
        tmp = _PROJECT_ROOT / "config" / "profiles" / "_test.txt"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp.write_text("hello: world")
            with pytest.raises(SystemExit) as exc:
                _load_profile(str(tmp))
            assert exc.value.code == 1
        finally:
            tmp.unlink(missing_ok=True)

    def test_raises_system_exit_for_invalid_yaml(self) -> None:
        """_load_profile exits with 1 for malformed YAML."""
        tmp = _PROJECT_ROOT / "config" / "profiles" / "_bad.yaml"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp.write_text(": bad : yaml: [")
            with pytest.raises(SystemExit) as exc:
                _load_profile(str(tmp))
            assert exc.value.code == 1
        finally:
            tmp.unlink(missing_ok=True)

    def test_raises_system_exit_for_non_dict_root(self) -> None:
        """_load_profile exits with 1 when root is not a dict."""
        tmp = _PROJECT_ROOT / "config" / "profiles" / "_list.yaml"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp.write_text("- item1\n- item2\n")
            with pytest.raises(SystemExit) as exc:
                _load_profile(str(tmp))
            assert exc.value.code == 1
        finally:
            tmp.unlink(missing_ok=True)

    def test_resolves_relative_paths_against_project_root(self) -> None:
        """Relative paths are resolved against _PROJECT_ROOT."""
        data = _load_profile("config/profiles/program1_profile.yaml")
        assert isinstance(data, dict)


# ===================================================================
# _inject_profile — Model injection from profile
# ===================================================================


class TestInjectProfile:
    """Tests for _inject_profile function."""

    @pytest.fixture
    def loaded_profile(self) -> dict:
        """Load the real program1_profile.yaml."""
        return _load_profile(str(_PROGRAM1_PROFILE))

    @pytest.fixture
    def empty_spec(self) -> CourseSpecification:
        """Empty spec with default values."""
        return CourseSpecification(course_context="", primary_language="")

    def test_injects_grading_scale(self, empty_spec, loaded_profile) -> None:
        """grading_scale is injected from profile."""
        spec = _inject_profile(empty_spec, loaded_profile)
        assert spec.grading_scale == "OVG"

    def test_injects_student_pathway(self, empty_spec, loaded_profile) -> None:
        """student_pathway is injected from profile."""
        spec = _inject_profile(empty_spec, loaded_profile)
        assert spec.student_pathway == "BOL"

    def test_injects_year_level(self, empty_spec, loaded_profile) -> None:
        """year_level is injected from profile."""
        spec = _inject_profile(empty_spec, loaded_profile)
        assert spec.year_level == 2

    def test_injects_hardware_constraints(self, empty_spec, loaded_profile) -> None:
        """hardware_constraints is injected from profile."""
        spec = _inject_profile(empty_spec, loaded_profile)
        assert "BYOD" in spec.hardware_constraints

    def test_injects_primary_language_from_tech_stack(self, empty_spec, loaded_profile) -> None:
        """primary_language is extracted from tech_stack.primary_language."""
        spec = _inject_profile(empty_spec, loaded_profile)
        assert spec.primary_language == "PHP"

    def test_does_not_overwrite_existing_values(self, loaded_profile) -> None:
        """Already-set spec fields are not overwritten by profile."""
        spec = CourseSpecification(
            course_context="ctx",
            primary_language="JavaScript",
            grading_scale="A-F",
            year_level=1,
        )
        spec = _inject_profile(spec, loaded_profile)
        assert spec.grading_scale == "A-F"
        assert spec.year_level == 1
        assert spec.student_pathway == "BOL"
        assert "BYOD" in spec.hardware_constraints

    def test_does_not_overwrite_non_default_primary_language(self, loaded_profile) -> None:
        """primary_language preserved when already set to a real value."""
        spec = CourseSpecification(
            course_context="ctx",
            primary_language="TypeScript",
        )
        spec = _inject_profile(spec, loaded_profile)
        assert spec.primary_language == "TypeScript"

    def test_does_overwrite_default_primary_language(self, loaded_profile) -> None:
        """primary_language overwritten when set to 'Python' (default)."""
        spec = CourseSpecification(
            course_context="ctx",
            primary_language="Python",
        )
        spec = _inject_profile(spec, loaded_profile)
        assert spec.primary_language == "PHP"

    def test_handles_missing_tech_stack(self, empty_spec) -> None:
        """Does not crash when profile has no tech_stack key."""
        profile = {"year_level": 3}
        spec = _inject_profile(empty_spec, profile)
        assert spec.year_level == 3
        assert spec.grading_scale is None

    def test_handles_empty_profile(self, empty_spec) -> None:
        """Does not modify spec when profile dict is empty."""
        spec = _inject_profile(empty_spec, {})
        assert spec.grading_scale is None
        assert spec.student_pathway is None
        assert spec.year_level is None
        assert spec.hardware_constraints is None
        assert spec.primary_language == ""

    def test_does_not_crash_on_none_values_in_profile(self, empty_spec) -> None:
        """None values in profile are skipped gracefully."""
        profile = {
            "grading_scale": None,
            "student_pathway": None,
            "year_level": 2,
        }
        spec = _inject_profile(empty_spec, profile)
        assert spec.grading_scale is None
        assert spec.student_pathway is None
        assert spec.year_level == 2


# ===================================================================
# _get_pre_populated_fields — Helper
# ===================================================================


class TestGetPrePopulatedFields:
    """Tests for _get_pre_populated_fields."""

    def test_returns_empty_list_when_nothing_set(self) -> None:
        """Empty list when all Optional fields are None."""
        spec = CourseSpecification(
            course_context="ctx",
            primary_language="Python",
        )
        assert _get_pre_populated_fields(spec) == []

    def test_returns_all_four_when_all_set(self) -> None:
        """All four field names returned when all set."""
        spec = CourseSpecification(
            course_context="ctx",
            primary_language="Python",
            grading_scale="OVG",
            student_pathway="BOL",
            year_level=2,
            hardware_constraints="BYOD",
        )
        fields = _get_pre_populated_fields(spec)
        assert sorted(fields) == sorted(
            [
                "grading_scale",
                "student_pathway",
                "year_level",
                "hardware_constraints",
            ]
        )

    def test_returns_partial_list(self) -> None:
        """Only actually-set fields are returned."""
        spec = CourseSpecification(
            course_context="ctx",
            primary_language="Python",
            year_level=1,
        )
        fields = _get_pre_populated_fields(spec)
        assert fields == ["year_level"]

    def test_primary_language_not_in_list(self) -> None:
        """primary_language and course_context are NOT in the list."""
        spec = CourseSpecification(
            course_context="ctx",
            primary_language="PHP",
            year_level=3,
            grading_scale="OVG",
        )
        fields = _get_pre_populated_fields(spec)
        assert "primary_language" not in fields
        assert "course_context" not in fields
        assert "year_level" in fields
        assert "grading_scale" in fields


# ===================================================================
# _run_intake — pre-populated skip behaviour (pure helpers, no LLM)
# ===================================================================


class TestRunIntakeSkipBehaviour:
    """Test the pure-function helpers that drive skip behaviour in _run_intake.

    Since _run_intake calls out to the Intake Specialist LLM (CrewAI),
    we test the helpers that determine skip behaviour:
    _get_pre_populated_fields and _inject_profile.
    """

    def test_skip_section_is_empty_when_no_pre_populated(self) -> None:
        """_get_pre_populated_fields returns [] for empty spec."""
        spec = CourseSpecification(course_context="x", primary_language="y")
        assert _get_pre_populated_fields(spec) == []

    def test_skip_section_contains_all_four_fields(self) -> None:
        """All four fields appear in the pre-populated list when set."""
        spec = CourseSpecification(
            course_context="x",
            primary_language="y",
            grading_scale="OVG",
            student_pathway="BOL",
            year_level=2,
            hardware_constraints="BYOD",
        )
        fields = _get_pre_populated_fields(spec)
        assert len(fields) == 4

    def test_partial_skip_when_only_some_fields_set(self) -> None:
        """Only two fields are returned when only two are set."""
        spec = CourseSpecification(
            course_context="x",
            primary_language="y",
            student_pathway="BBL",
            hardware_constraints="School devices only",
        )
        fields = _get_pre_populated_fields(spec)
        assert sorted(fields) == ["hardware_constraints", "student_pathway"]

    def test_injected_fields_are_tracked_by_get_pre_populated(self) -> None:
        """After _inject_profile, _get_pre_populated_fields sees them."""
        spec = CourseSpecification(course_context="", primary_language="")
        profile = {
            "grading_scale": "OVG",
            "student_pathway": "BOL",
            "year_level": 2,
            "hardware_constraints": "BYOD",
            "tech_stack": {"primary_language": "PHP"},
        }
        spec = _inject_profile(spec, profile)
        fields = _get_pre_populated_fields(spec)
        assert len(fields) == 4


# ===================================================================
# Integration: end-to-end profile loading & injection
# ===================================================================


class TestEndToEndProfileFlow:
    """End-to-end tests for the profile loading and injection flow."""

    def test_full_profile_load_and_inject(self) -> None:
        """Load real profile, inject, verify all fields."""
        data = _load_profile(str(_PROGRAM1_PROFILE))
        spec = CourseSpecification(course_context="", primary_language="")
        spec = _inject_profile(spec, data)

        assert spec.grading_scale == "OVG"
        assert spec.student_pathway == "BOL"
        assert spec.year_level == 2
        assert "BYOD" in spec.hardware_constraints
        assert spec.primary_language == "PHP"

    def test_school_defaults_loads_without_error(self) -> None:
        """school_defaults.yaml loads and has all required sections."""
        data = _load_profile(str(_SCHOOL_DEFAULTS))
        assert "organisation" in data
        assert "grading_scale" in data
        assert "hardware_constraints" in data
        assert "kerntaken" in data
        assert "student_pathways" in data
        assert "year_levels" in data

    def test_profile_values_are_correct_types(self) -> None:
        """Profile values have the expected Python types after YAML parsing."""
        data = _load_profile(str(_PROGRAM1_PROFILE))
        assert isinstance(data["year_level"], int)
        assert isinstance(data["grading_scale"], str)
        assert isinstance(data["student_pathway"], str)
        assert isinstance(data["hardware_constraints"], str)
        assert isinstance(data["tech_stack"], dict)
        assert isinstance(data["tech_stack"]["primary_language"], str)

    def test_injected_spec_serializes_correctly(self) -> None:
        """Injected spec model_dump() contains all expected fields."""
        data = _load_profile(str(_PROGRAM1_PROFILE))
        spec = CourseSpecification(course_context="", primary_language="")
        spec = _inject_profile(spec, data)

        dump = spec.model_dump()
        assert dump["grading_scale"] == "OVG"
        assert dump["student_pathway"] == "BOL"
        assert dump["year_level"] == 2
        assert "BYOD" in dump["hardware_constraints"]
        assert dump["primary_language"] == "PHP"

    def test_deserialize_round_trip(self) -> None:
        """Spec can go through dict round-trip without data loss."""
        spec1 = CourseSpecification(
            course_context="Test context",
            primary_language="Java",
            grading_scale="OVG",
            student_pathway="BBL",
            year_level=3,
            hardware_constraints="Thin clients only",
        )
        dump = spec1.model_dump()
        spec2 = CourseSpecification.model_validate(dump)
        assert spec2.course_context == spec1.course_context
        assert spec2.primary_language == spec1.primary_language
        assert spec2.grading_scale == spec1.grading_scale
        assert spec2.student_pathway == spec1.student_pathway
        assert spec2.year_level == spec1.year_level
        assert spec2.hardware_constraints == spec1.hardware_constraints
