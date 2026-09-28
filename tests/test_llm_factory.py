"""
test_llm_factory.py — Unit tests for the shared LLM factory
============================================================

Issue #5: Per-Agent Model Configuration — Specialized LLMs for Each Agent

Validates the 4-tier fallback chain implemented by
:func:`src.llm_factory.build_llm_for_agent`:

    1. Per-agent override   -> ``AGENT_{ROLE}_{PROPERTY}``
    2. Agent-wide default   -> ``AGENT_DEFAULT_{PROPERTY}``
    3. Legacy globals       -> ``OPENROUTER_MODEL`` / ``AGENT_TEMPERATURE`` /
                               ``AGENT_MAX_TOKENS`` (deprecated, kept for
                               backward compatibility)
    4. Hardcoded defaults   -> values baked into ``llm_factory.py``

Every test isolates the environment with ``unittest.mock.patch.dict`` using
``clear=True`` so that the ``.env`` values loaded at import time (via
``load_dotenv``) cannot leak into the assertions.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from src.llm_factory import (
    CURRICULUM_ARCHITECT,
    EDUCATION_DIRECTOR,
    LAB_DEVELOPER,
    OUTPUT_EXPORTER,
    QA_REVIEWER,
    THEORY_INSTRUCTOR,
    ConfigIssue,
    audit_agent_config,
    audit_agent_configs,
    build_llm_for_agent,
    format_config_issues,
    get_effective_config,
    has_fatal_config_issues,
)

# ---------------------------------------------------------------------------
# Expected hardcoded defaults (tier 4) — mirrored from src/llm_factory.py.
# NOTE: crewai.LLM strips the "openrouter/" provider prefix from model strings
# (e.g. "openrouter/deepseek/deepseek-v4-pro" becomes "deepseek/deepseek-v4-pro").
# Tests that inspect the raw resolved value use get_effective_config(); tests that
# inspect the constructed LLM use the stripped form.
# ---------------------------------------------------------------------------
HARDCODED_MODEL: str = "openrouter/deepseek/deepseek-v4-pro"
HARDCODED_MODEL_STRIPPED: str = "deepseek/deepseek-v4-pro"
HARDCODED_TEMPERATURE: float = 0.2
HARDCODED_TOP_P: float = 0.1
HARDCODED_MAX_TOKENS: int = 32768
EXPECTED_DEFAULT_BASE_URL: str = "https://openrouter.ai/api/v1"

_DUMMY_API_KEY: str = "sk-test-dummy-key-for-unit-tests"
_BASE_ENV: dict[str, str] = {"OPENROUTER_API_KEY": _DUMMY_API_KEY}

ALL_ROLES: tuple[str, ...] = (
    CURRICULUM_ARCHITECT,
    LAB_DEVELOPER,
    OUTPUT_EXPORTER,
    EDUCATION_DIRECTOR,
)

# ═══════════════════════════════════════════════════════════════════════
# Issue #3 RED PHASE — Model-Agnosticism Enforcement
# These tests MUST fail until hardcoded provider URLs are purged.
# ═══════════════════════════════════════════════════════════════════════


class TestModelAgnosticBaseUrl:
    """Verify the LLM factory reads base_url from env, not hardcoded constants."""

    def test_no_hardcoded_openrouter_url_in_source(self) -> None:
        """src/llm_factory.py uses env var for base URL (fallbacks allowed)."""
        from pathlib import Path

        factory_path = Path(__file__).resolve().parent.parent / "src" / "llm_factory.py"
        source = factory_path.read_text()

        # The fallback default inside os.getenv() is acceptable.
        # But there should be exactly ONE such URL string — the fallback.
        # Count occurrences of the OpenRouter URL:
        occurrences = source.count("https://openrouter.ai/api/v1")
        assert occurrences == 1, (
            f"Expected exactly 1 fallback URL (in os.getenv), "
            f"found {occurrences}. All other references must use _BASE_URL."
        )

        # Verify BASE_URL env var is referenced
        assert "BASE_URL" in source, (
            "src/llm_factory.py must reference the BASE_URL environment variable"
        )

    def test_base_url_env_var_overrides_default(self) -> None:
        """BASE_URL env var MUST override the default base URL."""
        custom_url = "https://custom-llm-proxy.example.com/v1"
        with patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "sk-test", "BASE_URL": custom_url},
            clear=True,
        ):
            config = get_effective_config(CURRICULUM_ARCHITECT)
            assert config["base_url"] == custom_url, (
                f"BASE_URL env var not respected. Expected {custom_url}, got {config['base_url']}"
            )

    def test_api_key_env_var_fallback_works(self) -> None:
        """API_KEY env var should also be checked as fallback for auth."""
        with patch.dict(
            os.environ,
            {"API_KEY": "sk-from-generic"},
            clear=True,
        ):
            config = get_effective_config(CURRICULUM_ARCHITECT)
            assert config["api_key_status"] == "set", (
                "API_KEY env var should be recognised as an auth credential"
            )


def _env(*extra: dict[str, str]) -> dict[str, str]:
    """Merge *extra* dicts on top of the base env (API key only)."""
    result: dict[str, str] = dict(_BASE_ENV)
    for d in extra:
        result.update(d)
    return result


# ---------------------------------------------------------------------------
# Model fallback chain
# ---------------------------------------------------------------------------


class TestModelFallbackChain:
    """Tests for the MODEL property's 4-tier fallback chain."""

    def test_per_agent_override_takes_highest_priority(self) -> None:
        """Tier 1: AGENT_{ROLE}_MODEL wins over default & legacy values."""
        env = _env(
            {
                "AGENT_CURRICULUM_ARCHITECT_MODEL": "openai/per-agent-model",
                "AGENT_DEFAULT_MODEL": "openai/default-model",
                "OPENROUTER_MODEL": "openai/legacy-model",
            }
        )
        with patch.dict(os.environ, env, clear=True):
            config = get_effective_config(CURRICULUM_ARCHITECT)
            assert config["model"] == "openai/per-agent-model"

    def test_fallback_to_agent_default_model(self) -> None:
        """Tier 2: AGENT_DEFAULT_MODEL used when no per-agent override set."""
        env = _env(
            {
                "AGENT_DEFAULT_MODEL": "openai/default-model",
                "OPENROUTER_MODEL": "openai/legacy-model",
            }
        )
        with patch.dict(os.environ, env, clear=True):
            config = get_effective_config(CURRICULUM_ARCHITECT)
            assert config["model"] == "openai/default-model"

    def test_fallback_to_hardcoded_default_model(self) -> None:
        """Tier 3: hardcoded default used when nothing is configured."""
        with patch.dict(os.environ, _env(), clear=True):
            config = get_effective_config(CURRICULUM_ARCHITECT)
            assert config["model"] == HARDCODED_MODEL


# ---------------------------------------------------------------------------
# Numeric fallback chain
# ---------------------------------------------------------------------------


class TestNumericFallbackChain:
    """Temperature and max_tokens follow the exact same chain as MODEL."""

    def test_temperature_follows_fallback_chain(self) -> None:
        # Tier 1: per-agent override
        with patch.dict(
            os.environ,
            _env(
                {
                    "AGENT_CURRICULUM_ARCHITECT_TEMPERATURE": "0.7",
                    "AGENT_DEFAULT_TEMPERATURE": "0.5",
                }
            ),
            clear=True,
        ):
            assert build_llm_for_agent(CURRICULUM_ARCHITECT).temperature == 0.7

        # Tier 2: agent-wide default
        with patch.dict(
            os.environ,
            _env({"AGENT_DEFAULT_TEMPERATURE": "0.5"}),
            clear=True,
        ):
            assert build_llm_for_agent(CURRICULUM_ARCHITECT).temperature == 0.5

        # Tier 3: hardcoded default
        with patch.dict(os.environ, _env(), clear=True):
            assert build_llm_for_agent(CURRICULUM_ARCHITECT).temperature == HARDCODED_TEMPERATURE

    def test_max_tokens_follows_fallback_chain(self) -> None:
        # Tier 1: per-agent override
        with patch.dict(
            os.environ,
            _env(
                {
                    "AGENT_CURRICULUM_ARCHITECT_MAX_TOKENS": "2048",
                    "AGENT_DEFAULT_MAX_TOKENS": "4096",
                }
            ),
            clear=True,
        ):
            assert build_llm_for_agent(CURRICULUM_ARCHITECT).max_tokens == 2048

        # Tier 2: agent-wide default
        with patch.dict(
            os.environ,
            _env({"AGENT_DEFAULT_MAX_TOKENS": "4096"}),
            clear=True,
        ):
            assert build_llm_for_agent(CURRICULUM_ARCHITECT).max_tokens == 4096

        # Tier 3: hardcoded default
        with patch.dict(os.environ, _env(), clear=True):
            assert build_llm_for_agent(CURRICULUM_ARCHITECT).max_tokens == HARDCODED_MAX_TOKENS


# ---------------------------------------------------------------------------
# Build LLM with no environment
# ---------------------------------------------------------------------------


class TestBuildLLMNoEnvironment:
    """build_llm_for_agent always returns a usable LLM with only API key set."""

    def test_returns_llm_with_all_defaults_when_only_api_key_set(self) -> None:
        with patch.dict(os.environ, _env(), clear=True):
            llm = build_llm_for_agent(CURRICULUM_ARCHITECT)
            assert llm is not None
            assert llm.model == HARDCODED_MODEL_STRIPPED
            assert llm.temperature == HARDCODED_TEMPERATURE
            assert llm.top_p == HARDCODED_TOP_P
            assert llm.max_tokens == HARDCODED_MAX_TOKENS
            assert llm.base_url == EXPECTED_DEFAULT_BASE_URL

    def test_all_known_agents_build_for_every_role(self) -> None:
        with patch.dict(os.environ, _env(), clear=True):
            for role in ALL_ROLES:
                llm = build_llm_for_agent(role)
                assert llm is not None, f"LLM should not be None for role {role}"
                assert llm.base_url == EXPECTED_DEFAULT_BASE_URL


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestAgentDefaultFallback:
    """AGENT_DEFAULT_* vars drive all agents without per-agent overrides."""

    def test_agent_default_drives_all_agents_without_per_agent_vars(self) -> None:
        env = _env(
            {
                "AGENT_DEFAULT_MODEL": "openai/default-model",
                "AGENT_DEFAULT_TEMPERATURE": "0.4",
                "AGENT_DEFAULT_TOP_P": "0.2",
                "AGENT_DEFAULT_MAX_TOKENS": "4096",
            }
        )
        with patch.dict(os.environ, env, clear=True):
            for role in ALL_ROLES:
                llm = build_llm_for_agent(role)
                assert llm.model == "default-model", f"Failed for role {role}"
                assert llm.temperature == 0.4, f"Failed for role {role}"
                assert llm.top_p == 0.2, f"Failed for role {role}"
                assert llm.max_tokens == 4096, f"Failed for role {role}"

    def test_per_agent_vars_do_not_leak_across_agents(self) -> None:
        """Per-agent override for one role does not affect other roles."""
        env = _env(
            {
                "AGENT_CURRICULUM_ARCHITECT_MODEL": "openai/architect-only",
                "AGENT_DEFAULT_MODEL": "openai/default-model",
            }
        )
        with patch.dict(os.environ, env, clear=True):
            architect = build_llm_for_agent(CURRICULUM_ARCHITECT)
            lab_dev = build_llm_for_agent(LAB_DEVELOPER)
            exporter = build_llm_for_agent(OUTPUT_EXPORTER)

            assert architect.model == "architect-only"
            assert lab_dev.model == "default-model"
            assert exporter.model == "default-model"


# ---------------------------------------------------------------------------
# get_effective_config
# ---------------------------------------------------------------------------


class TestGetEffectiveConfig:
    """get_effective_config() mirrors build_llm_for_agent resolution."""

    def test_effective_config_resolves_the_same_chain(self) -> None:
        env = _env(
            {
                "AGENT_CURRICULUM_ARCHITECT_MODEL": "openai/per-agent-model",
                "AGENT_DEFAULT_TEMPERATURE": "0.6",
            }
        )
        with patch.dict(os.environ, env, clear=True):
            config = get_effective_config(CURRICULUM_ARCHITECT)
            assert config["role"] == CURRICULUM_ARCHITECT
            assert config["model"] == "openai/per-agent-model"
            assert config["temperature"] == 0.6
            assert config["max_tokens"] == HARDCODED_MAX_TOKENS
            assert config["base_url"] == EXPECTED_DEFAULT_BASE_URL

    def test_effective_config_matches_built_llm(self) -> None:
        env = _env({"AGENT_DEFAULT_MODEL": "openai/default-model"})
        with patch.dict(os.environ, env, clear=True):
            config = get_effective_config(LAB_DEVELOPER)
            llm = build_llm_for_agent(LAB_DEVELOPER)
            assert config["model"] == "openai/default-model"
            assert llm.model == "default-model"
            assert llm.temperature == config["temperature"]
            assert llm.top_p == config["top_p"]
            assert llm.max_tokens == config["max_tokens"]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """build_llm_for_agent rejects invalid agent_role inputs."""

    def test_empty_role_raises_value_error(self) -> None:
        with patch.dict(os.environ, _env(), clear=True):
            with pytest.raises(ValueError):
                build_llm_for_agent("")

    def test_non_string_role_raises_value_error(self) -> None:
        with patch.dict(os.environ, _env(), clear=True):
            with pytest.raises(ValueError):
                build_llm_for_agent(None)  # type: ignore[arg-type]

# ---------------------------------------------------------------------------
# Pre-flight configuration audit — early warning before credits are spent
# ---------------------------------------------------------------------------


class TestAuditAgentConfig:
    """The audit must flag the exact misconfigurations that cause NULL content."""

    def test_reasoning_model_with_low_max_tokens_is_fatal(self) -> None:
        env = _env(
            {
                "AGENT_THEORY_INSTRUCTOR_MODEL": "openrouter/anthropic/claude-opus-5.5",
                "AGENT_THEORY_INSTRUCTOR_MAX_TOKENS": "8192",
                "AGENT_THEORY_INSTRUCTOR_TOP_P": "0.9",
            }
        )
        with patch.dict(os.environ, env, clear=True):
            issues = audit_agent_config(THEORY_INSTRUCTOR)

        fatal = [i for i in issues if i.severity == "fatal"]
        assert len(fatal) == 1
        assert "claude-opus-5.5" in fatal[0].message
        assert "AGENT_THEORY_INSTRUCTOR_MAX_TOKENS" in fatal[0].fix
        assert has_fatal_config_issues(issues)

    def test_reasoning_model_with_safe_max_tokens_is_clean(self) -> None:
        env = _env(
            {
                "AGENT_THEORY_INSTRUCTOR_MODEL": "openrouter/anthropic/claude-opus-5.5",
                "AGENT_THEORY_INSTRUCTOR_MAX_TOKENS": "32768",
                "AGENT_THEORY_INSTRUCTOR_TOP_P": "0.9",
            }
        )
        with patch.dict(os.environ, env, clear=True):
            issues = audit_agent_config(THEORY_INSTRUCTOR)

        assert issues == []
        assert not has_fatal_config_issues(issues)

    def test_reasoning_model_with_low_top_p_is_warning_only(self) -> None:
        env = _env(
            {
                "AGENT_THEORY_INSTRUCTOR_MODEL": "openrouter/anthropic/claude-opus-5.5",
                "AGENT_THEORY_INSTRUCTOR_MAX_TOKENS": "32768",
                "AGENT_THEORY_INSTRUCTOR_TOP_P": "0.1",
            }
        )
        with patch.dict(os.environ, env, clear=True):
            issues = audit_agent_config(THEORY_INSTRUCTOR)

        assert len(issues) == 1
        assert issues[0].severity == "warning"
        assert "top_p" in issues[0].message
        assert not has_fatal_config_issues(issues)

    def test_non_reasoning_model_is_not_flagged_for_low_tokens(self) -> None:
        env = _env(
            {
                "AGENT_LAB_DEVELOPER_MODEL": "openrouter/z-ai/glm-5.3",
                "AGENT_LAB_DEVELOPER_MAX_TOKENS": "8192",
                "AGENT_LAB_DEVELOPER_TOP_P": "0.1",
            }
        )
        with patch.dict(os.environ, env, clear=True):
            issues = audit_agent_config(LAB_DEVELOPER)

        assert issues == []

    def test_blank_model_is_fatal(self) -> None:
        env = _env({"AGENT_QA_REVIEWER_MODEL": "   "})
        with patch.dict(os.environ, env, clear=True):
            issues = audit_agent_config(QA_REVIEWER)

        assert len(issues) == 1
        assert issues[0].severity == "fatal"
        assert "no model is configured" in issues[0].message

    def test_audited_agents_are_resolved_independently(self) -> None:
        """A healthy non-reasoning default keeps every role clean."""
        with patch.dict(os.environ, _env(), clear=True):
            issues = audit_agent_configs([CURRICULUM_ARCHITECT, EDUCATION_DIRECTOR])
        assert issues == []


class TestConfigIssueReporting:
    def test_audit_agent_configs_aggregates_findings(self) -> None:
        env = _env(
            {
                "AGENT_DEFAULT_MODEL": "openrouter/anthropic/claude-opus-5.5",
                "AGENT_DEFAULT_MAX_TOKENS": "4096",
                "AGENT_DEFAULT_TOP_P": "0.9",
            }
        )
        with patch.dict(os.environ, env, clear=True):
            issues = audit_agent_configs([CURRICULUM_ARCHITECT, LAB_DEVELOPER])

        assert {i.role for i in issues} == {CURRICULUM_ARCHITECT, LAB_DEVELOPER}
        assert has_fatal_config_issues(issues)

    def test_has_fatal_config_issues_false_for_warnings_only(self) -> None:
        warnings = [
            ConfigIssue(role="X", severity="warning", message="m", fix="f"),
        ]
        assert not has_fatal_config_issues(warnings)

    def test_format_config_issues_includes_severity_and_fix(self) -> None:
        out = format_config_issues(
            [ConfigIssue(role="X", severity="fatal", message="boom", fix="set the var")]
        )
        assert "FATAL" in out
        assert "boom" in out
        assert "set the var" in out

    def test_format_config_issues_empty_input(self) -> None:
        assert format_config_issues([]) == ""

