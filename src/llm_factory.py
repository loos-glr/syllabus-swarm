"""
llm_factory.py — Shared LLM Builder with Per-Agent Configuration
================================================================

Issue #5: Per-Agent Model Configuration — Specialized LLMs for Each Agent

Provides a single, canonical source of truth for building ``crewai.LLM``
instances wired to **OpenRouter**.  Every agent in the swarm should obtain
its LLM through :func:`build_llm_for_agent` rather than reading environment
variables directly.

Key features
------------
* Agent role constants to eliminate magic strings across the codebase.
* 3-tier fallback chain for every property (MODEL, TEMPERATURE, TOP_P,
  MAX_TOKENS):
  1. Per-agent override  → ``AGENT_{ROLE}_{PROPERTY}``
  2. Agent-wide default   → ``AGENT_DEFAULT_{PROPERTY}``
  3. Hardcoded sensible defaults
* Always targets the provider specified by ``BASE_URL``.
* :func:`list_agent_configs` prints the effective configuration of every
  known agent for diagnostics and debugging.

Usage
-----
    from src.llm_factory import build_llm_for_agent, CURRICULUM_ARCHITECT

    llm = build_llm_for_agent(CURRICULUM_ARCHITECT)
    agent = Agent(role="...", goal="...", llm=llm, ...)
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass

# Fail fast with a clear message instead of a confusing ``ModuleNotFoundError``
# (or an ``ImportError`` from ``datetime.UTC``) when an older interpreter is
# used.  The project requires Python 3.12+.
if sys.version_info < (3, 12):
    raise SystemExit(
        "syllabus-swarm requires Python 3.12+ "
        f"(running {'.'.join(map(str, sys.version_info[:3]))}).\n"
        "Activate the project .venv or run with `python3.12`."
    )

from crewai import LLM
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap environment — load .env so os.getenv picks up values.
# Safe to call multiple times; dotenv ignores already-loaded files.
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Constants — Provider configuration (env-var driven, model-agnostic)
# ---------------------------------------------------------------------------
# BASE_URL is read from the environment so the factory is portable across
# any OpenAI-compatible provider (OpenRouter, Together, Fireworks, local
# Ollama/vLLM, etc.).  The fallback value ensures backward compatibility
# with existing OpenRouter deployments.
_BASE_URL: str = os.getenv("BASE_URL", "https://openrouter.ai/api/v1")


def _get_base_url() -> str:
    """Return the effective base URL, respecting the BASE_URL env var."""
    return os.getenv("BASE_URL", _BASE_URL)


# ---------------------------------------------------------------------------
# Agent role constants
# ---------------------------------------------------------------------------
CURRICULUM_ARCHITECT: str = "CURRICULUM_ARCHITECT"
LAB_DEVELOPER: str = "LAB_DEVELOPER"
OUTPUT_EXPORTER: str = "OUTPUT_EXPORTER"
INTAKE_SPECIALIST: str = "INTAKE_SPECIALIST"
QA_REVIEWER: str = "QA_REVIEWER"
THEORY_INSTRUCTOR: str = "THEORY_INSTRUCTOR"
EDUCATION_DIRECTOR: str = "EDUCATION_DIRECTOR"
MEDIA_STRATEGIST: str = "MEDIA_STRATEGIST"
VIDEO_ENGINEER: str = "VIDEO_ENGINEER"
INSTRUCTIONAL_COORDINATOR: str = "INSTRUCTIONAL_COORDINATOR"
PRESENTATION_DESIGNER: str = "PRESENTATION_DESIGNER"

# All known agent roles (used by list_agent_configs).
_KNOWN_ROLES: tuple[str, ...] = (
    CURRICULUM_ARCHITECT,
    LAB_DEVELOPER,
    OUTPUT_EXPORTER,
    INTAKE_SPECIALIST,
    QA_REVIEWER,
    THEORY_INSTRUCTOR,
    EDUCATION_DIRECTOR,
    MEDIA_STRATEGIST,
    VIDEO_ENGINEER,
    INSTRUCTIONAL_COORDINATOR,
    PRESENTATION_DESIGNER,
)

# ---------------------------------------------------------------------------
# Property names used in environment variable construction.
# ---------------------------------------------------------------------------
_PROPERTIES: tuple[str, ...] = ("MODEL", "TEMPERATURE", "TOP_P", "MAX_TOKENS")

# Iteration / rate-limit properties — same 3-tier fallback chain,
# but resolved via dedicated helpers so callers don't repeat
# ``_resolve_numeric`` everywhere.  These are NOT in _PROPERTIES
# because they map to Agent() constructor kwargs, not LLM() kwargs.
_MAX_ITER_ENV: str = "MAX_ITER"
_MAX_RPM_ENV: str = "MAX_RPM"

# Cached defaults after env resolution (lazily populated).
_iter_default: int | None = None
_rpm_default: int | None = None

# ---------------------------------------------------------------------------
# Hardcoded defaults — the final fallback when nothing is configured.
# These are sensible catch-all values; per-agent overrides (via env vars) are
# the recommended path.  Agent modules MUST NOT hardcode model IDs — they
# always delegate to build_llm_for_agent().
# ---------------------------------------------------------------------------
_DEFAULT_MODEL: str = "openrouter/deepseek/deepseek-v4-pro"
_DEFAULT_TEMPERATURE: float = 0.2
_DEFAULT_TOP_P: float = 0.1
_DEFAULT_MAX_TOKENS: int = 32768
# ---------------------------------------------------------------------------
# Models with built-in reasoning that consume tokens internally before
# producing visible output.  When these models are configured with
# max_tokens < 16384, they commonly return NULL content because all
# tokens are spent on reasoning.
# ---------------------------------------------------------------------------
_REASONING_MODEL_PATTERNS: tuple[str, ...] = (
    "claude-opus-5.5",
    "claude-opus-5",
    "claude-sonnet-5",
    "gpt-5",
)
_MIN_SAFE_MAX_TOKENS_REASONING: int = 16384
# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_property(
    role: str,
    property_name: str,
    *,
    hardcoded_default: str,
) -> str:
    """Resolve a single string property through the 3-tier fallback chain.

    Tier order (highest to lowest priority):
      1. ``AGENT_{role}_{property_name}``  — per-agent override
      2. ``AGENT_DEFAULT_{property_name}`` — agent-wide default
      3. *hardcoded_default*               — baked-in fallback
    """
    # Tier 1: per-agent override
    value = os.getenv(f"AGENT_{role}_{property_name}")
    if value is not None:
        return value

    # Tier 2: agent-wide default
    value = os.getenv(f"AGENT_DEFAULT_{property_name}")
    if value is not None:
        return value

    # Tier 3: hardcoded default
    return hardcoded_default


def _resolve_numeric(
    role: str,
    property_name: str,
    *,
    hardcoded_default: float | int,
) -> float | int:
    """Resolve a numeric property through the 3-tier fallback chain.

    Same semantics as _resolve_property but returns a float (or int,
    inferred from hardcoded_default).
    """
    raw = _resolve_property(
        role,
        property_name,
        hardcoded_default=str(hardcoded_default),
    )
    try:
        if isinstance(hardcoded_default, int):
            return int(raw)
        return float(raw)
    except (ValueError, TypeError):
        return hardcoded_default


def resolve_max_iter(agent_role: str, hardcoded_default: int) -> int:
    """Resolve the ``max_iter`` value for *agent_role*.

    Follows the same 3-tier fallback chain as ``build_llm_for_agent``:

    1. ``AGENT_{ROLE}_MAX_ITER``  — per-agent override
    2. ``AGENT_DEFAULT_MAX_ITER`` — agent-wide default
    3. *hardcoded_default*         — caller-supplied fallback

    Parameters
    ----------
    agent_role : str
        Uppercase snake_case agent identifier (e.g. ``QA_REVIEWER``).
    hardcoded_default : int
        The value to use when no env var is set.

    Returns
    -------
    int
        The resolved iteration limit.
    """
    return int(
        _resolve_numeric(
            agent_role,
            _MAX_ITER_ENV,
            hardcoded_default=hardcoded_default,
        )
    )


def resolve_max_rpm(agent_role: str, hardcoded_default: int) -> int:
    """Resolve the ``max_rpm`` value for *agent_role*.

    Same 3-tier fallback as :func:`resolve_max_iter`:

    1. ``AGENT_{ROLE}_MAX_RPM``  — per-agent override
    2. ``AGENT_DEFAULT_MAX_RPM`` — agent-wide default
    3. *hardcoded_default*        — caller-supplied fallback

    Parameters
    ----------
    agent_role : str
        Uppercase snake_case agent identifier.
    hardcoded_default : int
        The value to use when no env var is set.

    Returns
    -------
    int
        The resolved requests-per-minute limit.
    """
    return int(
        _resolve_numeric(
            agent_role,
            _MAX_RPM_ENV,
            hardcoded_default=hardcoded_default,
        )
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_llm_for_agent(
    agent_role: str,
    *,
    api_key: str | None = None,
) -> LLM:
    """Build a ``crewai.LLM`` instance configured for a specific agent.

    All agents connect through the provider specified by ``BASE_URL``.
    Model, temperature, top_p, and max_tokens are resolved through a 3-tier
    fallback chain that allows per-agent customisation while always falling
    back to a working configuration — even when no environment variables are
    set.

    Parameters
    ----------
    agent_role : str
        Uppercase snake_case identifier for the agent (use the module-level
        constants, e.g. ``CURRICULUM_ARCHITECT``).
    api_key : str or None
        OpenRouter API key.  When omitted the key is read from the
        ``OPENROUTER_API_KEY`` environment variable.

    Returns
    -------
    LLM
        A fully-configured ``crewai.LLM`` instance wired to OpenRouter.

    Raises
    ------
    ValueError
        If *agent_role* is empty or not a string.
    """
    if not agent_role or not isinstance(agent_role, str):
        raise ValueError(f"agent_role must be a non-empty string, got {agent_role!r}")

    # Resolve the API key — use explicit argument first, then env var.
    resolved_api_key: str = (
        api_key
        if api_key is not None
        else os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEY", "")
    )

    model: str = _resolve_property(
        agent_role,
        "MODEL",
        hardcoded_default=_DEFAULT_MODEL,
    )

    temperature: float = _resolve_numeric(
        agent_role,
        "TEMPERATURE",
        hardcoded_default=_DEFAULT_TEMPERATURE,
    )

    top_p: float = _resolve_numeric(
        agent_role,
        "TOP_P",
        hardcoded_default=_DEFAULT_TOP_P,
    )

    max_tokens: int = int(
        _resolve_numeric(
            agent_role,
            "MAX_TOKENS",
            hardcoded_default=_DEFAULT_MAX_TOKENS,
        )
    )

    # Warn if a reasoning model is configured with low max_tokens.
    model_lower = model.lower()
    if any(pattern in model_lower for pattern in _REASONING_MODEL_PATTERNS):
        if max_tokens < _MIN_SAFE_MAX_TOKENS_REASONING:
            import warnings

            warnings.warn(
                f"Agent '{agent_role}' uses reasoning model '{model}' with "
                f"max_tokens={max_tokens}. Reasoning models consume tokens "
                f"internally before producing output; values below "
                f"{_MIN_SAFE_MAX_TOKENS_REASONING} commonly return NULL "
                f"content. Set AGENT_{agent_role}_MAX_TOKENS >= "
                f"{_MIN_SAFE_MAX_TOKENS_REASONING} in .env.",
                stacklevel=2,
            )

    return LLM(
        model=model,
        api_key=resolved_api_key,
        base_url=_get_base_url(),
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


def get_effective_config(agent_role: str) -> dict[str, object]:
    """Return the effective configuration dict for a single agent.

    This is the programmatic counterpart to list_agent_configs — useful
    when you need the resolved *LLM* values in code rather than printed.

    Note: ``max_iter`` and ``max_rpm`` are Agent-level settings (not LLM
    settings), resolved separately via :func:`resolve_max_iter` and
    :func:`resolve_max_rpm`.
    """
    resolved_api_key: str = os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEY", "")
    api_key_status: str = "set" if resolved_api_key else "missing — authentication will fail"

    return {
        "role": agent_role,
        "model": _resolve_property(
            agent_role,
            "MODEL",
            hardcoded_default=_DEFAULT_MODEL,
        ),
        "temperature": _resolve_numeric(
            agent_role,
            "TEMPERATURE",
            hardcoded_default=_DEFAULT_TEMPERATURE,
        ),
        "top_p": _resolve_numeric(
            agent_role,
            "TOP_P",
            hardcoded_default=_DEFAULT_TOP_P,
        ),
        "max_tokens": int(
            _resolve_numeric(
                agent_role,
                "MAX_TOKENS",
                hardcoded_default=_DEFAULT_MAX_TOKENS,
            )
        ),
        "base_url": _get_base_url(),
        "api_key_status": api_key_status,
    }


# ---------------------------------------------------------------------------
# Configuration pre-flight audit — early warning before any credits are spent
# ---------------------------------------------------------------------------
# The single most common cause of the CrewAI error
# ``"Invalid response from LLM call - None or empty"`` is a *reasoning* model
# configured with a ``max_tokens`` budget that is too small: the model spends
# its entire allowance on internal reasoning and emits no visible content.
# Retrying that call cannot help and only burns credits, so the audit below
# detects the misconfiguration **before** the pipeline starts and surfaces it
# as a fatal, actionable finding.


@dataclass(frozen=True)
class ConfigIssue:
    """A single pre-flight finding for one agent's LLM configuration.

    Attributes
    ----------
    role : str
        Agent role the finding applies to (e.g. ``THEORY_INSTRUCTOR``).
    severity : str
        Either ``"fatal"`` (the run must not start) or ``"warning"``
        (the run may start, but the configuration is risky).
    message : str
        Human-readable description of the detected problem.
    fix : str
        Concrete remediation, including the exact ``.env`` key to change.
    """

    role: str
    severity: str
    message: str
    fix: str


def _is_reasoning_model(model: str) -> bool:
    """Return True when *model* is a known built-in-reasoning model."""
    lowered = model.lower()
    return any(pattern in lowered for pattern in _REASONING_MODEL_PATTERNS)


def audit_agent_config(agent_role: str) -> list[ConfigIssue]:
    """Inspect one agent's effective LLM config for known failure modes.

    This is a *static* check — it makes no API calls and costs nothing.  It
    catches the two configuration problems that reliably produce the
    ``"Invalid response from LLM call - None or empty"`` failure:

    1. **Reasoning model with a token budget that is too small** (fatal).
       Reasoning tokens are consumed before any visible output, so a budget
       below :data:`_MIN_SAFE_MAX_TOKENS_REASONING` yields NULL content.
    2. **Reasoning model with very restrictive ``top_p``** (warning).
       Extremely low values can degenerate into empty completions,
       particularly on tool-calling steps.

    Parameters
    ----------
    agent_role : str
        Uppercase snake_case agent identifier.

    Returns
    -------
    list[ConfigIssue]
        Zero or more findings (empty when the configuration is healthy).
    """
    cfg = get_effective_config(agent_role)
    model = str(cfg["model"])
    max_tokens = int(cfg["max_tokens"])  # type: ignore[arg-type]
    top_p = float(cfg["top_p"])  # type: ignore[arg-type]

    issues: list[ConfigIssue] = []

    if not model.strip():
        issues.append(
            ConfigIssue(
                role=agent_role,
                severity="fatal",
                message=f"{agent_role}: no model is configured.",
                fix=f"Set AGENT_{agent_role}_MODEL (or AGENT_DEFAULT_MODEL) in .env.",
            )
        )
        return issues

    is_reasoning = _is_reasoning_model(model)

    if is_reasoning and max_tokens < _MIN_SAFE_MAX_TOKENS_REASONING:
        issues.append(
            ConfigIssue(
                role=agent_role,
                severity="fatal",
                message=(
                    f"{agent_role}: reasoning model '{model}' is configured with "
                    f"max_tokens={max_tokens}, below the safe minimum of "
                    f"{_MIN_SAFE_MAX_TOKENS_REASONING}."
                ),
                fix=(
                    f"Set AGENT_{agent_role}_MAX_TOKENS="
                    f"{_MIN_SAFE_MAX_TOKENS_REASONING} (or higher) in .env. "
                    "Reasoning tokens are consumed before visible output, so a "
                    "smaller budget makes the model return NULL content and "
                    "aborts the run with 'Invalid response from LLM call - "
                    "None or empty'."
                ),
            )
        )

    if is_reasoning and top_p < 0.5:
        issues.append(
            ConfigIssue(
                role=agent_role,
                severity="warning",
                message=(
                    f"{agent_role}: reasoning model '{model}' uses top_p="
                    f"{top_p} (very restrictive sampling)."
                ),
                fix=(
                    f"Consider raising AGENT_{agent_role}_TOP_P to ~0.7-0.9. "
                    "Extremely low top_p can degenerate into empty completions "
                    "on tool-calling steps."
                ),
            )
        )

    return issues


def audit_agent_configs(
    agent_roles: Iterable[str] | None = None,
) -> list[ConfigIssue]:
    """Audit several agents at once.

    Parameters
    ----------
    agent_roles : Iterable[str] or None
        Roles to audit.  Defaults to every known role
        (:data:`_KNOWN_ROLES`) when None.

    Returns
    -------
    list[ConfigIssue]
        Concatenated findings for every audited role.
    """
    roles = tuple(agent_roles) if agent_roles is not None else _KNOWN_ROLES
    issues: list[ConfigIssue] = []
    for role in roles:
        issues.extend(audit_agent_config(role))
    return issues


def has_fatal_config_issues(issues: Iterable[ConfigIssue]) -> bool:
    """Return True when *issues* contains at least one ``"fatal"`` finding."""
    return any(issue.severity == "fatal" for issue in issues)


def format_config_issues(issues: Iterable[ConfigIssue]) -> str:
    """Render *issues* as a human-readable, copy-pasteable block."""
    lines: list[str] = []
    for issue in issues:
        icon = "⛔" if issue.severity == "fatal" else "⚠️ "
        lines.append(f"  {icon}  [{issue.severity.upper()}] {issue.message}")
        lines.append(f"       → {issue.fix}")
    return "\n".join(lines)


def list_agent_configs() -> None:
    """Print the effective configuration of every known agent to stdout.

    Useful for debugging environment-variable resolution and confirming
    that per-agent overrides are being picked up correctly.
    """
    print()
    print("=" * 60)
    print("  LLM Factory — Agent Configuration Summary")
    print("=" * 60)
    print()

    print(f"  OpenRouter Base URL:  {_get_base_url()}")
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    status = "set" if api_key else "missing — authentication will fail"
    print(f"  API Key:              {status}")
    print()

    # --- LLM env vars ---
    print("-- LLM Environment " + "-" * 41)
    for prop in _PROPERTIES:
        env_key = f"AGENT_DEFAULT_{prop}"
        value = os.getenv(env_key)
        label = value if value is not None else "(not set)"
        print(f"  {env_key:.<40} {label}")

    print()

    # --- Agent iteration / RPM env vars ---
    print("-- Agent Runtime Limits (max_iter / max_rpm) " + "-" * 13)
    for suffix in (_MAX_ITER_ENV, _MAX_RPM_ENV):
        def_key = f"AGENT_DEFAULT_{suffix}"
        def_val = os.getenv(def_key)
        label = def_val if def_val is not None else "(not set — uses per-agent hardcoded default)"
        print(f"  {def_key:.<40} {label}")

    print()

    for role in _KNOWN_ROLES:
        config = get_effective_config(role)
        padding = max(1, 45 - len(role))
        print(f"-- Agent: {role} " + "-" * padding)
        print(f"  model:        {config['model']}")
        print(f"  temperature:  {config['temperature']}")
        print(f"  top_p:        {config['top_p']}")
        print(f"  max_tokens:   {config['max_tokens']}")
        # Show per-agent override status for iteration limits
        for suffix in (_MAX_ITER_ENV, _MAX_RPM_ENV):
            key = f"AGENT_{role}_{suffix}"
            val = os.getenv(key)
            tag = val if val is not None else "(not set)"
            print(f"  {suffix.lower()}:    {tag}")

    print()
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("llm_factory module loaded successfully.\n")

    list_agent_configs()

    try:
        llm = build_llm_for_agent(CURRICULUM_ARCHITECT)
        print(f"build_llm_for_agent({CURRICULUM_ARCHITECT}) succeeded.")
        print(f"   model:           {llm.model}")
        print(f"   temperature:     {llm.temperature}")
        print(f"   top_p:           {llm.top_p}")
        print(f"   max_tokens:      {llm.max_tokens}")
        print(f"   base_url:        {llm.base_url}")
    except Exception as exc:
        print(f"build_llm_for_agent() raised: {exc}")
