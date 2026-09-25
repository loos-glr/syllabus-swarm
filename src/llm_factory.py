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
_DEFAULT_MODEL: str = "deepseek/deepseek-v4-pro"
_DEFAULT_TEMPERATURE: float = 0.2
_DEFAULT_TOP_P: float = 0.1
_DEFAULT_MAX_TOKENS: int = 8192
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
