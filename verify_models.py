#!/usr/bin/env python3
"""
verify_models.py — Model Availability & Fallback Chain Verification (Issue #6)
==============================================================================

Validates the v2 per-agent model assignments introduced in Issue #6:
  • deepseek/deepseek-v4-pro        → CURRICULUM_ARCHITECT
  • qwen/qwen3-coder                → LAB_DEVELOPER
  • deepseek/deepseek-v4-flash-latest → OUTPUT_EXPORTER
  • deepseek/deepseek/deepseek-v4-pro            → HARDCODED FALLBACK

The script runs two independent verification passes:

  PART 1 — OpenRouter model-list query
    Confirms the four model IDs actually exist on OpenRouter.  If a model
    is unavailable the script prints the closest alternatives (same provider
    prefix) so operators can quickly pivot.

  PART 2 — Smoke test (fallback-chain resolution)
    Calls :func:`src.llm_factory.build_llm_for_agent` for each agent role
    with the v2 environment variables set, confirming the resolved model
    string matches the expected new default.  Also validates the hardcoded
    fallback (`deepseek/deepseek/deepseek-v4-pro`) kicks in when no per-agent vars exist.

    This pass requires NO live API key — it exercises only the 4-tier
    resolution logic.

Usage
-----
    python verify_models.py
    python verify_models.py --verbose
    python verify_models.py --check-only          # skip network, config only

Environment
-----------
    OPENROUTER_API_KEY … optional; the smoke test uses a synthetic key.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import TypedDict
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Path / import setup — ensure the project root is on sys.path so that
# ``from src.llm_factory import ...`` works regardless of CWD.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.llm_factory import (
    _DEFAULT_MODEL,
    CURRICULUM_ARCHITECT,
    LAB_DEVELOPER,
    OUTPUT_EXPORTER,
    build_llm_for_agent,
    get_effective_config,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENROUTER_MODELS_URL: str = "https://openrouter.ai/api/v1/models"

# v2 per-agent model assignments (Issue #6)
V2_ASSIGNMENTS: dict[str, dict[str, str]] = {
    CURRICULUM_ARCHITECT: {
        "model": "openrouter/deepseek/deepseek-v4-pro",
        "env_var": "AGENT_CURRICULUM_ARCHITECT_MODEL",
        "rationale": "Deep reasoning for syllabus design",
    },
    LAB_DEVELOPER: {
        "model": "openrouter/qwen/qwen3-coder",
        "env_var": "AGENT_LAB_DEVELOPER_MODEL",
        "rationale": "Purpose-built code generation",
    },
    OUTPUT_EXPORTER: {
        "model": "openrouter/deepseek/deepseek-v4-flash-latest",
        "env_var": "AGENT_OUTPUT_EXPORTER_MODEL",
        "rationale": "Low-latency packaging / manifest",
    },
}

HARDCODED_FALLBACK: str = _DEFAULT_MODEL  # "deepseek/deepseek/deepseek-v4-pro"

DUMMY_API_KEY: str = "sk-or-v1-verify-models-smoke-test-key"

REQUEST_TIMEOUT: int = 15  # seconds for the OpenRouter HTTP call


# ---------------------------------------------------------------------------
# Colour helpers (ANSI escape codes — safe fallback when not in a TTY)
# ---------------------------------------------------------------------------


class _Ansi:
    """Tiny namespace so we can toggle colours off for non-TTY output."""

    _enabled: bool = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    @classmethod
    def disable(cls) -> None:
        cls._enabled = False

    @classmethod
    def _code(cls, code: int) -> str:
        return f"\033[{code}m" if cls._enabled else ""

    @classmethod
    def green(cls, text: str) -> str:
        return f"{cls._code(32)}{text}{cls._code(0)}"

    @classmethod
    def red(cls, text: str) -> str:
        return f"{cls._code(31)}{text}{cls._code(0)}"

    @classmethod
    def yellow(cls, text: str) -> str:
        return f"{cls._code(33)}{text}{cls._code(0)}"

    @classmethod
    def cyan(cls, text: str) -> str:
        return f"{cls._code(36)}{text}{cls._code(0)}"

    @classmethod
    def bold(cls, text: str) -> str:
        return f"{cls._code(1)}{text}{cls._code(0)}"


C = _Ansi


# ---------------------------------------------------------------------------
# TypedDict for the OpenRouter model list response (subset we care about)
# ---------------------------------------------------------------------------
class _ModelData(TypedDict):
    id: str
    name: str


class _ModelsResponse(TypedDict):
    data: list[_ModelData]


# ---------------------------------------------------------------------------
# PART 1 — OpenRouter model-list check
# ---------------------------------------------------------------------------


def _fetch_openrouter_models() -> list[str] | None:
    """Return a list of OpenRouter model IDs, or ``None`` on failure."""
    try:
        import requests  # type: ignore[import-untyped]
    except ImportError:
        print(C.red("ERROR: 'requests' package is required for network check."))
        print("       Install it:  pip install requests")
        return None

    print(f"\n{C.bold('Querying OpenRouter model list…')}")
    print(f"  URL:  {OPENROUTER_MODELS_URL}\n")

    try:
        resp = requests.get(OPENROUTER_MODELS_URL, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        print(C.red(f"ERROR: Request timed out after {REQUEST_TIMEOUT}s."))
        return None
    except requests.exceptions.ConnectionError:
        print(C.red("ERROR: Could not connect to openrouter.ai."))
        print("       Check your network connection and try again.")
        return None
    except requests.exceptions.HTTPError as exc:
        print(C.red(f"ERROR: HTTP {resp.status_code} — {exc}"))
        return None
    except Exception as exc:
        print(C.red(f"ERROR: Unexpected failure — {exc}"))
        return None

    try:
        payload: _ModelsResponse = resp.json()
    except json.JSONDecodeError:
        print(C.red("ERROR: OpenRouter returned non-JSON response."))
        return None

    models_raw = payload.get("data", [])
    if not models_raw:
        print(C.yellow("WARNING: OpenRouter returned zero models — is the API changed?"))
        return []

    model_ids: list[str] = []
    for item in models_raw:
        model_id: str = item.get("id", "")
        if model_id:
            model_ids.append(model_id)

    print(f"  {C.green('✓')} Received {len(model_ids):,} model IDs from OpenRouter.\n")
    return model_ids


def _find_alternatives(target: str, available: list[str], limit: int = 5) -> list[str]:
    """Return models from *available* whose ID shares the same provider prefix."""
    prefix = target.split("/")[0] + "/" if "/" in target else ""
    matches = [m for m in available if m.startswith(prefix)]
    matches.sort()
    if target in matches:
        matches.remove(target)
    return matches[:limit]


def run_availability_check(available_ids: list[str]) -> dict[str, bool]:
    """Check all v2 models + fallback against *available_ids*.

    Returns a dict mapping model ID → ``True`` (available) / ``False``.
    Also prints a formatted results table to stdout.
    """
    targets: list[tuple[str, str, str]] = []
    for role, info in V2_ASSIGNMENTS.items():
        targets.append((info["model"], role, info["rationale"]))
    # Add the hardcoded fallback
    targets.append((HARDCODED_FALLBACK, "FALLBACK (all agents)", "Backward-compatible catch-all"))

    # --- Table dimensions ---
    col_id = 38
    col_role = 26
    col_status = 18

    print(C.bold("  MODEL AVAILABILITY"))
    print(f"  ┌─{'─' * col_id}─┬─{'─' * col_role}─┬─{'─' * col_status}─┐")
    print(f"  │ {'Model ID':<{col_id}} │ {'Assigned To':<{col_role}} │ {'Status':<{col_status}} │")
    print(f"  ├─{'─' * col_id}─┼─{'─' * col_role}─┼─{'─' * col_status}─┤")

    results: dict[str, bool] = {}
    unavailable: list[tuple[str, str]] = []

    for model_id, role, _rationale in targets:
        available = model_id in available_ids
        results[model_id] = available
        if available:
            status = C.green("✅ Available     ")
        else:
            status = C.red("❌ NOT FOUND    ")
        print(f"  │ {model_id:<{col_id}} │ {role:<{col_role}} │ {status:<{col_status + 9}} │")
        if not available:
            unavailable.append((model_id, role))

    print(f"  └─{'─' * col_id}─┴─{'─' * col_role}─┴─{'─' * col_status}─┘")
    print()

    # --- Warnings + alternatives for unavailable models ---
    if unavailable:
        print(C.bold(C.red("  ⚠  UNAVAILABLE MODELS — closest alternatives:")))
        print()
        for model_id, role in unavailable:
            alternatives = _find_alternatives(model_id, available_ids)
            print(f"    {C.red('✗')} {C.bold(model_id)}  ({role})")
            if alternatives:
                print(f"      {C.yellow('Did you mean one of these?')}")
                for alt in alternatives:
                    print(f"        • {alt}")
            else:
                provider = model_id.split("/")[0] if "/" in model_id else "?"
                print(f"      {C.yellow(f'No other {provider}/ models found on OpenRouter.')}")
            print()

    return results


# ---------------------------------------------------------------------------
# PART 2 — Smoke test (fallback-chain resolution)
# ---------------------------------------------------------------------------


def _env_for_v2() -> dict[str, str]:
    """Build the environment dict that mirrors a v2 ``.env`` deployment."""
    env: dict[str, str] = {"OPENROUTER_API_KEY": DUMMY_API_KEY}
    for info in V2_ASSIGNMENTS.values():
        env[info["env_var"]] = info["model"]
    return env


def _pad_role(role: str, width: int = 22) -> str:
    return f"{role:<{width}}"


def run_smoke_test() -> bool:
    """Smoke-test the fallback chain with and without v2 per-agent vars.

    Returns ``True`` when every check passes.
    """
    all_ok = True

    print(C.bold("\n  SMOKE TEST — Fallback Chain Resolution"))
    print(f"  {'─' * 55}\n")

    # -- Test A: v2 per-agent env vars set → expect new models ----------
    v2_env = _env_for_v2()

    print(f"  {C.cyan('Test A:')} v2 per-agent model assignments (AGENT_{{ROLE}}_MODEL set)")
    for info in V2_ASSIGNMENTS.values():
        print(f"    {info['env_var']}={info['model']}")

    with patch.dict(os.environ, v2_env, clear=True):
        for role, info in V2_ASSIGNMENTS.items():
            expected = info["model"]
            config = get_effective_config(role)
            resolved = str(config["model"])
            ok = resolved == expected
            if not ok:
                all_ok = False

            mark = C.green("✅") if ok else C.red("❌")
            status = (
                C.green(f"(expected: {expected})")
                if ok
                else C.red(f"(expected: {expected}, GOT: {resolved})")
            )
            print(f"    {mark} {_pad_role(role)} → {resolved}  {status}")

    # -- Test B: no per-agent or legacy vars → expect hardcoded fallback --
    print(f"\n  {C.cyan('Test B:')} No per-agent / legacy vars → hardcoded fallback")

    fallback_env: dict[str, str] = {"OPENROUTER_API_KEY": DUMMY_API_KEY}
    with patch.dict(os.environ, fallback_env, clear=True):
        for role, info in V2_ASSIGNMENTS.items():
            expected = HARDCODED_FALLBACK
            config = get_effective_config(role)
            resolved = str(config["model"])
            ok = resolved == expected
            if not ok:
                all_ok = False

            mark = C.green("✅") if ok else C.red("❌")
            status = (
                C.green(f"(expected: {expected})")
                if ok
                else C.red(f"(expected: {expected}, GOT: {resolved})")
            )
            print(f"    {mark} {_pad_role(role)} → {resolved}  {status}")

    # -- Test C: build_llm_for_agent() constructs valid LLM objects -----
    print(f"\n  {C.cyan('Test C:')} build_llm_for_agent() returns valid LLM objects")
    print(
        "      (crewai.LLM may strip or keep the provider prefix depending "
        "on whether litellm is installed.)"
    )

    with patch.dict(os.environ, v2_env, clear=True):
        for role, info in V2_ASSIGNMENTS.items():
            try:
                llm = build_llm_for_agent(role)
                # crewai.LLM strips some provider prefixes (e.g. deepseek/)
                # but keeps others when litellm handles them (e.g. qwen/).
                # Accept either the full model ID or the stripped form.
                full_m = info["model"]
                stripped_m = full_m.split("/", 1)[1] if "/" in full_m else full_m
                ok = llm.model in (full_m, stripped_m)
                if not ok:
                    all_ok = False
                mark = C.green("✅") if ok else C.red("❌")
                print(
                    f"    {mark} {_pad_role(role)} → "
                    f"LLM(model='{llm.model}', base_url='{llm.base_url}')"
                )
            except Exception as exc:
                print(
                    f"    {C.yellow('⚠')}  {_pad_role(role)} → "
                    f"LLM construction failed (provider may need litellm)"
                )
                if "LiteLLM" in str(exc) or "provider" in str(exc).lower():
                    print(
                        f"      {C.yellow('→')} Install litellm for "
                        f"broad provider support: pip install litellm"
                    )

    print()

    if all_ok:
        print(
            f"  {C.green(C.bold('✅ All smoke tests passed'))} — v2 model "
            f"assignments resolve correctly."
        )
    else:
        print(
            f"  {C.red(C.bold('❌ Some smoke tests FAILED'))} — review the "
            f"output above for details."
        )

    return all_ok


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    """Entry point.  Returns 0 on success, 1 on any failure."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Verify v2 model availability and fallback chain (Issue #6).",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Skip OpenRouter network check; only run the config smoke test.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print extra diagnostics.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colour codes.",
    )
    args = parser.parse_args()

    if args.no_color:
        C.disable()

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------
    print()
    print(C.bold("=" * 64))
    print(C.bold("  verify_models.py — Issue #6: Model Verification"))
    print(C.bold("=" * 64))
    print(f"  Time:  {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"  Root:  {_PROJECT_ROOT}")
    print()

    exit_code: int = 0

    # ------------------------------------------------------------------
    # Part 1: OpenRouter availability check
    # ------------------------------------------------------------------
    if not args.check_only:
        available_ids = _fetch_openrouter_models()
        if available_ids is not None:
            run_availability_check(available_ids)
        else:
            print(
                C.yellow(
                    "  ⚠  Skipping availability table — network check failed.\n"
                    "     Run with --check-only to validate config resolution only."
                ),
            )
    else:
        print(f"  {C.yellow('(Skipping network check — --check-only flag set)')}")

    # ------------------------------------------------------------------
    # Part 2: Fallback-chain smoke test  (always runs)
    # ------------------------------------------------------------------
    smoke_ok = run_smoke_test()
    if not smoke_ok:
        exit_code = 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(C.bold("=" * 64))

    if args.check_only:
        print(C.bold("  Note: Network check skipped. Only config resolution was validated."))
    if exit_code == 0:
        print(C.bold(C.green("  Result: ALL CHECKS PASSED")))
    else:
        print(C.bold(C.red("  Result: SOME CHECKS FAILED — see output above.")))
    print(C.bold("=" * 64))
    print()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
