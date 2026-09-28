"""
syllabus_crew.py — Full Syllabus + Labs Orchestration Crew
==========================================================

Issue #2: Core Agent — The Curriculum Architect (Humanics Alignment)
Issue #3: Core Agent — The Lab & Project Developer (Tiered Coding Challenges)
Issue #7: AI-Driven Feedback Loop — QA Reviewer Agent

Wires together three agents into a sequential CrewAI Crew:
  1. **Curriculum Architect** — generates a Humanics-aligned syllabus in
     Markdown and saves it to ``output/syllabus/<course_name>.md``.
  2. **Lab & Project Developer** — receives the syllabus as context and
     generates tiered coding labs saved to ``output/labs/<course_name>/``.
  3. **QA Reviewer** — reviews all generated labs for technical correctness
     and MBO4 didactic appropriateness.  Can delegate fixes back to the
     Lab Developer via CrewAI's delegation mechanism.

The crew runs sequentially so each agent can use the previous agent's
output as grounding context.
"""

from __future__ import annotations

import os
import re
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from crewai import Agent, Crew, Process, Task

from src.agents.curriculum_architect import get_architect
from src.agents.education_director import get_education_director
from src.agents.instructional_coordinator import get_instructional_coordinator
from src.agents.lab_developer import get_lab_developer
from src.agents.presentation_designer import get_presentation_designer
from src.agents.qa_reviewer import get_qa_reviewer
from src.agents.theory_instructor import get_theory_instructor
from src.exporters import (
    update_output_manifest,
    write_file,
)
from src.exporters.file_writer import _sanitize_filename
from src.exporters.theory_validator import (
    format_validation_report,
    validate_theory_directory,
)
from src.llm_factory import (
    ConfigIssue,
    audit_agent_configs,
    format_config_issues,
    has_fatal_config_issues,
)
from src.models import GenerationState, TierState
from src.preflight import (
    ProbeResult,
    fatal_probe_results,
    format_probe_report,
    probe_agent_models,
    probe_enabled,
)
from src.tasks.lab_generation import create_lab_generation_task
from src.tasks.lesson_plan_generation import create_lesson_plan_task
from src.tasks.presentation_generation import create_presentation_task
from src.tasks.qa_review import create_qa_review_task
from src.tasks.syllabus_generation import create_syllabus_generation_task
from src.tasks.syllabus_review import create_syllabus_review_task
from src.tasks.theory_generation import create_theory_task


class SwarmState(str, Enum):
    """Execution states for the HITL cyclic syllabus swarm.

    .. rubric:: Issue #6 — Cyclic State Machine

    Values
    ------
    GENERATING
        Initial generation in progress (all agents run).
    AWAITING_FEEDBACK
        Generation complete; waiting for human review.
    EXPORTING
        Human approved; proceed to manifest export.
    """

    GENERATING = "generating"
    AWAITING_FEEDBACK = "awaiting_feedback"
    EXPORTING = "exporting"
    VIDEO_GENERATING = "video_generating"
    LESSON_PLAN_GENERATING = "lesson_plan_generating"
    PRESENTATION_GENERATING = "presentation_generating"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
OUTPUT_ROOT: Path = _PROJECT_ROOT / "output"

# Tier names used for the lab directory scaffolding.
_TIERS: list[tuple[str, str]] = [
    ("tier1_foundations", "Tier 1 — Foundations"),
    ("tier2_application", "Tier 2 — Application"),
    ("tier3_architecture", "Tier 3 — Architecture"),
]

# Known pattern emitted by CrewAI when an agent hits its ``max_iter``
# ceiling.  We scan exception messages for this substring so we can
# surface a clear, actionable hint (which env var to bump) instead of
# forcing the operator to decode three separate error lines.
_MAX_ITER_CREWAI_MARKER: str = "Maximum iterations reached"

# Known markers emitted by CrewAI when an LLM returns no usable content
# (see ``_validate_and_finalize_llm_response`` in crewai/utilities/agent_utils).
# This is usually transient — a provider blip, rate limiting, or a model that
# spent its whole max_tokens budget on internal reasoning before producing
# visible output — and is distinct from "Maximum iterations reached" above.
_EMPTY_LLM_RESPONSE_MARKERS: tuple[str, ...] = (
    "Invalid response from LLM call - None or empty",
    "Received None or empty response from LLM call",
)

# ---------------------------------------------------------------------------
# LLM error taxonomy — decides whether a failure is worth retrying or whether
# the run must be aborted immediately.
# ---------------------------------------------------------------------------
# "retryable"     — transient infrastructure problem (rate limit, timeout,
#                   connection reset).  A bounded retry with backoff is cheap
#                   and frequently succeeds.
# "futile"        — the model returned NULL/empty content.  This almost always
#                   means a *configuration* or *model* problem (e.g. a
#                   reasoning model whose token budget was fully consumed by
#                   internal reasoning), so repeating the call burns credits
#                   for no benefit.  Abort on the first indication by default.
# "deterministic" — anything else (e.g. a bad tool name).  Retrying would fail
#                   identically, so it is re-raised immediately.
_LLM_ERROR_FUTILE: str = "futile"
_LLM_ERROR_RETRYABLE: str = "retryable"
_LLM_ERROR_DETERMINISTIC: str = "deterministic"

# Number of attempts allowed for a *futile* failure before the run aborts.
# Default 1 == "abort on the first indication", which is what we want: a
# reasoning model that answered with NULL content will do so again, and every
# repeat call is pure credit burn.  Set AGENT_LLM_FUTILE_RETRIES=2 (or higher)
# only when the provider is known to emit occasional one-off empty responses.
_FUTILE_RETRY_ATTEMPTS_ENV: str = "AGENT_LLM_FUTILE_RETRIES"
_DEFAULT_FUTILE_RETRY_ATTEMPTS: int = 1

# Agent roles whose configuration is audited before the run starts.  Only the
# agents that actually appear in the pipeline are included so the pre-flight
# report stays focused and actionable.
_PREFLIGHT_ROLES: tuple[str, ...] = (
    "CURRICULUM_ARCHITECT",
    "EDUCATION_DIRECTOR",
    "THEORY_INSTRUCTOR",
    "INSTRUCTIONAL_COORDINATOR",
    "PRESENTATION_DESIGNER",
    "LAB_DEVELOPER",
    "QA_REVIEWER",
)


def _resolve_futile_attempts() -> int:
    """Return the configured number of attempts for futile LLM failures."""
    raw = os.getenv(_FUTILE_RETRY_ATTEMPTS_ENV)
    if raw is None:
        return _DEFAULT_FUTILE_RETRY_ATTEMPTS
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return _DEFAULT_FUTILE_RETRY_ATTEMPTS


class FatalLLMError(RuntimeError):
    """Raised when an LLM failure is diagnosed as *futile*.

    A futile failure is one that reproduces on every retry — typically a model
    returning ``None``/empty content because its token budget was spent on
    internal reasoning, or because the configured model is unavailable.
    Continuing to hammer the API wastes credits, so this exception unwinds to
    the run-level circuit breaker (:class:`_FatalAbortGuard`), which skips all
    remaining generation stages instead of starting work that is guaranteed to
    fail the same way.
    """

    def __init__(
        self,
        message: str,
        *,
        agent_role: str | None = None,
        stage: str | None = None,
    ) -> None:
        super().__init__(message)
        self.agent_role = agent_role
        self.stage = stage

# Patterns used by :func:`_scan_for_stray_generated_files` to detect
# agent-generated artefacts written straight into the ``output/`` root
# instead of into a per-run directory (``output/<run_id>/``).
#
# NOTE: these patterns are matched against the *immediate children of*
# ``OUTPUT_ROOT`` only — never against project-root files such as
# ``README.md``, ``DESIGN.md`` or ``docs/*``, which are permanent
# repository files and can never be strays.
_STRAY_PATTERNS: list[tuple[str, str]] = [
    (r"^tier\d", "directory"),
    (r"^tier\d.*", "directory"),
    (r"\.js$", "file"),
    (r"^package\.json$", "file"),
    (r"^Makefile$", "file"),
    (r"^Dockerfile$", "file"),
    (r"^docker-compose\.yml$", "file"),
    (r"^docker-compose\.yaml$", "file"),
    (r"^\.gitignore$", "file"),
    (r"\.html$", "file"),
    (r"\.css$", "file"),
    (r"\.sh$", "file"),
    (r"\.yml$", "file"),
    (r"\.md$", "file"),
]

# Files that legitimately live *directly* under ``output/`` and must therefore
# never be reported as strays.  Both are produced by
# :func:`src.exporters.manifest.update_output_manifest`:
#   * ``README.md``         — the aggregated output manifest.
#   * ``course_graph.json`` — the optional course-graph export.
_OUTPUT_ROOT_SAFE_FILES: frozenset[str] = frozenset(
    {
        "README.md",
        "course_graph.json",
    }
)

# OS / editor artefacts that regularly appear at the output root and must be
# ignored by the stray scan.  Listed explicitly (instead of skipping every
# dotfile) so that genuine strays such as a stray ``.gitignore`` are still
# detected via ``_STRAY_PATTERNS``.
_OUTPUT_ROOT_IGNORED_FILES: frozenset[str] = frozenset(
    {
        ".DS_Store",
        "Thumbs.db",
        "desktop.ini",
    }
)


def _annotate_iter_exhaustion(
    raw_error: str,
    agent_role_env_key: str,
    *,
    parent_error: BaseException | None = None,
) -> str:
    """If *raw_error* signals iteration exhaustion, append a hint.

    CrewAI emits ``"Maximum iterations reached"`` (or wraps it in an
    ``Invalid response`` chain) when an agent consumes all of its
    ``max_iter`` budget.  This function detects that marker and appends
    a human-readable suggestion pinned to the appropriate env var::

        AGENT_{agent_role_env_key}_MAX_ITER

    Parameters
    ----------
    raw_error : str
        The original error message text.
    agent_role_env_key : str
        Uppercase role constant used in env-var names
        (e.g. ``"QA_REVIEWER"``).
    parent_error : BaseException or None
        When available, its class name is included in the hint so the
        operator can distinguish a timeout from a max-iter ceiling.

    Returns
    -------
    str
        *raw_error* unchanged if no exhaustion marker is found; otherwise
        *raw_error* plus a newline-separated hint.
    """
    if _MAX_ITER_CREWAI_MARKER not in raw_error:
        return raw_error

    env_var = f"AGENT_{agent_role_env_key}_MAX_ITER"

    parts: list[str] = [
        raw_error,
        "",
        "─" * 60,
        "⚠️  ITERATION LIMIT EXHAUSTION DETECTED",
        "",
        "   The agent exceeded its max_iter budget.  CrewAI emitted:",
        f'   "{_MAX_ITER_CREWAI_MARKER}"',
        "",
        "   👉  Increase the limit by setting this in your .env file:",
        f"       {env_var}=<higher_value>",
    ]

    if parent_error is not None:
        parts.append(f"   (Wrapped exception: {type(parent_error).__name__})")

    return "\n".join(parts)


def _classify_llm_error(exc: BaseException) -> str:
    """Classify *exc* into one of three buckets.

    Returns
    -------
    str
        ``"futile"`` when the model returned NULL/empty content (a repeat call
        will almost certainly fail identically — abort, don't burn credits);
        ``"retryable"`` for transient infrastructure failures (rate limiting,
        timeouts, connection resets); ``"deterministic"`` for everything else
        (e.g. a bad tool name), which is re-raised immediately.
    """
    msg = str(exc)

    # An exhausted reasoning budget / unreliable model surfaces as an empty
    # LLM response.  This is the failure mode that motivated the fail-fast
    # behaviour: retrying it is pure credit burn.
    if any(marker in msg for marker in _EMPTY_LLM_RESPONSE_MARKERS):
        return _LLM_ERROR_FUTILE

    if getattr(exc.__class__, "__module__", "").startswith("litellm"):
        return _LLM_ERROR_RETRYABLE

    lowered = msg.lower()
    for fragment in (
        "rate limit",
        "429",
        "too many requests",
        "timed out",
        "timeout",
        "connection",
    ):
        if fragment in lowered:
            return _LLM_ERROR_RETRYABLE

    return _LLM_ERROR_DETERMINISTIC


def _is_futile_llm_error(exc: BaseException) -> bool:
    """Return True when *exc* is a futile (non-recoverable) LLM failure.

    Futile failures are not worth retrying: the model has already proven it
    cannot produce usable output for this request.
    """
    return _classify_llm_error(exc) == _LLM_ERROR_FUTILE


def _is_transient_llm_error(exc: BaseException) -> bool:
    """Return True when *exc* signals an LLM failure that *may* resolve on retry.

    Kept as the union of the ``retryable`` and ``futile`` buckets for backward
    compatibility with callers that only ask "is this an LLM hiccup?".  New
    code should prefer :func:`_classify_llm_error` so that futile failures can
    be aborted instead of retried.
    """
    return _classify_llm_error(exc) in (_LLM_ERROR_RETRYABLE, _LLM_ERROR_FUTILE)


def _annotate_empty_llm_response(raw_error: str, agent_role_env_key: str) -> str:
    """Append an actionable hint when *raw_error* is an empty LLM response.

    CrewAI raises ``ValueError("Invalid response from LLM call - None or
    empty.")`` when the model returns no usable content.  Unlike the max-iter
    hint (see :func:`_annotate_iter_exhaustion`), this is usually a *model or
    token-budget* problem rather than an iteration-budget problem, so the
    suggested fixes are different.
    """
    if not any(marker in raw_error for marker in _EMPTY_LLM_RESPONSE_MARKERS):
        return raw_error

    parts: list[str] = [
        raw_error,
        "",
        "─" * 60,
        "⚠️  EMPTY LLM RESPONSE DETECTED",
        "",
        "   CrewAI received None/empty content from the model.  Common causes:",
        "   1. The model spent its whole max_tokens budget on internal reasoning",
        "      and produced no visible output (raise the token limit).",
        "   2. The configured model is unreliable or overloaded (switch models).",
        "   3. The review context grew too large (review files in batches).",
        "",
        "   👉  Try these in .env:",
        f"       AGENT_{agent_role_env_key}_MAX_TOKENS=16384",
        f"       AGENT_{agent_role_env_key}_MODEL=<a more stable model>",
    ]
    return "\n".join(parts)


def _kickoff_with_retry(
    build_crew: Callable[[], Crew],
    *,
    attempts: int = 3,
    backoff_seconds: float = 2.0,
    verbose: bool = False,
    agent_role: str = "AGENT",
    futile_attempts: int | None = None,
) -> object:
    """Run a crew with bounded retries, aborting fast on *futile* failures.

    CrewAI's per-agent ``max_retry_limit`` only re-runs a task *within* a
    single crew.  A whole-crew failure (e.g. the QA Reviewer hitting an empty
    LLM response after many file reads) currently aborts the run.  This helper
    re-builds and re-runs the crew a small number of times, sleeping with
    exponential backoff between attempts, so a single transient provider error
    no longer kills the pipeline.

    Crucially, it distinguishes two very different failure classes
    (see :func:`_classify_llm_error`):

    * **retryable** (rate limit, timeout, connection reset) — retried up to
      *attempts* times, because the next attempt genuinely may succeed.
    * **futile** (``None``/empty model response) — retried at most
      *futile_attempts* times (default 1, i.e. abort immediately).  A model
      that answered with NULL content does so again, so repeating the call is
      pure credit burn.  On exhaustion a :class:`FatalLLMError` is raised,
      which unwinds to the run-level circuit breaker.

    Parameters
    ----------
    build_crew : Callable[[], Crew]
        A zero-arg factory that constructs a fresh ``Crew``.  Rebuilding on
        each attempt avoids reusing per-execution state from a failed run.
    attempts : int
        Maximum number of kickoff attempts for *retryable* failures (default 3).
    backoff_seconds : float
        Initial backoff delay, doubled on each retry.
    verbose : bool
        When True, log each retry to stderr.
    agent_role : str
        Agent role used to build the actionable ``.env`` hint on a futile abort.
    futile_attempts : int or None
        Maximum attempts for *futile* failures.  Defaults to
        ``AGENT_LLM_FUTILE_RETRIES`` (1 = abort on the first indication).

    Returns
    -------
    object
        The result of ``crew.kickoff()`` on success (typically a
        ``CrewOutput``).

    Raises
    ------
    FatalLLMError
        When a futile (empty-response) failure exhausts *futile_attempts*.
    Exception
        The last exception when retryable attempts are exhausted, or
        immediately for deterministic errors.
    """
    futile_budget = (
        futile_attempts
        if futile_attempts is not None
        else _resolve_futile_attempts()
    )
    last_exc: BaseException | None = None
    futile_seen = 0

    for attempt in range(1, attempts + 1):
        try:
            crew = build_crew()
            return crew.kickoff()
        except Exception as exc:  # noqa: BLE001 — re-raised after attempts
            last_exc = exc
            bucket = _classify_llm_error(exc)

            # Deterministic failures (bad tool name, schema mismatch, …) would
            # reproduce identically — never retry, never classify as futile.
            if bucket == _LLM_ERROR_DETERMINISTIC:
                raise

            if bucket == _LLM_ERROR_FUTILE:
                futile_seen += 1
                # Fail fast: an empty response means the model could not
                # produce usable output for this request.  Retrying it burns
                # credits without changing the outcome.
                if futile_seen >= futile_budget:
                    raise FatalLLMError(
                        _annotate_empty_llm_response(str(exc), agent_role),
                        agent_role=agent_role,
                    ) from exc
                delay = backoff_seconds * (2 ** (futile_seen - 1))
                if verbose:
                    print(
                        f"  ⚠️  Empty LLM response on attempt "
                        f"{futile_seen}/{futile_budget} ({type(exc).__name__}); "
                        f"retrying in {delay:.0f}s…",
                        file=sys.stderr,
                    )
                time.sleep(delay)
                continue

            # Retryable infrastructure failure (rate limit / timeout / reset).
            if attempt >= attempts:
                raise
            delay = backoff_seconds * (2 ** (attempt - 1))
            if verbose:
                print(
                    f"  ⚠️  Transient LLM failure on attempt {attempt}/{attempts} "
                    f"({type(exc).__name__}); retrying in {delay:.0f}s…",
                    file=sys.stderr,
                )
            time.sleep(delay)

    # Reached only when the loop budget is exhausted without returning.
    assert last_exc is not None
    if _is_futile_llm_error(last_exc):
        raise FatalLLMError(
            _annotate_empty_llm_response(str(last_exc), agent_role),
            agent_role=agent_role,
        ) from last_exc
    raise last_exc


# ---------------------------------------------------------------------------
# Run-level circuit breaker — stops the pipeline on the first futile failure
# ---------------------------------------------------------------------------


def _print_abort_banner(*, stage: str, agent_role: str, detail: str) -> None:
    """Print a loud, unmissable banner explaining why the run was aborted."""
    print(
        "\n"
        + "=" * 74
        + "\n"
        + "  ⛔  RUN ABORTED — FUTILE LLM FAILURE (NO FURTHER CREDITS SPENT)\n"
        + "=" * 74
        + "\n"
        + f"  Stage      : {stage}\n"
        + f"  Agent      : {agent_role}\n"
        + "  Diagnosis  : the model returned None/empty content.  Repeating\n"
        + "               this call cannot succeed, so every remaining stage\n"
        + "               (theory tiers, lesson plans, presentations, labs,\n"
        + "               QA) is skipped instead of burning credits.\n"
        + "\n"
        + detail
        + "\n"
        + "=" * 74
        + "\n",
        file=sys.stderr,
    )


class _FatalAbortGuard:
    """Run-scoped circuit breaker for futile LLM failures.

    Once :meth:`trip` is called, every remaining generation stage is skipped.
    This is the credit-saving mechanism: a broken model configuration, an
    exhausted reasoning budget, or an unavailable model would otherwise cause
    each subsequent agent to spend a full — and equally futile — API call.
    """

    def __init__(self, *, verbose: bool = False) -> None:
        self.verbose = verbose
        self._stage: str | None = None
        self._agent_role: str | None = None
        self._detail: str | None = None

    @property
    def tripped(self) -> bool:
        """True once a futile failure has been recorded."""
        return self._stage is not None

    @property
    def stage(self) -> str | None:
        """Human-readable label of the stage that tripped the breaker."""
        return self._stage

    @property
    def agent_role(self) -> str | None:
        """Agent role that produced the futile failure."""
        return self._agent_role

    @property
    def detail(self) -> str | None:
        """Annotated error text (includes the actionable ``.env`` hints)."""
        return self._detail

    @property
    def reason(self) -> str | None:
        """One-line summary suitable for ``CrewResult.abort_reason``."""
        if not self.tripped:
            return None
        return (
            f"Aborted during {self._stage} ({self._agent_role}): the model "
            "returned None/empty content. Remaining stages were skipped to "
            "avoid burning credits — see stderr for the actionable fix."
        )

    def trip(
        self,
        *,
        stage: str,
        agent_role: str,
        exc: BaseException,
    ) -> str:
        """Record a futile failure (idempotent) and return :attr:`reason`."""
        if not self.tripped:
            self._stage = stage
            self._agent_role = agent_role
            self._detail = _annotate_empty_llm_response(str(exc), agent_role)
            _print_abort_banner(
                stage=stage, agent_role=agent_role, detail=self._detail
            )
        return self.reason or ""

    def should_skip(self, stage_label: str) -> bool:
        """Return True (and log) when *stage_label* must be skipped."""
        if not self.tripped:
            return False
        if self.verbose:
            print(
                f"  ⏭️  {stage_label}: skipped — run aborted during "
                f"{self._stage} to avoid burning credits."
            )
        return True


# ---------------------------------------------------------------------------
# Pre-flight configuration gate — early warning before any credits are spent
# ---------------------------------------------------------------------------

_SKIP_PREFLIGHT_ENV: str = "SYLLABUS_SKIP_PREFLIGHT"


def _preflight_enabled() -> bool:
    """Return True unless the operator explicitly disabled the pre-flight audit."""
    raw = os.getenv(_SKIP_PREFLIGHT_ENV, "").strip().lower()
    return raw not in ("1", "true", "yes", "on")


def _run_preflight_audit(
    *,
    verbose: bool = False,
    strict: bool = True,
) -> list[ConfigIssue]:
    """Statically audit every pipeline agent's LLM configuration.

    Costs nothing (no API calls) and runs *before* any agent is invoked, so a
    configuration that is known to produce ``"Invalid response from LLM call -
    None or empty"`` is caught while the credit balance is still untouched.

    Parameters
    ----------
    verbose : bool
        When True, print every finding (fatal and warning) to stderr.
    strict : bool
        When True, a fatal finding blocks the run entirely.

    Returns
    -------
    list[ConfigIssue]
        Every finding produced by the audit (possibly empty).
    """
    issues: list[ConfigIssue] = audit_agent_configs(_PREFLIGHT_ROLES)
    fatal = has_fatal_config_issues(issues)

    if issues and verbose:
        print("\n  🔎  Pre-flight configuration audit", file=sys.stderr)
        print(format_config_issues(issues), file=sys.stderr)

    if fatal and strict:
        print(
            "\n"
            + "=" * 74
            + "\n"
            + "  ⛔  PRE-FLIGHT CHECK FAILED — RUN NOT STARTED (0 credits spent)\n"
            + "=" * 74
            + "\n"
            + format_config_issues(issues)
            + "\n\n"
            + "  Fix the .env entries above, then re-run.\n"
            + f"  To bypass this gate: export {_SKIP_PREFLIGHT_ENV}=1\n"
            + "=" * 74
            + "\n",
            file=sys.stderr,
        )

    return issues


def _run_live_model_probe(
    *,
    verbose: bool = False,
    strict: bool = True,
) -> list[ProbeResult]:
    """Probe every configured model with one cheap tool-calling request.

    This is the live counterpart to :func:`_run_preflight_audit`.  Where the
    audit catches *misconfiguration*, the probe catches a *broken model*: one
    that accepts a tool-calling request (CrewAI's ``call_llm_native_tools``
    listener) and answers with nothing.  Detecting that here costs a single
    tiny request per model instead of an entire generation cycle.

    Only an ``empty`` outcome is fatal; transport/HTTP problems are surfaced as
    warnings, never as a hard stop, so a flaky probe cannot block a run that
    would otherwise succeed.

    Parameters
    ----------
    verbose : bool
        When True, print the full probe report to stderr.
    strict : bool
        When True, an empty-response probe blocks the run.

    Returns
    -------
    list[ProbeResult]
        One result per distinct model (models are deduplicated).
    """
    results = probe_agent_models(_PREFLIGHT_ROLES)
    fatal = fatal_probe_results(results)

    if verbose and results:
        print("\n  🔎  Pre-flight model probe (tool-calling path)", file=sys.stderr)
        print(format_probe_report(results), file=sys.stderr)

        for result in results:
            if result.is_unreachable:
                print(
                    f"  ⚠️  Probe could not reach '{result.model}': {result.detail}",
                    file=sys.stderr,
                )

    if fatal and strict:
        details = "\n".join(
            f"  ⛔  {result.model}  ({result.detail})" for result in fatal
        )
        print(
            "\n"
            + "=" * 74
            + "\n"
            + "  ⛔  PRE-FLIGHT MODEL PROBE FAILED — RUN NOT STARTED (0 credits)\n"
            + "=" * 74
            + "\n"
            + "  The following models returned neither content nor a tool call for\n"
            + "  a trivial tool-calling request.  This is the exact condition that\n"
            + "  CrewAI reports as 'Invalid response from LLM call - None or\n"
            + "  empty.', so running the pipeline would burn credits for nothing:\n"
            + "\n"
            + details
            + "\n\n"
            + "  👉  Switch the affected agent(s) to a different model in .env, e.g.\n"
            + "      AGENT_THEORY_INSTRUCTOR_MODEL=<a more reliable model>\n"
            + f"  To bypass this gate: export {_SKIP_PREFLIGHT_ENV}=1\n"
            + "=" * 74
            + "\n",
            file=sys.stderr,
        )

    return results


def _scan_for_stray_generated_files(
    *,
    run_id: str,
    verbose: bool = False,
) -> list[str]:
    """Detect agent-generated files written straight into the ``output/`` root.

    Every generated artefact belongs **inside** a per-run directory
    (``output/<run_id>/``).  When an agent bypasses the ``write-labs`` /
    ``write-syllabus`` commands and writes directly into ``output/``, the
    shared output root is polluted with loose artefacts (``index.html``,
    ``tier1_foundations/``, …) that are mistaken for — or mixed up with —
    run folders.

    The scan is deliberately scoped to the **immediate children of
    ``OUTPUT_ROOT``** and never looks anywhere else.  Permanent repository
    files at the project root (``README.md``, ``DESIGN.md``, ``.gitignore``,
    ``docs/``, …) are therefore never inspected and can never be reported,
    and content nested inside a run directory is — by definition — in the
    correct location and is not inspected either.

    The two files that *do* legitimately live at the output root
    (``output/README.md`` and ``output/course_graph.json``, both written by
    :func:`src.exporters.manifest.update_output_manifest`) are whitelisted
    via ``_OUTPUT_ROOT_SAFE_FILES``.

    Parameters
    ----------
    run_id : str
        The per-run identifier used for this pipeline execution.  Its
        directory (``output/<run_id>/``) is never reported.
    verbose : bool
        When ``True``, prints the warnings to stderr immediately.

    Returns
    -------
    list[str]
        Human-readable warning strings (one per stray).  Empty if clean.
    """
    warnings_list: list[str] = []
    output_root = OUTPUT_ROOT

    # Nothing to scan when no run has produced output yet.
    if not output_root.is_dir():
        return warnings_list

    for entry in sorted(output_root.iterdir()):
        name = entry.name

        # 1. The current run directory is exactly where generated artefacts
        #    belong, so it is never a stray.
        if name == run_id:
            continue
        # 2. Files that legitimately live directly under output/ (the
        #    aggregate manifest and the optional course graph).
        if name in _OUTPUT_ROOT_SAFE_FILES:
            continue
        # 3. OS / editor artefacts (e.g. .DS_Store) are not generated by us.
        #    NOTE: this is an explicit ignore-list rather than a blanket
        #    "skip dotfiles" rule, so genuine strays such as a stray
        #    ``.gitignore`` are still detected via ``_STRAY_PATTERNS``.
        if name in _OUTPUT_ROOT_IGNORED_FILES:
            continue

        for pattern, kind in _STRAY_PATTERNS:
            if re.search(pattern, name):
                msg = (
                    f"⚠️  Stray generated {kind} detected: {entry}\n"
                    f"   This {kind} was written directly into output/ instead\n"
                    f"   of under output/{run_id}/\n"
                    f"   To clean up:  rm -rf '{entry}'\n"
                    f"   To prevent recurrence: ensure all agents use the\n"
                    f"   'write-labs' command with 'run_id' instead of 'write-file'."
                )
                warnings_list.append(msg)
                if verbose:
                    print(msg, file=sys.stderr)
                break

    return warnings_list


def _create_lab_scaffolding(labs_base_path: Path) -> Path:
    """Create the tiered lab directory scaffolding under *labs_base_path*."""
    base = labs_base_path
    base.mkdir(parents=True, exist_ok=True)

    for tier_dir_name, tier_label in _TIERS:
        tier_path = base / tier_dir_name
        starter_path = tier_path / "starter"
        solution_path = tier_path / "solution"
        starter_path.mkdir(parents=True, exist_ok=True)
        solution_path.mkdir(parents=True, exist_ok=True)

        (starter_path / ".gitkeep").touch(exist_ok=True)
        (solution_path / ".gitkeep").touch(exist_ok=True)

        tier_readme = tier_path / "README.md"
        if not tier_readme.exists():
            tier_readme.write_text(
                f"# {tier_label}\n\n"
                f"Labs for this tier will be generated here.\n\n"
                f"- **starter/** — Scaffolded exercises with TODO markers.\n"
                f"- **solution/** — Fully-commented reference implementations.\n",
                encoding="utf-8",
            )

    return base


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class CrewResult:
    """Holds the result of a full syllabus + theory + labs crew run."""

    def __init__(
        self,
        syllabus_path: Path,
        labs_base_path: Path,
        syllabus_ok: bool = True,
        labs_ok: bool = True,
        syllabus_error: str | None = None,
        labs_error: str | None = None,
        manifest_path: Path | None = None,
        qa_ok: bool = True,
        qa_error: str | None = None,
        qa_report: str | None = None,
        theory_ok: bool = True,
        theory_error: str | None = None,
        syllabus_review_ok: bool = True,
        syllabus_review_error: str | None = None,
        syllabus_review_report: str | None = None,
        human_feedback_requested: bool = False,
        human_feedback_summary: str | None = None,
        lesson_plan_ok: bool = True,
        lesson_plan_error: str | None = None,
        presentation_ok: bool = True,
        presentation_error: str | None = None,
        aborted: bool = False,
        abort_reason: str | None = None,
    ) -> None:
        self.syllabus_path = syllabus_path
        self.labs_base_path = labs_base_path
        self.syllabus_ok = syllabus_ok
        self.labs_ok = labs_ok
        self.syllabus_error = syllabus_error
        self.labs_error = labs_error
        self.manifest_path = manifest_path
        self.qa_ok = qa_ok
        self.qa_error = qa_error
        self.qa_report = qa_report
        self.theory_ok = theory_ok
        self.theory_error = theory_error
        self.syllabus_review_ok = syllabus_review_ok
        self.syllabus_review_error = syllabus_review_error
        self.syllabus_review_report = syllabus_review_report
        self.human_feedback_requested = human_feedback_requested
        self.human_feedback_summary = human_feedback_summary
        self.lesson_plan_ok = lesson_plan_ok
        self.lesson_plan_error = lesson_plan_error
        self.presentation_ok = presentation_ok
        self.presentation_error = presentation_error
        self.aborted = aborted
        self.abort_reason = abort_reason

    @property
    def all_succeeded(self) -> bool:
        return (
            not self.aborted
            and self.syllabus_ok
            and self.syllabus_review_ok
            and self.theory_ok
            and self.labs_ok
            and self.qa_ok
            and self.lesson_plan_ok
            and self.presentation_ok
        )


# ---------------------------------------------------------------------------
# Crew runner
# ---------------------------------------------------------------------------


def generate_run_id(course_safe_name: str) -> str:
    """Generate a unique, human-readable run identifier.

    Format: ``YYYY-MM-DD_HHMMSS_<course_safe_name>``
    """
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")
    return f"{timestamp}_{course_safe_name}"


def _find_syllabus_in_dir(resume_dir: Path) -> Path:
    """Locate the syllabus markdown file inside a resume directory.

    Looks for a ``.md`` file inside ``<resume_dir>/syllabus/``.
    Returns the first match found.

    Raises
    ------
    FileNotFoundError
        If the resume directory or syllabus subdirectory doesn't exist,
        or if no ``.md`` file is found.
    """
    if not resume_dir.exists():
        raise FileNotFoundError(f"Resume directory not found: {resume_dir}")

    if not resume_dir.is_dir():
        raise FileNotFoundError(
            f"Resume path is not a directory (expected a run directory): {resume_dir}"
        )

    syllabus_subdir = resume_dir / "syllabus"
    if not syllabus_subdir.is_dir():
        raise FileNotFoundError(
            f"No 'syllabus/' subdirectory found in resume directory: {resume_dir}"
        )

    md_files = sorted(syllabus_subdir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md syllabus file found in: {syllabus_subdir}")

    return md_files[0]


# ===========================================================================
# Generation State Management — resume/restart progress tracking
# ===========================================================================

_STATE_FILE_NAME: str = "_generation_state.json"


def load_generation_state(run_dir: Path) -> GenerationState | None:
    """Load the generation progress state from a run directory.

    If ``_generation_state.json`` exists, deserialises it.  Otherwise,
    auto-generates a state by scanning the filesystem for existing lab
    and theory files (backward-compatible with pre-state-file runs).

    Parameters
    ----------
    run_dir : Path
        The run output directory (e.g. ``output/2026-09-02_191946_MyCourse/``).

    Returns
    -------
    GenerationState or None
        The loaded or auto-generated state, or ``None`` if the run
        directory does not exist at all.
    """
    if not run_dir.exists():
        return None

    state_path = run_dir / _STATE_FILE_NAME
    course_name = (
        run_dir.name.split("_", 2)[-1] if len(run_dir.name.split("_", 2)) >= 3 else run_dir.name
    )

    if state_path.exists():
        try:
            return GenerationState.model_validate_json(state_path.read_text(encoding="utf-8"))
        except Exception:
            # Corrupt state file — fall back to auto-detection.
            pass

    # ── Auto-generate from filesystem (backward compat) ──────────────
    labs_path = run_dir / "labs"
    state = GenerationState(run_id=run_dir.name, course_name=course_name)

    for tier_dir_name, _label in _TIERS:
        tier_labs_path = labs_path / tier_dir_name

        # Lab files check
        if _is_tier_labs_complete(tier_labs_path):
            state.tiers[tier_dir_name] = TierState(
                status="complete",
                files=_count_lab_files(tier_labs_path),
            )
        else:
            state.tiers[tier_dir_name] = TierState(status="incomplete")

        # Theory files check
        state.theory[tier_dir_name] = (
            "complete" if _is_tier_theory_complete(tier_labs_path) else "incomplete"
        )

    return state


def save_generation_state(run_dir: Path, state: GenerationState) -> None:
    """Persist the generation state to disk atomically.

    Writes to a temp file first, then renames to avoid corruption on
    partial writes.
    """
    state_path = run_dir / _STATE_FILE_NAME
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    tmp_path.rename(state_path)


def _is_tier_labs_complete(tier_labs_path: Path) -> bool:
    """Check whether a tier has real lab files (not just .gitkeep).

    Returns ``True`` when both ``starter/`` and ``solution/`` contain
    at least one file that is not a hidden/dot file.
    """
    for subdir in ("starter", "solution"):
        sub_path = tier_labs_path / subdir
        if not sub_path.exists():
            return False
        real_files = [f for f in sub_path.iterdir() if f.is_file() and not f.name.startswith(".")]
        if not real_files:
            return False
    return True


def _is_tier_theory_complete(tier_labs_path: Path) -> bool:
    """Check whether a tier has theory artifacts.

    Returns ``True`` when ``theory/`` exists and contains at least one
    non-hidden file.
    """
    theory_path = tier_labs_path / "theory"
    if not theory_path.exists():
        return False
    real_files = [f for f in theory_path.iterdir() if f.is_file() and not f.name.startswith(".")]
    return bool(real_files)


def _count_lab_files(tier_labs_path: Path) -> int:
    """Count non-hidden lab files across starter/ and solution/.

    Returns the total number of real (non-``.gitkeep``, non-hidden)
    files found in the tier's lab directories.
    """
    count = 0
    for subdir in ("starter", "solution"):
        sub_path = tier_labs_path / subdir
        if not sub_path.exists():
            continue
        count += sum(1 for f in sub_path.iterdir() if f.is_file() and not f.name.startswith("."))
    return count


def _build_top_level_lab_readme(
    course_name: str,
    primary_language: str,
    labs_base_path: Path,
) -> str:
    """Build a top-level README.md index for all generated labs.

    Scans the lab directory tree and produces a Markdown index with
    a numbered list of every lab, its tier, and a one-line description.
    """
    lines: list[str] = [
        f"# {course_name} — Coding Labs",
        "",
        f"**Primary Language:** {primary_language}",
        "",
        "## Overview",
        "",
        "This directory contains hands-on coding labs organised into three "
        "progressive tiers. Each lab includes a **starter/** scaffold with "
        "TODO markers and a **solution/** reference implementation.",
        "",
        "## Lab Index",
        "",
    ]

    tier_labels = {
        "tier1_foundations": "Tier 1 — Foundations",
        "tier2_application": "Tier 2 — Application",
        "tier3_architecture": "Tier 3 — Architecture",
    }

    lab_num = 0
    for tier_dir_name in ["tier1_foundations", "tier2_application", "tier3_architecture"]:
        tier_path = labs_base_path / tier_dir_name
        if not tier_path.exists():
            continue

        label = tier_labels.get(tier_dir_name, tier_dir_name)
        lines.append(f"### {label}")
        lines.append("")

        # Look for lab files in the solution directory.
        solution_dir = tier_path / "solution"
        if solution_dir.exists():
            for f in sorted(solution_dir.iterdir()):
                if f.name.startswith(".") or f.name == "README.md":
                    continue
                if f.is_file():
                    lab_num += 1
                    # Derive a readable lab name from the filename.
                    lab_name = f.stem.replace("_", " ").replace("-", " ").title()
                    lines.append(
                        f"{lab_num}. **{lab_name}** — "
                        f"`{tier_dir_name}/starter/{f.name}` / "
                        f"`{tier_dir_name}/solution/{f.name}`"
                    )
        lines.append("")

    if lab_num == 0:
        lines.append("*Labs are still being generated. Check back after the pipeline completes.*")

    lines.extend(
        [
            "## Getting Started",
            "",
            "1. Navigate to any lab's `starter/` directory.",
            "2. Read the `README.md` for learning objectives and instructions.",
            "3. Complete the TODO markers in the starter files.",
            "4. Compare your solution with the `solution/` directory.",
            "",
            "## Humanics Literacies",
            "",
            "- **[T] Technological Literacy** — Every lab requires writing, "
            "debugging, and running real code.",
            "- **[D] Data Literacy** — Labs include data processing, analysis, "
            "and evidence-based decision making.",
            "- **[H] Human Literacy** — Every README includes ethics, "
            "accessibility, and collaboration reflection prompts.",
        ]
    )

    return "\n".join(lines) + "\n"


def run_syllabus_crew(
    course_context: str,
    *,
    course_name: str = "",
    primary_language: str = "Python",
    material_language: str = "Dutch",
    verbose: bool = False,
    architect_agent: Agent | None = None,
    lab_dev_agent: Agent | None = None,
    skip_syllabus_review: bool = False,
    skip_theory: bool = False,
    skip_labs: bool = False,
    skip_qa: bool = False,
    skip_lesson_plans: bool = False,
    skip_presentations: bool = False,
    resume_dir: str | Path | None = None,
    run_id: str | None = None,
    human_feedback: str | None = None,
) -> CrewResult:
    """Run all agents sequentially and return a full result summary.

    Execution order:
      1. Curriculum Architect generates a syllabus (or loads from disk if
         *resume_dir* is provided).
      2. Education Director audits the syllabus for time-budget math,
         workload realism, scheduling sanity, and MBO4 appropriateness.
         Can delegate fixes back to the Curriculum Architect.
      3. Theory Instructor generates interactive theory artifacts from the
         syllabus.
      4. Lab & Project Developer processes the syllabus to generate tiered labs.
      5. QA Reviewer inspects all generated labs and delegates fixes if needed.

    All output is scoped under ``output/<run_id>/`` where *run_id* is a
    timestamp + course-slug combination, ensuring every pipeline run
    produces a unique, non-overlapping directory.

    Parameters
    ----------
    course_context : str
        Rich course context string (from the Intake Specialist) containing
        tech stack, kerntaken emphasis, student profile, and pedagogical
        notes.  This is the primary input for syllabus generation.
    course_name : str
        Short course name / title used for file naming and directory
        scaffolding.  When empty, extracted from *course_context*.
    primary_language : str
        The exact programming language for coding labs (e.g. 'JavaScript',
        'Python').  Passed through to the Lab Developer task so file
        extensions, linters, and tooling references match the language.
    verbose : bool
        Enable detailed agent and task logging.
    architect_agent : Agent or None
        Pre-built Curriculum Architect agent; lazily created when None.
    lab_dev_agent : Agent or None
        Pre-built Lab Developer agent; lazily created when None.
    skip_syllabus_review : bool
        If True, skip the Education Director feasibility audit step.
    skip_theory : bool
        If True, skip the Theory Instructor step.
    skip_labs : bool
        If True, only run the Curriculum Architect (backward-compatible).
    skip_qa : bool
        If True, skip the QA review step.
    resume_dir : str, Path, or None
        Path to a previous run directory (e.g.
        ``output/2026-08-22_153000_Course_Name``). When provided, the
        Curriculum Architect is **skipped** and the existing syllabus is
        loaded from disk instead. This saves API costs and enables a
        human-in-the-loop workflow where the syllabus can be manually
        edited before generating labs.

    Returns
    -------
    CrewResult
        Container with paths, status flags, and any error messages.
    """
    # ── Pre-flight: audit every agent's LLM config before spending anything ─
    # A reasoning model with an undersized max_tokens budget (or a missing
    # model) reliably produces "Invalid response from LLM call - None or
    # empty".  Catching that here costs nothing and saves a full pipeline run.
    if _preflight_enabled():
        preflight_issues = _run_preflight_audit(verbose=verbose, strict=True)
        if has_fatal_config_issues(preflight_issues):
            raise FatalLLMError(
                "Pre-flight configuration audit failed: "
                + "; ".join(
                    issue.message
                    for issue in preflight_issues
                    if issue.severity == "fatal"
                ),
                stage="pre-flight",
            )

    # ── Pre-flight: live probe of the tool-calling path ────────────────────
    # Catches a *broken model* — one that answers a tool-calling request with
    # nothing — before a single generation credit is spent.
    if probe_enabled():
        fatal_probes = fatal_probe_results(
            _run_live_model_probe(verbose=verbose, strict=True)
        )
        if fatal_probes:
            raise FatalLLMError(
                "Pre-flight model probe failed: "
                + "; ".join(
                    f"{result.model} ({result.role}) returned an empty response"
                    for result in fatal_probes
                ),
                stage="pre-flight probe",
            )

    # Run-scoped circuit breaker: once tripped, every remaining generation
    # stage is skipped, so a broken model cannot burn credits on work that is
    # guaranteed to fail exactly the same way.
    abort_guard = _FatalAbortGuard(verbose=verbose)

    def _trip_guard(stage: str, agent_role: str, exc: BaseException) -> None:
        """Record a futile failure so all remaining stages are skipped."""
        abort_guard.trip(stage=stage, agent_role=agent_role, exc=exc)

    # Extract course_name from context if not explicitly provided.
    if not course_name:
        # Use the first line or first 80 chars as a fallback name.
        first_line = course_context.strip().split("\n")[0]
        course_name = first_line.replace("Course Name:", "").strip()
        if not course_name or len(course_name) > 100:
            course_name = course_context.strip()[:80]

    safe_name = _sanitize_filename(course_name)

    # ── 0. Handle resume mode ──────────────────────────────────────────
    syllabus_ok = False
    syllabus_error: str | None = None
    syllabus_raw: str = ""
    syllabus_path: Path
    _active_run_id: str  # Always populated below — used for lab task context.

    if resume_dir is not None:
        # ── In-place resume: reuse the SAME run_id and directory ────
        resume_path = Path(resume_dir)
        _active_run_id = resume_path.name
        run_dir = resume_path

        try:
            syllabus_path = _find_syllabus_in_dir(resume_path)
            syllabus_raw = syllabus_path.read_text(encoding="utf-8").strip()

            if not syllabus_raw:
                raise RuntimeError(f"Syllabus file is empty: {syllabus_path}")

            syllabus_ok = True
            if verbose:
                print(f"  📄  Loaded syllabus from: {syllabus_path}")
                print(f"      ({len(syllabus_raw):,} characters)")

        except Exception as exc:
            # Fail loudly instead of fabricating a broken fallback path.  The
            # previous approach (``output/resume_failed/<name>.md``) did not
            # create its parent directory, so the later ``shutil.copy2`` would
            # crash with an unrelated ``FileNotFoundError``.  Surface the real
            # cause with actionable guidance instead.
            raise RuntimeError(
                "Failed to load syllabus from resume directory "
                f"{resume_path}: {exc}\n"
                "  → Pass the run *directory* (e.g. output/<run_id>/) to "
                "--resume-from, not a file such as intake_session.json.  Use "
                "--load-session to reuse a saved intake session."
            ) from exc

        # Use the existing labs directory (files are already in place).
        labs_base_path = resume_path / "labs"
        labs_base_path.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"  📁  Resuming in-place from: {resume_path}")

        # ── In resume mode, create an architect agent so the
        # syllabus review delegation pool has access to it.
        # The architect's LLM is instantiated here but only used
        # if the Education Director delegates back to it.
        if architect_agent is None:
            architect: Agent = get_architect(verbose=verbose)
        else:
            architect = architect_agent

    else:
        # ── Fresh run: create new run directory ────────────────────────
        if run_id is not None:
            # Use the run_id provided by the caller (e.g. main.py for
            # intake-session persistence).  The directory should already
            # exist at this point.
            _active_run_id = run_id
        else:
            _active_run_id = generate_run_id(safe_name)
        run_dir = OUTPUT_ROOT / _active_run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        syllabus_dir = run_dir / "syllabus"
        syllabus_dir.mkdir(parents=True, exist_ok=True)
        syllabus_path = syllabus_dir / f"{safe_name}.md"

        labs_dir = run_dir / "labs"
        labs_dir.mkdir(parents=True, exist_ok=True)
        labs_base_path = labs_dir

        # ── 1. Curriculum Architect ────────────────────────────────────
        if architect_agent is None:
            architect = get_architect(verbose=verbose)
        else:
            architect = architect_agent

        syllabus_task = create_syllabus_generation_task(
            agent=architect,
            course_name=course_name,
            course_context=course_context,
            material_language=material_language,
            human_feedback=human_feedback,
        )

        try:
            architect_crew = Crew(
                agents=[architect],
                tasks=[syllabus_task],
                process=Process.sequential,
                verbose=verbose,
            )
            architect_result = architect_crew.kickoff()
            syllabus_raw = (
                architect_result.raw if hasattr(architect_result, "raw") else str(architect_result)
            ).strip()

            if not syllabus_raw:
                raise RuntimeError(
                    "Curriculum Architect produced no output. "
                    "Check your API key, model availability, and network."
                )

            write_file(syllabus_path, syllabus_raw, force=True)
            syllabus_ok = True

        except Exception as exc:
            if _is_futile_llm_error(exc) or isinstance(exc, FatalLLMError):
                _trip_guard(
                    "Curriculum Architect (syllabus)",
                    "CURRICULUM_ARCHITECT",
                    exc,
                )
                syllabus_error = abort_guard.reason
            else:
                syllabus_error = _annotate_iter_exhaustion(
                    str(exc), "CURRICULUM_ARCHITECT", parent_error=exc
                )
            write_file(
                syllabus_path,
                f"# {course_name} — Syllabus Generation Failed\n\n**Error:** {syllabus_error}\n",
                force=True,
            )

    # ── 1.5. Education Director — Syllabus Feasibility Audit ───────────
    syllabus_review_ok = False
    syllabus_review_error: str | None = None
    syllabus_review_report: str | None = None

    if skip_syllabus_review:
        syllabus_review_ok = True
    elif abort_guard.should_skip("Syllabus Feasibility Audit"):
        syllabus_review_error = abort_guard.reason
    elif syllabus_raw:
        try:
            education_director = get_education_director(verbose=verbose)
            review_task = create_syllabus_review_task(
                agent=education_director,
                course_name=course_name,
                syllabus_context=syllabus_raw,
                material_language=material_language,
                human_feedback=human_feedback,
                verbose=verbose,
            )

            # CRITICAL: Both agents must be in the SAME Crew array so
            # the Education Director can delegate fixes back to the
            # Curriculum Architect.
            review_crew = Crew(
                agents=[architect, education_director],
                tasks=[review_task],
                process=Process.sequential,
                verbose=verbose,
            )
            review_result = review_crew.kickoff()
            syllabus_review_report = (
                review_result.raw if hasattr(review_result, "raw") else str(review_result)
            ).strip()

            if syllabus_review_report:
                syllabus_review_ok = True
                if verbose:
                    print("  ✅  Syllabus Feasibility Audit completed.")
            else:
                syllabus_review_error = "Education Director produced no output."

        except Exception as exc:
            if _is_futile_llm_error(exc):
                _trip_guard(
                    "Syllabus Feasibility Audit", "EDUCATION_DIRECTOR", exc
                )
                syllabus_review_error = abort_guard.reason
            else:
                syllabus_review_error = _annotate_iter_exhaustion(
                    str(exc), "EDUCATION_DIRECTOR", parent_error=exc
                )
            if verbose:
                print(f"  ❌  Syllabus Feasibility Audit failed: {exc}", file=sys.stderr)
    else:
        syllabus_review_error = "Skipped — Curriculum Architect produced no syllabus to audit."

    # ── 2. Theory Instructor (per-tier, with resume detection) ──────────
    theory_ok = False
    theory_error: str | None = None
    theory_instructor: Agent | None = None
    all_theory_tiers_ok = True

    if skip_theory:
        theory_ok = True
    elif abort_guard.should_skip("Theory artifacts"):
        theory_error = abort_guard.reason
    elif syllabus_raw:
        # ── Load generation state for resume detection ────────────────
        state = None if human_feedback else load_generation_state(run_dir)

        for tier_dir_name, tier_label in _TIERS:
            # ── Skip completed theory ──────────────────────────────────
            if state:
                tier_theory_status = state.theory.get(tier_dir_name)
                if tier_theory_status == "complete":
                    if verbose:
                        print(f"  ⏭️  Theory for {tier_dir_name}: already complete, skipping.")
                    continue

            # Also check filesystem (backward compat / first resume)
            tier_labs_path = labs_base_path / tier_dir_name
            if _is_tier_theory_complete(tier_labs_path):
                if state:
                    state.theory[tier_dir_name] = "complete"
                    save_generation_state(run_dir, state)
                if verbose:
                    print(f"  ⏭️  Theory for {tier_dir_name}: already complete, skipping.")
                continue

            # ── Generate theory for this tier ─────────────────────────
            try:
                if theory_instructor is None:
                    theory_instructor = get_theory_instructor(verbose=verbose)

                tier_theory_task = create_theory_task(
                    agent=theory_instructor,
                    course_name=course_name,
                    syllabus_context=syllabus_raw,
                    run_id=_active_run_id,
                    tier=tier_dir_name,
                    material_language=material_language,
                    human_feedback=human_feedback,
                    verbose=verbose,
                )

                def _build_tier_theory_crew(
                    _agent: Agent = theory_instructor,
                    _task: Task = tier_theory_task,
                ) -> Crew:
                    """Rebuild the per-tier theory crew for each retry attempt.

                    Default-argument binding is deliberate: it snapshots the
                    loop-scoped agent/task so the rebuilder never picks up a
                    later iteration's values.
                    """
                    return Crew(
                        agents=[_agent],
                        tasks=[_task],
                        process=Process.sequential,
                        verbose=verbose,
                    )

                # Fail fast on empty LLM responses (futile) while still
                # absorbing genuine provider blips (rate limits, timeouts).
                _kickoff_with_retry(
                    _build_tier_theory_crew,
                    attempts=3,
                    backoff_seconds=2.0,
                    verbose=verbose,
                    agent_role="THEORY_INSTRUCTOR",
                )

                # ── Validate the generated theory file ────────────────
                theory_dir = tier_labs_path / "theory"
                if theory_dir.exists():
                    tier_results = validate_theory_directory(theory_dir)
                    if tier_results:
                        report = format_validation_report(tier_results)
                        report_dir = run_dir / "theory"
                        report_dir.mkdir(parents=True, exist_ok=True)
                        report_path = report_dir / f"VALIDATION_REPORT_{tier_dir_name}.md"
                        report_path.write_text(report, encoding="utf-8")

                        tier_has_errors = any(r.error_count > 0 for r in tier_results)
                        if tier_has_errors:
                            all_theory_tiers_ok = False
                            if verbose:
                                for r in tier_results:
                                    if r.error_count > 0:
                                        print(
                                            f"  ⚠️  Theory validation: "
                                            f"{r.file_path.name} has "
                                            f"{r.error_count} error(s)"
                                        )
                            if state:
                                state.theory[tier_dir_name] = "failed"
                                save_generation_state(run_dir, state)
                            continue

                if state:
                    state.theory[tier_dir_name] = "complete"
                    save_generation_state(run_dir, state)
                if verbose:
                    print(f"  ✅  Theory for {tier_dir_name}: generated successfully.")

            except Exception as exc:
                all_theory_tiers_ok = False
                if state:
                    state.theory[tier_dir_name] = "failed"
                    save_generation_state(run_dir, state)

                if _is_futile_llm_error(exc) or isinstance(exc, FatalLLMError):
                    # Stop immediately: every remaining tier would fail exactly
                    # the same way and burn credits for nothing.
                    _trip_guard(
                        f"Theory artifacts ({tier_dir_name})",
                        "THEORY_INSTRUCTOR",
                        exc,
                    )
                    if verbose:
                        print(
                            f"  ⛔  Theory for {tier_dir_name} aborted — "
                            "remaining tiers skipped.",
                            file=sys.stderr,
                        )
                    break

                # Deterministic/other failure: annotate with both hints so the
                # operator sees the actionable .env fix either way.
                exc_msg = _annotate_iter_exhaustion(
                    _annotate_empty_llm_response(str(exc), "THEORY_INSTRUCTOR"),
                    "THEORY_INSTRUCTOR",
                    parent_error=exc,
                )
                if verbose:
                    print(
                        f"  ❌  Theory for {tier_dir_name} failed: {exc_msg}",
                        file=sys.stderr,
                    )

        theory_ok = all_theory_tiers_ok
        if not theory_ok:
            theory_error = (
                abort_guard.reason
                if abort_guard.tripped
                else "One or more theory tiers failed. See logs above for details."
            )

    else:
        theory_error = "Skipped — Curriculum Architect produced no syllabus to use as context."

    # ── 2.5. Lesson Plan Generation — Instructional Coordinator ──────────
    lesson_plan_ok = False
    lesson_plan_error: str | None = None

    if skip_lesson_plans:
        lesson_plan_ok = True
    elif abort_guard.should_skip("Lesson plans"):
        lesson_plan_error = abort_guard.reason
    elif syllabus_raw:
        lp_state = None if human_feedback else load_generation_state(run_dir)
        all_lp_tiers_ok = True
        lp_coordinator: Agent | None = None

        for tier_dir_name, tier_label in _TIERS:
            if lp_state:
                tier_lp_status = lp_state.lesson_plan.get(tier_dir_name)
                if tier_lp_status == "complete":
                    if verbose:
                        print(f"  ⏭️  Lesson Plan for {tier_dir_name}: already complete, skipping.")
                    continue

            try:
                if lp_coordinator is None:
                    lp_coordinator = get_instructional_coordinator(verbose=verbose)

                lp_task = create_lesson_plan_task(
                    agent=lp_coordinator,
                    course_name=course_name,
                    syllabus_context=syllabus_raw,
                    module_name=tier_label,
                    run_id=_active_run_id,
                    material_language=material_language,
                    human_feedback=human_feedback,
                    verbose=verbose,
                )
                lp_crew = Crew(
                    agents=[lp_coordinator],
                    tasks=[lp_task],
                    process=Process.sequential,
                    verbose=verbose,
                )
                lp_crew.kickoff()

                if lp_state:
                    lp_state.lesson_plan[tier_dir_name] = "complete"
                    save_generation_state(run_dir, lp_state)

                if verbose:
                    print(f"  ✅  Lesson Plan for {tier_dir_name} completed.")

            except Exception as exc:
                all_lp_tiers_ok = False
                if lp_state:
                    lp_state.lesson_plan[tier_dir_name] = "failed"
                    save_generation_state(run_dir, lp_state)
                if _is_futile_llm_error(exc):
                    _trip_guard(
                        f"Lesson plan ({tier_dir_name})",
                        "INSTRUCTIONAL_COORDINATOR",
                        exc,
                    )
                    if verbose:
                        print(
                            f"  ⛔  Lesson Plan for {tier_dir_name} aborted — "
                            "remaining tiers skipped.",
                            file=sys.stderr,
                        )
                    break
                if verbose:
                    print(f"  ❌  Lesson Plan for {tier_dir_name} failed: {exc}", file=sys.stderr)

        lesson_plan_ok = all_lp_tiers_ok
        if not lesson_plan_ok:
            lesson_plan_error = (
                abort_guard.reason
                if abort_guard.tripped
                else "One or more lesson plan tiers failed."
            )

    # ── 2.6. Presentation Generation — Presentation Designer ────────────
    presentation_ok = False
    presentation_error: str | None = None

    if skip_presentations:
        presentation_ok = True
    elif abort_guard.should_skip("Presentations"):
        presentation_error = abort_guard.reason
    elif syllabus_raw:
        pres_state = None if human_feedback else load_generation_state(run_dir)
        all_pres_tiers_ok = True
        pres_designer: Agent | None = None

        for tier_dir_name, tier_label in _TIERS:
            if pres_state:
                tier_pres_status = pres_state.presentation.get(tier_dir_name)
                if tier_pres_status == "complete":
                    if verbose:
                        print(f"  ⏭️  Presentation for {tier_dir_name}: already complete, skipping.")
                    continue

            try:
                if pres_designer is None:
                    pres_designer = get_presentation_designer(verbose=verbose)

                pres_task = create_presentation_task(
                    agent=pres_designer,
                    course_name=course_name,
                    syllabus_context=syllabus_raw,
                    module_name=tier_label,
                    run_id=_active_run_id,
                    material_language=material_language,
                    human_feedback=human_feedback,
                    verbose=verbose,
                )
                pres_crew = Crew(
                    agents=[pres_designer],
                    tasks=[pres_task],
                    process=Process.sequential,
                    verbose=verbose,
                )
                pres_crew.kickoff()

                if pres_state:
                    pres_state.presentation[tier_dir_name] = "complete"
                    save_generation_state(run_dir, pres_state)

                if verbose:
                    print(f"  ✅  Presentation for {tier_dir_name} completed.")

            except Exception as exc:
                all_pres_tiers_ok = False
                if pres_state:
                    pres_state.presentation[tier_dir_name] = "failed"
                    save_generation_state(run_dir, pres_state)
                if _is_futile_llm_error(exc):
                    _trip_guard(
                        f"Presentation ({tier_dir_name})",
                        "PRESENTATION_DESIGNER",
                        exc,
                    )
                    if verbose:
                        print(
                            f"  ⛔  Presentation for {tier_dir_name} aborted — "
                            "remaining tiers skipped.",
                            file=sys.stderr,
                        )
                    break
                if verbose:
                    print(f"  ❌  Presentation for {tier_dir_name} failed: {exc}", file=sys.stderr)

        presentation_ok = all_pres_tiers_ok
        if not presentation_ok:
            presentation_error = (
                abort_guard.reason
                if abort_guard.tripped
                else "One or more presentation tiers failed."
            )

    # ── 3. Lab & Project Developer ─────────────────────────────────────
    labs_ok = False
    labs_error: str | None = None
    # Initialised here (not only inside the generate branch) so the later
    # delegation pool check `lab_dev is not None` cannot raise NameError when
    # labs are skipped via skip_labs=True.
    lab_dev: Agent | None = None

    if skip_labs:
        _create_lab_scaffolding(labs_base_path)
        labs_ok = True
    elif abort_guard.should_skip("Labs"):
        _create_lab_scaffolding(labs_base_path)
        labs_error = abort_guard.reason
    else:
        _create_lab_scaffolding(labs_base_path)
        lab_dev = None

        try:
            if lab_dev_agent is not None:
                lab_dev = lab_dev_agent
            else:
                lab_dev = get_lab_developer(verbose=verbose)
        except RuntimeError as exc:
            labs_error = str(exc)

        if lab_dev is not None and syllabus_raw:
            # Run three separate per-tier tasks to keep prompt sizes
            # manageable for the LLM.  Each task focuses on a single
            # tier and writes its files via the output_export_tool.

            def _generate_tier(tier_name: str) -> bool:
                try:
                    tier_task = create_lab_generation_task(
                        agent=lab_dev,
                        course_name=course_name,
                        syllabus_context=syllabus_raw,
                        language=primary_language,
                        run_id=_active_run_id,
                        tier=tier_name,
                        material_language=material_language,
                        human_feedback=human_feedback,
                        verbose=verbose,
                    )
                except Exception as exc:
                    # Task construction used to sit outside the try block, so a
                    # config/schema problem here escaped run_syllabus_crew
                    # entirely and crashed the whole pipeline.
                    print(f"  ❌  {tier_name}: could not build the lab task: {exc}")
                    return False

                def _build_tier_lab_crew(
                    _agent: Agent = lab_dev,
                    _task: Task = tier_task,
                ) -> Crew:
                    """Rebuild the per-tier lab crew for each retry attempt."""
                    return Crew(
                        agents=[_agent],
                        tasks=[_task],
                        process=Process.sequential,
                        verbose=verbose,
                    )

                try:
                    # Fail fast on empty LLM responses (futile) while still
                    # absorbing genuine provider blips (rate limits, timeouts).
                    tier_result = _kickoff_with_retry(
                        _build_tier_lab_crew,
                        attempts=3,
                        backoff_seconds=2.0,
                        verbose=verbose,
                        agent_role="LAB_DEVELOPER",
                    )
                    tier_raw = (
                        tier_result.raw if hasattr(tier_result, "raw") else str(tier_result)
                    ).strip()

                    if not tier_raw:
                        print(f"  ⚠️  {tier_name}: produced no output.")
                        return False

                    # Write the tier-level README (summary).
                    tier_readme = labs_base_path / tier_name / "README.md"
                    write_file(tier_readme, tier_raw, force=True)

                    # Verify actual lab files were written, not just text output.
                    starter_glob = list((labs_base_path / tier_name).rglob("starter/**/*"))
                    solution_glob = list((labs_base_path / tier_name).rglob("solution/**/*"))
                    real_files = [
                        f
                        for f in starter_glob + solution_glob
                        if f.is_file() and f.suffix != ".gitkeep" and f.name != ".gitkeep"
                    ]
                    if not real_files:
                        print(
                            f"  ⚠️  {tier_name}: agent produced text but wrote no "
                            f"lab files to starter/ or solution/."
                        )
                        return False

                    print(f"  ✅  {tier_name}: generated successfully ({len(real_files)} files).")
                    return True

                except Exception as exc:
                    if _is_futile_llm_error(exc) or isinstance(exc, FatalLLMError):
                        _trip_guard(
                            f"Labs ({tier_name})", "LAB_DEVELOPER", exc
                        )
                        print(
                            f"  ⛔  {tier_name}: aborted — remaining tiers skipped."
                        )
                    else:
                        print(f"  ❌  {tier_name}: {exc}")
                    return False

            tiers = [
                "tier1_foundations",
                "tier2_application",
                "tier3_architecture",
            ]
            all_tier_ok = True
            # ── Load state for resume detection ────────────────────────
            lab_state = None if human_feedback else load_generation_state(run_dir)

            # Sequential execution avoids race conditions on the shared
            # Agent singleton (the LLM client and iteration tracker are
            # not thread-safe).
            for tier_name in tiers:
                # ── Skip completed tiers ──────────────────────────────
                if lab_state:
                    tier_state = lab_state.tiers.get(tier_name)
                    if tier_state and tier_state.status == "complete":
                        if verbose:
                            print(
                                f"  ⏭️  {tier_name}: already complete "
                                f"({tier_state.files} files), skipping."
                            )
                        continue

                # Also check filesystem (backward compat) — skipped when human_feedback
                tier_labs_path = labs_base_path / tier_name
                if not human_feedback and _is_tier_labs_complete(tier_labs_path):
                    if lab_state:
                        lab_state.tiers[tier_name] = TierState(
                            status="complete",
                            files=_count_lab_files(tier_labs_path),
                        )
                        save_generation_state(run_dir, lab_state)
                    if verbose:
                        print(f"  ⏭️  {tier_name}: already complete (filesystem check), skipping.")
                    continue

                if not _generate_tier(tier_name):
                    all_tier_ok = False
                    if lab_state:
                        lab_state.tiers[tier_name] = TierState(
                            status="failed",
                        )
                        save_generation_state(run_dir, lab_state)
                    # A futile failure means the model cannot serve this run:
                    # stop spending credits on the remaining tiers.
                    if abort_guard.tripped:
                        break
                elif lab_state:
                    lab_state.tiers[tier_name] = TierState(
                        status="complete",
                        files=_count_lab_files(tier_labs_path),
                    )
                    save_generation_state(run_dir, lab_state)

            if all_tier_ok:
                # Write a top-level index README.
                top_readme = _build_top_level_lab_readme(
                    course_name, primary_language, labs_base_path
                )
                write_file(labs_base_path / "README.md", top_readme, force=True)
                labs_ok = True
            else:
                labs_error = (
                    abort_guard.reason
                    if abort_guard.tripped
                    else "One or more tier lab tasks failed.  Check the per-tier output above."
                )
        elif not syllabus_raw:
            labs_error = "Skipped — Curriculum Architect produced no syllabus to use as context."

        if labs_error and not (labs_base_path / "README.md").exists():
            write_file(
                labs_base_path / "README.md",
                f"# {course_name} — Lab Generation Failed\n\n**Error:** {labs_error}\n",
                force=True,
            )

    # ── 4. QA Reviewer ─────────────────────────────────────────────────
    qa_ok = False
    qa_error: str | None = None
    qa_report: str | None = None

    # QA runs when either labs or theory were generated (or both).
    # Skip only if explicitly disabled or nothing was produced.
    if skip_qa or (not labs_ok and not theory_ok):
        qa_ok = True  # Nothing to review, or explicitly skipped.
    elif abort_guard.should_skip("QA review"):
        qa_error = abort_guard.reason
    elif lab_dev is not None or theory_instructor is not None:
        # The QA Reviewer is a singleton (get_qa_reviewer) and the Lab
        # Developer / Theory Instructor objects are reused across the run, so
        # the agent list is built once.  The Crew + Task are rebuilt on each
        # retry attempt by the factory passed to _kickoff_with_retry, so a
        # single transient LLM failure doesn't abort the whole review.
        qa_reviewer = get_qa_reviewer(verbose=verbose)

        # CRITICAL: All agents that may receive delegation MUST be in
        # the SAME Crew array.  The QA Reviewer delegates lab fixes to
        # the Lab Developer and theory fixes to the Theory Instructor.
        qa_agents: list[Agent] = []
        if lab_dev is not None:
            qa_agents.append(lab_dev)
        if theory_instructor is not None:
            qa_agents.append(theory_instructor)
        qa_agents.append(qa_reviewer)

        def _build_qa_crew() -> Crew:
            qa_task = create_qa_review_task(
                agent=qa_reviewer,
                course_name=course_name,
                run_id=_active_run_id,
                material_language=material_language,
                lab_developer_role=lab_dev.role if lab_dev is not None else None,
                theory_instructor_role=(
                    theory_instructor.role if theory_instructor is not None else None
                ),
                verbose=verbose,
            )
            return Crew(
                agents=qa_agents,
                tasks=[qa_task],
                process=Process.sequential,
                verbose=verbose,
            )

        try:
            qa_result = _kickoff_with_retry(
                _build_qa_crew,
                attempts=3,
                backoff_seconds=2.0,
                verbose=verbose,
                agent_role="QA_REVIEWER",
            )
            qa_report = (qa_result.raw if hasattr(qa_result, "raw") else str(qa_result)).strip()

            if qa_report:
                qa_ok = True
                if verbose:
                    print("  ✅  QA Review completed.")

                # ── Mark QA as complete in the state file.  Load
                # the existing state file directly (NOT re-scanning the
                # filesystem) to preserve tier statuses set by the
                # pipeline — auto-detection would incorrectly mark
                # QA-found broken files as "complete".
                try:
                    state_path = run_dir / _STATE_FILE_NAME
                    if state_path.exists():
                        existing = GenerationState.model_validate_json(
                            state_path.read_text(encoding="utf-8")
                        )
                        existing.qa_review = "complete"
                        save_generation_state(run_dir, existing)
                except Exception:
                    pass  # Non-critical — state file is best-effort
            else:
                qa_error = _annotate_iter_exhaustion(
                    "QA Reviewer produced no output.",
                    "QA_REVIEWER",
                    parent_error=None,
                )

        except Exception as exc:
            if _is_futile_llm_error(exc) or isinstance(exc, FatalLLMError):
                _trip_guard("QA review", "QA_REVIEWER", exc)
                qa_error = abort_guard.reason
            else:
                qa_error = _annotate_iter_exhaustion(
                    _annotate_empty_llm_response(str(exc), "QA_REVIEWER"),
                    "QA_REVIEWER",
                    parent_error=exc,
                )
            if verbose:
                print(f"  ❌  QA Review failed: {exc}", file=sys.stderr)

    # ── 5. Post-run sanity: scan output/ for stray generated files ─────
    # NOTE: ``_active_run_id`` (not the raw ``run_id`` argument) is the
    # resolved identifier — it is always populated, whereas the argument is
    # None for fresh runs that let the crew generate its own run id.
    strays = _scan_for_stray_generated_files(run_id=_active_run_id, verbose=verbose)
    if strays and verbose:
        print(f"\n  🔍  Stray file scan: {len(strays)} issue(s) found above.", file=sys.stderr)

    # ── 6. Generate output manifest ────────────────────────────────────
    try:
        manifest_path = update_output_manifest(
            course_name,
            syllabus_path=syllabus_path,
            labs_base_path=labs_base_path,
        )
    except Exception as exc:
        if verbose:
            print(f"  [Warning] Manifest generation failed: {exc}", file=sys.stderr)
        manifest_path = None

    # ── 7. Return combined result ──────────────────────────────────────
    return CrewResult(
        syllabus_path=syllabus_path,
        labs_base_path=labs_base_path,
        syllabus_ok=syllabus_ok,
        labs_ok=labs_ok,
        syllabus_error=syllabus_error,
        labs_error=labs_error,
        manifest_path=manifest_path,
        qa_ok=qa_ok,
        qa_error=qa_error,
        qa_report=qa_report,
        theory_ok=theory_ok,
        theory_error=theory_error,
        syllabus_review_ok=syllabus_review_ok,
        syllabus_review_error=syllabus_review_error,
        syllabus_review_report=syllabus_review_report,
        lesson_plan_ok=lesson_plan_ok,
        lesson_plan_error=lesson_plan_error,
        presentation_ok=presentation_ok,
        presentation_error=presentation_error,
        aborted=abort_guard.tripped,
        abort_reason=abort_guard.reason,
    )
