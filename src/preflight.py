"""preflight.py — Live model probe (early-warning system).
=========================================================

The static audit in :mod:`src.llm_factory` catches *misconfiguration* (e.g. a
reasoning model with an undersized token budget).  This module catches *broken
models*: a provider that is unavailable, or — far more insidiously — one that
accepts a tool-calling request and answers with nothing.

That second failure is exactly what CrewAI reports as::

    ERROR:crewai.flow.runtime:Error executing listener call_llm_native_tools:
    Invalid response from LLM call - None or empty.

Repeating that call cannot succeed, so every further attempt is wasted credit.
By issuing ONE cheap, *tool-calling* probe request per distinct model **before**
the pipeline starts, the swarm can refuse to run at all instead of burning a
full generation cycle.

Why tool-calling?
-----------------
The failure surfaces on CrewAI's ``call_llm_native_tools`` listener, i.e. the
path where the agent calls the LLM *with* tool schemas.  A plain text
completion can succeed while the tool-calling path returns NULL, so a probe
that does not send tool schemas would give false assurance.  The probe
therefore sends a trivial no-op tool — matching production behaviour with
negligible cost.

Enabling the probe
------------------
The probe is opt-in because it adds one network round-trip per distinct model::

    export SYLLABUS_PREFLIGHT_PROBE=1

Recommended for CI/nightly runs and whenever a run has previously failed with
the empty-response error.  An "unreachable/HTTP error" outcome is reported as a
warning only; an "empty response" outcome is fatal, because it is precisely the
failure this swarm must never retry.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from src.llm_factory import get_effective_config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENABLE_PROBE_ENV: str = "SYLLABUS_PREFLIGHT_PROBE"
PROBE_TIMEOUT_ENV: str = "SYLLABUS_PREFLIGHT_PROBE_TIMEOUT"
_DEFAULT_TIMEOUT_SECONDS: float = 30.0
_PROBE_MAX_TOKENS: int = 64
_PROBE_TIMEOUT_ATTEMPTS: int = 2

STATUS_OK: str = "ok"
STATUS_EMPTY: str = "empty"
STATUS_ERROR: str = "error"

# A trivial, side-effect-free tool.  Sending a tool schema is the whole point:
# it forces the request down the same ``call_llm_native_tools`` path that the
# production failure occurs on.
_PROBE_TOOL: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "noop",
            "description": "Acknowledge the request without doing anything.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }
]


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of probing a single model.

    Attributes
    ----------
    role : str
        Representative agent role that uses this model.
    roles : tuple[str, ...]
        Every agent role sharing this model (deduplicated probes).
    model : str
        Model identifier exactly as configured (provider prefix stripped).
    status : str
        One of :data:`STATUS_OK`, :data:`STATUS_EMPTY`, :data:`STATUS_ERROR`.
    detail : str
        Human-readable explanation of the outcome.
    elapsed_ms : int
        Wall-clock duration of the probe request.
    """

    role: str
    model: str
    status: str
    detail: str = ""
    elapsed_ms: int = 0
    roles: tuple[str, ...] = ()

    @property
    def is_fatal(self) -> bool:
        """True when the model proved it cannot answer a tool-calling request."""
        return self.status == STATUS_EMPTY

    @property
    def is_unreachable(self) -> bool:
        """True when the probe failed at the transport/HTTP level (non-fatal)."""
        return self.status == STATUS_ERROR


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def probe_enabled() -> bool:
    """Return True when the live probe is switched on via the environment."""
    raw = os.getenv(ENABLE_PROBE_ENV, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _probe_timeout() -> float:
    """Resolve the per-request probe timeout in seconds."""
    raw = os.getenv(PROBE_TIMEOUT_ENV)
    if raw is None:
        return _DEFAULT_TIMEOUT_SECONDS
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT_SECONDS


def api_model_id(model: str) -> str:
    """Strip the ``openrouter/`` routing prefix CrewAI passes to litellm.

    CrewAI removes this prefix before calling the provider, so the probe must
    do the same or OpenRouter will reject the request.
    """
    return model[len("openrouter/") :] if model.startswith("openrouter/") else model


# ---------------------------------------------------------------------------
# Payload / response handling (pure — unit-testable without any network)
# ---------------------------------------------------------------------------


def build_probe_payload(model_id: str) -> bytes:
    """Build the JSON request body for a single probe, as bytes."""
    return json.dumps(
        {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": "Call the noop tool to acknowledge this request.",
                }
            ],
            "max_tokens": _PROBE_MAX_TOKENS,
            "tools": _PROBE_TOOL,
            "tool_choice": "auto",
        }
    ).encode()


def interpret_probe_response(body: object) -> tuple[str, str]:
    """Map a chat-completion response body to ``(status, detail)``.

    Returns
    -------
    tuple[str, str]
        ``(STATUS_OK, detail)`` when the model produced content or a tool call;
        ``(STATUS_EMPTY, detail)`` when it produced neither — the exact
        condition that aborts production runs; ``(STATUS_ERROR, detail)`` for a
        malformed/unexpected payload.
    """
    if not isinstance(body, dict):
        return STATUS_ERROR, "response body was not a JSON object"

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return STATUS_ERROR, "response contained no choices"

    choice = choices[0]
    if not isinstance(choice, dict):
        return STATUS_ERROR, "response choice was not an object"

    message = choice.get("message")
    if not isinstance(message, dict):
        return STATUS_ERROR, "response choice contained no message object"

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        return STATUS_OK, "model returned a tool call"

    content = message.get("content")
    if content is not None and str(content).strip():
        return STATUS_OK, "model returned text content"

    finish_reason = choice.get("finish_reason", "unknown")
    return (
        STATUS_EMPTY,
        "model returned neither content nor a tool call "
        f"(finish_reason={finish_reason})",
    )



# ---------------------------------------------------------------------------
# Network probe
# ---------------------------------------------------------------------------


def probe_model(
    role: str,
    *,
    model: str,
    base_url: str,
    api_key: str,
    timeout: float,
) -> ProbeResult:
    """Issue one tool-calling probe request for *model*.

    Never raises: transport problems are reported as :data:`STATUS_ERROR` so a
    flaky pre-flight check can never block a run that would otherwise work.
    """
    payload = build_probe_payload(api_model_id(model))
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    start = time.monotonic()
    last_error = "unknown error"
    for _ in range(_PROBE_TIMEOUT_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read())
            status, detail = interpret_probe_response(body)
            elapsed = int((time.monotonic() - start) * 1000)
            return ProbeResult(
                role=role, model=model, status=status, detail=detail, elapsed_ms=elapsed
            )
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}: {exc.reason}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = f"could not parse response: {exc}"
        except Exception as exc:  # noqa: BLE001 — the probe must never crash a run
            last_error = f"{type(exc).__name__}: {exc}"

    elapsed = int((time.monotonic() - start) * 1000)
    return ProbeResult(
        role=role,
        model=model,
        status=STATUS_ERROR,
        detail=last_error,
        elapsed_ms=elapsed,
    )


def probe_agent_models(
    roles: Iterable[str],
    *,
    timeout: float | None = None,
    api_key: str | None = None,
) -> list[ProbeResult]:
    """Probe every distinct model used by *roles*.

    Models are deduplicated so a shared model costs a single request; the
    returned :class:`ProbeResult` records every role that shares it.

    When no API key is configured, a single upfront
    :data:`STATUS_ERROR` result is returned for each model rather than
    attempting (and failing) a network call per role.
    """
    resolved_key = api_key if api_key is not None else os.getenv("OPENROUTER_API_KEY", "")
    resolved_timeout = timeout if timeout is not None else _probe_timeout()

    # model -> (representative role, all roles)
    by_model: dict[str, list[str]] = {}
    for role in roles:
        cfg = get_effective_config(role)
        model = str(cfg["model"])
        by_model.setdefault(model, []).append(role)

    results: list[ProbeResult] = []
    for model, model_roles in by_model.items():
        base_url = str(get_effective_config(model_roles[0])["base_url"])

        if not resolved_key:
            results.append(
                ProbeResult(
                    role=model_roles[0],
                    model=model,
                    status=STATUS_ERROR,
                    detail="OPENROUTER_API_KEY is not set — probe skipped",
                    roles=tuple(model_roles),
                )
            )
            continue

        result = probe_model(
            model_roles[0],
            model=model,
            base_url=base_url,
            api_key=resolved_key,
            timeout=resolved_timeout,
        )
        results.append(
            ProbeResult(
                role=result.role,
                model=result.model,
                status=result.status,
                detail=result.detail,
                elapsed_ms=result.elapsed_ms,
                roles=tuple(model_roles),
            )
        )

    return results


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def fatal_probe_results(results: Iterable[ProbeResult]) -> list[ProbeResult]:
    """Return only the results that must abort the run."""
    return [r for r in results if r.is_fatal]


def format_probe_report(results: Sequence[ProbeResult]) -> str:
    """Render probe results as a human-readable, multi-line block."""
    lines: list[str] = []
    for result in results:
        if result.status == STATUS_OK:
            icon = "✅"
        elif result.status == STATUS_EMPTY:
            icon = "⛔"
        else:
            icon = "⚠️ "
        roles = ", ".join(result.roles) if result.roles else result.role
        lines.append(f"  {icon}  {result.model}  ({result.elapsed_ms} ms)")
        lines.append(f"      agents: {roles}")
        lines.append(f"      {result.detail}")
    return "\n".join(lines)

