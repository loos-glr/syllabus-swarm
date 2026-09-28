"""test_preflight.py — Unit tests for the live model probe / early warning.

The probe exists to catch the failure that motivated the fail-fast work: a
model that accepts CrewAI's tool-calling request (the
``call_llm_native_tools`` listener) and answers with ``None``/empty content.
Every network call is mocked — these tests never touch the internet.
"""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import patch

import pytest

from src.preflight import (
    ENABLE_PROBE_ENV,
    PROBE_TIMEOUT_ENV,
    STATUS_EMPTY,
    STATUS_ERROR,
    STATUS_OK,
    ProbeResult,
    api_model_id,
    build_probe_payload,
    fatal_probe_results,
    format_probe_report,
    interpret_probe_response,
    probe_agent_models,
    probe_enabled,
    probe_model,
)


class _FakeResponse:
    """Minimal stand-in for the ``urlopen`` context manager."""

    def __init__(self, body: object) -> None:
        self._body = json.dumps(body).encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _completion(*, content: object = None, tool_calls: object = None, finish: str = "stop"):
    """Build a minimal chat-completion body."""
    message: dict[str, object] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message, "finish_reason": finish}]}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestApiModelId:
    def test_strips_openrouter_prefix(self) -> None:
        assert api_model_id("openrouter/anthropic/claude-opus-5.5") == (
            "anthropic/claude-opus-5.5"
        )

    def test_leaves_other_models_untouched(self) -> None:
        assert api_model_id("anthropic/claude-opus-5.5") == "anthropic/claude-opus-5.5"


class TestBuildProbePayload:
    def test_payload_sends_tool_schemas(self) -> None:
        """A tool schema is mandatory: it exercises the tool-calling path."""
        payload = json.loads(build_probe_payload("openai/gpt-6-sol"))
        assert payload["model"] == "openai/gpt-6-sol"
        assert payload["tools"]
        assert payload["tools"][0]["function"]["name"] == "noop"
        assert payload["max_tokens"] <= 128


class TestInterpretProbeResponse:
    def test_tool_call_is_ok(self) -> None:
        status, detail = interpret_probe_response(
            _completion(tool_calls=[{"id": "1", "function": {"name": "noop"}}])
        )
        assert status == STATUS_OK
        assert "tool call" in detail

    def test_text_content_is_ok(self) -> None:
        status, _ = interpret_probe_response(_completion(content="hello"))
        assert status == STATUS_OK

    def test_null_content_without_tool_calls_is_empty(self) -> None:
        """This is the exact production failure mode."""
        status, detail = interpret_probe_response(
            _completion(content=None, finish="stop")
        )
        assert status == STATUS_EMPTY
        assert "neither content nor a tool call" in detail

    def test_whitespace_only_content_is_empty(self) -> None:
        status, _ = interpret_probe_response(_completion(content="   "))
        assert status == STATUS_EMPTY

    def test_empty_tool_call_list_is_empty(self) -> None:
        status, _ = interpret_probe_response(_completion(content=None, tool_calls=[]))
        assert status == STATUS_EMPTY

    @pytest.mark.parametrize(
        "body",
        [
            None,
            "not-a-dict",
            {},
            {"choices": []},
            {"choices": ["bad-choice"]},
            {"choices": [{"message": "not-a-dict"}]},
        ],
    )
    def test_malformed_payloads_are_errors(self, body: object) -> None:
        status, _ = interpret_probe_response(body)
        assert status == STATUS_ERROR


class TestProbeEnabled:
    def test_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENABLE_PROBE_ENV, raising=False)
        assert probe_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy_values_enable(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv(ENABLE_PROBE_ENV, value)
        assert probe_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
    def test_falsy_values_disable(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv(ENABLE_PROBE_ENV, value)
        assert probe_enabled() is False


class TestProbeTimeout:
    def test_invalid_value_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.preflight import _probe_timeout

        monkeypatch.setenv(PROBE_TIMEOUT_ENV, "not-a-number")
        assert _probe_timeout() == 30.0

    def test_value_is_clamped_to_at_least_one_second(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.preflight import _probe_timeout

        monkeypatch.setenv(PROBE_TIMEOUT_ENV, "0")
        assert _probe_timeout() == 1.0


# ---------------------------------------------------------------------------
# probe_model — network behaviour (urlopen mocked)
# ---------------------------------------------------------------------------


class TestProbeModel:
    def _call(self, side_effect: object, **kwargs: object) -> ProbeResult:
        defaults: dict[str, object] = {
            "model": "openrouter/anthropic/claude-opus-5.5",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-test",
            "timeout": 5.0,
        }
        defaults.update(kwargs)
        with patch("src.preflight.urllib.request.urlopen", side_effect=side_effect):
            return probe_model("THEORY_INSTRUCTOR", **defaults)

    def test_ok_response(self) -> None:
        result = self._call(lambda *a, **k: _FakeResponse(_completion(content="hi")))
        assert result.status == STATUS_OK
        assert result.role == "THEORY_INSTRUCTOR"

    def test_empty_response_is_fatal(self) -> None:
        result = self._call(lambda *a, **k: _FakeResponse(_completion(content=None)))
        assert result.status == STATUS_EMPTY
        assert result.is_fatal is True

    def test_http_error_is_not_fatal(self) -> None:
        def boom(*args: object, **kwargs: object):
            raise urllib.error.HTTPError("url", 503, "Service Unavailable", {}, None)

        result = self._call(boom)
        assert result.status == STATUS_ERROR
        assert result.is_fatal is False
        assert "503" in result.detail

    def test_urlerror_is_not_fatal(self) -> None:
        def boom(*args: object, **kwargs: object):
            raise urllib.error.URLError("dns failure")

        result = self._call(boom)
        assert result.status == STATUS_ERROR
        assert result.is_fatal is False

    def test_unparseable_body_is_not_fatal(self) -> None:
        class _Bad(_FakeResponse):
            def read(self) -> bytes:
                return b"<html>not json</html>"

        result = self._call(lambda *a, **k: _Bad({}))
        assert result.status == STATUS_ERROR

    def test_never_raises_on_unexpected_exception(self) -> None:
        result = self._call(RuntimeError("unexpected"))
        assert result.status == STATUS_ERROR


# ---------------------------------------------------------------------------
# probe_agent_models — deduplication and key handling
# ---------------------------------------------------------------------------


class TestProbeAgentModels:
    def test_models_are_deduplicated(self) -> None:
        """Two roles sharing one model cost exactly one probe request."""
        configs = {
            "A": {"model": "openai/gpt-6-sol", "base_url": "https://x/v1"},
            "B": {"model": "openai/gpt-6-sol", "base_url": "https://x/v1"},
            "C": {"model": "z-ai/glm-5.3", "base_url": "https://x/v1"},
        }
        calls: list[str] = []

        def fake_probe_model(role, *, model, base_url, api_key, timeout):  # noqa: ANN001
            calls.append(model)
            return ProbeResult(role=role, model=model, status=STATUS_OK)

        with (
            patch("src.preflight.get_effective_config", side_effect=lambda r: configs[r]),
            patch("src.preflight.probe_model", side_effect=fake_probe_model),
        ):
            results = probe_agent_models(["A", "B", "C"], timeout=1.0, api_key="sk-test")

        assert len(calls) == 2
        assert len(results) == 2
        shared = next(r for r in results if r.model == "openai/gpt-6-sol")
        assert set(shared.roles) == {"A", "B"}

    def test_missing_api_key_short_circuits(self) -> None:
        configs = {"A": {"model": "m", "base_url": "https://x/v1"}}

        with (
            patch("src.preflight.get_effective_config", side_effect=lambda r: configs[r]),
            patch("src.preflight.probe_model") as mock_probe,
        ):
            results = probe_agent_models(["A"], timeout=1.0, api_key="")

        mock_probe.assert_not_called()
        assert len(results) == 1
        assert results[0].status == STATUS_ERROR
        assert "OPENROUTER_API_KEY" in results[0].detail


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


class TestReporting:
    def test_fatal_probe_results_filters(self) -> None:
        results = [
            ProbeResult(role="A", model="m1", status=STATUS_OK),
            ProbeResult(role="B", model="m2", status=STATUS_EMPTY),
            ProbeResult(role="C", model="m3", status=STATUS_ERROR),
        ]
        fatal = fatal_probe_results(results)
        assert [r.model for r in fatal] == ["m2"]

    def test_format_probe_report_includes_model_and_detail(self) -> None:
        out = format_probe_report(
            [
                ProbeResult(
                    role="THEORY_INSTRUCTOR",
                    model="anthropic/claude-opus-5.5",
                    status=STATUS_EMPTY,
                    detail="no content",
                    roles=("THEORY_INSTRUCTOR",),
                )
            ]
        )
        assert "anthropic/claude-opus-5.5" in out
        assert "no content" in out
        assert "THEORY_INSTRUCTOR" in out

    def test_format_probe_report_empty(self) -> None:
        assert format_probe_report([]) == ""

