#!/usr/bin/env python3
"""Direct OpenRouter Model Diagnostic — tests models w/out CrewAI/litellm.

Usage:
    python src/debug/debug_model.py --all --prompt-size small
    python src/debug/debug_model.py --model anthropic/claude-opus-5.5 --prompt-size full
    python src/debug/debug_model.py --opus55-test
"""

from __future__ import annotations

import argparse, json, os, sys, time, urllib.request, urllib.error
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)

API_KEY = os.getenv("OPENROUTER_API_KEY", "")
BASE_URL = "https://openrouter.ai/api/v1"

MODELS_TO_TEST = [
    {"id": "anthropic/claude-opus-5.5", "label": "Claude Opus 5.5"},
    {"id": "anthropic/claude-opus-4.8", "label": "Claude Opus 4.8"},
    {"id": "openai/gpt-6-sol", "label": "GPT-6 Sol"},
    {"id": "deepseek/deepseek-v4-pro", "label": "DeepSeek V4 Pro"},
]


def _build_prompts() -> tuple[str, str]:
    """Build the system + user prompts matching what CrewAI sends."""
    from src.agents.curriculum_architect import create_curriculum_architect
    from src.tasks.syllabus_generation import create_syllabus_generation_task

    agent = create_curriculum_architect()
    system = f"You are {agent.role}. {agent.backstory}\nYour personal goal is: {agent.goal}"

    sessions = sorted((_PROJECT_ROOT / "output").glob("*/intake_session.json"))
    if sessions:
        with open(sessions[-1]) as f:
            s = json.load(f)
        ctx = s["course_specification"]["course_context"]
        lang = s["course_specification"].get("material_language", "Dutch")
        cname = s.get("course_name", "Test")
    else:
        ctx = "Test course context."
        lang = "Dutch"
        cname = "Test"

    task = create_syllabus_generation_task(
        course_name=cname, agent=agent, course_context=ctx, material_language=lang
    )
    user = f"\nCurrent Task: {task.description}"
    return system, user


def call_api(model, messages, max_tokens=4096, temperature=0.2, top_p=0.1, timeout=300):
    """Call OpenRouter chat/completions and return parsed result."""
    payload = json.dumps(dict(
        model=model, messages=messages,
        temperature=temperature, max_tokens=max_tokens, top_p=top_p
    )).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions", data=payload,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read())
            msg = body["choices"][0]["message"]
            return {
                "ok": True, "status": r.status,
                "elapsed": round((time.monotonic() - start) * 1000),
                "content": msg.get("content"),
                "finish": body["choices"][0].get("finish_reason", "?"),
                "usage": body.get("usage", {}),
                "msg_keys": list(msg.keys()),
            }
    except urllib.error.HTTPError as e:
        return {
            "ok": False, "status": e.code,
            "error": str(e),
            "body": e.read().decode(errors="replace")[:500],
        }
    except Exception as e:
        return {"ok": False, "status": 0, "error": str(e)}


def test_small(model_info):
    """Tiny prompt — sanity check that model responds at all."""
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Say hi."},
    ]
    return call_api(model_info["id"], msgs, max_tokens=50)


def test_full(model_info):
    """Full syllabus prompt — replicates CrewAI task exactly."""
    sys_p, usr_p = _build_prompts()
    return call_api(
        model_info["id"],
        [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
        max_tokens=8192,
    )


def opus55_experiment():
    """Test Claude Opus 5.5 with increasing max_tokens to find threshold."""
    print("=" * 60)
    print("  Claude Opus 5.5 - max_tokens Experiment")
    print("=" * 60)
    sys_p, usr_p = _build_prompts()
    print(f"System: {len(sys_p):,} chars, User: {len(usr_p):,} chars")
    msgs = [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": usr_p},
    ]
    for mt in [8192, 16384, 24576, 32768, 49152]:
        print(f"\n-- max_tokens={mt:,} --")
        r = call_api("anthropic/claude-opus-5.5", msgs, max_tokens=mt)
        if r["ok"]:
            c = r["content"]
            print(f"  status={r['status']}  {r['elapsed']}ms  finish={r['finish']}")
            print(f"  content is None: {c is None}")
            u = r["usage"]
            print(f"  tokens: {u.get('prompt_tokens','?')} in / {u.get('completion_tokens','?')} out")
            if c is not None:
                print(f"  content length: {len(c):,} chars")
                print(f"  preview: {c[:200]}...")
            else:
                print(f"  msg_keys: {r['msg_keys']}")
        else:
            print(f"  FAILED HTTP {r['status']}: {r.get('error','?')}")
            if r.get("body"):
                print(f"  body: {r['body'][:300]}")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--prompt-size", choices=["small", "full"], default="small")
    p.add_argument("--opus55-test", action="store_true")
    p.add_argument("--all", action="store_true")
    a = p.parse_args()

    if a.opus55_test:
        opus55_experiment()
        return

    models = [{"id": a.model, "label": a.model}] if a.model else MODELS_TO_TEST
    test_fn = test_full if a.prompt_size == "full" else test_small

    print("=" * 60)
    print(f"  Model Diagnostic  |  prompt={a.prompt_size}")
    print("=" * 60)

    for mi in models:
        print(f"\n-- {mi['label']} ({mi['id']}) --")
        r = test_fn(mi)
        if r["ok"]:
            c = r["content"]
            print(f"  OK {r['status']} {r['elapsed']}ms  finish={r['finish']}")
            if c is None:
                print(f"  WARN content=NULL  msg_keys={r['msg_keys']}")
            else:
                print(f"  content: {len(c):,} chars")
                if len(c) < 100:
                    print(f"  preview: {c[:100]}")
            u = r["usage"]
            if u:
                print(f"  tokens: {u.get('prompt_tokens','?')} in / {u.get('completion_tokens','?')} out")
        else:
            print(f"  FAIL HTTP {r['status']}: {r.get('error','?')}")
            if r.get("body"):
                print(f"  body: {r['body'][:300]}")
    print()


if __name__ == "__main__":
    main()