#!/usr/bin/env python
"""
test_gateway_api.py — smoke test for the EXTERNAL API backend of the *_api.py runners.

Verifies, against the Salesforce AI gateway (or any OpenAI-compatible endpoint), that:
  (a) an EXECUTOR-style call (litellm.completion with the vLLM-only extra_body fields SUPPRESSED)
      returns non-empty text and does NOT 400 on `top_k` / `chat_template_kwargs`; and
  (b) a SKILLOS-style NATIVE tool-calling curation call returns parsed `tool_calls`
      (new_skill_insert / skill_update / skill_delete).

Run this BEFORE launching a full external-backend sweep.

Required env:
    export OPENAI_API_BASE="https://gateway.salesforceresearch.ai/openai/process/v1/"
    export OPENAI_API_KEY="<your gateway key>"       # bearer
    # optional (Salesforce gateway header auth):
    export X_API_KEY="<gateway x-api-key>"

Usage:
    python test_gateway_api.py                       # default model gpt-5.4-mini
    python test_gateway_api.py gpt-5.4-mini
    python test_gateway_api.py gemini-3-lite         # passed WITHOUT gemini/ prefix -> via gateway

Notes:
  - Model is passed to litellm with an `openai/` prefix so it routes to base_url as
    OpenAI-compatible (this is how the _api runners pass --model openai/<name>).
  - MEMORY_TOOL_SCHEMAS is imported from run_unified_dev_async_api so the exact schemas the
    runner uses are exercised here.
"""
import os
import sys
import json


def _extra_headers():
    x = os.environ.get("X_API_KEY") or None
    return {"X-Api-Key": x} if x else None


def test_executor(model: str) -> bool:
    """Executor path: NO top_k / chat_template_kwargs in extra_body (external gateway)."""
    from litellm import completion
    kwargs = dict(
        model=f"openai/{model}",
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_API_BASE"],
        temperature=0.7,
        num_retries=3,
    )
    hdrs = _extra_headers()
    if hdrs:
        kwargs["extra_headers"] = hdrs
    resp = completion(**kwargs)
    text = resp.choices[0].message.content or ""
    print(f"[executor] reply: {text.strip()[:200]!r}")
    return bool(text.strip())


def test_curator_tool_calls(model: str) -> bool:
    """Skillos native tool-calling: pass tools=, expect message.tool_calls back."""
    from litellm import completion
    # Import the exact tool schemas the runner uses.
    try:
        from run_unified_dev_async_api import MEMORY_TOOL_SCHEMAS
    except Exception as e:
        print(f"[curator] could not import MEMORY_TOOL_SCHEMAS: {e}")
        return False

    messages = [
        {"role": "system",
         "content": "You are a skill librarian. Given a task and a successful trajectory, call "
                    "the appropriate tool to record a reusable skill. You MUST make at least one "
                    "tool call."},
        {"role": "user",
         "content": "# Task\nHeat a mug and put it on the desk.\n\n# Trajectory (success)\n"
                    "go to microwave 1; open microwave 1; put mug in microwave; heat mug; "
                    "take mug; go to desk 1; put mug on desk 1.\n\n# Result: Success\n\n"
                    "Record a new skill capturing this procedure."},
    ]
    kwargs = dict(
        model=f"openai/{model}",
        messages=messages,
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_API_BASE"],
        temperature=0.7,
        num_retries=3,
        tools=MEMORY_TOOL_SCHEMAS,
        tool_choice="auto",
    )
    hdrs = _extra_headers()
    if hdrs:
        kwargs["extra_headers"] = hdrs
    resp = completion(**kwargs)
    msg = resp.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None) or []
    print(f"[curator] {len(tool_calls)} tool_call(s) returned:")
    for tc in tool_calls:
        args_preview = (tc.function.arguments or "")[:160]
        print(f"    - {tc.function.name}({args_preview})")
        # confirm arguments parse as JSON (what _apply_parsed_function_calls needs)
        try:
            json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError as e:
            print(f"      [WARN] arguments not valid JSON: {e}")
    return len(tool_calls) > 0


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt-5.4-mini"
    missing = [k for k in ("OPENAI_API_BASE", "OPENAI_API_KEY") if not os.environ.get(k)]
    if missing:
        print(f"[FAIL] missing env vars: {', '.join(missing)}")
        sys.exit(2)

    print(f"[info] base_url = {os.environ['OPENAI_API_BASE']}")
    print(f"[info] x_api_key = {'set' if os.environ.get('X_API_KEY') else 'unset'}")
    print(f"[info] model    = {model}\n")

    ok_exec = False
    ok_cur = False
    try:
        print("=== (a) executor call ===")
        ok_exec = test_executor(model)
    except Exception as e:
        print(f"[executor] ERROR: {type(e).__name__}: {e}")

    try:
        print("\n=== (b) skillos native tool-calling curation ===")
        ok_cur = test_curator_tool_calls(model)
    except Exception as e:
        print(f"[curator] ERROR: {type(e).__name__}: {e}")

    print(f"\n[result] executor={'PASS' if ok_exec else 'FAIL'}  "
          f"curator_tool_calls={'PASS' if ok_cur else 'FAIL'}")
    sys.exit(0 if (ok_exec and ok_cur) else 1)


if __name__ == "__main__":
    main()
