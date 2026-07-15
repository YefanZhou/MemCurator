#!/usr/bin/env python
"""
test_gemini_api.py — smoke test for the Gemini (Vertex AI) executor path.

Mirrors run_unified_dev.py's `llm_vertexai()` so a pass here means the real
eval executor path works. Run BEFORE launching a gemini/... executor job.

Required env (same vars run_unified_dev.py reads):
    export GOOGLE_CLOUD_PROJECT="salesforce-research-internal"
    export GOOGLE_CLOUD_LOCATION="global"
    export GOOGLE_GENAI_USE_VERTEXAI="True"
    # auth: `gcloud auth application-default login` (ADC), or GOOGLE_APPLICATION_CREDENTIALS

Usage:
    python test_gemini_api.py                      # default model gemini-3.1-flash-lite
    python test_gemini_api.py gemini-2.5-pro       # override model (name WITHOUT the gemini/ prefix)
"""

import os
import sys


def llm_vertexai(prompt, model="gemini-2.5-pro"):
    """Copy of run_unified_dev.py:llm_vertexai — keep in sync with the real one."""
    from google import genai
    from google.genai import types

    if isinstance(prompt, list):
        text = "\n".join(m["content"] for m in prompt if m.get("role") != "system")
    elif isinstance(prompt, str):
        text = prompt
    else:
        raise ValueError(f"prompt must be a list or a string, got {type(prompt)}")

    client = genai.Client(
        vertexai=True,
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.environ["GOOGLE_CLOUD_LOCATION"],
    )
    response = client.models.generate_content(
        model=model,
        contents=text,
        config=types.GenerateContentConfig(temperature=0.7),
    )
    return response.text or "Output Error"


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "gemini-3.1-flash-lite"

    # Fail early with a clear message if the required env is missing.
    missing = [k for k in ("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION")
               if not os.environ.get(k)]
    if missing:
        print(f"[FAIL] missing env vars: {', '.join(missing)}")
        print("       export GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION "
              "(and GOOGLE_GENAI_USE_VERTEXAI=True) first.")
        sys.exit(2)

    print(f"[info] project  = {os.environ['GOOGLE_CLOUD_PROJECT']}")
    print(f"[info] location = {os.environ['GOOGLE_CLOUD_LOCATION']}")
    print(f"[info] model    = {model}")
    print("[info] sending test prompt ...")

    # 1) string prompt
    reply = llm_vertexai("Reply with exactly: OK", model=model)
    print(f"[reply:str ] {reply.strip()[:200]}")

    # 2) chat-style list prompt (the shape run_unified passes)
    reply2 = llm_vertexai(
        [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "In one word, what is 2+2?"},
        ],
        model=model,
    )
    print(f"[reply:list] {reply2.strip()[:200]}")

    if reply and reply != "Output Error":
        print("[PASS] Gemini Vertex executor path works.")
        sys.exit(0)
    print("[FAIL] empty / error response.")
    sys.exit(1)


if __name__ == "__main__":
    main()
