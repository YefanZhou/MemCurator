# Environment Caveats — single `memory` env

Reference for the **single-conda-env** setup (`memory`) that serves Qwen3 via vLLM
**and** runs the eval harness. This is what the codebase authors did (all training +
28/36 eval scripts point at `.../envs/memory/...`; `scripts/memory_server_config.sh`
launches `vllm serve` in the same env).

Server context: Linux x86_64, glibc 2.35, CUDA 12.4, Python 3.10, torch 2.6.0+cu124.

> **TL;DR:** every warning below is a pip *metadata-range* complaint, not a runtime
> break. All critical imports succeed (`vllm`, `chromadb`, `litellm`, `langchain`,
> `openai`, `alfworld`, `textworld`). None affect eval **results**. The definitive test
> is `import`, not `pip check` — and imports pass.

---

## How to read `pip check` / resolver warnings

pip verifies **declared** version ranges ("package X says it wants Y≥N"). It does **not**
check whether the code actually breaks. A mismatch surfaces at runtime as a loud
`ImportError`/exception you'd see in logs — **never** as a silently-wrong accuracy score.
So the real correctness gate is: *does it import and does a smoke run produce output?*
Both pass here.

---

## Warnings present in this env — all benign

### 1. protobuf 4.25.9 vs google-api-core / grpcio-status
```
google-api-core 2.31.0 requires protobuf>=5.29.6, but you have 4.25.9
grpcio-status 1.81.1 requires protobuf>=6.33.5, but you have 4.25.9
```
- **Source:** `google-api-core`/`grpcio-status` come from `langchain-google-genai` — the
  **Gemini curator path** (`--curation_model gemini/...`).
- **Do you hit it?** No — you run **local Qwen3-8B**, not Gemini. That path is never imported.
- **Why protobuf stays at 4.x:** vllm/ray require protobuf 4.x. Keeping it there is
  **correct** for the serving stack. Do not "fix" it.
- **Effect on results:** None. Would only matter with a `gemini/` curator, and even then
  it'd crash at import (visible), not corrupt numbers.

### 2. opentelemetry-instrumentation 0.64b0 vs semantic-conventions 0.47b0
```
opentelemetry-instrumentation-{,asgi,fastapi} 0.64b0 requires semantic-conventions==0.64b0,
but you have 0.47b0
```
- **Source:** OpenTelemetry = tracing/metrics telemetry. The `instrumentation-fastapi/asgi`
  bits were dragged in by **chromadb 0.6.3**. Core otel was pinned down to **1.26 / 0.47b0**
  so **vllm 0.8.5** (which requires `opentelemetry<1.27`) is satisfied.
- **The underlying tension:** chromadb wants newer otel, vllm wants older. In one env they
  coexist only because chromadb's telemetry submodule is optional + lazy-loaded.
- **Effect on results:** None. Telemetry records timing/traces; it does **not** touch BM25
  retrieval, memory storage, or LLM scoring. Worst case: a log warning or a no-op usage-stats call.
- **Proof:** `import chromadb` succeeds. If the mismatch mattered for used paths, import would fail.

### 3. wheel 0.47.0 vs packaging 23.2
```
wheel 0.47.0 requires packaging>=24.0, but you have packaging 23.2
```
- **Source:** root `requirements.txt` pins `packaging==23.2`; only the `wheel` build-tool wants ≥24.
- **Effect:** matters only when **building wheels from source**. flash-attn was installed from a
  **prebuilt wheel**, so it never triggered. Nothing in torch/vllm/eval needs `packaging>=24`.
- **Fix if ever needed:** `pip install 'packaging>=24'` (only before a source build). Otherwise ignore.

### 4. textworld 1.7.0 "is not supported on this platform"
```
textworld 1.7.0 is not supported on this platform
```
- **Source:** `pip check` reading textworld's conservative platform metadata. Pulled by `alfworld==0.4.2`.
- **Reality:** on Linux x86_64 it works. Verified:
  `import textworld` → 1.7.0, `from textworld import gym` → OK, full alfworld env chain builds
  (134 eval_out_of_distribution games load).
- **Effect on results:** None — this is the ALFWorld engine and it runs. Warning is noise.

---

## The circular conflict at the root of #1 and #2

- **chromadb 0.6.3** (eval) → pulls **opentelemetry ≥1.30** and newer protobuf.
- **vllm 0.8.5** (serve) → requires **opentelemetry <1.27** and protobuf 4.x.

In a **single env** these coexist only by luck (imports work despite the pins). This is the
core reason a **two-env split** (`memory-train` + `memory-eval`, see `INSTALL_TWO_ENVS.md`)
is the lower-risk alternative: the serving env keeps otel 1.26 for vllm; the eval env lets
otel/protobuf float for chromadb, and has no vllm to conflict with.

---

## What ACTUALLY affects results (watch these, not the warnings)

The warnings above are telemetry + an unused cloud path. The packages that change **numbers**
are all correctly pinned:

| Package | Version | Why it matters | Status |
|---|---|---|---|
| `openai` | **1.78.1** | real inference path (litellm → vllm server); eval reqs try to downgrade to 1.75.0 | ✅ restored |
| `litellm` | 1.67.0.post1 | the `completion()` call to the served model | ✅ works |
| `rank-bm25` | 0.2.2 | BM25 skill/memory retrieval | ✅ pinned |
| `tiktoken` | 0.9.0 | tokenization / compression counting | ✅ pinned |
| `transformers` | 4.51.3 | curator tokenizer (SkillOS in-process path) | ✅ pinned |
| `torch` | 2.6.0+cu124 | vllm backend | ✅ |
| `vllm` | 0.8.5 | serving | ✅ imports |

**Critical rule for the single env:** any time you run `pip install -r evaluation/requirements.txt`
(or reinstall eval deps), it re-downgrades `openai` to 1.75.0 and can bump `opentelemetry`.
**Always re-run afterward:**
```bash
pip install 'openai==1.78.1' \
    'opentelemetry-api==1.26.0' 'opentelemetry-sdk==1.26.0' \
    'opentelemetry-semantic-conventions==0.47b0' 'opentelemetry-proto==1.26.0' \
    'opentelemetry-exporter-otlp-proto-common==1.26.0' \
    'opentelemetry-exporter-otlp-proto-grpc==1.26.0'
```
Then confirm: `python -c "import vllm, chromadb, litellm, langchain_community, openai; print('OK')"`

---

## Verified-working import checklist

```bash
python -c "import vllm; print('vllm OK', vllm.__version__)"                       # 0.8.5
python -c "import chromadb; print('chromadb OK', chromadb.__version__)"           # 0.6.3
python -c "import litellm, langchain_community, openai; print('eval imports OK')"
python -c "import flash_attn; print('flash_attn OK', flash_attn.__version__)"     # 2.7.4.post1
python -c "import alfworld.agents.environment; from alfworld.agents.environment import get_environment; print('alfworld env chain OK')"
python -c "import textworld; from textworld import gym; print('textworld OK', textworld.__version__)"  # 1.7.0
```
All six pass → env is good. **Bottom line: proceed; ignore the pip-check noise.**
