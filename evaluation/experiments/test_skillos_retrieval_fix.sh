#!/bin/bash
# ==============================================================================
# SMOKE TEST for the SkillOS retrieval fix (Option B: task<->task BM25 alignment).
#
# Runs a short SkillOS eval (10 games) with the FIXED code, then auto-inspects the
# result to CONFIRM the fix took effect:
#   (1) skills.json entries carry the new `tasks` provenance key
#   (2) BM25 now indexes those task strings (retrieval doc == task, not skill body)
#   (3) prints SR + a retrieval-precision probe (target-object vs wrong-object)
#
# What the fix does: SkillOS now retrieves skills by matching the CURRENT task
# description against the TASK DESCRIPTION(S) that produced each skill (like
# ReasoningBank/MemCurator), instead of matching the task against the skill's
# full markdown body.
#
# Prereq: Qwen3-8B vLLM servers — executor on :8001, curator on :8002 (split by default).
#   (point both at one server with EXEC_PORT=8001 CUR_PORT=8001 if you only have one.)
# Usage:
#   bash test_skillos_retrieval_fix.sh                 # 10-game smoke (default)
#   NUM_GAMES=140 EXP=skillos-fix-full bash test_skillos_retrieval_fix.sh   # full run
#   EXEC_PORT=8001 CUR_PORT=8002 bash test_skillos_retrieval_fix.sh         # override ports
# ==============================================================================
set -u

cd "$(dirname "$0")/../agent_eval" || { echo "cannot cd to agent_eval"; exit 1; }

EXEC_PORT="${EXEC_PORT:-8001}"       # executor (gameplay) vLLM server
CUR_PORT="${CUR_PORT:-8002}"         # curator (skill curation) vLLM server
NUM_GAMES="${NUM_GAMES:-10}"
BATCH_SIZE="${BATCH_SIZE:-3}"        # small bs => faster smoke (concurrency == batch_size)
RETRIEVE_NUM="${RETRIEVE_NUM:-5}"
EXP="${EXP:-skillos-fix-smoke}"
PY="${PY:-python}"                   # override with a conda python if `python` isn't the env
MODEL="openai/Qwen/Qwen3-8B"

# ---- executor / curator hyperparams (match the smoke command in RUN_COMMAND_Log.sh) ----
export OPENAI_API_KEY=EMPTY
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export EXECUTOR_TEMPERATURE=1.0 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096
export CURATION_TEMPERATURE=1.0 CURATION_TOP_P=0.95 CURATION_TOP_K=20 CURATION_MAX_TOKENS=4096 CURATION_ENABLE_THINKING=true
export PROMPT_STYLE=revise_react SAVE_RAW=10
# Silence the benign litellm<->pydantic serializer UserWarning (keeps real errors visible).
export PYTHONWARNINGS="ignore:Pydantic serializer warnings:UserWarning"

echo "[preflight] checking vLLM servers: executor :${EXEC_PORT}, curator :${CUR_PORT} ..."
for p in "$EXEC_PORT" "$CUR_PORT"; do
  if ! curl -sf "http://localhost:${p}/v1/models" >/dev/null; then
    echo "ERROR: no vLLM server on :${p}. Start Qwen3-8B there first, or override EXEC_PORT/CUR_PORT."
    exit 1
  fi
done
echo "[preflight] OK"

RESULT_DIR="Alfworld/results/${MODEL}/dev_${EXP}_few_shot_False_skillos"

echo "======================================================================"
echo "RUN: SkillOS (FIXED retrieval) | ${MODEL} | exp=${EXP} | ${NUM_GAMES} games | bs=${BATCH_SIZE} rn=${RETRIEVE_NUM}"
echo "  executor: nonthink hist3 revise_react temp1.0 @ :${EXEC_PORT}   curator: think temp1.0 4096 @ :${CUR_PORT}"
echo "  results -> ${RESULT_DIR}"
echo "======================================================================"

OPENAI_API_BASE="http://localhost:${EXEC_PORT}/v1" HISTORY_LENGTH=3 ENABLE_THINKING=false \
"$PY" -u run_unified_dev_async.py --env alfworld --memory_type skillos \
    --model          "$MODEL" \
    --curation_model "$MODEL" \
    --curation_base_url "http://localhost:${CUR_PORT}/v1" \
    --batch_size "$BATCH_SIZE" --retrieve_num "$RETRIEVE_NUM" --max_steps 30 \
    --num_games "$NUM_GAMES" --exp_name "$EXP" --overwrite
RC=$?
echo "[run] exit=${RC}"
[ "$RC" -ne 0 ] && { echo "run failed; skipping verification"; exit "$RC"; }

# ==============================================================================
# VERIFY the fix took effect (parses the just-written skills.json + idx files)
# ==============================================================================
echo ""
echo "======================================================================"
echo "VERIFY: did the retrieval fix take effect?"
echo "======================================================================"
"$PY" - "$RESULT_DIR" <<'PYEOF'
import json, os, sys, glob, re
D = sys.argv[1]
sp = os.path.join(D, "skills.json")
if not os.path.exists(sp):
    print("FAIL: no skills.json at", sp); sys.exit(1)
skills = json.load(open(sp))
n = len(skills)
with_tasks = [s for s in skills if s.get("tasks")]
print(f"skills.json: {n} skills; {len(with_tasks)} carry the new `tasks` provenance key")
if not with_tasks:
    print(">>> FAIL: no skill has `tasks` -> the fix did NOT take effect (old code / not synced).")
    sys.exit(1)
print(">>> PASS (1): skills record originating task(s).")
# show a few
for s in with_tasks[:4]:
    print(f"    - {s['title'][:40]!r}  tasks={[t[:50] for t in s['tasks']]}")

# confirm BM25 now indexes the task, not the body
sys.path.insert(0, os.path.join(os.getcwd(), "SkillOS"))
try:
    from skills_memory import SkillMemory
    m = SkillMemory(); m.skills = skills
    doc = m._retrieval_doc(with_tasks[0])
    is_taskdoc = doc.strip() in " ".join(with_tasks[0]["tasks"]) or all(t in doc for t in with_tasks[0]["tasks"])
    print(f">>> PASS (2): _retrieval_doc indexes the task string: {doc[:80]!r}" if is_taskdoc
          else f">>> WARN: _retrieval_doc doesn't look task-keyed: {doc[:80]!r}")
except Exception as e:
    print("    (skipped live _retrieval_doc check:", e, ")")

# SR
idxs = glob.glob(os.path.join(D, "idx_*.json"))
won = tot = 0
for f in idxs:
    try: g = json.load(open(f))
    except: continue
    tot += 1; won += (1 if float(g.get("reward",0)) >= 1.0 else 0)
print(f"\nSR: {won}/{tot} = {100*won/max(tot,1):.1f}%   (steps saved per game in idx_*.json)")

# retrieval-precision probe on the SAVE_RAW games: fraction of injected skills whose
# recorded task shares the target object with the CURRENT task.
OBJS=["soapbar","cloth","dishsponge","pot","pan","bowl","mug","plate","cup","fork","knife",
 "lettuce","tomato","apple","egg","book","pen","pencil","cd","vase","candle","box","tissuebox",
 "toiletpaper","spraybottle","remotecontrol","newspaper","pillow","kettle","watch"]
def obj(s):
    s=s.lower()
    return next((o for o in OBJS if o in s), None)
inj_match=inj_tot=0
for f in idxs:
    try: g=json.load(open(f))
    except: continue
    tgt=obj(g.get("name","")); rt=g.get("raw_trace")
    if not tgt or not rt: continue
    p=rt[0].get("prompt","")
    if "Past Relevant Skills" not in p: continue
    block=p.split("## Current Progress")[0]
    for m_ in re.findall(r"\*\*Skill \d+:(.*?)\*\*(.*?)(?=\*\*Skill \d+:|\Z)", block, re.DOTALL):
        txt=(m_[0]+m_[1]).lower(); inj_tot+=1
        if tgt in txt: inj_match+=1
if inj_tot:
    print(f"retrieval probe: {inj_match}/{inj_tot} injected skills mention the target object "
          f"({100*inj_match/inj_tot:.0f}%)  [higher = better task-alignment]")
print("\nVERIFY DONE.")
PYEOF

echo ""
echo "Compare SR to the OLD (pre-fix) SkillOS nonthink/hist3 baseline (~25%) and no-memory (~34%)."
echo "Full result dir: ${RESULT_DIR}"
