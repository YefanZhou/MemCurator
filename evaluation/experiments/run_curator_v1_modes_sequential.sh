#!/bin/bash
# ==============================================================================
# Sequential curation-mode sweep for a curator_v1 API sweep script.
#
# Runs the SAME base sweep script three times, back-to-back, once per --curation_mode:
#     success_only_v1  ->  success_and_fail  ->  success_and_fail_v1
# Each invocation blocks until it finishes before the next mode starts (so all three
# don't hammer the gateway/Vertex at once). Nothing runs in the background here.
#
# It just sets CURATION_MODE and calls the base script — which already:
#   * honors CURATION_MODE as an env override (default success_only_v1), and
#   * pins ${CURATION_MODE} in both the exp-name AND the already_done glob,
# so the three modes land in DISTINCT result folders and never skip one another.
#
# The base script is picked by BASE (default = the gemini success_v1 sweep). Any env the base
# script reads (MODELS, HIST_LIST, RUNS, TC_LIST, STAMP, SKIP_DONE, JOB_PARALLEL, NUM_GAMES, …)
# passes straight through — export it before calling this, and it applies to all three modes.
#
# Usage:
#   # gemini, all three modes, hist 3&5 x seeds 1/2/3 (base defaults):
#   tmux new -s curv1_modes 'bash run_curator_v1_modes_sequential.sh'
#
#   # gpt base instead:
#   BASE=server_api_curator_v1_gpt_jul16th_extra_success_v1.sh bash run_curator_v1_modes_sequential.sh
#
#   # a custom mode order / subset:
#   MODES="success_and_fail success_and_fail_v1" bash run_curator_v1_modes_sequential.sh
#
#   # quick end-to-end check (6 games, one seed) before the real run:
#   NUM_GAMES=6 RUNS=1 bash run_curator_v1_modes_sequential.sh --dry-run
#
#   bash run_curator_v1_modes_sequential.sh --dry-run    # show what each mode would launch
# ==============================================================================
set -u

DRY_RUN=""
[ "${1:-}" = "--dry-run" ] && DRY_RUN="--dry-run"

cd "$(dirname "$0")" || { echo "cannot cd to experiments dir"; exit 1; }

BASE="${BASE:-server_api_curator_v1_gemini_jul16th_success_v1.sh}"
MODES="${MODES:-success_only_v1 success_and_fail success_and_fail_v1}"

[ -f "$BASE" ] || { echo "ERROR: base script not found: $BASE"; exit 1; }

echo "=============================================================="
echo "[curator_v1 modes] base=${BASE}"
echo "[curator_v1 modes] modes (in order): ${MODES}"
echo "[curator_v1 modes] dry_run=${DRY_RUN:-0}   (each mode runs to completion before the next)"
echo "=============================================================="

rc_total=0
for mode in $MODES; do
  echo ""
  echo "######################################################################"
  echo "### [$(date +%H:%M:%S)] START mode=${mode}  (base=${BASE})"
  echo "######################################################################"
  CURATION_MODE="$mode" bash "$BASE" $DRY_RUN
  rc=$?
  echo "### [$(date +%H:%M:%S)] DONE  mode=${mode}  (exit ${rc})"
  [ "$rc" -ne 0 ] && rc_total=$rc
done

echo ""
echo "=============================================================="
echo "[$(date +%H:%M:%S)] ALL MODES COMPLETE  (worst exit=${rc_total})"
echo "=============================================================="
exit "$rc_total"
