# Experiment TODO

Running list of experiments to run / in flight / done. Add absolute-dated notes.

---

## TODO

### [ ] reasoningbank thinking @ temp=0.7  (the temp ablation of the best configs)
The two best ALFWorld configs so far are reasoningbank + thinking + revise_react at **temp=1.0**:

| config | temp=1.0 result |
|--------|-----------------|
| rb thinking wo`<think>` hist3 | 56.90 ± 3.21 |
| rb thinking wo`<think>` hist5 | **64.05 ± 0.89** |

**Goal:** re-run both at **temp=0.7** (executor) to see if lower sampling temperature helps.
6 runs: hist{3,5} × seed{1,2,3}, async, Qwen3-8B exec+curator, ENABLE_THINKING=true, revise_react.
Curator temp stays 1.0 (only executor temp is the variable; override with `CUR_TEMP=` if wanted).

- **Driver (direct bash, needs 2 running 8B vLLM servers on :8001/:8002):**
  `evaluation/experiments/server_8b_rb_think_temp0.7.sh`
  - dry-run: `bash server_8b_rb_think_temp0.7.sh --dry-run`
  - real:    `tmux new -s rb07 'bash evaluation/experiments/server_8b_rb_think_temp0.7.sh'`
- **Slurm (self-contained: serves both servers + runs the driver):**
  `slurm_script/rb_think_temp0.7_8b.sh`
  - submit: `sbatch slurm_script/rb_think_temp0.7_8b.sh`
- exp_names: `rb-async_think_temp0.7_hist{3,5}_run{1,2,3}_<STAMP>` → distinct from all temp=1.0 runs.
- Compare against temp=1.0 in `analysis/analysis_new.ipynb` (matched on hist; add a temp column to the pivot).

---

## IN FLIGHT
<!-- move items here when launched; note node + job id + start time -->

---

## DONE
<!-- move items here with final numbers + result path -->

- **[done 2026-07-12] Batch memory-vs-none ablation** — `server_8b_nonthink_batch_box1.sh`, 24 runs,
  nonthinking, mem{rb,none}×temp{0.7,1.0}×hist{3,5}×seed{1,2,3}. Finding: at nonthinking, rb does
  NOT beat none (Δ −3.1 to +0.2). See analysis_new.ipynb `batch_cmp`.
- **[done 2026-07-12] reasoningbank thinking @ temp=1.0** — `server_8b_8b_8b_jul12th.sh`. hist5=64.05,
  hist3=56.90 (the configs this temp=0.7 ablation targets).

---

## Notes / gotchas
- Every driver appends a `_<STAMP>` timestamp to exp_names so re-launches never overwrite results.
- reasoningbank runs wipe `Alfworld/memory/reasoningbank_<exp>` per run to match `--overwrite`.
- Slurm: 8×H200 split 4/4 across :8001/:8002, data-parallel-size 4 each; partition ml.p5en.48xlarge, account sfr-rl.
- GPT-via-gateway experiments use the separate `server_gateway_*.sh` scripts + the `*_api.py` runners.
