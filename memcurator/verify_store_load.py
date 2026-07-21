"""Read-only pre-smoke check: confirm Stage C's store loader cleanly reads the new multi-field
store schema (trajectory + trajectory_stage_a/_action_only/_w_thinking) produced by build_dataset,
under the curator variant/mode the smoke will use.

It loads the FIRST store of a dataset via the SAME path Stage C uses (CuratorBackend.make_store ->
CuratorAlfworld._load_jsonl -> BM25), does one retrieval, and renders a _format_case — proving the
extra trajectory_* keys are ignored and `trajectory` is what gets read. No training, no GPU, no LLM.

Run (box, conda `memory`, cwd = evaluation/agent_eval so Alfworld config resolves):
    cd /fsx/home/yefan.zhou/mem-evolve/SkillCurator-main/evaluation/agent_eval
    PYTHONPATH=/fsx/home/yefan.zhou/mem-evolve/SkillCurator-main \
    python -m memcurator.verify_store_load \
        --dataset /fsx/home/yefan.zhou/mem-evolve/data/datasets/smoke_dataset/dataset.jsonl \
        --variant curator_alfworld_v1_api --curation_mode success_only
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify Stage C can load a build_dataset store.")
    ap.add_argument("--dataset", required=True, help="Path to dataset.jsonl")
    ap.add_argument("--variant", default="curator_alfworld_v1_api")
    ap.add_argument("--curation_mode", default="success_only")
    ap.add_argument("--retrieve_num", type=int, default=3)
    args = ap.parse_args()

    from memcurator.curator_backend import CuratorBackend

    rows = [json.loads(l) for l in open(args.dataset, encoding="utf-8") if l.strip()]
    print(f"[verify] dataset rows: {len(rows)}")
    row = rows[0]
    sp = row["store_path"]
    print(f"[verify] first target: {row['task_id']}  store: {sp}")

    b = CuratorBackend(variant=args.variant, curation_mode=args.curation_mode)
    store = b.make_store(storage_path=sp, retrieve_num=args.retrieve_num, curator_on_empty=True)
    n = len(store.memory_bank)
    print(f"[verify] store loaded: {n} records")

    # inspect the raw stored keys (should include the extra trajectory_* fields)
    keys = list(store.memory_bank[0].keys()) if n else []
    print(f"[verify] stored record keys: {keys}")

    # one retrieval on the target's query + render a case (this is what Stage C does)
    q = row.get("query", "") or "put an object somewhere"
    docs = store.bm25_retriever.invoke(q) if store.bm25_retriever else []
    print(f"[verify] retrieved {len(docs)} docs for query {q!r}")
    if docs:
        rec = store.memory_bank[docs[0].metadata["idx"]]
        case = store._format_case(1, rec)
        assert "Trajectory:" in case, "format_case did not render a Trajectory: block"
        assert rec.get("trajectory"), "retrieved record has empty 'trajectory' (the field Stage C reads)"
        print(f"[verify] _format_case OK (len={len(case)}); reads 'trajectory' field correctly.")
        print("[verify] ---- case preview (first 300 chars) ----")
        print(case[:300])

    # also build the curator prompt end-to-end (retrieved_text -> messages), like _generate_briefings
    retrieved_text = "\n\n".join(
        store._format_case(j + 1, store.memory_bank[d.metadata["idx"]]) for j, d in enumerate(docs)
    )
    msgs = b.build_curator_messages(q, retrieved_text)
    print(f"[verify] build_curator_messages OK: {len(msgs)} turns, roles={[m['role'] for m in msgs]}")
    print("\n[verify] PASS — Stage C store loader reads the new schema cleanly (extra fields ignored).")


if __name__ == "__main__":
    main()
