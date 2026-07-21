# curator_vendor — provenance

**Byte-identical copies** of the MemCurator memory library from sea-mem-policy, vendored so the
transferred `curator_v1` method stays faithful to the original (and to any trained curator
checkpoint's expected prompt/format).

## Source
- Repo: `~/Research/sea-mem-policy`
- Package: **`curator/`** (the newer of the two sibling packages; `curator/` == `curator_react/`
  for everything here EXCEPT `memory_strategies/curator.py`, where `curator/` adds the
  `sampling` passthrough to `chat.completions.create`. We vendored from `curator/`.)
- Copied: 2026-07-16.

## Files (all verbatim `cp`, verified with `cmp`)
| vendored path | source |
|---|---|
| `prompt_loader.py` | `curator/prompt_loader.py` |
| `memory_module/{__init__,schema,store,retriever}.py` | `curator/memory_module/…` (identical in both packages) |
| `memory_strategies/{__init__,strategies,curator,synapse}.py` | `curator/memory_strategies/…` |
| `templates/curator_system.txt` | `curator/templates/curator_system.txt` |

`__init__.py` at this dir's root is the ONLY added file (makes `curator_vendor` a package so the
original `from ..memory_module …` / `from ..prompt_loader …` relative imports resolve unchanged).

## DO NOT EDIT
These files are frozen. All adaptation lives OUTSIDE this package, in
`../curator_v1_alfworld.py` (the runner adapter):
- retrieval swapped to BM25 (via a `MemoryStore` subclass that overrides `search()`),
- sync bridge over the async `CuratorReader`,
- CURATION_* → vendored `sampling` dict,
- our `curator_on_empty` / `task_context` knobs + `curator_calls.jsonl` logging.

If you need to re-sync from an updated source, re-`cp` and update the date above.

## Dependency note
`memory_module/{store,retriever}.py` import `torch`/`transformers` (for the original's dense
SimCSE retriever). `curator_v1` uses BM25 and never calls `search()` on the base class, but
importing the package still triggers `import torch` at module load. The box `memory` env has
torch, so this is fine on the box; locally, tests stub torch.
