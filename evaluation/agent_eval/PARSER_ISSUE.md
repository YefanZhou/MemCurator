# Action Parser Robustness — ALFWorld executor

**Scope:** the `<action>...</action>` extraction in the ALFWorld ReAct loop of
`run_unified_nonthink*.py` (and every copy derived from it).
**Status:** not a bug in the *current* non-thinking runs, but fragile under two
configuration changes (thinking mode, and adding a stop token). Documented here so it
is fixed before either change lands.

---

## Current parser

```python
# run_unified_nonthink_revise_react_step0bug_fix.py, ~line 457
if '<action>' in response and '</action>' in response:
    action_list[idx] = response.split('<action>')[-1].split('</action>')[0].strip()
```

Logic: require both tags to be present, then take the substring after the **last**
`<action>` and before the following `</action>`.

**Why it is fine right now (non-thinking, no stop token):** in `revise_react` mode the
model emits plain-text reasoning followed by a single `<action>...</action>`. Exactly
one action tag pair, both tags present, no native `<think>` block. The parser extracts
cleanly. Verified on the 2-game probe: 60/60 steps produced a parsed action.

---

## Failure mode 1 — closing tag consumed by a stop token

If a stop token `["</action>"]` is added to the generation call (recommended, to prevent
runaway output after the action), vLLM by default does **not** include the stop string in
the returned text. The response then contains `<action>go to cabinet 1` with **no**
`</action>`.

Effect: the guard `and '</action>' in response` evaluates False, the `if` body never
runs, and `action_list[idx]` stays `""`. The environment receives an empty action for
that step. Silent — no error, just a wasted step.

**Fix:** drop the closing-tag requirement and tolerate a missing `</action>`:

```python
if '<action>' in response:
    action_list[idx] = response.split('<action>')[-1].split('</action>')[0].strip()
```

`split('</action>')[0]` returns the whole remainder when `</action>` is absent, which is
the desired behavior when the stop token ate the closing tag.

---

## Failure mode 2 — `<action>` string appears inside reasoning (thinking mode)

Under thinking mode, or any prompt that encourages the model to talk about the action it
will take, the model frequently writes the literal string `<action>` inside its reasoning,
e.g. *"I should output `<action>go to cabinet 1</action>` next."* When that reasoning and
the real action both reach the parser in the same `content` string, there are two or more
`<action>` occurrences.

`.split('<action>')[-1]` takes the **last** one, which is usually (but not always) the
real action. It breaks when the model's final real action is followed by more prose that
also mentions `<action>`, or when the reasoning's action mention is the last one and the
real action was never emitted. Extraction then grabs reasoning text instead of a valid
admissible action, and the env rejects it.

This does not occur in the current non-thinking `revise_react` runs (0 `<think>` tags,
one action per step), but will surface if native reasoning is not stripped from `content`.

---

## Interaction with vLLM reasoning parser (thinking mode only)

Two sub-cases when running thinking mode:

- **`--reasoning-parser qwen3` enabled:** vLLM routes the `<think>...</think>` block into
  the separate `reasoning_content` field; `content` holds only the post-`</think>` text.
  The action parser sees clean text and works. Confirm the deployed vLLM version strips
  the block fully (some versions mishandle streaming — reasoning can leak into
  `delta.content` or the whole output can land in `reasoning_content`, leaving `content`
  empty). Inspect one raw response before trusting it.
- **No reasoning parser (or completion endpoint):** the full
  `<think>...</think><action>...</action>` arrives in `content`. Failure mode 2 applies.

---

## Recommended robust parser (safe for all three configs)

Strip any native reasoning block first, then extract the action from what remains. This is
immune to both `<action>` mentions inside reasoning and a stop-token-truncated closing tag.

```python
def parse_action(response: str) -> str:
    # 1. drop anything up to and including the last </think> (native reasoning, if any)
    tail = response.rsplit('</think>', 1)[-1]
    # 2. take the last <action> block in the remaining text
    if '<action>' not in tail:
        return ""                      # no action emitted this step
    seg = tail.split('<action>')[-1]
    # 3. closing tag optional (stop token may have removed it)
    return seg.split('</action>')[0].strip()
```

Behavior by config:
- non-thinking `revise_react` (current): `rsplit('</think>')` is a no-op, identical result
  to today's parser. Safe to adopt now with zero behavior change.
- non-thinking + stop token: closing tag optional, extraction still works.
- thinking mode: native reasoning removed before searching for `<action>`, so any
  `<action>` mentioned inside reasoning is excluded.

---

## Recommended action

1. Adopt the robust parser now — it is a no-op for the current runs and removes the trap
   before thinking mode / stop tokens are introduced.
2. If a stop token is added, this parser already handles the missing closing tag; no
   further change needed.
3. If thinking mode is run with `--reasoning-parser qwen3`, still adopt this parser as a
   defense in case a vLLM version leaks reasoning into `content`.
