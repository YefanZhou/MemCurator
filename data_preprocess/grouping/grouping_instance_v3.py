"""
PairSamplerV3: semantic embedding-based pair sampling

Improvements over V2:
1. Sentence-transformer embeddings (captures "Pigeonhole Principle" ≈ "Counting Argument")
2. Exact string matching fast path (instant for identical phrases)
3. Pre-computed inverted index for O(1) candidate lookup
4. Flagged fallback — never silently returns random garbage

Usage:
    tasks = [TaskItem(...), ...]
    sampler = PairSamplerV3(tasks)
    pair = sampler.sample_pair()       # returns PairSample or None
    batch = sampler.sample_batch(100)  # returns list[PairSample]
"""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


# ------------------------------------------------------------------
# Data schema (same as V2 — drop-in compatible)
# ------------------------------------------------------------------

@dataclass(frozen=True)
class TaskItem:
    task_id: str
    topic: List[str]
    skills: List[str]
    concepts: List[str]
    strategies: List[str]
    pitfalls: List[str]
    difficulty: float
    payload: Any = None  # optional: raw prompt / answer / metadata


@dataclass
class PairSample:
    source: TaskItem
    target: TaskItem
    negative: Optional[TaskItem] = None
    meta: Optional[Dict[str, Any]] = None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _clean(s: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


# ------------------------------------------------------------------
# 1) Embedding-based phrase encoder
# ------------------------------------------------------------------

class PhraseEmbedder:
    """
    Encode phrases using a sentence-transformer model.
    All embeddings are L2-normalised → cosine similarity = dot product.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self._cache: Dict[str, np.ndarray] = {}

    def fit(self, phrases: Sequence[str]) -> "PhraseEmbedder":
        """Batch-encode all unique phrases and cache the vectors."""
        unique = sorted(set(_clean(p) for p in phrases if _clean(p)))
        if not unique:
            return self

        embeddings = self.model.encode(
            unique,
            batch_size=512,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        for phrase, emb in zip(unique, embeddings):
            self._cache[phrase] = emb

        print(f"PhraseEmbedder: cached {len(unique)} unique phrases "
              f"(dim={embeddings.shape[1]})")
        return self

    def encode(self, phrase: str) -> Optional[np.ndarray]:
        phrase = _clean(phrase)
        if not phrase:
            return None
        if phrase in self._cache:
            return self._cache[phrase]
        # Encode on-the-fly for unseen phrases
        emb = self.model.encode([phrase], normalize_embeddings=True)[0]
        self._cache[phrase] = emb
        return emb


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product of two L2-normalised vectors == cosine similarity."""
    return float(np.dot(a, b))


# ------------------------------------------------------------------
# 2) Soft-Jaccard with exact-match fast path
# ------------------------------------------------------------------

def soft_jaccard_phrases(
    A: Sequence[str],
    B: Sequence[str],
    *,
    encoder: PhraseEmbedder,
    tau: float = 0.65,
) -> Tuple[float, float, int]:
    """
    Improved Soft-Jaccard:
      Step 1 – exact string match (cleaned) → contributes sim=1.0 each
      Step 2 – embedding-based greedy 1-to-1 matching on remaining phrases
    Returns: (soft_jaccard, soft_intersection, matched_pairs_count)
    """
    A_clean = [_clean(x) for x in A if _clean(x)]
    B_clean = [_clean(x) for x in B if _clean(x)]

    if not A_clean or not B_clean:
        return 0.0, 0.0, 0

    # ---- Step 1: exact matches ----
    A_set, B_set = set(A_clean), set(B_clean)
    exact = A_set & B_set

    soft_inter = float(len(exact))
    matched = len(exact)

    A_rem = [a for a in A_clean if a not in exact]
    B_rem = [b for b in B_clean if b not in exact]

    # ---- Step 2: embedding fuzzy matches on leftovers ----
    if A_rem and B_rem:
        A_emb = [(i, encoder.encode(a)) for i, a in enumerate(A_rem)]
        B_emb = [(j, encoder.encode(b)) for j, b in enumerate(B_rem)]
        A_emb = [(i, e) for i, e in A_emb if e is not None]
        B_emb = [(j, e) for j, e in B_emb if e is not None]

        if A_emb and B_emb:
            edges: List[Tuple[float, int, int]] = []
            for i, ae in A_emb:
                for j, be in B_emb:
                    sim = _cosine(ae, be)
                    if sim >= tau:
                        edges.append((sim, i, j))

            edges.sort(reverse=True, key=lambda x: x[0])
            used_i: Set[int] = set()
            used_j: Set[int] = set()
            for sim, i, j in edges:
                if i in used_i or j in used_j:
                    continue
                used_i.add(i)
                used_j.add(j)
                soft_inter += sim
                matched += 1

    total = len(A_clean) + len(B_clean)
    denom = total - soft_inter
    sj = soft_inter / denom if denom > 1e-9 else 0.0
    return sj, soft_inter, matched


# ------------------------------------------------------------------
# 3) Inverted index for fast candidate lookup
# ------------------------------------------------------------------

class InvertedIndex:
    """
    Cleaned-phrase → set[task_index].
    Look up which tasks share at least one identical phrase with a query task.
    """

    def __init__(self, tasks: Sequence[TaskItem], fields: Sequence[str]):
        self._index: Dict[str, Set[int]] = defaultdict(set)
        for idx, task in enumerate(tasks):
            for field in fields:
                for p in (getattr(task, field, None) or []):
                    c = _clean(p)
                    if c:
                        self._index[c].add(idx)
        print(f"InvertedIndex: {len(self._index)} unique phrases "
              f"across {len(tasks)} tasks")

    def candidates_for(
        self,
        task: TaskItem,
        fields: Sequence[str],
    ) -> Set[int]:
        """Task indices that share ≥1 exact phrase in *fields*."""
        hits: Set[int] = set()
        for field in fields:
            for p in (getattr(task, field, None) or []):
                c = _clean(p)
                if c and c in self._index:
                    hits.update(self._index[c])
        return hits


# ------------------------------------------------------------------
# 4) PairSamplerV3
# ------------------------------------------------------------------

class PairSamplerV3:
    """
    Pair sampler with semantic matching, inverted index, and flagged fallback.

    Candidate search order in _pick_target:
      Pass 1 – inverted-index candidates + strict difficulty + dependency gate
      Pass 2 – inverted-index candidates + relaxed difficulty + dependency gate
      Pass 3 – random pool + full semantic dependency gate  ← flagged
      None   – if nothing found, returns None (caller decides what to do)

    ALL returned pairs pass the full dependency gate (no gate-free fallback).
    """

    FIELDS = ("topic", "skills", "concepts", "strategies", "pitfalls")
    DEPENDENCY_FIELDS = ("concepts", "strategies", "pitfalls")

    def __init__(
        self,
        tasks: Sequence[TaskItem],
        *,
        seed: int = None,
        # encoder
        embedding_model: str = "all-MiniLM-L6-v2",
        tau: float = 0.65,
        # dependency gates
        min_shared_concepts: int = 1,
        min_shared_skills: int = 0,
        require_shared_strategy_or_pitfall: bool = False,
        require_progression: bool = False,
        # similarity filters
        min_overall_similarity: float = 0.0,
        max_topic_soft_jaccard: float = 0.65,
        max_overall_similarity: float = 0.85,
        # difficulty curriculum
        p_easy_to_hard: float = 0.75,
        p_same: float = 0.20,
        p_hard_to_easy: float = 0.05,
        easy_to_hard_gap: Tuple[float, float] = (0.5, 3.0),
        same_gap_abs: float = 0.3,
        hard_to_easy_gap: Tuple[float, float] = (-3.0, -0.5),
        # scoring weights
        w_concepts: float = 4.0,
        w_strategies: float = 3.0,
        w_pitfalls: float = 2.0,
        w_skills: float = 1.0,
        w_topic: float = 2.0,
        w_diff_bonus: float = 2.0,
        # absolute minimum difficulty delta (applied in ALL passes)
        # None = no floor; 0.0 = never allow negative deltas
        min_difficulty_delta: Optional[float] = None,
        # inverted-index candidate cap (adds randomness via subsampling)
        max_inv_candidates: int = 2000,
        # fallback random pool size
        fallback_pool: int = 200,
        # negatives
        include_hard_negative: bool = False,
        negative_scan: int = 80,
        negative_min_topic_or_skill_sj: float = 0.4,
        negative_max_concept_matches: int = 0,
        negative_topic_weight: float = 0.7,
        negative_skill_weight: float = 0.3,
    ):
        self.rng = random.Random(seed)
        self.tasks = list(tasks)
        if not self.tasks:
            raise ValueError("tasks is empty")

        self.tau = tau
        self.min_shared_concepts = min_shared_concepts
        self.min_shared_skills = min_shared_skills
        self.require_shared_strategy_or_pitfall = require_shared_strategy_or_pitfall
        self.require_progression = require_progression
        self.max_topic_soft_jaccard = max_topic_soft_jaccard
        self.min_overall_similarity = min_overall_similarity
        self.max_overall_similarity = max_overall_similarity

        # difficulty
        total = p_easy_to_hard + p_same + p_hard_to_easy
        if total <= 0:
            raise ValueError("difficulty mode probs must sum > 0")
        self.p_easy_to_hard = p_easy_to_hard / total
        self.p_same = p_same / total
        self.p_hard_to_easy = p_hard_to_easy / total
        self.easy_to_hard_gap = easy_to_hard_gap
        self.same_gap_abs = same_gap_abs
        self.hard_to_easy_gap = hard_to_easy_gap

        # scoring
        self.w_concepts = w_concepts
        self.w_strategies = w_strategies
        self.w_pitfalls = w_pitfalls
        self.w_skills = w_skills
        self.w_topic = w_topic
        self.w_diff_bonus = w_diff_bonus
        self.min_difficulty_delta = min_difficulty_delta

        self.max_inv_candidates = max_inv_candidates
        self.fallback_pool = fallback_pool

        self.include_hard_negative = include_hard_negative
        self.negative_scan = max(10, negative_scan)
        self.negative_min_topic_or_skill_sj = negative_min_topic_or_skill_sj
        self.negative_max_concept_matches = negative_max_concept_matches
        self.negative_topic_weight = negative_topic_weight
        self.negative_skill_weight = negative_skill_weight

        # ── task_id → index map ──
        self._id_to_idx: Dict[str, int] = {
            t.task_id: i for i, t in enumerate(self.tasks)
        }

        # ── 1. Fit phrase embedder ──
        print("Fitting phrase embedder …")
        all_phrases: List[str] = []
        for t in self.tasks:
            for f in self.FIELDS:
                all_phrases.extend(getattr(t, f) or [])
        self.encoder = PhraseEmbedder(model_name=embedding_model).fit(all_phrases)

        # ── 2. Build inverted index ──
        print("Building inverted index …")
        self.inv_index = InvertedIndex(self.tasks, self.DEPENDENCY_FIELDS)

    # ================================================================
    # Public API
    # ================================================================

    def sample_pair(
        self,
        exclude_ids: Optional[Set[str]] = None,
        source: Optional[TaskItem] = None,
    ) -> Optional[PairSample]:
        """
        Returns a PairSample or None if no valid pair is found.

        Args:
            exclude_ids: task_ids to skip for both source and target.
            source:      if provided, use this as the source instead of
                         picking one at random.
        Check meta["is_fallback"] to know if dependency is guaranteed.
        """
        _excl = exclude_ids or set()
        if source is not None:
            src = source
        else:
            eligible = [t for t in self.tasks if t.task_id not in _excl]
            if not eligible:
                return None
            src = self.rng.choice(eligible)
        mode = self._sample_mode()
        tgt, is_fallback = self._pick_target(src, mode, exclude_ids=_excl)
        if tgt is None:
            return None
        neg = self._pick_negative(src, tgt) if self.include_hard_negative else None
        meta = self._build_meta(src, tgt, neg, mode, is_fallback)
        return PairSample(source=src, target=tgt, negative=neg, meta=meta)

    def sample_batch(
        self,
        n: int,
        max_attempts_factor: int = 3,
        exclude_ids: Optional[Set[str]] = None,
    ) -> List[PairSample]:
        results: List[PairSample] = []
        attempts = 0
        while len(results) < n and attempts < n * max_attempts_factor:
            pair = self.sample_pair(exclude_ids=exclude_ids)
            if pair is not None:
                results.append(pair)
            attempts += 1
        return results

    # ================================================================
    # Difficulty helpers
    # ================================================================

    def _sample_mode(self) -> str:
        r = self.rng.random()
        if r < self.p_easy_to_hard:
            return "easy_to_hard"
        if r < self.p_easy_to_hard + self.p_same:
            return "same"
        return "hard_to_easy"

    def _difficulty_ok(self, d1: float, d2: float, mode: str) -> bool:
        delta = d2 - d1
        if mode == "easy_to_hard":
            return self.easy_to_hard_gap[0] <= delta <= self.easy_to_hard_gap[1]
        if mode == "same":
            return abs(delta) <= self.same_gap_abs
        return self.hard_to_easy_gap[0] <= delta <= self.hard_to_easy_gap[1]

    def _difficulty_floor_ok(self, d_src: float, d_tgt: float) -> bool:
        """Universal minimum delta floor — applied across ALL passes."""
        if self.min_difficulty_delta is None:
            return True
        return (d_tgt - d_src) >= self.min_difficulty_delta

    # ================================================================
    # Overlap primitives (embedding-based)
    # ================================================================

    def _softj(
        self, A: Sequence[str], B: Sequence[str],
    ) -> Tuple[float, float, int]:
        return soft_jaccard_phrases(A, B, encoder=self.encoder, tau=self.tau)

    def _match_count(self, A: Sequence[str], B: Sequence[str]) -> int:
        _, _, m = self._softj(A, B)
        return m

    def _overall_similarity(self, src: TaskItem, tgt: TaskItem) -> float:
        """Weighted average derived from scoring weights (consistent ratios)."""
        sj_con, _, _ = self._softj(src.concepts, tgt.concepts)
        sj_str, _, _ = self._softj(src.strategies, tgt.strategies)
        sj_pit, _, _ = self._softj(src.pitfalls, tgt.pitfalls)
        sj_skl, _, _ = self._softj(src.skills, tgt.skills)
        sj_top, _, _ = self._softj(src.topic, tgt.topic)
        # Topic uses the same w_topic as pair scoring
        total = (self.w_concepts + self.w_strategies + self.w_pitfalls +
                 self.w_skills + self.w_topic)
        return ((self.w_concepts * sj_con + self.w_strategies * sj_str +
                 self.w_pitfalls * sj_pit + self.w_skills * sj_skl +
                 self.w_topic * sj_top) / total)

    # ================================================================
    # Dependency gate & scoring
    # ================================================================

    def _dependency_gate(self, src: TaskItem, cand: TaskItem) -> bool:
        """Strict dependency gate — ensures genuine prerequisite relationship.

        Checks (in order):
        1. Not same task
        2. Shared concepts  ≥ min_shared_concepts
        3. Shared skills    ≥ min_shared_skills
        4. Shared strategy or pitfall (if required)
        5. Topic not too similar (avoid near-duplicates)
        6. Overall similarity cap (avoid near-duplicates)
        7. Progression: target has ≥1 concept or skill that source lacks
           (ensures the pair represents learning progression, not repetition)
        """
        if src.task_id == cand.task_id:
            return False
        # ── Foundation overlap: concepts + skills ──
        concept_matches = self._match_count(src.concepts, cand.concepts)
        if concept_matches < self.min_shared_concepts:
            return False
        skill_matches = self._match_count(src.skills, cand.skills)
        if skill_matches < self.min_shared_skills:
            return False
        # ── Strategy / pitfall overlap ──
        if self.require_shared_strategy_or_pitfall:
            if (self._match_count(src.strategies, cand.strategies) <= 0 and
                    self._match_count(src.pitfalls, cand.pitfalls) <= 0):
                return False
        # ── Diversity: not too similar; Relevance: not too different ──
        sj_topic, _, _ = self._softj(src.topic, cand.topic)
        if sj_topic > self.max_topic_soft_jaccard:
            return False
        overall = self._overall_similarity(src, cand)
        if overall > self.max_overall_similarity:
            return False
        if overall < self.min_overall_similarity:
            return False
        # ── Progression: target introduces something new ──
        if self.require_progression:
            # Target must have at least 1 concept OR 1 skill not matched in source
            tgt_has_new_concept = len(cand.concepts) > concept_matches
            tgt_has_new_skill = len(cand.skills) > skill_matches
            if not (tgt_has_new_concept or tgt_has_new_skill):
                return False
        return True

    def _pair_score(self, src: TaskItem, cand: TaskItem, mode: str) -> float:
        sj_con, _, _ = self._softj(src.concepts, cand.concepts)
        sj_str, _, _ = self._softj(src.strategies, cand.strategies)
        sj_pit, _, _ = self._softj(src.pitfalls, cand.pitfalls)
        sj_skl, _, _ = self._softj(src.skills, cand.skills)
        sj_top, _, _ = self._softj(src.topic, cand.topic)

        base = (self.w_concepts * sj_con +
                self.w_strategies * sj_str +
                self.w_pitfalls * sj_pit +
                self.w_skills * sj_skl +
                self.w_topic * sj_top)

        # Difficulty bonus capped: max contribution ≈ 30% of semantic range
        delta = cand.difficulty - src.difficulty
        if mode == "easy_to_hard":
            base += self.w_diff_bonus * max(0.0, min(delta, self.easy_to_hard_gap[1]))
        elif mode == "same":
            base += 0.2 * self.w_diff_bonus
        else:
            base -= 0.5 * self.w_diff_bonus * max(0.0, -delta)
        return base

    # ================================================================
    # Target selection (inverted index → semantic fallback → None)
    # ================================================================

    def _pick_target(
        self,
        src: TaskItem,
        mode: str,
        exclude_ids: Optional[Set[str]] = None,
    ) -> Tuple[Optional[TaskItem], bool]:
        """
        Returns (target_task, is_fallback).
        is_fallback=False means the pair came from the inverted-index pool.
        is_fallback=True  means it came from the random fallback pool.
        None              means nothing useful was found at all.

        Subsamples up to max_inv_candidates from the inverted-index pool
        (adds randomness), then picks the highest-scoring gate-passing pair.
        """
        _excl = exclude_ids or set()
        src_idx = self._id_to_idx.get(src.task_id, -1)

        # ── Inverted-index candidates (share ≥1 exact phrase) ──
        inv_hits = self.inv_index.candidates_for(src, self.DEPENDENCY_FIELDS)
        inv_hits.discard(src_idx)
        # Remove excluded task indices
        if _excl:
            inv_hits = {i for i in inv_hits
                        if self.tasks[i].task_id not in _excl}

        # Subsample for randomness when candidate pool is large
        if len(inv_hits) > self.max_inv_candidates:
            inv_hits = set(self.rng.sample(sorted(inv_hits),
                                           k=self.max_inv_candidates))

        inv_cands = [self.tasks[i] for i in inv_hits]
        self.rng.shuffle(inv_cands)

        # ── Single sweep: difficulty floor + gate → globally best score ──
        best: Optional[Tuple[float, TaskItem]] = None
        for cand in inv_cands:
            if not self._difficulty_floor_ok(src.difficulty, cand.difficulty):
                continue
            if not self._dependency_gate(src, cand):
                continue
            score = self._pair_score(src, cand, mode)
            if best is None or score > best[0]:
                best = (score, cand)
        if best is not None:
            return best[1], False

        # ── Fallback: random pool + full semantic gate ──
        #   (catches semantically similar pairs missed by exact-phrase index)
        eligible = [t for t in self.tasks
                    if t.task_id != src.task_id and t.task_id not in _excl]
        pool = self.rng.sample(
            eligible, k=min(self.fallback_pool, len(eligible))) if eligible else []
        for cand in pool:
            if not self._difficulty_floor_ok(src.difficulty, cand.difficulty):
                continue
            if self._dependency_gate(src, cand):
                score = self._pair_score(src, cand, mode)
                if best is None or score > best[0]:
                    best = (score, cand)
        if best is not None:
            return best[1], True  # flagged

        # ── Nothing found ──
        return None, True

    # ================================================================
    # Negative selection (unchanged from V2)
    # ================================================================

    def _pick_negative(
        self, src: TaskItem, tgt: TaskItem,
    ) -> Optional[TaskItem]:
        best: Optional[Tuple[float, TaskItem]] = None
        for _ in range(self.negative_scan):
            cand = self.rng.choice(self.tasks)
            if cand.task_id in (src.task_id, tgt.task_id):
                continue
            sj_topic, _, _ = self._softj(src.topic, cand.topic)
            sj_skill, _, _ = self._softj(src.skills, cand.skills)
            if max(sj_topic, sj_skill) < self.negative_min_topic_or_skill_sj:
                continue
            concept_matches = self._match_count(src.concepts, cand.concepts)
            if concept_matches > self.negative_max_concept_matches:
                continue
            sj_t_topic, _, _ = self._softj(tgt.topic, cand.topic)
            sj_t_skill, _, _ = self._softj(tgt.skills, cand.skills)
            confusable = (self.negative_topic_weight * sj_t_topic +
                         self.negative_skill_weight * sj_t_skill)
            if best is None or confusable > best[0]:
                best = (confusable, cand)
        return best[1] if best else None

    # ================================================================
    # Meta / inspection
    # ================================================================

    def _build_meta(
        self,
        src: TaskItem,
        tgt: TaskItem,
        neg: Optional[TaskItem],
        mode: str,
        is_fallback: bool,
    ) -> Dict[str, Any]:
        sj_topic, _, _ = self._softj(src.topic, tgt.topic)
        overall = self._overall_similarity(src, tgt)

        con_sj, con_inter, con_m = self._softj(src.concepts, tgt.concepts)
        str_sj, str_inter, str_m = self._softj(src.strategies, tgt.strategies)
        pit_sj, pit_inter, pit_m = self._softj(src.pitfalls, tgt.pitfalls)
        skl_sj, skl_inter, skl_m = self._softj(src.skills, tgt.skills)

        meta: Dict[str, Any] = {
            "mode": mode,
            "is_fallback": is_fallback,
            "difficulty_delta": tgt.difficulty - src.difficulty,
            "softj_topic": sj_topic,
            "overall_similarity": overall,
            "matches": {
                "concepts": {
                    "softj": con_sj,
                    "soft_inter": con_inter,
                    "matched_pairs": con_m,
                },
                "strategies": {
                    "softj": str_sj,
                    "soft_inter": str_inter,
                    "matched_pairs": str_m,
                },
                "pitfalls": {
                    "softj": pit_sj,
                    "soft_inter": pit_inter,
                    "matched_pairs": pit_m,
                },
                "skills": {
                    "softj": skl_sj,
                    "soft_inter": skl_inter,
                    "matched_pairs": skl_m,
                },
            },
            "tau": self.tau,
        }

        if neg is not None:
            neg_topic, _, _ = self._softj(src.topic, neg.topic)
            neg_skill, _, _ = self._softj(src.skills, neg.skills)
            neg_con_m = self._match_count(src.concepts, neg.concepts)
            meta["negative"] = {
                "task_id": neg.task_id,
                "softj_topic": neg_topic,
                "softj_skills": neg_skill,
                "concept_matches": neg_con_m,
            }

        return meta


# ------------------------------------------------------------------
# Quick smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import json

    tasks = []
    with open("./group_annotation_DeepMath.jsonl", "r") as f:
        data = [json.loads(line) for line in f]

    tasks = [
        TaskItem(
            task_id=str(i),
            topic=d["annotations"]["Topic"],
            skills=d["annotations"]["Skills or Capabilities"],
            concepts=d["annotations"]["Math Concepts or Theorems"],
            strategies=d["annotations"]["Heuristic Strategy"],
            pitfalls=d["annotations"]["Common Pitfalls"],
            difficulty=d["difficulty"],
            payload=d,
        )
        for i, d in enumerate(data)
    ]

    sampler = PairSamplerV3(
        tasks,
        seed=None,
        tau=0.60,
        # ── tight dependency gates ──
        min_shared_concepts=1,
        min_shared_skills=1,
        require_shared_strategy_or_pitfall=True,
        require_progression=True,
        # ── similarity filters ──
        min_overall_similarity=0.3,
        # ── difficulty curriculum ──
        p_easy_to_hard=0.8,
        p_same=0.2,
        p_hard_to_easy=0.0,
        min_difficulty_delta=0.0,       # never allow negative Δd
        # ── scoring weights ──
        w_concepts=5.0,
        w_strategies=3.0,
        w_pitfalls=1.0,
        w_skills=4.0,
        w_topic=2.0,
        w_diff_bonus=1.0,
    )

    # ── Diagnostic (set DIAG=True to profile gate bottlenecks) ──
    DIAG = False
    if DIAG:
        print("\n=== Gate Diagnostic (50 random sources × up to 50 inv-index cands) ===")
        from collections import Counter
        fail_reasons = Counter()
        pass_count = 0
        total_checked = 0
        for _ in range(50):
            src = sampler.rng.choice(sampler.tasks)
            src_idx = sampler._id_to_idx.get(src.task_id, -1)
            inv_hits = sampler.inv_index.candidates_for(src, sampler.DEPENDENCY_FIELDS)
            inv_hits.discard(src_idx)
            cands = [sampler.tasks[i] for i in list(inv_hits)[:50]]
            total_checked += len(cands)
            for cand in cands:
                cm = sampler._match_count(src.concepts, cand.concepts)
                sm = sampler._match_count(src.skills, cand.skills)
                stm = sampler._match_count(src.strategies, cand.strategies)
                pm = sampler._match_count(src.pitfalls, cand.pitfalls)
                sj_top, _, _ = sampler._softj(src.topic, cand.topic)
                overall = sampler._overall_similarity(src, cand)
                d_ok = sampler._difficulty_floor_ok(src.difficulty, cand.difficulty)
                tgt_new_c = len(cand.concepts) > cm
                tgt_new_s = len(cand.skills) > sm

                reasons = []
                if cm < sampler.min_shared_concepts:
                    reasons.append("concepts < 2")
                if sm < sampler.min_shared_skills:
                    reasons.append("skills < 1")
                if sampler.require_shared_strategy_or_pitfall and stm <= 0 and pm <= 0:
                    reasons.append("no strat/pitfall")
                if sj_top > sampler.max_topic_soft_jaccard:
                    reasons.append("topic too similar")
                if overall > sampler.max_overall_similarity:
                    reasons.append("overall too similar")
                if sampler.require_progression and not (tgt_new_c or tgt_new_s):
                    reasons.append("no progression")
                if not d_ok:
                    reasons.append("neg difficulty")
                if reasons:
                    for r in reasons:
                        fail_reasons[r] += 1
                else:
                    pass_count += 1

        print(f"  Checked {total_checked} (src, cand) pairs")
        print(f"  PASSED all conditions: {pass_count}")
        for reason, cnt in fail_reasons.most_common():
            print(f"  FAIL: {reason:25s} {cnt:6d} ({100*cnt/max(total_checked,1):.1f}%)")
        print()

    # Sample 5 pairs
    for trial in range(5):
        sample = sampler.sample_pair()
        if sample is None:
            print(f"\n[Trial {trial}] No valid pair found.")
            continue

        m = sample.meta
        print(f"\n[Trial {trial}]  "
              f"SRC={sample.source.task_id}  TGT={sample.target.task_id}  "
              f"fallback={m['is_fallback']}  mode={m['mode']}  "
              f"Δd={m['difficulty_delta']}")
        print(f"  Concepts  matched={m['matches']['concepts']['matched_pairs']}  "
              f"sj={m['matches']['concepts']['softj']:.3f}")
        print(f"  Strategies matched={m['matches']['strategies']['matched_pairs']}  "
              f"sj={m['matches']['strategies']['softj']:.3f}")
        print(f"  Pitfalls  matched={m['matches']['pitfalls']['matched_pairs']}  "
              f"sj={m['matches']['pitfalls']['softj']:.3f}")
        print(f"  Topic sj={m['softj_topic']:.3f}  "
              f"overall={m['overall_similarity']:.3f}")

        src_q = (sample.source.payload or {}).get("question", "N/A")
        tgt_q = (sample.target.payload or {}).get("question", "N/A")

        print(f"  ── Source (id={sample.source.task_id}, diff={sample.source.difficulty}) ──")
        print(f"    Question:   {src_q[:200]}{'…' if len(src_q) > 200 else ''}")
        print(f"    Topic:      {sample.source.topic}")
        print(f"    Skills:     {sample.source.skills}")
        print(f"    Concepts:   {sample.source.concepts}")
        print(f"    Strategies: {sample.source.strategies}")
        print(f"    Pitfalls:   {sample.source.pitfalls}")
        print(f"  ── Target (id={sample.target.task_id}, diff={sample.target.difficulty}) ──")
        print(f"    Question:   {tgt_q[:200]}{'…' if len(tgt_q) > 200 else ''}")
        print(f"    Topic:      {sample.target.topic}")
        print(f"    Skills:     {sample.target.skills}")
        print(f"    Concepts:   {sample.target.concepts}")
        print(f"    Strategies: {sample.target.strategies}")
        print(f"    Pitfalls:   {sample.target.pitfalls}")
