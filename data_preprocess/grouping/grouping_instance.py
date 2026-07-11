"""
PairSamplerV2: data-driven fuzzy matching

What you get:
- TaskItem schema with your 5 label dimensions + difficulty
- Char n-gram TF-IDF phrase encoder fit on *your* phrases (unsupervised)
- Soft-Jaccard overlap between two lists of phrases (fuzzy, 1-to-1 greedy matching)
- PairSamplerV2 that samples (source -> target) with guaranteed dependency:
    * shared_concepts >= 1 (fuzzy)
    * and (shared_strategies >= 1 OR shared_pitfalls >= 1) (fuzzy)
  while discouraging near-duplicates and controlling difficulty direction (easy->hard, etc.)
- Optional hard negative sampling (topic/skills similar but concept disjoint)

Usage:
1) Build a list[TaskItem]
2) sampler = PairSamplerV2(tasks, seed=42)
3) sample = sampler.sample_pair()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
import random
import re
from collections import Counter


# ---------------------------
# Data schema
# ---------------------------

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


# ---------------------------
# Data-driven fuzzy matching
# ---------------------------

def _clean(s: str) -> str:
    s = (s or "").lower().strip()
    # normalize whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def _char_ngrams(s: str, n: int) -> List[str]:
    # remove spaces for char ngrams
    s = _clean(s).replace(" ", "")
    if not s:
        return []
    if len(s) < n:
        return [s]
    return [s[i:i + n] for i in range(len(s) - n + 1)]


@dataclass
class CharTfidf:
    """
    Unsupervised char-ngram TF-IDF encoder (sparse dict vectors).
    Fit on your corpus of phrases; no manual synonym/abbrev tables.
    """
    ngram_range: Tuple[int, int] = (3, 5)
    min_df: int = 2

    idf: Dict[str, float] = None
    _fitted: bool = False

    def fit(self, phrases: Sequence[str]) -> "CharTfidf":
        df = Counter()
        docs = 0

        for p in phrases:
            p = _clean(p)
            if not p:
                continue
            docs += 1
            seen = set()
            for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
                for g in _char_ngrams(p, n):
                    seen.add(g)
            for g in seen:
                df[g] += 1

        self.idf = {}
        for g, c in df.items():
            if c >= self.min_df:
                # smooth idf
                self.idf[g] = math.log((docs + 1) / (c + 1)) + 1.0

        self._fitted = True
        return self

    def encode(self, phrase: str) -> Dict[str, float]:
        if not self._fitted:
            raise RuntimeError("CharTfidf not fitted; call fit() first.")
        phrase = _clean(phrase)
        if not phrase:
            return {}

        tf = Counter()
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for g in _char_ngrams(phrase, n):
                if g in self.idf:
                    tf[g] += 1

        if not tf:
            return {}

        # log-tf * idf
        vec = {g: (1.0 + math.log(c)) * self.idf[g] for g, c in tf.items()}

        # L2 normalize for cosine
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            for g in list(vec.keys()):
                vec[g] /= norm
        return vec


def cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    # iterate smaller
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def soft_jaccard_phrases(
    A: Sequence[str],
    B: Sequence[str],
    *,
    encoder: CharTfidf,
    tau: float,
    max_pairs: int = 20_000,
) -> Tuple[float, float, int]:
    """
    Soft-Jaccard between two phrase lists with fuzzy matching:
    - Compute cosine similarity between phrase vectors
    - Keep edges >= tau
    - Greedy 1-to-1 matching to avoid double counting
    - soft_intersection = sum(sim_matched)
    - soft_jaccard = soft_intersection / (|A| + |B| - soft_intersection)

    Returns: (soft_jaccard, soft_intersection, matched_pairs_count)
    """
    A = [x for x in A if x and x.strip()]
    B = [x for x in B if x and x.strip()]
    if not A and not B:
        return 0.0, 0.0, 0
    if not A or not B:
        return 0.0, 0.0, 0

    Avec = [encoder.encode(x) for x in A]
    Bvec = [encoder.encode(x) for x in B]

    pairs: List[Tuple[float, int, int]] = []
    cnt = 0
    for i in range(len(A)):
        for j in range(len(B)):
            cnt += 1
            if cnt > max_pairs:
                break
            sim = cosine_sparse(Avec[i], Bvec[j])
            if sim >= tau:
                pairs.append((sim, i, j))
        if cnt > max_pairs:
            break

    pairs.sort(reverse=True, key=lambda x: x[0])

    used_i, used_j = set(), set()
    soft_inter = 0.0
    matched = 0
    for sim, i, j in pairs:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        soft_inter += sim
        matched += 1

    denom = (len(A) + len(B) - soft_inter)
    sj = soft_inter / denom if denom > 1e-9 else 0.0
    return sj, soft_inter, matched


# ---------------------------
# PairSamplerV2
# ---------------------------

class PairSamplerV2:
    """
    Samples (instance1 -> instance2) pairs with *guaranteed* memory dependency using fuzzy overlap.
    No hard-coded phrase tables; similarity is learned from your phrase corpus via char-ngram TF-IDF.

    Dependency gate:
      - fuzzy_shared_concepts >= 1 (matched phrase pairs under concept tau)
      - and (fuzzy_shared_strategies >=1 OR fuzzy_shared_pitfalls >=1)

    Discourage:
      - too-high topic overlap (fuzzy soft-jaccard)
      - near-duplicate overall similarity

    Difficulty modes:
      - easy_to_hard (default heavy)
      - same
      - hard_to_easy (small)
    """

    FIELDS = ("topic", "skills", "concepts", "strategies", "pitfalls")

    def __init__(
        self,
        tasks: Sequence[TaskItem],
        *,
        seed: int = 0,
        # TF-IDF encoder
        ngram_range: Tuple[int, int] = (3, 5),
        min_df: int = 2,
        # auto-thresholding (tau) settings
        tau_quantile: float = 0.995,     # near-max noise baseline
        tau_floor: float = 0.45,
        tau_cap: float = 0.85,
        tau_margin: float = 0.05,        # set tau = q + margin, clipped
        tau_samples_per_field: int = 4000,

        # dependency gates (counts are fuzzy matches >= tau)
        min_shared_concepts: int = 1,
        require_shared_strategy_or_pitfall: bool = True,

        # similarity filters
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
        w_topic_penalty: float = 3.0,
        w_diff_bonus: float = 2.0,

        # candidate search
        candidate_pool_multiplier: int = 30,  # how many candidates to scan per source
        # negatives
        include_hard_negative: bool = True,
        negative_scan: int = 80,
        negative_min_topic_or_skill_sj: float = 0.4,
        negative_max_concept_matches: int = 0,  # must be concept-disjoint
    ):
        self.rng = random.Random(seed)
        self.tasks = list(tasks)
        if not self.tasks:
            raise ValueError("tasks is empty")

        self.min_shared_concepts = min_shared_concepts
        self.require_shared_strategy_or_pitfall = require_shared_strategy_or_pitfall
        self.max_topic_soft_jaccard = max_topic_soft_jaccard
        self.max_overall_similarity = max_overall_similarity

        # difficulty distribution normalization
        total = p_easy_to_hard + p_same + p_hard_to_easy
        if total <= 0:
            raise ValueError("difficulty mode probabilities must sum to > 0")
        self.p_easy_to_hard = p_easy_to_hard / total
        self.p_same = p_same / total
        self.p_hard_to_easy = p_hard_to_easy / total
        self.easy_to_hard_gap = easy_to_hard_gap
        self.same_gap_abs = same_gap_abs
        self.hard_to_easy_gap = hard_to_easy_gap

        self.w_concepts = w_concepts
        self.w_strategies = w_strategies
        self.w_pitfalls = w_pitfalls
        self.w_skills = w_skills
        self.w_topic_penalty = w_topic_penalty
        self.w_diff_bonus = w_diff_bonus

        self.candidate_pool_multiplier = max(5, candidate_pool_multiplier)

        self.include_hard_negative = include_hard_negative
        self.negative_scan = max(10, negative_scan)
        self.negative_min_topic_or_skill_sj = negative_min_topic_or_skill_sj
        self.negative_max_concept_matches = negative_max_concept_matches

        # Fit encoder on all phrases (all fields)
        all_phrases: List[str] = []
        for t in self.tasks:
            for f in self.FIELDS:
                all_phrases.extend(getattr(t, f) or [])
        self.encoder = CharTfidf(ngram_range=ngram_range, min_df=min_df).fit(all_phrases)

        # Auto-set tau per field (data-driven)
        self.tau: Dict[str, float] = {}
        self.tau_meta: Dict[str, Any] = {}
        for field in self.FIELDS:
            field_phrases = [p for t in self.tasks for p in (getattr(t, field) or []) if _clean(p)]
            tau_val, stats = self._auto_tau(
                field_phrases,
                tau_quantile=tau_quantile,
                tau_floor=tau_floor,
                tau_cap=tau_cap,
                tau_margin=tau_margin,
                samples=tau_samples_per_field,
            )
            self.tau[field] = tau_val
            self.tau_meta[field] = stats

    # -------- public API --------

    def sample_pair(self) -> PairSample:
        src = self.rng.choice(self.tasks)
        mode = self._sample_mode()
        tgt = self._pick_target(src, mode)
        neg = self._pick_negative(src, tgt) if self.include_hard_negative else None
        meta = self._build_meta(src, tgt, neg, mode)
        return PairSample(source=src, target=tgt, negative=neg, meta=meta)

    def sample_batch(self, n: int) -> List[PairSample]:
        return [self.sample_pair() for _ in range(n)]

    # -------- tau estimation --------

    def _auto_tau(
        self,
        phrases: List[str],
        *,
        tau_quantile: float,
        tau_floor: float,
        tau_cap: float,
        tau_margin: float,
        samples: int,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Estimate a similarity threshold tau for a field without hard-coding.
        We sample random phrase pairs and compute cosine sims.
        We take q-quantile as "noise high end" and set tau = q + margin.
        """
        phrases = [p for p in phrases if _clean(p)]
        if len(phrases) < 50:
            # too small to estimate; pick a conservative default
            tau = min(tau_cap, max(tau_floor, 0.65))
            return tau, {"note": "too_few_phrases", "tau": tau}

        sims: List[float] = []
        n = len(phrases)
        # pre-encode a subset for speed
        # sample phrases with replacement
        for _ in range(samples):
            a = phrases[self.rng.randrange(n)]
            b = phrases[self.rng.randrange(n)]
            if a == b:
                continue
            sim = cosine_sparse(self.encoder.encode(a), self.encoder.encode(b))
            sims.append(sim)

        if not sims:
            tau = min(tau_cap, max(tau_floor, 0.65))
            return tau, {"note": "no_sims", "tau": tau}

        sims.sort()
        idx = int(min(len(sims) - 1, max(0, round(tau_quantile * (len(sims) - 1)))))
        q = sims[idx]
        tau = q + tau_margin
        tau = max(tau_floor, min(tau_cap, tau))
        return tau, {
            "tau": tau,
            "q": q,
            "quantile": tau_quantile,
            "margin": tau_margin,
            "samples_used": len(sims),
            "p50": sims[len(sims) // 2],
            "p90": sims[int(0.90 * (len(sims) - 1))],
            "p99": sims[int(0.99 * (len(sims) - 1))],
        }

    # -------- difficulty logic --------

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
            lo, hi = self.easy_to_hard_gap
            return lo <= delta <= hi
        if mode == "same":
            return abs(delta) <= self.same_gap_abs
        lo, hi = self.hard_to_easy_gap
        return lo <= delta <= hi

    # -------- overlap primitives --------

    def _softj(self, field: str, A: Sequence[str], B: Sequence[str]) -> Tuple[float, float, int]:
        return soft_jaccard_phrases(A, B, encoder=self.encoder, tau=self.tau[field])

    def _match_count(self, field: str, A: Sequence[str], B: Sequence[str]) -> int:
        _, _, matched = self._softj(field, A, B)
        return matched

    def _overall_similarity(self, src: TaskItem, tgt: TaskItem) -> float:
        # weighted fuzzy overlap across fields
        sj_con, _, _ = self._softj("concepts", src.concepts, tgt.concepts)
        sj_str, _, _ = self._softj("strategies", src.strategies, tgt.strategies)
        sj_pit, _, _ = self._softj("pitfalls", src.pitfalls, tgt.pitfalls)
        sj_skl, _, _ = self._softj("skills", src.skills, tgt.skills)
        sj_top, _, _ = self._softj("topic", src.topic, tgt.topic)
        return (
            0.30 * sj_con +
            0.25 * sj_str +
            0.15 * sj_pit +
            0.15 * sj_skl +
            0.15 * sj_top
        )

    # -------- dependency gates & scoring --------

    def _dependency_gate(self, src: TaskItem, cand: TaskItem) -> bool:
        if src.task_id == cand.task_id:
            return False

        # Must share concepts
        shared_concepts = self._match_count("concepts", src.concepts, cand.concepts)
        if shared_concepts < self.min_shared_concepts:
            return False

        # Must share strategy or pitfall (optional gate)
        if self.require_shared_strategy_or_pitfall:
            shared_str = self._match_count("strategies", src.strategies, cand.strategies)
            shared_pit = self._match_count("pitfalls", src.pitfalls, cand.pitfalls)
            if shared_str <= 0 and shared_pit <= 0:
                return False

        # Avoid too much topic overlap (to promote transfer)
        sj_topic, _, _ = self._softj("topic", src.topic, cand.topic)
        if sj_topic > self.max_topic_soft_jaccard:
            return False

        # Avoid near-duplicates overall
        if self._overall_similarity(src, cand) > self.max_overall_similarity:
            return False

        return True

    def _pair_score(self, src: TaskItem, cand: TaskItem, mode: str) -> float:
        sj_con, _, _ = self._softj("concepts", src.concepts, cand.concepts)
        sj_str, _, _ = self._softj("strategies", src.strategies, cand.strategies)
        sj_pit, _, _ = self._softj("pitfalls", src.pitfalls, cand.pitfalls)
        sj_skl, _, _ = self._softj("skills", src.skills, cand.skills)
        sj_top, _, _ = self._softj("topic", src.topic, cand.topic)

        base = (
            self.w_concepts * sj_con +
            self.w_strategies * sj_str +
            self.w_pitfalls * sj_pit +
            self.w_skills * sj_skl -
            self.w_topic_penalty * sj_top
        )

        # difficulty bonus
        d1, d2 = src.difficulty, cand.difficulty
        delta = d2 - d1
        if mode == "easy_to_hard":
            base += self.w_diff_bonus * max(0.0, min(delta, self.easy_to_hard_gap[1]))
        elif mode == "same":
            base += 0.2 * self.w_diff_bonus
        else:
            base -= 0.5 * self.w_diff_bonus * max(0.0, -delta)

        return base

    # -------- target / negative selection --------

    def _pick_target(self, src: TaskItem, mode: str) -> TaskItem:
        # Scan a random subset; simplest robust approach without building heavy indexes.
        # If you need speed later, we can add an approximate index keyed by hashed TF-IDF.
        pool_size = min(len(self.tasks), self.candidate_pool_multiplier)
        candidates = self.rng.sample(self.tasks, k=pool_size) if len(self.tasks) > pool_size else list(self.tasks)
        self.rng.shuffle(candidates)

        best: Optional[Tuple[float, TaskItem]] = None

        # First pass: strict difficulty + dependency
        for cand in candidates:
            if not self._difficulty_ok(src.difficulty, cand.difficulty, mode):
                continue
            if not self._dependency_gate(src, cand):
                continue
            score = self._pair_score(src, cand, mode)
            if best is None or score > best[0]:
                best = (score, cand)

        if best is not None:
            return best[1]

        # Second pass: relax difficulty, keep dependency
        for cand in candidates:
            if not self._dependency_gate(src, cand):
                continue
            score = self._pair_score(src, cand, mode)
            if best is None or score > best[0]:
                best = (score, cand)

        if best is not None:
            return best[1]

        # Fallback: return random different
        for _ in range(50):
            cand = self.rng.choice(self.tasks)
            if cand.task_id != src.task_id:
                return cand
        return src  # extremely unlikely

    def _pick_negative(self, src: TaskItem, tgt: TaskItem) -> Optional[TaskItem]:
        # Hard negative: topic/skills similar, but concept-disjoint (no concept matches)
        best: Optional[Tuple[float, TaskItem]] = None

        for _ in range(self.negative_scan):
            cand = self.rng.choice(self.tasks)
            if cand.task_id in (src.task_id, tgt.task_id):
                continue

            # looks similar by topic or skills
            sj_topic, _, _ = self._softj("topic", src.topic, cand.topic)
            sj_skill, _, _ = self._softj("skills", src.skills, cand.skills)
            if max(sj_topic, sj_skill) < self.negative_min_topic_or_skill_sj:
                continue

            # but concept disjoint
            concept_matches = self._match_count("concepts", src.concepts, cand.concepts)
            if concept_matches > self.negative_max_concept_matches:
                continue

            # prefer confusable w.r.t target
            sj_t_topic, _, _ = self._softj("topic", tgt.topic, cand.topic)
            sj_t_skill, _, _ = self._softj("skills", tgt.skills, cand.skills)
            confusable = 0.7 * sj_t_topic + 0.3 * sj_t_skill

            if best is None or confusable > best[0]:
                best = (confusable, cand)

        return best[1] if best else None

    # -------- meta / inspection --------

    def _build_meta(self, src: TaskItem, tgt: TaskItem, neg: Optional[TaskItem], mode: str) -> Dict[str, Any]:
        sj_topic, _, _ = self._softj("topic", src.topic, tgt.topic)
        overall = self._overall_similarity(src, tgt)

        con_sj, con_inter, con_m = self._softj("concepts", src.concepts, tgt.concepts)
        str_sj, str_inter, str_m = self._softj("strategies", src.strategies, tgt.strategies)
        pit_sj, pit_inter, pit_m = self._softj("pitfalls", src.pitfalls, tgt.pitfalls)

        meta = {
            "mode": mode,
            "difficulty_delta": tgt.difficulty - src.difficulty,
            "softj_topic": sj_topic,
            "overall_similarity": overall,
            "matches": {
                "concepts": {"softj": con_sj, "soft_inter": con_inter, "matched_pairs": con_m},
                "strategies": {"softj": str_sj, "soft_inter": str_inter, "matched_pairs": str_m},
                "pitfalls": {"softj": pit_sj, "soft_inter": pit_inter, "matched_pairs": pit_m},
            },
            "tau": dict(self.tau),          # thresholds actually used
            "tau_meta": dict(self.tau_meta) # how they were estimated
        }

        if neg is not None:
            neg_topic, _, _ = self._softj("topic", src.topic, neg.topic)
            neg_skill, _, _ = self._softj("skills", src.skills, neg.skills)
            neg_con_m = self._match_count("concepts", src.concepts, neg.concepts)
            meta["negative"] = {
                "task_id": neg.task_id,
                "softj_topic": neg_topic,
                "softj_skills": neg_skill,
                "concept_matches": neg_con_m,
            }

        return meta


# ---------------------------
# Minimal example (delete in production)
# ---------------------------

if __name__ == "__main__":
    # Tiny toy data (replace with yours)

    tasks = []

    with open("./group_annotation_DeepMath.jsonl", "r") as f:
        import json
        data = [json.loads(line) for line in f][:3000]

        tasks.extend([TaskItem(
            task_id=str(i),
            topic=data[i]['annotations']['Topic'],
            skills=data[i]['annotations']['Skills or Capabilities'],
            concepts=data[i]['annotations']['Math Concepts or Theorems'],
            strategies=data[i]['annotations']['Heuristic Strategy'],
            pitfalls=data[i]['annotations']['Common Pitfalls'],
            difficulty=data[i]['difficulty'],
        ) for i in range(len(data))])

        # for i in data:
        #     print(i['annotations'])
        #     input()


    sampler = PairSamplerV2(
        tasks,
        seed=None,  # Random seed for different pairs each run
        # Lower tau thresholds for better matches
        tau_floor=0.30,
        tau_cap=0.70,
        # Scan ALL candidates for best match
        candidate_pool_multiplier=len(tasks),  # Check every task
        # Relax dependency requirements
        require_shared_strategy_or_pitfall=False,
        # Difficulty curriculum
        p_easy_to_hard=0.8,
        p_same=0.2,
        p_hard_to_easy=0.0,
        include_hard_negative=True,
    )

    sample = sampler.sample_pair()
    print("SOURCE:", sample.source.task_id)
    print("TARGET:", sample.target.task_id)
    print("NEG:", sample.negative.task_id if sample.negative else None)
    print("META keys:", list(sample.meta.keys()))
    print("MODE:", sample.meta["mode"], "Δd:", sample.meta["difficulty_delta"])
    print("MATCHES:", sample.meta["matches"])
    print("TAU(concepts):", sample.meta["tau"]["concepts"])
    
    print("\n=== SOURCE TASK ===")
    print(f"Task ID: {sample.source.task_id}")
    print(f"Topic: {sample.source.topic}")
    print(f"Skills: {sample.source.skills}")
    print(f"Concepts: {sample.source.concepts}")
    print(f"Strategies: {sample.source.strategies}")
    print(f"Pitfalls: {sample.source.pitfalls}")
    print(f"Difficulty: {sample.source.difficulty}")
    
    print("\n=== TARGET TASK ===")
    print(f"Task ID: {sample.target.task_id}")
    print(f"Topic: {sample.target.topic}")
    print(f"Skills: {sample.target.skills}")
    print(f"Concepts: {sample.target.concepts}")
    print(f"Strategies: {sample.target.strategies}")
    print(f"Pitfalls: {sample.target.pitfalls}")
    print(f"Difficulty: {sample.target.difficulty}")
    