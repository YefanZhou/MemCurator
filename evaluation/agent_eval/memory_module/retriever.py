"""SimCSE-based dense retriever with lazy model loading."""

from __future__ import annotations

import logging
from typing import List, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "princeton-nlp/sup-simcse-bert-base-uncased"


class SimCSERetriever:
    """Embed texts with SimCSE and retrieve by cosine similarity.

    The transformer model is loaded lazily on the first call to
    :meth:`embed` or :meth:`search`, not at import time.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "auto",
        batch_size: int = 64,
        max_length: int = 256,
    ) -> None:
        self.model_name = model_name
        self._device_str = device
        self.batch_size = batch_size
        self.max_length = max_length

        self._tokenizer = None
        self._model = None
        self._device: torch.device | None = None

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    def _resolve_device(self) -> torch.device:
        if self._device_str == "cpu":
            return torch.device("cpu")
        if self._device_str in ("cuda",) and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        logger.info("Loading SimCSE model %s …", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._device = self._resolve_device()
        self._model = self._model.to(self._device)
        self._model.eval()
        logger.info("SimCSE model loaded on %s", self._device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed(self, texts: List[str]) -> torch.Tensor:
        """Return L2-normalised embeddings (N, D) on CPU."""
        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None
        vecs: list[torch.Tensor] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self._device) for k, v in enc.items()}
            out = self._model(**enc, return_dict=True)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                e = out.pooler_output
            else:
                e = out.last_hidden_state[:, 0, :]
            vecs.append(F.normalize(e, p=2, dim=1).cpu())
        return torch.cat(vecs, dim=0)

    def search(
        self,
        query: str,
        keys: List[str],
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """Return ``(index, score)`` tuples for the *top_k* closest keys.

        Re-embeds all *keys* on every call.  Prefer :meth:`search_cached`
        when key embeddings are maintained externally.
        """
        if not keys:
            return []
        key_vecs = self.embed(keys)
        return self.search_cached(query, key_vecs, top_k=top_k)

    def search_cached(
        self,
        query: str,
        key_vecs: torch.Tensor,
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """Like :meth:`search` but with pre-computed *key_vecs* ``(N, D)``."""
        if key_vecs.shape[0] == 0:
            return []
        # TODO: use fast knn search library instead of torch.topk
        query_vec = self.embed([query])[0].unsqueeze(0)
        sims = (query_vec @ key_vecs.T).squeeze(0)
        k = min(top_k, key_vecs.shape[0])
        topk_scores, topk_idx = torch.topk(sims, k)
        return list(zip(topk_idx.tolist(), topk_scores.tolist()))
