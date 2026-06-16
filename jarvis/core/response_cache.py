"""Semantic response cache.

Caches simple chat answers so repeated or near-identical questions return
instantly instead of re-running the model. It is deliberately conservative:

- Only the tool-free "fast chat" path uses it (answers that depend on tools,
  web search, time, weather, prices, etc. are never cached or served).
- An exact text match is tried first (free); otherwise a cosine-similarity
  match over query embeddings is used, gated by a high threshold.
- Entries expire after a TTL and the cache is capped in size.

Embeddings reuse the local Ollama embedding model. If embeddings are not
available (e.g. a cloud provider), the cache safely degrades to exact-match.
"""

from __future__ import annotations

import json
import time
import threading
from pathlib import Path

import numpy as np

# Queries containing these hints are time/context-sensitive and must stay fresh.
_SKIP_HINTS = (
    "today", "now", "current", "currently", "latest", "right now", "this week",
    "this month", "tomorrow", "yesterday", "time", "date", "weather", "news",
    "price", "stock", "score", "who won", "breaking",
)


class ResponseCache:
    def __init__(
        self,
        provider=None,
        embed_model: str = "nomic-embed-text",
        path: str | Path | None = None,
        threshold: float = 0.94,
        max_entries: int = 500,
        ttl_seconds: int = 86_400,
        min_chars: int = 12,
    ):
        self.provider = provider
        self.embed_model = embed_model
        self.threshold = threshold
        self.max_entries = max_entries
        self.ttl = ttl_seconds
        self.min_chars = min_chars
        self.path = Path(path) if path else (Path.home() / ".jarvis" / "cache" / "response_cache.json")
        self._lock = threading.Lock()
        self._entries: list[dict] = []  # {"q": str, "response": str, "emb": list|None, "ts": float}
        self._load()

    # ---- public API -------------------------------------------------------
    def is_cacheable(self, query: str) -> bool:
        if not query:
            return False
        q = query.strip().lower()
        if len(q) < self.min_chars or len(q) > 2000:
            return False
        return not any(h in q for h in _SKIP_HINTS)

    def get(self, query: str):
        """Return a cached response for an equivalent query, or None."""
        if not self.is_cacheable(query):
            return None
        norm = self._normalize(query)
        now = time.time()
        with self._lock:
            self._evict_expired(now)
            # 1) exact (normalized) match
            for e in self._entries:
                if e["q"] == norm:
                    return e["response"]
            # 2) semantic match
            emb = self._embed(query)
            if emb is None:
                return None
            best, best_sim = None, 0.0
            for e in self._entries:
                if not e.get("emb"):
                    continue
                sim = self._cosine(emb, e["emb"])
                if sim > best_sim:
                    best, best_sim = e, sim
            if best is not None and best_sim >= self.threshold:
                return best["response"]
        return None

    def put(self, query: str, response: str) -> None:
        if not response or not response.strip() or not self.is_cacheable(query):
            return
        norm = self._normalize(query)
        emb = self._embed(query)
        with self._lock:
            # replace existing entry for the same normalized query
            self._entries = [e for e in self._entries if e["q"] != norm]
            self._entries.append({"q": norm, "response": response, "emb": emb, "ts": time.time()})
            if len(self._entries) > self.max_entries:
                self._entries = self._entries[-self.max_entries:]
            self._save()

    # ---- internals --------------------------------------------------------
    @staticmethod
    def _normalize(query: str) -> str:
        return " ".join(query.strip().lower().split())

    @staticmethod
    def _cosine(a, b) -> float:
        va, vb = np.asarray(a, dtype="float32"), np.asarray(b, dtype="float32")
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))

    def _embed(self, text: str):
        """Embed text via the local Ollama embedding model; None on failure."""
        client = getattr(self.provider, "client", None)
        if client is None:
            return None
        try:
            resp = client.embeddings(model=self.embed_model, prompt=text)
            vec = resp.get("embedding") if isinstance(resp, dict) else getattr(resp, "embedding", None)
            return list(vec) if vec else None
        except Exception:
            return None

    def _evict_expired(self, now: float) -> None:
        if self.ttl > 0:
            self._entries = [e for e in self._entries if now - e.get("ts", 0) <= self.ttl]

    def _load(self) -> None:
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self._entries = data
        except Exception:
            self._entries = []

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._entries), encoding="utf-8")
        except Exception:
            pass
