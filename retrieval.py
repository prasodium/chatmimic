#!/usr/bin/env python3
"""
Tiny dependency-free TF-IDF retrieval over your own past replies.

For each contact we index *their* past messages; a new incoming message
finds the most similar past incoming messages, and we suggest *your*
replies to those. Pure standard library -- no numpy, no sklearn.
"""

import json
import math
import re
from collections import Counter
from pathlib import Path


def tokenize(text: str):
    return re.findall(r"[a-z0-9']+", text.lower())


class TfIdfIndex:
    def __init__(self, docs: list[str]):
        self.docs = docs
        self.N = len(docs)
        tfs = [Counter(tokenize(d)) for d in docs]
        df = Counter()
        for tf in tfs:
            for tok in tf:
                df[tok] += 1
        self.idf = {tok: math.log((1 + self.N) / (1 + c)) + 1.0
                    for tok, c in df.items()}
        self.vecs, self.norms = [], []
        for tf in tfs:
            total = sum(tf.values())
            vec = {tok: (c / total) * self.idf[tok] for tok, c in tf.items()}
            n = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self.vecs.append(vec)
            self.norms.append(n)

    def query(self, text: str, k: int = 3):
        qtf = Counter(tokenize(text))
        if not qtf:
            return []
        total = sum(qtf.values())
        q = {tok: (c / total) * self.idf.get(tok, 0.0)
             for tok, c in qtf.items()}
        qn = math.sqrt(sum(v * v for v in q.values())) or 1.0
        scored = []
        for i, vec in enumerate(self.vecs):
            dot = sum(qv * vec.get(tok, 0.0) for tok, qv in q.items())
            s = dot / (qn * self.norms[i])
            if s > 0:
                scored.append((i, s))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


class ReplyEngine:
    """Per-contact reply suggester backed by one JSON pairs file."""

    def __init__(self, pairs_path: str | Path):
        self.pairs = json.loads(Path(pairs_path).read_text(encoding="utf-8"))
        self.index = TfIdfIndex([p["incoming"] for p in self.pairs])
        # fallback: your most common reply (generic style default)
        replies = Counter(p["reply"] for p in self.pairs)
        self.fallback = replies.most_common(1)[0][0] if replies else "..."

    def suggest(self, incoming: str, k: int = 3):
        """Return [(reply, score, pair_index)] best-first."""
        return [(self.pairs[i]["reply"], s, i)
                for i, s in self.index.query(incoming, k=k)]

    def __len__(self):
        return len(self.pairs)
