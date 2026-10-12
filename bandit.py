#!/usr/bin/env python3
"""
Contextual bandit -- the 'reinforcement' in v1.

Arms = which candidate to send: rank0 / rank1 / rank2 (the top TF-IDF
candidates) or the generic fallback. Reward comes from YOUR feedback
in the draft loop:

    approve as-is  -> +1.0
    edit then send -> +0.4
    reject         -> -1.0

Epsilon-greedy with incremental-average Q updates. State persists per
contact in JSON, so learning survives restarts. This is genuine
reinforcement learning -- just the lightweight (bandit) kind instead
of deep RL.
"""

import json
import random
from pathlib import Path

ARMS = ["rank0", "rank1", "rank2", "fallback"]
REWARDS = {"approve": 1.0, "edit": 0.4, "reject": -1.0}


class Bandit:
    def __init__(self, state_path: str | Path, epsilon: float = 0.15,
                 seed: int = 7):
        self.state_path = Path(state_path)
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.q = {a: 0.0 for a in ARMS}
        self.n = {a: 0 for a in ARMS}
        if self.state_path.exists():
            s = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.q.update(s.get("q", {}))
            self.n.update(s.get("n", {}))

    def select(self, n_candidates: int) -> str:
        """Pick an arm. n_candidates = how many retrieval candidates exist."""
        valid = [f"rank{i}" for i in range(min(n_candidates, 3))] + ["fallback"]
        if self.rng.random() < self.epsilon:
            return self.rng.choice(valid)  # explore
        # exploit: highest Q, random tiebreak so fresh arms get a chance
        return max(valid, key=lambda a: (self.q[a], self.rng.random()))

    def update(self, arm: str, reward: float):
        self.n[arm] += 1
        self.q[arm] += (reward - self.q[arm]) / self.n[arm]  # incremental mean
        self.save()

    def save(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps({"q": self.q, "n": self.n}, indent=1), encoding="utf-8")

    def summary(self):
        return {a: {"q": round(self.q[a], 3), "n": self.n[a]} for a in ARMS}
