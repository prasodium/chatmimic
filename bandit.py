#!/usr/bin/env python3
"""
Contextual bandit -- the 'reinforcement' in the system.

Arms = which candidate to send: rank0 / rank1 / rank2 (the top TF-IDF
candidates) or the generic fallback. Reward comes from YOUR feedback
in the draft loop:

    approve as-is  -> +1.0
    edit then send -> +0.4
    reject         -> -1.0

Epsilon-greedy with incremental-average Q updates. Q-values are kept
PER SITUATION per contact (see situations.py) -- that is what makes it
a *contextual* bandit: it can learn that rank1 wins for questions while
rank0 wins for greetings, for the same person, separately.

State persists per contact in JSON, so learning survives restarts.
Old flat state files ({"q": ..., "n": ...}) migrate automatically
under the "chat" situation.
"""

import json
import random
from pathlib import Path

ARMS = ["rank0", "rank1", "rank2", "fallback"]
REWARDS = {"approve": 1.0, "edit": 0.4, "reject": -1.0}
DEFAULT_SITUATION = "chat"


def _blank():
    return {"q": {a: 0.0 for a in ARMS}, "n": {a: 0 for a in ARMS}}


class Bandit:
    def __init__(self, state_path: str | Path, epsilon: float = 0.15,
                 seed: int = 7):
        self.state_path = Path(state_path)
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.situations: dict = {}
        if self.state_path.exists():
            s = json.loads(self.state_path.read_text(encoding="utf-8"))
            if "situations" in s:
                self.situations = s["situations"]
            else:
                # old flat format -> migrate under "chat"
                self.situations = {DEFAULT_SITUATION: {
                    "q": {a: float(s.get("q", {}).get(a, 0.0)) for a in ARMS},
                    "n": {a: int(s.get("n", {}).get(a, 0)) for a in ARMS},
                }}
        self.use_situation(DEFAULT_SITUATION)

    def use_situation(self, situation: str):
        """Point q/n at this situation's tables. Call before select/update
        (make_draft does this automatically after classifying)."""
        self.situation = situation
        tbl = self.situations.setdefault(situation, _blank())
        for a in ARMS:  # forward-compat: fill any missing arms
            tbl["q"].setdefault(a, 0.0)
            tbl["n"].setdefault(a, 0)
        self.q = tbl["q"]
        self.n = tbl["n"]

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
            json.dumps({"situations": self.situations}, indent=1),
            encoding="utf-8")

    def summary(self):
        return {s: {a: {"q": round(t["q"][a], 3), "n": t["n"][a]}
                    for a in ARMS}
                for s, t in self.situations.items()}
