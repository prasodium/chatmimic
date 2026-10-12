#!/usr/bin/env python3
"""
End-to-end smoke test with synthetic data (no Instagram export needed).

1. Builds a fake contact corpus and checks retrieval returns the right
   replies for similar incoming messages.
2. Simulates a user whose consistent taste is "always prefer the rank1
   candidate" and checks the bandit converges on rank1.

Run:  python simulate.py
"""

import json
from pathlib import Path

from bandit import Bandit
from retrieval import ReplyEngine

ROOT = Path(__file__).resolve().parent
PAIRS = ROOT / "data" / "pairs" / "_demo.json"
STATE = ROOT / "data" / "bandit" / "_demo.json"


def build_corpus():
    topics = [
        ("want to grab lunch tomorrow?", "yeah sure, what time?"),
        ("are we still on for lunch?", "yep, see you at 1"),
        ("did you watch the match last night?", "yeah that last over was insane"),
        ("cricket match was crazy right", "fr, i thought we lost it"),
        ("send me the notes pls", "sending in 2 mins"),
        ("can you share yesterday's notes?", "yeah gimme a sec"),
        ("happy birthday!!", "thank youu!!"),
        ("happy bday bro", "thanks man means a lot"),
    ]
    pairs = []
    for i in range(200):
        inc, rep = topics[i % len(topics)]
        pairs.append({"incoming": inc, "reply": rep, "ts": i})
    PAIRS.parent.mkdir(parents=True, exist_ok=True)
    PAIRS.write_text(json.dumps(pairs), encoding="utf-8")


def main():
    build_corpus()
    engine = ReplyEngine(PAIRS)

    # 1) retrieval sanity: similar incoming -> your real reply to that topic
    checks = {
        "lunch tomorrow?": ("yeah sure, what time?", "yep, see you at 1"),
        "that match was mad": ("yeah that last over was insane", "fr, i thought we lost it"),
        "send notes": ("sending in 2 mins", "yeah gimme a sec"),
    }
    for msg, ok_replies in checks.items():
        top = engine.suggest(msg, k=1)[0][0]
        assert top in ok_replies, f"retrieval miss for {msg!r}: {top!r}"
        print(f"retrieval OK: {msg!r} -> {top!r}")

    # 2) bandit learning: simulated user always approves rank1, rejects else
    if STATE.exists():
        STATE.unlink()
    bandit = Bandit(STATE, epsilon=0.15, seed=1)
    counts = {"rank0": 0, "rank1": 0, "rank2": 0, "fallback": 0}
    test_msgs = ["lunch tomorrow?", "that match was mad", "send notes",
                 "happy birthday"]
    for t in range(300):
        msg = test_msgs[t % len(test_msgs)]
        cands = engine.suggest(msg, k=3)
        arm = bandit.select(len(cands))
        bandit.update(arm, 1.0 if arm == "rank1" else -1.0)
        counts[arm] += 1
        if (t + 1) % 100 == 0:
            qs = ", ".join(f"{a}={bandit.q[a]:+.2f}" for a in bandit.q)
            print(f"round {t + 1}: {qs}")

    print("\nfinal Q:", bandit.summary())
    print("selections:", counts)
    assert bandit.q["rank1"] == max(bandit.q.values()), \
        "bandit did not learn the preferred arm"
    assert counts["rank1"] > counts["rank0"], \
        "preferred arm was not selected most"
    print("\nOK: bandit converged on the user's preferred reply rank.")


if __name__ == "__main__":
    main()
