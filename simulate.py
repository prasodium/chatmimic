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

    test_autosend(engine)
    test_lstm()


def test_autosend(engine):
    """v3: the gate must hold everything back until confident + opted in."""
    import tempfile
    from datetime import datetime
    import autosend as A

    noon = datetime.now().astimezone().replace(hour=12, minute=0)
    night = noon.replace(hour=2)
    with tempfile.TemporaryDirectory() as d:
        p = A.Paths(Path(d))
        bandit = Bandit(p.bandit / "demo.json", epsilon=0.0)
        outbox = lambda *_: None
        run = lambda now=noon: A.process(
            p, "demo", "want to grab lunch tomorrow?", engine, bandit, send=outbox, now=now)
        trained = lambda: [bandit.update("rank0", 1.0) for _ in range(A.DEFAULTS["min_n"])]

        assert run() == "queued", "must be draft-only before opt-in"
        A.save_config(p, {**A.DEFAULTS, "contacts": ["demo"]})
        assert run() == "queued", "untrained bandit must not auto-send"
        trained()
        assert run() == "sent", "confident + whitelisted should send"
        assert run(night) == "queued", "quiet hours must hold"
        p.kill.write_text("x")
        assert run() == "queued", "kill switch must hold"
        p.kill.unlink()
        A.save_config(p, {**A.DEFAULTS, "contacts": ["demo"], "max_per_day": 1})
        assert run() == "queued", "daily rate limit must hold"
        assert len(A.read_jsonl(p.send_log)) == 1
    print("OK: send gate holds on opt-in, confidence, quiet hours, "
          "kill switch, rate limit.")


def test_lstm():
    """v2: LSTM trains and the hybrid never drifts from the retrieved reply."""
    try:
        import torch  # noqa: F401
    except ImportError:
        print("SKIP: v2 LSTM test (pip install torch to enable)")
        return
    from generate import StyleModel, rephrase
    from train_lstm import train_model

    texts = [json.loads(PAIRS.read_text())[i]["reply"] for i in range(200)]
    ckpt = train_model(texts, epochs=15, verbose=False)
    assert ckpt["val_loss"] < 1.5, f"LSTM failed to learn: {ckpt['val_loss']:.2f}"
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        torch.save(ckpt, Path(d) / "demo.pt")
        style = StyleModel(Path(d) / "demo.pt")
    sample = style.generate("yeah", 0.5)
    print(f"LSTM sample: {sample!r}  (val loss {ckpt['val_loss']:.2f})")
    out = rephrase(style, "yeah sure, what time?", seed=0)
    assert {"yeah", "sure", "what", "time"} & set(out.lower().replace(",", " ").replace("?", "").split()), out
    print(f"OK: LSTM learned your voice; hybrid rephrase -> {out!r}")


if __name__ == "__main__":
    main()
