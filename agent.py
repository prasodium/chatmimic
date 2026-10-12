#!/usr/bin/env python3
"""
Draft-approval loop (v1 agent).

    python agent.py --contact "Aarav Sharma"

Paste an incoming message -> the engine retrieves your most similar past
replies -> the bandit picks one -> you approve (Enter), edit (type your
version), or reject (r). Your choice is the reward signal; the bandit
learns per contact.

Nothing is ever sent anywhere in v1 -- this is a draft loop only.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from bandit import REWARDS, Bandit
from retrieval import ReplyEngine

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "feedback_log.jsonl"


def make_draft(engine, bandit, incoming, style=None):
    """Retrieve -> bandit picks -> (optional) LSTM rephrase.

    Returns (draft, arm, similarity, label). similarity is None for the
    fallback arm. `style` is a generate.StyleModel or None (v1 behaviour).
    """
    cands = engine.suggest(incoming, k=3)
    arm = bandit.select(len(cands))
    if arm == "fallback":
        return engine.fallback, arm, None, "fallback (your most common reply)"
    draft, score, _ = cands[int(arm[4:])]
    label = f"{arm} (similarity {score:.2f})"
    if style is not None:
        from generate import rephrase
        new = rephrase(style, draft)
        if new != draft:
            draft, label = new, label + " + lstm"
    return draft, arm, score, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contact", required=True,
                    help="Contact name (matches data/pairs/<name>.json)")
    ap.add_argument("--pairs-dir", default=str(ROOT / "data" / "pairs"))
    ap.add_argument("--state-dir", default=str(ROOT / "data" / "bandit"))
    ap.add_argument("--hybrid", action="store_true",
                    help="v2: rephrase retrieved replies with your char-LSTM "
                         "(needs: python train_lstm.py --all)")
    args = ap.parse_args()

    pairs_path = Path(args.pairs_dir) / f"{args.contact}.json"
    if not pairs_path.exists():
        avail = sorted(p.stem for p in Path(args.pairs_dir).glob("*.json"))
        raise SystemExit(
            f"no data for {args.contact!r}. "
            f"available: {avail or 'none -- run parse_instagram.py first'}")

    engine = ReplyEngine(pairs_path)
    bandit = Bandit(Path(args.state_dir) / f"{args.contact}.json")
    style = None
    if args.hybrid:
        from generate import StyleModel
        style = StyleModel.for_contact(args.contact)
        if style is None:
            raise SystemExit("no LSTM model -- run: python train_lstm.py --all")
    print(f"loaded {len(engine)} pairs for {args.contact}"
          + (" (hybrid: retrieval + LSTM)" if style else ""))
    print("commands: [Enter] approve as-is | type your edit | r reject | q quit\n")

    while True:
        try:
            incoming = input("them> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if incoming.lower() == "q":
            break
        if not incoming:
            continue

        draft, arm, _, src = make_draft(engine, bandit, incoming, style)

        print(f"\ndraft [{src}]: {draft}")
        try:
            action = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if action.lower() == "q":
            break
        if action.lower() == "r":
            kind, final = "reject", None
        elif action == "":
            kind, final = "approve", draft
        else:
            kind, final = "edit", action

        reward = REWARDS[kind]
        bandit.update(arm, reward)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(),
                "contact": args.contact,
                "incoming": incoming,
                "arm": arm,
                "draft": draft,
                "decision": kind,
                "final": final,
                "reward": reward,
            }, ensure_ascii=False) + "\n")
        qs = ", ".join(f"{a}={bandit.q[a]:+.2f}" for a in bandit.q)
        print(f"  -> {kind} (reward {reward:+.1f})   bandit: {qs}\n")


if __name__ == "__main__":
    main()
