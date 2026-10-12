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
from situations import POLICY, classify, human_only

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "feedback_log.jsonl"


def make_draft(engine, bandit, incoming, style=None, gen=None):
    """Retrieve -> classify situation -> bandit (per-situation) picks ->
    (optional) LSTM rephrase.

    If gen (a seq2seq.Seq2SeqModel) is given, a brand-new generated reply
    competes as the "gen" arm -- the bandit learns per situation whether
    you prefer generated or retrieved replies.

    Returns (draft, arm, similarity, label, situation). similarity is None
    for the fallback and gen arms. `style` is a generate.StyleModel or None.
    """
    cands = engine.suggest(incoming, k=3)
    sim = cands[0][1] if cands else None
    situation = classify(incoming, sim)
    bandit.use_situation(situation)
    gen_text = gen.generate(incoming) if gen is not None else None
    arm = bandit.select(len(cands), has_gen=gen_text is not None)
    if arm == "gen":
        return (gen_text, arm, None,
                "gen (newly generated in your style)", situation)
    if arm == "fallback":
        return (engine.fallback, arm, None,
                "fallback (your most common reply)", situation)
    draft, score, _ = cands[int(arm[4:])]
    label = f"{arm} (similarity {score:.2f})"
    if style is not None:
        from generate import rephrase
        new = rephrase(style, draft)
        if new != draft:
            draft, label = new, label + " + lstm"
    return draft, arm, sim, label, situation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contact", required=True,
                    help="Contact name (matches data/pairs/<name>.json)")
    ap.add_argument("--pairs-dir", default=str(ROOT / "data" / "pairs"))
    ap.add_argument("--state-dir", default=str(ROOT / "data" / "bandit"))
    ap.add_argument("--hybrid", action="store_true",
                    help="v2: rephrase retrieved replies with your char-LSTM "
                         "(needs: python train_lstm.py --all)")
    ap.add_argument("--gen", action="store_true",
                    help="true generation: brand-new replies from your seq2seq "
                         "model compete as the 'gen' bandit arm "
                         "(needs: python seq2seq.py train --all)")
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
    gen = None
    if args.gen:
        from seq2seq import Seq2SeqModel
        gen = Seq2SeqModel.for_contact(args.contact)
        if gen is None:
            raise SystemExit("no seq2seq model -- run: python seq2seq.py train --all")
    modes = "".join([
        " (hybrid: retrieval + LSTM)" if style else "",
        " (gen: true generation as bandit arm)" if gen else "",
    ])
    print(f"loaded {len(engine)} pairs for {args.contact}{modes}")
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

        draft, arm, _, src, situation = make_draft(engine, bandit,
                                                       incoming, style, gen)

        flag = ""
        if human_only(situation):
            flag = (f"  ⚠ '{situation}' → human only "
                    f"({POLICY[situation]['note']})")
        print(f"\ndraft [{src} | situation: {situation}]{flag}: {draft}")
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
                "situation": situation,
                "draft": draft,
                "decision": kind,
                "final": final,
                "reward": reward,
            }, ensure_ascii=False) + "\n")
        qs = ", ".join(f"{a}={bandit.q[a]:+.2f}" for a in bandit.q)
        print(f"  -> {kind} (reward {reward:+.1f})"
              f"   bandit[{situation}]: {qs}\n")


if __name__ == "__main__":
    main()
