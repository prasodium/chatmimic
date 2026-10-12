#!/usr/bin/env python3
"""
Teach the bot from your own answers — the day-by-day loop.

When the bot can't handle a situation and YOU answer instead, that answer
is new knowledge. This script appends the (incoming -> your reply) pair
to the contact's data file so the retrieval engine finds it next time.

    python learn.py --contact "Aarav Sharma"                 # interactive
    python learn.py --contact "Aarav Sharma" \\
        --incoming "are we still on for lunch?" --reply "yep, see you at 1"

How the loop compounds daily:
  - Retrieval needs NO retraining: the index rebuilds from data/pairs/
    on every startup, so taught pairs are live immediately.
  - The bandit keeps ranking: the new reply becomes a candidate and your
    future approve/edit/reject decisions teach its Q-value per person.
  - v2 LSTM: re-run train_lstm.py (e.g. nightly) to fold taught pairs
    into the style model.

Pairs are deduplicated (exact incoming+reply) and timestamped.
"""

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAIRS_DIR = ROOT / "data" / "pairs"


def teach(contact: str, incoming: str, reply: str) -> bool:
    """Append a new training pair. Returns True if it was new."""
    path = PAIRS_DIR / f"{contact}.json"
    pairs = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    if any(p["incoming"] == incoming and p["reply"] == reply for p in pairs):
        print("already known — skipping duplicate.")
        return False
    pairs.append({
        "incoming": incoming,
        "reply": reply,
        "ts": int(time.time() * 1000),
    })
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pairs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"taught: {contact} now has {len(pairs)} pairs.")
    return True


def main():
    ap = argparse.ArgumentParser(description="Teach the bot your manual answer.")
    ap.add_argument("--contact", required=True, help="Contact name")
    ap.add_argument("--incoming", default=None, help="Their message you answered")
    ap.add_argument("--reply", default=None, help="The answer you gave")
    args = ap.parse_args()

    incoming = args.incoming or input("their message> ").strip()
    reply = args.reply or input("your answer> ").strip()
    if not incoming or not reply:
        raise SystemExit("need both an incoming message and your reply.")
    teach(args.contact, incoming, reply)
    print("Done. The next similar message will retrieve this reply, and the "
          "bandit will learn how well it lands from your feedback.")


if __name__ == "__main__":
    main()
