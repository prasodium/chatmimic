#!/usr/bin/env python3
"""
Stage 6 (v3): gated auto-send. Default is draft-only; nothing is automatic
until you opt a contact in.

    python autosend.py enable "Aarav Sharma"     # per-contact opt-in
    python autosend.py disable "Aarav Sharma"
    python autosend.py kill                      # kill switch ON  (all sends stop)
    python autosend.py unkill
    python autosend.py status
    python autosend.py run --contact "Aarav Sharma" [--incoming-file f.txt]
    python autosend.py review --contact "Aarav Sharma"   # clear the review queue

`run` reads incoming messages (one per line, stdin by default), drafts a
reply, and the SEND GATE decides: auto-send, or queue for human review.
A message is auto-sent only if ALL hold:

  - contact is on the whitelist
  - kill switch is off, not in quiet hours, under the per-day rate limit
  - arm is a retrieval arm (never the generic fallback)
  - retrieval similarity >= min_similarity
  - bandit Q for that arm >= min_q, with >= min_n feedback rounds behind it

Everything else goes to data/review_queue.jsonl, never sent. Exploration is
off for auto-sends (epsilon=0). Every sent message is appended to
data/send_log.jsonl.

"Sending" is a pluggable function. Instagram has no official API for
personal DMs, so the default sender appends to data/outbox.jsonl -- you
(or your own local tool) deliver it. This script makes no network calls.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from agent import make_draft
from bandit import REWARDS, Bandit
from retrieval import ReplyEngine

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

DEFAULTS = {
    "contacts": [],            # whitelist; empty = draft-only everywhere
    "min_similarity": 0.60,
    "min_q": 0.50,
    "min_n": 20,
    "max_per_day": 20,
    "quiet_hours": [23, 7],    # local time; no auto-send from 23:00 to 07:00
}


class Paths:
    def __init__(self, data: Path = DATA):
        self.data = data
        self.config = data / "autosend.json"
        self.kill = data / "KILL"
        self.send_log = data / "send_log.jsonl"
        self.queue = data / "review_queue.jsonl"
        self.outbox = data / "outbox.jsonl"
        self.pairs = data / "pairs"
        self.bandit = data / "bandit"


def load_config(p: Paths) -> dict:
    cfg = dict(DEFAULTS)
    if p.config.exists():
        cfg.update(json.loads(p.config.read_text(encoding="utf-8")))
    return cfg


def save_config(p: Paths, cfg: dict):
    p.data.mkdir(parents=True, exist_ok=True)
    p.config.write_text(json.dumps(cfg, indent=1), encoding="utf-8")


def append(path: Path, rec: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def in_quiet_hours(cfg: dict, now: datetime) -> bool:
    start, end = cfg["quiet_hours"]
    h = now.astimezone().hour
    return (start <= h or h < end) if start > end else (start <= h < end)


def sent_today(p: Paths, contact: str, now: datetime) -> int:
    today = now.astimezone().date()
    return sum(
        1 for r in read_jsonl(p.send_log)
        if r["contact"] == contact
        and datetime.fromisoformat(r["ts"]).astimezone().date() == today)


def gate(p: Paths, cfg: dict, contact: str, arm: str, sim, bandit: Bandit,
         now: datetime | None = None) -> tuple[bool, str]:
    """The send gate. Returns (allowed, reason)."""
    now = now or now_utc()
    if p.kill.exists():
        return False, "kill switch is on"
    if contact not in cfg["contacts"]:
        return False, "contact not whitelisted (draft-only)"
    if in_quiet_hours(cfg, now):
        return False, "quiet hours"
    if sent_today(p, contact, now) >= cfg["max_per_day"]:
        return False, "daily rate limit reached"
    if arm == "fallback" or sim is None:
        return False, "fallback reply is never auto-sent"
    if sim < cfg["min_similarity"]:
        return False, f"similarity {sim:.2f} < {cfg['min_similarity']}"
    if bandit.n[arm] < cfg["min_n"]:
        return False, f"only {bandit.n[arm]} feedback rounds on {arm} (< {cfg['min_n']})"
    if bandit.q[arm] < cfg["min_q"]:
        return False, f"{arm} Q {bandit.q[arm]:+.2f} < {cfg['min_q']}"
    return True, "confident"


def deliver_to_outbox(p: Paths, contact: str, text: str):
    """Default sender: write to the outbox. Swap for your own local tool."""
    append(p.outbox, {"ts": now_utc().isoformat(), "contact": contact,
                      "text": text})


def process(p: Paths, contact: str, incoming: str, engine, bandit, style=None,
            send=deliver_to_outbox, now: datetime | None = None) -> str:
    """Handle one incoming message. Returns 'sent' or 'queued'."""
    cfg = load_config(p)
    # No exploration when real messages are on the line.
    bandit.epsilon = 0.0
    draft, arm, sim, label = make_draft(engine, bandit, incoming, style)
    ok, reason = gate(p, cfg, contact, arm, sim, bandit, now)
    rec = {"ts": (now or now_utc()).isoformat(), "contact": contact,
           "incoming": incoming, "draft": draft, "arm": arm,
           "similarity": sim, "label": label,
           "q": None if arm == "fallback" else round(bandit.q[arm], 3)}
    if ok:
        send(p, contact, draft)
        append(p.send_log, rec)
        return "sent"
    append(p.queue, {**rec, "reason": reason})
    return "queued"


def cmd_review(p: Paths, contact: str):
    """Walk the review queue; your decision is still reward for the bandit."""
    queue = read_jsonl(p.queue)
    mine = [r for r in queue if r["contact"] == contact]
    rest = [r for r in queue if r["contact"] != contact]
    if not mine:
        print("review queue is empty")
        return
    bandit = Bandit(p.bandit / f"{contact}.json")
    keep = []
    print("[Enter] approve+send | type edit+send | r reject | s skip | q quit\n")
    for i, r in enumerate(mine):
        print(f"them: {r['incoming']}\ndraft ({r['label']}; held: {r['reason']}):"
              f" {r['draft']}")
        a = input("you> ").strip()
        if a.lower() == "q":
            keep += mine[i:]
            break
        if a.lower() == "s":
            keep.append(r)
            continue
        kind = "reject" if a.lower() == "r" else "edit" if a else "approve"
        bandit.update(r["arm"], REWARDS[kind])
        if kind != "reject":
            deliver_to_outbox(p, contact, a or r["draft"])
            append(p.send_log, {**r, "draft": a or r["draft"], "reviewed": True})
    p.queue.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rest + keep),
        encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("enable", "disable"):
        sub.add_parser(name).add_argument("contact")
    for name in ("kill", "unkill", "status"):
        sub.add_parser(name)
    for name in ("run", "review"):
        sp = sub.add_parser(name)
        sp.add_argument("--contact", required=True)
        if name == "run":
            sp.add_argument("--incoming-file")
            sp.add_argument("--hybrid", action="store_true")
    args = ap.parse_args()

    p = Paths()
    cfg = load_config(p)
    if args.cmd in ("enable", "disable"):
        s = set(cfg["contacts"])
        (s.add if args.cmd == "enable" else s.discard)(args.contact)
        cfg["contacts"] = sorted(s)
        save_config(p, cfg)
        print(f"{args.cmd}d auto-send for {args.contact}")
    elif args.cmd == "kill":
        p.data.mkdir(parents=True, exist_ok=True)
        p.kill.write_text("stop\n")
        print("KILL SWITCH ON: nothing will be auto-sent")
    elif args.cmd == "unkill":
        p.kill.unlink(missing_ok=True)
        print("kill switch off")
    elif args.cmd == "status":
        print(f"kill switch : {'ON' if p.kill.exists() else 'off'}")
        print(f"whitelist   : {cfg['contacts'] or 'none (draft-only everywhere)'}")
        print(f"thresholds  : sim>={cfg['min_similarity']} Q>={cfg['min_q']} "
              f"n>={cfg['min_n']} max/day={cfg['max_per_day']} "
              f"quiet={cfg['quiet_hours']}")
        now = now_utc()
        today = {c: sent_today(p, c, now) for c in cfg["contacts"]}
        print(f"sent today  : {today or 'none'} (cap {cfg['max_per_day']}/contact)")
        print(f"sent total  : {len(read_jsonl(p.send_log))}   "
              f"awaiting review: {len(read_jsonl(p.queue))}")
    elif args.cmd == "review":
        cmd_review(p, args.contact)
    elif args.cmd == "run":
        pairs = p.pairs / f"{args.contact}.json"
        if not pairs.exists():
            raise SystemExit(f"no data for {args.contact!r}")
        engine = ReplyEngine(pairs)
        bandit = Bandit(p.bandit / f"{args.contact}.json")
        style = None
        if args.hybrid:
            from generate import StyleModel
            style = StyleModel.for_contact(args.contact)
        src = open(args.incoming_file, encoding="utf-8") if args.incoming_file \
            else sys.stdin
        for line in src:
            line = line.strip()
            if line:
                result = process(p, args.contact, line, engine, bandit, style)
                print(f"[{result}] {line}")


if __name__ == "__main__":
    main()
