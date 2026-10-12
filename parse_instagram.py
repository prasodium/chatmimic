#!/usr/bin/env python3
"""
Parse an Instagram 'Download Your Information' export (JSON format) into
per-contact (incoming -> reply) training pairs.

Usage:
    python parse_instagram.py /path/to/export --me "Your Name" --out data/pairs

Input layout:
    <export>/messages/inbox/<thread>/message_1.json ...

Output:
    data/pairs/<Contact Name>.json  ->  [{"incoming": ..., "reply": ..., "ts": ...}, ...]

Notes:
- Only 1-on-1 threads are used (group chats are skipped in v1).
- Consecutive messages from the same sender are joined with newlines.
- Non-text content (photos, videos, shares, reactions) is skipped.
- Instagram sometimes mojibakes UTF-8 as latin-1; we repair that.
"""

import argparse
import json
import re
import sys
from pathlib import Path


def fix_text(s: str) -> str:
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def thread_files(thread_dir: Path):
    return sorted(
        thread_dir.glob("message_*.json"),
        key=lambda p: int(re.search(r"message_(\d+)", p.name).group(1)),
    )


def parse_thread(thread_dir: Path, me: str | None):
    files = thread_files(thread_dir)
    if not files:
        return None
    first = json.loads(files[0].read_text(encoding="utf-8"))
    participants = [fix_text(p["name"]) for p in first.get("participants", [])]
    if len(participants) != 2:
        return None  # group chat: skip in v1

    if me is None:
        # inbox folders are usually named after the *other* person
        folder = fix_text(thread_dir.name)
        others = [p for p in participants if folder.lower() not in p.lower()]
        if len(others) != 1:
            return None
        me = others[0]
    contact = next(p for p in participants if p != me)

    msgs = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        msgs.extend(data.get("messages", []))
    msgs.sort(key=lambda m: m.get("timestamp_ms", 0))

    pairs = []
    pending, pending_ts = [], None
    mine, mine_ts = [], None

    def flush():
        nonlocal pending, mine, pending_ts, mine_ts
        if pending and mine:
            pairs.append({
                "incoming": "\n".join(pending),
                "reply": "\n".join(mine),
                "ts": mine_ts,
            })
        pending, mine, pending_ts, mine_ts = [], [], None, None

    for m in msgs:
        sender = fix_text(m.get("sender_name", ""))
        text = m.get("content")
        if not text:
            continue  # photo / video / share / reaction: skip in v1
        text = fix_text(text).strip()
        if not text:
            continue
        ts = m.get("timestamp_ms", 0)
        if sender == me:
            if pending:
                mine.append(text)
                mine_ts = mine_ts or ts
            # else: I spoke first with nothing to reply to -> ignore
        else:
            if mine:
                flush()  # they answered my reply: close the previous pair
            pending.append(text)
            pending_ts = pending_ts or ts

    return contact, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir", help="Root of the Instagram JSON export")
    ap.add_argument("--me", default=None,
                    help="Your display name as it appears in the export")
    ap.add_argument("--out", default="data/pairs")
    ap.add_argument("--min-pairs", type=int, default=20,
                    help="Skip contacts with fewer pairs than this")
    args = ap.parse_args()

    inbox = Path(args.export_dir) / "messages" / "inbox"
    if not inbox.is_dir():
        sys.exit(f"not found: {inbox} (expected <export>/messages/inbox)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    total = 0
    for thread_dir in sorted(inbox.iterdir()):
        if not thread_dir.is_dir():
            continue
        res = parse_thread(thread_dir, args.me)
        if not res:
            continue
        contact, pairs = res
        if len(pairs) < args.min_pairs:
            continue
        safe = re.sub(r"[^\w\- ]", "_", contact).strip()
        (out / f"{safe}.json").write_text(
            json.dumps(pairs, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{contact}: {len(pairs)} pairs")
        total += len(pairs)
    print(f"\nwrote {total} pairs total -> {out}")


if __name__ == "__main__":
    main()
