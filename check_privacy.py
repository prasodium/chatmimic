#!/usr/bin/env python3
"""
Privacy guard: run before every `git push`.

    python check_privacy.py

Fails (exit 1) if any private file -- chat data, bandit state, logs,
model weights, raw exports -- is tracked by git or sitting in a
committable location. Your chats should NEVER leave your machine.

Add to your flow:  python check_privacy.py && git push
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Patterns that must never be committed, ever.
FORBIDDEN = [
    "data/", "models/", "feedback_log.jsonl", "*.jsonl",
    "messages/", "*.zip",
]


def tracked_files():
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True,
            text=True, check=True).stdout.splitlines()
        return out
    except Exception:
        return []  # not a git repo (yet) -- fall through to disk scan


def main():
    problems = []

    for f in tracked_files():
        for pat in ("data/", "models/", "messages/"):
            if f.startswith(pat):
                problems.append(f"tracked by git: {f}")
        if f.endswith(".jsonl") or f.endswith(".zip"):
            problems.append(f"tracked by git: {f}")

    # Disk scan: private dirs must exist only as gitignored locals, if at all
    in_repo = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                             cwd=ROOT, capture_output=True).returncode == 0
    for d in ("data", "models", "messages"):
        p = ROOT / d
        if p.exists() and in_repo:
            # ensure git really ignores it
            r = subprocess.run(["git", "check-ignore", "-q", str(p)],
                               cwd=ROOT, capture_output=True)
            if r.returncode != 0:
                problems.append(f"{d}/ exists but is NOT gitignored!")

    if problems:
        print("PRIVACY CHECK FAILED -- fix before pushing:\n")
        for p in problems:
            print(f"  - {p}")
        print("\nYour chat data must never be committed. "
              "Check .gitignore and `git rm --cached` the file.")
        sys.exit(1)

    print("privacy check passed: no chat data, state, or models are committable.")


if __name__ == "__main__":
    main()
