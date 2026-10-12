#!/usr/bin/env python3
"""
Stage 5 (v2): train a tiny char-level LSTM on YOUR replies, per contact.

    python train_lstm.py --contact "Aarav Sharma"
    python train_lstm.py --all            # every contact + generic "_all" model

Architecture: char-embed(64) -> 2x LSTM(256) -> linear -> softmax over the
char vocab (~100). Next-char prediction on your own messages only, from
scratch -- no pretrained weights. CPU-only. Weights go to models/ (gitignored).

Contacts with too little history fall back to the generic "_all" model,
which is trained on your replies across every contact.
"""

import argparse
import json
import random
import re
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
BOS, EOS = "\x02", "\x03"  # message start / end markers
WORD_RE = re.compile(r"[a-z0-9']+")


class CharLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed: int = 64, hidden: int = 256,
                 layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.cfg = dict(vocab_size=vocab_size, embed=embed, hidden=hidden,
                        layers=layers, dropout=dropout)
        self.embed = nn.Embedding(vocab_size, embed)
        self.lstm = nn.LSTM(embed, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, x, state=None):
        h, state = self.lstm(self.embed(x), state)
        return self.out(self.drop(h)), state


def load_replies(pairs_path: Path) -> list[str]:
    pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    return [p["reply"] for p in pairs if p.get("reply")]


def build_vocab(texts: list[str]) -> list[str]:
    return [BOS, EOS] + sorted({c for t in texts for c in t})


def encode(text: str, stoi: dict) -> list[int]:
    return [stoi[BOS]] + [stoi[c] for c in text if c in stoi] + [stoi[EOS]]


def make_batches(seqs, batch_size, rng):
    seqs = seqs[:]
    rng.shuffle(seqs)
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        L = max(len(s) for s in chunk)
        x = torch.full((len(chunk), L - 1), -1, dtype=torch.long)
        y = torch.full((len(chunk), L - 1), -1, dtype=torch.long)
        for j, s in enumerate(chunk):
            t = torch.tensor(s)
            x[j, :len(s) - 1] = t[:-1]
            y[j, :len(s) - 1] = t[1:]
        yield x.clamp(min=0), y  # -1 targets are ignored by the loss


def train_model(texts: list[str], epochs: int = 30, batch_size: int = 32,
                lr: float = 3e-3, max_len: int = 200, seed: int = 0,
                verbose: bool = True):
    """Train on a list of your messages. Returns a checkpoint dict."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    texts = [t[:max_len] for t in texts]
    vocab = build_vocab(texts)
    stoi = {c: i for i, c in enumerate(vocab)}
    seqs = [encode(t, stoi) for t in texts]
    split = max(1, int(len(seqs) * 0.9))
    rng.shuffle(seqs)
    train, val = seqs[:split], seqs[split:] or seqs[:1]

    model = CharLSTM(len(vocab))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss(ignore_index=-1)
    best, best_state = float("inf"), None

    def run(data, training):
        model.train(training)
        tot, n = 0.0, 0
        with torch.set_grad_enabled(training):
            for x, y in make_batches(data, batch_size, rng):
                logits, _ = model(x)
                loss = lossf(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                if training:
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                tot += loss.item()
                n += 1
        return tot / max(n, 1)

    for ep in range(1, epochs + 1):
        tr, va = run(train, True), run(val, False)
        if va < best:
            best = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 5 == 0 or ep == 1):
            print(f"  epoch {ep:3d}  train {tr:.3f}  val {va:.3f}")
    model.load_state_dict(best_state)
    return {"cfg": model.cfg, "vocab": vocab, "state": model.state_dict(),
            "val_loss": best, "n_msgs": len(texts),
            "words": sorted({w for x in texts for w in WORD_RE.findall(x.lower())})}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--contact")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--pairs-dir", default=str(ROOT / "data" / "pairs"))
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--min-msgs", type=int, default=200,
                    help="skip per-contact training below this many replies")
    args = ap.parse_args()

    pairs_dir, models_dir = Path(args.pairs_dir), Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    files = ([pairs_dir / f"{args.contact}.json"] if args.contact
             else sorted(p for p in pairs_dir.glob("*.json")
                         if not p.stem.startswith("_")))
    per_contact = {f.stem: load_replies(f) for f in files if f.exists()}
    if not per_contact:
        raise SystemExit("no pairs found -- run parse_instagram.py first")

    jobs = dict(per_contact)
    if args.all:
        jobs["_all"] = [t for ts in per_contact.values() for t in ts]
    for name, texts in jobs.items():
        if name != "_all" and len(texts) < args.min_msgs:
            print(f"{name}: only {len(texts)} replies (< {args.min_msgs}); "
                  "will use the generic _all model")
            continue
        print(f"training {name} on {len(texts)} of your replies")
        ckpt = train_model(texts, epochs=args.epochs)
        torch.save(ckpt, models_dir / f"{name}.pt")
        print(f"  saved models/{name}.pt (val loss {ckpt['val_loss']:.3f})")


if __name__ == "__main__":
    main()
