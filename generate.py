#!/usr/bin/env python3
"""
Stage 5 (v2): hybrid inference -- retrieval picks WHAT to say, the LSTM
rephrases HOW you say it.

    python generate.py --contact "Aarav Sharma" --seed "yeah sure"

Rephrasing = seeded generation: keep the first part of the retrieved reply
as a prefix, let your char-LSTM continue it with temperature sampling, and
accept the result only if it still carries the retrieved reply's meaning
(word overlap check). Otherwise the original retrieved reply is returned,
so the LSTM can polish but never derail a draft.
"""

import argparse
import re
from pathlib import Path

import torch

from train_lstm import BOS, EOS, CharLSTM

ROOT = Path(__file__).resolve().parent


class StyleModel:
    def __init__(self, ckpt_path: str | Path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.vocab = ckpt["vocab"]
        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        self.model = CharLSTM(**ckpt["cfg"])
        self.model.load_state_dict(ckpt["state"])
        self.model.eval()

    @classmethod
    def for_contact(cls, contact: str, models_dir: str | Path = ROOT / "models"):
        """Per-contact model, else generic _all, else None."""
        for name in (contact, "_all"):
            p = Path(models_dir) / f"{name}.pt"
            if p.exists():
                return cls(p)
        return None

    @torch.no_grad()
    def generate(self, prefix: str = "", temperature: float = 0.7,
                 max_len: int = 120, rng: torch.Generator | None = None) -> str:
        ids = [self.stoi[BOS]] + [self.stoi[c] for c in prefix if c in self.stoi]
        x = torch.tensor([ids])
        logits, state = self.model(x)
        out = list(prefix)
        for _ in range(max_len):
            probs = torch.softmax(logits[0, -1] / max(temperature, 1e-3), -1)
            nxt = torch.multinomial(probs, 1, generator=rng).item()
            if self.vocab[nxt] == EOS:
                break
            if self.vocab[nxt] != BOS:
                out.append(self.vocab[nxt])
            logits, state = self.model(torch.tensor([[nxt]]), state)
        return "".join(out)

    @torch.no_grad()
    def avg_logprob(self, text: str) -> float:
        """Mean per-char log-prob of `text` under your style (higher = more you)."""
        ids = [self.stoi[BOS]] + [self.stoi[c] for c in text if c in self.stoi] \
            + [self.stoi[EOS]]
        x = torch.tensor([ids])
        logits, _ = self.model(x[:, :-1])
        lp = torch.log_softmax(logits[0], -1)
        return lp[range(len(ids) - 1), x[0, 1:]].mean().item()


def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", s.lower()))


def rephrase(style: StyleModel, retrieved: str, temperature: float = 0.7,
             tries: int = 8, keep_frac: float = 0.4,
             min_overlap: float = 0.5, seed: int | None = None) -> str:
    """Rephrase a retrieved reply in your char-level style, or return it
    unchanged if no sample stays faithful to its meaning."""
    if len(retrieved) < 6:
        return retrieved
    rng = torch.Generator().manual_seed(seed) if seed is not None else None
    prefix = retrieved[:max(3, int(len(retrieved) * keep_frac))]
    want = _words(retrieved)
    best, best_score = None, -1e9
    for _ in range(tries):
        cand = style.generate(prefix, temperature, rng=rng).strip()
        if not cand or cand == retrieved:
            continue
        got = _words(cand)
        if not want or len(want & got) / len(want) < min_overlap:
            continue
        score = style.avg_logprob(cand)
        if score > best_score:
            best, best_score = cand, score
    return best or retrieved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contact", required=True)
    ap.add_argument("--seed", default="", help="prefix to continue, or a "
                    "retrieved reply to rephrase with --rephrase")
    ap.add_argument("--rephrase", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("-n", type=int, default=5)
    args = ap.parse_args()

    style = StyleModel.for_contact(args.contact)
    if style is None:
        raise SystemExit("no model -- run: python train_lstm.py --all")
    for _ in range(args.n):
        print(rephrase(style, args.seed, args.temperature) if args.rephrase
              else style.generate(args.seed, args.temperature))


if __name__ == "__main__":
    main()
