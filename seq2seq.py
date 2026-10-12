#!/usr/bin/env python3
"""
True generation: a conditional seq2seq model that READS the incoming
message and WRITES a brand-new reply in your style.

Unlike retrieval (copies your old replies) or the char-LSTM (an
*unconditional* style model -- it can't see the incoming message),
this is an encoder-decoder: the encoder reads THEIR message, the
decoder writes YOUR reply, word by word. Every weight is trained from
scratch on your (incoming -> reply) pairs. No pretrained anything.

    python seq2seq.py train --all                    # global + per-contact
    python seq2seq.py train --contact "Aarav Sharma"
    python seq2seq.py gen --contact "Aarav Sharma" --incoming "are we still on?"

Architecture (word-level, CPU-friendly):
    vocab: ~8k most common words + <pad>/<unk>/<bos>/<eos>
    encoder: embedding(128) -> LSTM(256) x2  -> context (h, c)
    decoder: embedding(128) -> LSTM(256) x2 (seeded with context)
             -> linear -> softmax over vocab        (~4M params)
    training: teacher forcing, cross-entropy, Adam. CPU-trainable.

Wiring: agent.py --gen adds the generated reply as a "gen" bandit arm,
so the RL loop learns per situation whether YOU prefer generated or
retrieved replies. autosend.py never auto-sends "gen" on similarity
(there is none) -- it gates on bandit Q instead.

Needs: pip install torch --cpu   (the only non-stdlib dependency)
Checkpoints -> models/seq2seq_<contact>.pt (gitignored, never published).

Honest note: trained on a few thousand casual chat pairs, this is
fluent and stylistically yours, but often generic ("haha yeah") on
unfamiliar inputs. Novelty: yes. Deep understanding: no -- that ceiling
is data, not architecture (see README "Honest limits").
"""

import argparse
import json
import random
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

WORD_RE = re.compile(r"[a-z0-9']+")
PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIALS = [PAD, UNK, BOS, EOS]

MAX_VOCAB = 8000
MAX_IN = 40
MAX_OUT = 40
MIN_PAIRS_PER_CONTACT = 500

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------- data prep
# (pure Python -- testable without torch)

def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def build_vocab(pairs: list[dict], max_vocab: int = MAX_VOCAB) -> list[str]:
    from collections import Counter
    counts: Counter = Counter()
    for p in pairs:
        counts.update(tokenize(p.get("incoming", "")))
        counts.update(tokenize(p.get("reply", "")))
    words = [w for w, _ in counts.most_common(max_vocab - len(SPECIALS))]
    return SPECIALS + words


def encode_pair(pair: dict, stoi: dict,
                max_in: int = MAX_IN, max_out: int = MAX_OUT):
    """-> (src_ids, tgt_ids) as plain int lists (no torch needed)."""
    unk = stoi[UNK]
    src = [stoi.get(w, unk) for w in tokenize(pair.get("incoming", ""))[:max_in]]
    body = [stoi.get(w, unk) for w in tokenize(pair.get("reply", ""))[:max_out - 2]]
    tgt = [stoi[BOS]] + body + [stoi[EOS]]
    return src or [unk], tgt


def pad_batch(seqs: list[list[int]], pad_id: int) -> list[list[int]]:
    n = max(len(s) for s in seqs)
    return [s + [pad_id] * (n - len(s)) for s in seqs]


def make_batches(encoded: list[tuple[list[int], list[int]]],
                 batch_size: int, rng: random.Random, pad_id: int):
    """Yield (src, tgt) as padded int lists. Torch conversion happens in
    the training loop, so batching itself is fully testable."""
    order = list(range(len(encoded)))
    rng.shuffle(order)
    for i in range(0, len(order), batch_size):
        chunk = [encoded[j] for j in order[i:i + batch_size]]
        yield (pad_batch([s for s, _ in chunk], pad_id),
               pad_batch([t for _, t in chunk], pad_id))


def detokenize(ids: list[int], vocab: list[str]) -> str:
    words = [vocab[i] for i in ids
             if vocab[i] not in (PAD, BOS, EOS)]
    text = " ".join(w for w in words if w != UNK)
    for a, b in ((" n't", "n't"), (" 's", "'s"), (" 're", "'re"),
                 (" 'm", "'m"), (" 'll", "'ll"), (" 'd", "'d")):
        text = text.replace(a, b)
    return text.strip()


# ---------------------------------------------------------------- model

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size: int, embed: int = 128, hidden: int = 256,
                 layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.cfg = dict(vocab_size=vocab_size, embed=embed, hidden=hidden,
                        layers=layers, dropout=dropout)
        self.enc_emb = nn.Embedding(vocab_size, embed)
        self.encoder = nn.LSTM(embed, hidden, layers, batch_first=True)
        self.dec_emb = nn.Embedding(vocab_size, embed)
        self.decoder = nn.LSTM(embed, hidden, layers, batch_first=True,
                               dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, src, tgt_in):
        """src: (B, S), tgt_in: (B, T) -> logits (B, T, V)."""
        _, (h, c) = self.encoder(self.enc_emb(src))
        h2, _ = self.decoder(self.dec_emb(tgt_in), (h, c))
        return self.out(self.drop(h2))


def train_model(pairs: list[dict], epochs: int = 25, batch_size: int = 32,
                lr: float = 1e-3, seed: int = 0, verbose: bool = True):
    """Train on (incoming -> reply) pairs. Returns a checkpoint dict."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    vocab = build_vocab(pairs)
    stoi = {w: i for i, w in enumerate(vocab)}
    pad_id = stoi[PAD]
    encoded = [encode_pair(p, stoi) for p in pairs]

    model = Seq2Seq(len(vocab))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        total, nb = 0.0, 0
        for src_b, tgt_b in make_batches(encoded, batch_size, rng, pad_id):
            src = torch.tensor(src_b)
            tgt = torch.tensor(tgt_b)
            logits = model(src, tgt[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, len(vocab)),
                                   tgt[:, 1:].reshape(-1),
                                   ignore_index=pad_id)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            nb += 1
        if verbose:
            print(f"epoch {ep + 1}/{epochs}  loss {total / max(nb, 1):.4f}")
    return {"vocab": vocab, "cfg": model.cfg,
            "state": model.state_dict()}


def save_checkpoint(ckpt: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    print(f"saved {path}")


# ---------------------------------------------------------------- inference

class Seq2SeqModel:
    """Loads a trained checkpoint; generates new replies."""

    def __init__(self, ckpt_path: str | Path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.vocab: list[str] = ckpt["vocab"]
        self.stoi = {w: i for i, w in enumerate(self.vocab)}
        self.model = Seq2Seq(**ckpt["cfg"])
        self.model.load_state_dict(ckpt["state"])
        self.model.eval()

    @classmethod
    def for_contact(cls, contact: str,
                    models_dir: str | Path = ROOT / "models"):
        """Per-contact model, else global _all, else None."""
        for name in (f"seq2seq_{contact}", "seq2seq__all"):
            p = Path(models_dir) / f"{name}.pt"
            if p.exists():
                return cls(p)
        return None

    @torch.no_grad()
    def generate(self, incoming: str, temperature: float = 0.8,
                 max_len: int = MAX_OUT,
                 rng: torch.Generator | None = None) -> str:
        src = encode_pair({"incoming": incoming, "reply": ""},
                          self.stoi)[0]
        src_t = torch.tensor([src])
        _, (h, c) = self.model.encoder(self.model.enc_emb(src_t))
        tok = torch.tensor([[self.stoi[BOS]]])
        out_ids: list[int] = []
        for _ in range(max_len):
            h2, (h, c) = self.model.decoder(self.model.dec_emb(tok), (h, c))
            logits = self.model.out(h2[:, -1, :])
            probs = torch.softmax(logits / max(temperature, 1e-3), dim=-1)
            tok = torch.multinomial(probs, 1, generator=rng)
            i = tok.item()
            if self.vocab[i] == EOS:
                break
            out_ids.append(i)
        text = detokenize(out_ids, self.vocab)
        return text or "..."


# ---------------------------------------------------------------- CLI

def load_pairs(pairs_dir: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for f in pairs_dir.glob("*.json"):
        if f.stem.startswith("_"):
            continue
        out[f.stem] = json.loads(f.read_text(encoding="utf-8"))
    return out


def main():
    ap = argparse.ArgumentParser(description="Conditional seq2seq: true generation.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--contact", default=None)
    tr.add_argument("--all", action="store_true",
                    help="train global _all + every contact with enough pairs")
    tr.add_argument("--epochs", type=int, default=25)
    tr.add_argument("--pairs-dir", default=str(ROOT / "data" / "pairs"))
    tr.add_argument("--models-dir", default=str(ROOT / "models"))

    gn = sub.add_parser("gen")
    gn.add_argument("--contact", required=True)
    gn.add_argument("--incoming", required=True)
    gn.add_argument("--temperature", type=float, default=0.8)

    args = ap.parse_args()
    pairs_dir = Path(args.pairs_dir)

    if args.cmd == "train":
        by_contact = load_pairs(pairs_dir)
        targets: dict[str, list[dict]] = {}
        if args.all:
            everything = [p for ps in by_contact.values() for p in ps]
            targets["_all"] = everything
            for contact, ps in by_contact.items():
                if len(ps) >= MIN_PAIRS_PER_CONTACT:
                    targets[contact] = ps
                else:
                    print(f"skip {contact}: only {len(ps)} pairs "
                          f"(< {MIN_PAIRS_PER_CONTACT}), _all covers it")
        elif args.contact:
            f = pairs_dir / f"{args.contact}.json"
            if not f.exists():
                raise SystemExit(f"no data for {args.contact!r}")
            targets[args.contact] = json.loads(f.read_text(encoding="utf-8"))
        else:
            raise SystemExit("pass --contact NAME or --all")
        for name, ps in targets.items():
            print(f"\ntraining seq2seq_{name} on {len(ps)} pairs…")
            ckpt = train_model(ps, epochs=args.epochs)
            save_checkpoint(ckpt, Path(args.models_dir) / f"seq2seq_{name}.pt")
    else:
        model = Seq2SeqModel.for_contact(args.contact)
        if model is None:
            raise SystemExit("no seq2seq model -- run: python seq2seq.py train --all")
        print(model.generate(args.incoming, temperature=args.temperature))


if __name__ == "__main__":
    main()
