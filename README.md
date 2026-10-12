# chatmimic

A lightweight, fully self-trained Instagram reply bot that learns **your**
chatting pattern and drafts replies **in your voice, to specific people** —
then keeps learning from your feedback.

No API keys. No pretrained models. No dependencies beyond the Python
standard library (v1). Your chats never leave your machine.

---

## Ground rules

Everything in this repo follows the owner's standing requirements.

### Always do

- **Train on your own chats only.** The model learns your pattern from
  your Instagram message history — nothing else.
- **Stay lightweight.** CPU-only, tiny models, minimal dependencies.
  If a component needs a GPU farm, it doesn't belong here.
- **Use BOTH supervised learning and reinforcement learning.**
  Supervised learning gives it your voice (trained on your past chats);
  reinforcement learning gives it your judgment (it keeps learning from
  every approve / edit / reject, per person, forever).
- **Reply per person.** Models and feedback state are tracked per
  contact — how you talk to one friend is not how you talk to another.
- **Keep code public, data private.** The repo is public code; your chat
  data, trained state, and model weights never get committed
  (see [Privacy architecture](#privacy-architecture)).
- **Train offline.** Everything runs on your own machine (VS Code
  terminal is fine) — no server, no API calls, no network needed.

### Never do

- **Never use API keys** (no OpenAI/Anthropic/Gemini calls anywhere).
- **Never use pretrained models** — no Qwen, no Hugging Face weights,
  no borrowed checkpoints. Every weight is learned from your chats.
- **Never publish your chat data** — not the raw export, not the parsed
  pairs, not the feedback log.
- **Never publish your trained model** — weights and bandit state stay
  local, always.
- **Never auto-send blindly.** v1 is draft-only. Auto-send (v3) is
  per-contact opt-in, gated by confidence, with a kill switch and a full
  send log.
- **Never reach for heavy architectures** to solve what a small one
  solves: no transformers for training from scratch on small chat data,
  no vanilla RNNs (they forget), no Markov chains (they garble).

---

## The full pipeline

```
 ┌─ STAGE 0 ──────────────────────────────────────────────┐
 │ Instagram → Download Your Information (JSON) → unzip   │
 │ LOCALLY. Nothing leaves your machine. Ever.            │
 └────────────────────────┬───────────────────────────────┘
                          │  parse_instagram.py
 ┌─ STAGE 1: PARSE ────────▼───────────────────────────────┐
 │ messages/inbox/<thread>/message_*.json                  │
 │  • keep 1-on-1 threads only (groups skipped in v1)      │
 │  • chronological sort, UTF-8 mojibake repair            │
 │  • text messages only (photos/shares/reactions skipped) │
 │  • pair: their consecutive msgs → your consecutive reply│
 └────────────────────────┬───────────────────────────────┘
                          ▼
              data/pairs/<Contact>.json        ← gitignored
              [{"incoming": ..., "reply": ..., "ts": ...}]
                          │
        ┌─────────────────┴──────────────────┐
        │  STAGE 2: SUPERVISED (your voice)   │
        │  retrieval.py — per-contact TF-IDF  │
        │  index over THEIR past messages.    │
        │  New incoming → most similar past   │
        │  incomings → YOUR historical replies│
        │  as candidates (top-k + fallback).  │
        │  Zero dependencies, pure stdlib.    │
        └─────────────────┬──────────────────┘
                          │ candidates
        ┌─────────────────▼──────────────────┐
        │  STAGE 3: REINFORCEMENT (judgment)  │
        │  bandit.py — contextual bandit.     │
        │  Arms: rank0 / rank1 / rank2 /       │
        │  fallback. Epsilon-greedy pick,     │
        │  incremental Q-updates.             │
        │  State: data/bandit/<Contact>.json  │
        └─────────────────┬──────────────────┘
                          │ draft
 ┌─ STAGE 4: DRAFT LOOP ──▼───────────────────────────────┐
 │ agent.py — interactive CLI:                            │
 │   them> <paste incoming message>                       │
 │   draft [rank1, sim 0.87]: yeah sure, what time?        │
 │   you>  <Enter>=approve  <type edit>  r=reject  q=quit │
 │ Reward: approve +1.0 · edit +0.4 · reject −1.0         │
 │ Every decision → bandit.update + feedback_log.jsonl    │
 │ → the NEXT draft for this person is smarter.           │
 └────────────────────────────────────────────────────────┘
```

**v2 adds:**

```
 ┌─ STAGE 5: STYLE MODEL (v2) ────────────────────────────┐
 │ train_lstm.py — char-level LSTM, PyTorch, CPU-only.    │
 │  • vocab ≈ 100 chars → tiny (1–2M params, a few MB)    │
 │  • arch: char-embed(64) → 2×LSTM(256) → linear →       │
 │    softmax. Next-char prediction on YOUR messages.     │
 │  • trains in minutes on CPU; no tokenizer needed —     │
 │    slang, typos and emoji habits come free.            │
 │  • (GRU is an accepted leaner swap: fewer params,      │
 │    similar quality.)                                   │
 │ HYBRID INFERENCE: retrieval picks WHAT to say,         │
 │ the LSTM rephrases HOW you say it (seeded generation,  │
 │ temperature sampling).                                 │
 │ Weights → models/  ← gitignored, never published.      │
 └────────────────────────────────────────────────────────┘
```

**v3 adds:**

```
 ┌─ STAGE 6: GATED AUTO-SEND (v3) ────────────────────────┐
 │  • per-contact opt-in whitelist (default: draft-only)  │
 │  • send gate: bandit Q + retrieval similarity must     │
 │    clear a confidence threshold, else back to drafts   │
 │  • kill-switch file, quiet hours, per-day rate limit   │
 │  • every sent message appended to a local send log     │
 │  • low-confidence → human review queue, never auto     │
 └────────────────────────────────────────────────────────┘
```

**v4 (optional, breaks a ground rule — clearly marked):**

```
 ┌─ STAGE 7 (OPTIONAL): Qwen + LoRA ─────────────────────┐
 │ Only if v2's understanding ever feels too shallow.    │
 │ Small Qwen + LoRA fine-tune on your pairs. Needs GPU, │
 │ starts from PRETRAINED weights — kept out of the core │
 │ plan on purpose.                                       │
 └────────────────────────────────────────────────────────┘
```

---

## The plan

| Version | Goal | Deliverables | Status |
|---|---|---|---|
| **v1** | Prove the loop: your chats → drafts that sound like you → RL from your feedback | `parse_instagram.py`, `retrieval.py`, `bandit.py`, `agent.py`, `simulate.py`, `check_privacy.py` | ✅ done & verified |
| **v2** | Add the LSTM style layer; hybrid retrieval+LSTM generation | `train_lstm.py`, `generate.py`, per-contact `models/*.pt` | planned |
| **v3** | Careful auto-send for whitelisted contacts | send gate, whitelist, kill switch, send log, review queue | planned |
| **v4** | (Optional) Qwen+LoRA if v2 understanding plateaus | LoRA fine-tune script | only if needed |

`simulate.py` is the standing regression test: synthetic corpus →
retrieval sanity checks → 300 simulated feedback rounds proving the
bandit converges on the user's taste. Run it after any change:
`python simulate.py`.

---

## Design decisions

**Why both supervised AND reinforcement learning?**
Supervised learning (your history → your voice) is the engine, but it's
static — once trained it never improves. The bandit (your live
approve/edit/reject → reward signal) is what makes it *keep* learning
which replies land with which person. One gives it something good to
say; the other teaches it your taste over time.

**Why LSTM and not something else?**
- *Vanilla RNN* — forgets across a message (vanishing gradients); can't
  hold your sentence rhythm and sign-offs.
- *Transformer* — needs far more data and compute to train from scratch;
  on small per-person chat data it performs worse, and it breaks the
  CPU-only constraint.
- *Markov chains / n-grams* — lighter, but generate garbled text; they
  count sequences, they don't learn style.
- *LSTM* — char-level means a ~100-char vocab and a few-MB model that
  trains on CPU in minutes; its gating remembers style across a whole
  message; slang/typos/emoji need no tokenizer. (GRU is an accepted
  leaner drop-in.)

**Why not Qwen / Hugging Face models?**
Three violations of the ground rules: they're pretrained (your rule is
own-weights-only), they're heavy (~1GB+, GPU to fine-tune), and out of
the box they reply like a generic assistant — not like you. Kept as an
explicitly-marked optional v4, not the core plan.

---

## Privacy architecture

This repo is public **code** — your chats never leave your machine.

- `data/` (chat pairs, bandit state), `models/` (v2 weights),
  `feedback_log.jsonl`, and raw Instagram exports are all in
  `.gitignore` and can never be committed.
- Before every push, run the guard:
  ```bash
  python check_privacy.py && git push
  ```
  It fails loudly if any private file is tracked or un-ignored.

---

## Train offline in VS Code (your machine only)

Everything runs locally — no server, no API, no network needed after setup.

1. Clone this repo on your laptop and open the folder in VS Code.
2. Instagram → Settings → Download Your Information → **JSON** format,
   include Messages. Unzip it **outside** the repo (e.g. `~/Downloads/ig`).
3. In VS Code's terminal (`` Ctrl+` ``):
   ```bash
   python parse_instagram.py ~/Downloads/ig --me "Your Name"
   ```
   Pairs land in `data/pairs/` — gitignored, invisible to git.
4. Draft loop (still fully offline):
   ```bash
   python agent.py --contact "Aarav Sharma"
   ```
   Paste an incoming message, then **Enter** = approve as-is, type your
   edit, **r** = reject, **q** = quit. Watch the bandit Q-values learn.
   Every decision is logged to `feedback_log.jsonl`; bandit state lives
   in `data/bandit/<contact>.json` and survives restarts.
5. `git status` will show only code changes. Your data simply doesn't
   exist as far as git is concerned.
6. v2 LSTM: `pip install torch --cpu`, then the training script reads
   from `data/` and writes weights to `models/` (also gitignored). The
   one network call in the whole project is the `pip install`.

---

## Try it in 2 minutes (no Instagram data needed)

```bash
python simulate.py
```

Builds a fake corpus, checks retrieval, then simulates 300 rounds
against a picky user and shows the bandit converging on their taste.

---

## Repo layout

```
chatmimic/
├── README.md            ← you are here (full plan + pipeline)
├── LICENSE              ← MIT
├── .gitignore            ← data/, models/, logs can never be committed
├── requirements.txt      ← v1: stdlib only, nothing to install
├── parse_instagram.py    ← Stage 1: export → per-contact pairs
├── retrieval.py          ← Stage 2: TF-IDF reply suggester (supervised)
├── bandit.py             ← Stage 3: epsilon-greedy bandit (reinforcement)
├── agent.py              ← Stage 4: draft-approval CLI loop
├── check_privacy.py      ← pre-push privacy guard
├── simulate.py           ← end-to-end verification on synthetic data
├── data/                 ← YOURS ONLY, gitignored (pairs, bandit state)
└── models/               ← YOURS ONLY, gitignored (v2 LSTM weights)
```

---

## Data requirements

| Stage | Minimum useful history per contact |
|---|---|
| v1 retrieval | a few hundred messages (falls back to your generic style below that) |
| v1 bandit | learns from your very first approve/edit — useful within dozens of rounds |
| v2 LSTM | a few thousand of your messages (trains in minutes on CPU) |

---

## Honest limits

- Retrieval mimics *what you said before*; it doesn't understand new
  topics. Great for casual chat, not for serious or emotional
  conversations — keep those supervised.
- The LSTM learns *style*, not deep meaning. That's why it's paired
  with retrieval (retrieval = what to say, LSTM = how you say it).
- Needs history: low-data contacts fall back to your generic style model.
- v1 is draft-only by design: nothing is ever sent anywhere.
