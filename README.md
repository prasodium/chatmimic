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

### System overview

```
 STAGE 0                    STAGE 1                    STAGE 2 + 3 + 4
 Instagram export   ──►   parse_instagram.py    ──►   retrieval (voice)
 (your machine)          (their msg → your            + bandit (judgment)
                          reply pairs)                + draft loop
                                                          │
                        ┌─────────────────────────────────┘
                        ▼
                  ┌─────────────┐
                  │  STAGE 5    │  v2: char-LSTM style model
                  │  (train +   │  (hybrid: retrieval picks WHAT,
                  │   generate) │   LSTM rephrases HOW)
                  └──────┬──────┘
                         ▼
                  ┌─────────────┐
                  │  STAGE 6    │  v3: gated auto-send
                  │  (autosend) │  (whitelist + confidence gate +
                  └─────────────┘   kill switch + review queue)
```

Data never leaves your machine at any stage. Everything below runs
offline after setup.

---

### Stage 0 — Data acquisition

1. Instagram app → profile → **Settings** → **Accounts Centre** →
   **Your information and permissions** → **Download your information**.
2. Choose **JSON** format, all-time date range, and tick **Messages**.
3. Instagram emails you a zip. **Unzip it outside the repo**
   (e.g. `~/Downloads/ig`) — the raw export must never sit inside
   the project folder.

Resulting layout:

```
<export>/
└── messages/
    └── inbox/
        ├── Aarav_Sharma/
        │   ├── message_1.json
        │   └── message_2.json
        └── ...
```

Each `message_N.json` looks like this:

```json
{
  "participants": [{"name": "Aarav Sharma"}, {"name": "Your Name"}],
  "messages": [
    {"sender_name": "Aarav Sharma", "timestamp_ms": 1728212345000,
     "content": "want to grab lunch tomorrow?"},
    {"sender_name": "Your Name", "timestamp_ms": 1728212400000,
     "content": "yeah sure, what time?"}
  ],
  "title": "Aarav Sharma"
}
```

---

### Stage 1 — Parsing: export → per-contact pairs

`parse_instagram.py /path/to/export --me "Your Name" --out data/pairs`

Step by step, per thread folder:

1. Read `message_1.json … message_N.json`, merge, sort by `timestamp_ms`.
2. **1-on-1 filter:** threads whose `participants` list ≠ 2 names are
   skipped (group chats stay out of v1).
3. **Identity:** `--me` tells the parser which sender is you; without
   it, the folder name (usually the *other* person) is used to infer it.
4. **Text repair:** Instagram sometimes stores UTF-8 as latin-1
   (`"cafÃ©"` instead of `"café"`) — every string is repaired.
5. **Text-only filter:** messages with no `content` (photos, videos,
   shares, reactions) are skipped.
6. **Pairing:** walk chronologically; consecutive messages from the same
   sender are joined with newlines. When *they* speak then *you* reply,
   one training pair is emitted:

```
them:  "want to grab lunch tomorrow?"
them:  "there's a new place near campus"
you:   "yeah sure"
you:   "what time?"
        ──►  pair: incoming="want to grab lunch tomorrow?\nthere's a new
              place near campus", reply="yeah sure\nwhat time?"
```

7. **Quality floor:** contacts with fewer than `--min-pairs` (default 20)
   pairs are skipped — too little history to learn from.

Output schema — `data/pairs/<Contact>.json` (gitignored):

```json
[
  {"incoming": "want to grab lunch tomorrow?", "reply": "yeah sure, what time?", "ts": 1728212400000},
  {"incoming": "did you watch the match?",     "reply": "that last over was insane", "ts": 1728216000000}
]
```

---

### Stage 2 — Supervised base: retrieval (your voice)

`retrieval.py` — per-contact TF-IDF index, pure standard library.

**Index build** (once per contact, at startup):

1. Tokenize every past *incoming* message: lowercase, keep `[a-z0-9']`.
2. Term frequency (TF): count of each token ÷ message length.
3. Inverse document frequency (IDF): `log((1 + N) / (1 + df)) + 1`,
   where `N` = number of messages, `df` = messages containing the token.
   Rare, distinctive words ("lunch") weigh more than common ones ("the").
4. Each message becomes a TF×IDF vector, L2-normalized.

**Query** (per incoming message):

1. Tokenize the new message the same way, weight by the same IDFs.
2. Cosine similarity against every indexed message: O(N) dot products —
   trivial for thousands of pairs, no GPU, no embeddings model.
3. Return top-k `(your_reply, similarity, pair_index)`, best first.

Worked example — contact's history contains
`"want to grab lunch tomorrow?" → "yeah sure, what time?"`.
New incoming: `"lunch tomorrow?"`. Shared distinctive tokens
(`lunch`, `tomorrow`) dominate the dot product → similarity ≈ 0.9 →
candidate reply: `"yeah sure, what time?"`.

**Fallback:** if nothing scores above zero (a topic you never discussed),
the engine returns your most common reply to that contact
(e.g. `"haha"`), flagged as the `fallback` arm so the bandit and the
send gate treat it with suspicion.

---

### Stage 3 — Reinforcement: the bandit (your judgment)

`bandit.py` — a contextual bandit. This is genuine reinforcement
learning, the lightweight (bandit, not deep-RL) kind.

**Arms** (the actions it chooses between):

| Arm | Meaning |
|---|---|
| `rank0` / `rank1` / `rank2` | send the 1st / 2nd / 3rd retrieval candidate |
| `fallback` | send your generic most-common reply |

**Selection — epsilon-greedy** (`epsilon = 0.15`):

```
with probability 0.15:  pick a random valid arm      # explore
otherwise:              pick the arm with highest Q  # exploit
```

Exploration never fully stops, so it can discover that your taste
changed.

**Update — incremental mean.** After your feedback gives reward `r`
for arm `a` (seen `n` times before):

```
n(a) ← n(a) + 1
Q(a) ← Q(a) + (r − Q(a)) / n(a)
```

Worked numbers: `Q(rank1) = 0.20` after 4 rounds, you approve (`r = +1.0`)
→ `Q = 0.20 + (1.0 − 0.20)/5 = 0.36`. One approval visibly moves the
needle; dozens of rounds converge it.

**Rewards** (from your decision in the draft loop):

| Your decision | Reward | Meaning |
|---|---|---|
| approve as-is (Enter) | **+1.0** | perfect draft |
| edit then send | **+0.4** | right idea, wrong wording |
| reject (`r`) | **−1.0** | wrong draft |

**State** — `data/bandit/<Contact>.json` (gitignored, survives restarts):

```json
{"q": {"rank0": -0.42, "rank1": 0.81, "rank2": -0.10, "fallback": -1.0},
 "n": {"rank0": 31, "rank1": 58, "rank2": 12, "fallback": 4}}
```

Reading it: *for this contact, the 2nd retrieval candidate is usually
the keeper* — that is the learned, per-person judgment no static model
has.

---

### Stage 4 — Draft loop: where the learning happens

`python agent.py --contact "Aarav Sharma"` — the interactive CLI.

Sequence per message:

```
1. you paste          them> are we still on for lunch?
2. engine suggests    candidates = [("yep, see you at 1", 0.91),
                                    ("yeah sure, what time?", 0.62),
                                    ("haha ok", 0.31)]
3. bandit selects     arm = rank0   (Q(rank0) highest; 15% explore)
4. draft shown        draft [rank0, sim 0.91]: yep, see you at 1
5. you decide         you> [Enter]               → approve
6. reward + update    bandit.update("rank0", +1.0) → Q rises
7. logged             feedback_log.jsonl += {ts, contact, incoming,
                                             arm, draft, decision, reward}
8. repeat             → the NEXT draft for this person is smarter
```

`feedback_log.jsonl` (gitignored) is the audit trail of everything the
bandit ever learned from — one JSON object per line, forever replayable.

Nothing is sent anywhere in v1/v2. The loop is the product: your taste,
distilled per person, one decision at a time.

---

### Stage 5 — Style model (v2): char-level LSTM

`train_lstm.py` — trains **only on your replies**, from scratch.

**Data prep:** your replies from `data/pairs/` → one long char stream,
each message wrapped in `BOS … EOS` markers (`\x02`, `\x03`) so the
model learns where messages start and end. Contacts with thin history
fall back to a generic `_all` model trained on your replies everywhere.

**Architecture** (tiny on purpose):

```
chars (~100 vocab) → embedding(64) → LSTM(256) ×2 → linear → softmax
≈ 1.5M parameters ≈ a few MB on disk
```

**Training:** next-character prediction (supervised cross-entropy),
Adam optimizer, minibatches of message chunks, ~10–30 minutes on CPU.
No tokenizer — slang, typos, and emoji habits are just characters to it.

**Hybrid inference** (`generate.py`) — retrieval picks **WHAT** to say,
the LSTM rephrases **HOW** you say it:

```
1. retrieval → top candidate reply, e.g. "yeah sure, what time?"
2. keep its first words as a seed prefix: "yeah sure,"
3. LSTM continues the seed with temperature sampling → "yeah sure, what time works?"
4. meaning check: word overlap between candidate and generation ≥ threshold?
   yes → use the rephrased version (your content, your style)
   no  → keep the original candidate (the LSTM may polish, never derail)
```

Weights → `models/<Contact>.pt` (gitignored, never published).
`agent.py --hybrid` wires it into the draft loop; the bandit's arms and
rewards work unchanged — the RL layer doesn't care where the wording
came from.

---

### Stage 6 — Gated auto-send (v3)

`autosend.py` — default is **draft-only**. Nothing is automatic until you
opt a contact in: `autosend.py enable "Aarav Sharma"`.

**The send gate** — a message is auto-sent only if ALL hold:

```
contact on whitelist?
  └─ no → draft mode (as before)
kill switch off?  (`autosend.py kill` stops EVERYTHING instantly)
outside quiet hours?
under the per-day rate limit for this contact?
chosen arm is rank0/1/2 (never the generic fallback)?
retrieval similarity ≥ min_similarity?
bandit Q(arm) ≥ min_q  AND  n(arm) ≥ min_n feedback rounds?
  └─ ALL YES → send (epsilon forced to 0: no exploration on auto-sends)
  └─ ANY NO  → data/review_queue.jsonl  (human review, never sent)
```

**Mechanics:**

- `autosend.py status` — whitelist, kill-switch state, today's send counts.
- `autosend.py review --contact …` — walk the review queue, approve/edit/
  reject each item (every decision still feeds the bandit).
- Every sent message is appended to a local send log: timestamp, contact,
  incoming, sent text, arm, similarity, Q — full accountability.
- Thresholds (`min_similarity`, `min_q`, `min_n`, quiet hours, daily cap)
  live in one config block at the top of `autosend.py` — tune them as the
  bandit matures. Start strict; loosen only with evidence.

---

### End-to-end worked example

One message traveling the whole system, contact "Aarav Sharma":

```
① incoming:            "are we still on for lunch tomorrow?"

② retrieval:           top candidates —
   rank0 (0.93): "yep, see you at 1"
   rank1 (0.71): "yeah sure, what time?"
   rank2 (0.44): "haha ok"

③ bandit:              Q = {rank0: 0.12, rank1: 0.81, rank2: -0.2}
   epsilon roll → exploit → arm = rank1
   (you've taught it: for Aarav, the 2nd candidate lands better)

④ v2 hybrid (optional): seed "yeah sure," → LSTM → "yeah sure, what time works?"
   meaning check passes → draft = "yeah sure, what time works?"

⑤ draft loop:          draft [rank1, sim 0.71]: yeah sure, what time works?
   you> [Enter] → approve → reward +1.0 → Q(rank1) rises again

⑥ v3 gate (if enabled): whitelisted? yes · sim 0.71 ≥ 0.6? yes ·
   Q(rank1)=0.83 ≥ 0.5? yes · n=59 ≥ 20? yes → auto-send.
   Otherwise → review queue, you decide.
```

That is the whole machine: your history teaches it your voice,
your feedback teaches it your judgment, and the gate decides when it's
earned the right to speak without you.

---

## The plan

| Version | Goal | Deliverables | Status |
|---|---|---|---|
| **v1** | Prove the loop: your chats → drafts that sound like you → RL from your feedback | `parse_instagram.py`, `retrieval.py`, `bandit.py`, `agent.py`, `simulate.py`, `check_privacy.py` | ✅ done & verified |
| **v2** | Add the LSTM style layer; hybrid retrieval+LSTM generation | `train_lstm.py`, `generate.py`, per-contact `models/*.pt` | ✅ built (`agent.py --hybrid`) |
| **v3** | Careful auto-send for whitelisted contacts | `autosend.py`: send gate, whitelist, kill switch, send log, review queue | ✅ built |

`simulate.py` is the standing regression test: synthetic corpus →
retrieval sanity checks → 300 simulated feedback rounds proving the
bandit converges on the user's taste → send-gate checks → LSTM training
(skipped if torch isn't installed). Run it after any change:
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
the box they reply like a generic assistant — not like you. Rejected
outright: no pretrained weights in this project, period.

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
6. v2 LSTM: `pip install torch` (CPU build is fine), then:
   ```bash
   python train_lstm.py --all            # per-contact models + generic _all
   python agent.py --contact "Aarav Sharma" --hybrid
   python generate.py --contact "Aarav Sharma" --seed "yeah" -n 5
   ```
   Weights are written to `models/` (gitignored). The one network call in
   the whole project is the `pip install`.
7. v3 gated auto-send (off by default, per-contact opt-in):
   ```bash
   python autosend.py enable "Aarav Sharma"
   python autosend.py status
   python autosend.py run --contact "Aarav Sharma" --incoming-file new.txt
   python autosend.py review --contact "Aarav Sharma"   # held drafts
   python autosend.py kill                              # stop everything
   ```
   Confident drafts go to `data/outbox.jsonl`; everything else waits in
   `data/review_queue.jsonl`. Instagram has no official personal-DM API,
   so nothing is delivered over the network: the outbox is the hand-off
   to whatever local tool you choose to send with.

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
├── agent.py              ← Stage 4: draft-approval CLI loop (--hybrid for v2)
├── train_lstm.py         ← Stage 5 (v2): char-LSTM trainer
├── generate.py           ← Stage 5 (v2): hybrid retrieval + LSTM rephrase
├── autosend.py           ← Stage 6 (v3): gated auto-send, kill switch, review
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
