# chatmimic — v1

A lightweight, fully self-trained Instagram reply bot. No API keys, no
pretrained models, no dependencies beyond the Python standard library.

**Design (the "both" we agreed on):**

- **Supervised part (your voice):** your exported Instagram chats become
  `(their message → your reply)` pairs per contact. A TF-IDF retriever
  finds your most similar past replies to any new incoming message.
- **Reinforcement part (your judgment):** a contextual bandit picks which
  candidate to draft. Your approve / edit / reject in the draft loop is
  the reward signal (+1.0 / +0.4 / −1.0), so it keeps learning per
  contact, forever. Epsilon-greedy, incremental Q-updates, JSON state.

## Try it in 2 minutes (no Instagram data needed)

```bash
cd ~/workspace/chatmimic
python simulate.py
```

This builds a fake corpus, checks retrieval, then simulates 300 rounds
against a picky user and shows the bandit converging on their taste.

## Use your real chats

1. Instagram → Settings → Download Your Information → request **JSON**
   format, include Messages. Unzip it.
2. Parse it into per-contact pairs:
   ```bash
   python parse_instagram.py /path/to/export --me "Your Name"
   ```
   (Skips group chats and contacts with < 20 pairs in v1.)
3. Run the draft loop:
   ```bash
   python agent.py --contact "Aarav Sharma"
   ```
   Paste an incoming message, then **Enter** = approve as-is, type your
   edit, **r** = reject, **q** = quit. Watch the bandit Q-values learn.
   Every decision is logged to `feedback_log.jsonl`; bandit state lives
   in `data/bandit/<contact>.json` and survives restarts.

## Files

| file | what it does |
|---|---|
| `parse_instagram.py` | export → per-contact `(incoming, reply)` pairs |
| `retrieval.py` | dependency-free TF-IDF reply suggester |
| `bandit.py` | epsilon-greedy contextual bandit (the RL layer) |
| `agent.py` | interactive draft-approval loop |
| `simulate.py` | synthetic end-to-end verification |

## Public repo, private data

This repo is public **code** — your chats never leave your machine.

- `data/` (chat pairs, bandit state), `models/` (v2 weights),
  `feedback_log.jsonl`, and raw Instagram exports are all in
  `.gitignore` and can never be committed.
- Before every push, run the guard:
  ```bash
  python check_privacy.py && git push
  ```
  It fails loudly if any private file is tracked or un-ignored.

## Train offline in VS Code (your machine only)

Everything runs locally — no server, no API, no network needed after setup.

1. Clone this repo on your laptop and open the folder in VS Code.
2. Instagram → Settings → Download Your Information → **JSON** format,
   include Messages. Unzip it **outside** the repo (e.g. `~/Downloads/ig`).
3. In VS Code's terminal (`Ctrl+`` `):
   ```bash
   python parse_instagram.py ~/Downloads/ig --me "Your Name"
   ```
   Pairs land in `data/pairs/` — gitignored, invisible to git.
4. Draft loop (still fully offline):
   ```bash
   python agent.py --contact "Aarav Sharma"
   ```
5. `git status` will show only code changes. Your data simply doesn't exist
   as far as git is concerned.
6. v2 LSTM: `pip install torch --cpu`, then the training script reads from
   `data/` and writes weights to `models/` (also gitignored). The one
   network call in the whole project is the `pip install`.

## Honest limits (v1)

- Retrieval mimics *what you said before*; it doesn't understand new
  topics. Great for casual chat, not for serious conversations.
- Needs history: ~hundreds of messages per contact to be useful.
- Draft-only: nothing is ever sent anywhere. Auto-send is a later,
  carefully-gated step.

## Roadmap

- **v2:** char-level LSTM trained on your messages (still CPU, still
  yours) for style rephrasing; retrieval picks *what*, LSTM rewrites
  *how*.
- **v3:** per-contact fine-tuning + supervised auto-send for whitelisted
  contacts with a kill switch and full send log.
