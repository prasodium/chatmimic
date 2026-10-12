#!/usr/bin/env python3
"""
Situation handling: WHAT kind of moment is this?

The bot behaves differently per situation instead of treating every
message the same. Classification is keyword/heuristic-based (stdlib
only, no ML model, no pretrained anything) -- the *learning* lives in
the bandit, which keeps separate Q-values per situation per contact.
That is what makes it a *contextual* bandit: it can learn that rank1
wins for questions while rank0 wins for greetings, separately.

Situations and policy:

    greeting   hi / hey / good morning            → auto-send OK (low risk)
    farewell   bye / goodnight / see you          → auto-send OK (low risk)
    thanks     thanks / thank you                 → auto-send OK (low risk)
    question   ends with ? / starts with wh-word  → gated (needs your history)
    plan       tomorrow / meet / lunch / time…    → gated (details matter)
    chat       everything else                    → gated (generic)
    emotional  sorry / sad / urgent / hurt…       → HUMAN ONLY, never auto-sent
    unknown    retrieval similarity too low       → HUMAN ONLY, teach via learn.py

POLICY[situation] = {"draft": bool, "autosend": True | "gated" | False,
                     "note": str}
"""

import re

UNKNOWN_SIM = 0.25  # below this retrieval similarity → "unknown"


def _words(text: str):
    return re.findall(r"[a-z']+", text.lower())


GREETING_WORDS = {"hi", "hii", "hiii", "hey", "heyy", "hello", "yo", "sup",
                  "namaste", "hola"}
GREETING_PHRASES = ("good morning", "good afternoon", "good evening")
FAREWELL_WORDS = {"bye", "byee", "goodnight", "gn", "ttyl", "cya", "later"}
FAREWELL_PHRASES = ("good night", "see you", "take care")
THANKS_WORDS = {"thanks", "thank", "thx", "thanku"}
EMOTIONAL_WORDS = {
    "sorry", "sad", "cry", "cried", "crying", "tears",
    "angry", "anger", "upset", "hurt", "hurts", "pain",
    "worried", "worry", "anxious", "anxiety", "scared", "fear",
    "depressed", "depression", "lonely", "alone",
    "emergency", "urgent", "help", "please",
    "died", "death", "dead", "hospital", "accident", "fight",
    "broke", "breakup", "divorce", "fired", "failed", "fail",
}
PLAN_WORDS = {
    # deliberately specific: generic time words ("morning", "night",
    # "time") misfire on past-tense chat like "last night was crazy"
    "tomorrow", "today", "tonight",
    "meet", "meeting", "lunch", "dinner", "breakfast", "coffee",
    "call", "plan", "plans", "planning", "come", "coming", "visit",
    "trip", "movie", "party", "weekend", "schedule", "appointment",
}
QUESTION_STARTS = {
    "what", "when", "where", "who", "whom", "whose", "which", "why", "how",
    "can", "could", "would", "will", "shall", "do", "does", "did",
    "is", "are", "was", "were", "have", "has", "had", "should", "may",
}

POLICY = {
    "greeting":  {"draft": True, "autosend": True,
                  "note": "low risk, formulaic"},
    "farewell":  {"draft": True, "autosend": True,
                  "note": "low risk, formulaic"},
    "thanks":    {"draft": True, "autosend": True,
                  "note": "low risk, formulaic"},
    "question":  {"draft": True, "autosend": "gated",
                  "note": "needs your historical answer to be right"},
    "plan":      {"draft": True, "autosend": "gated",
                  "note": "details matter — the gate decides"},
    "chat":      {"draft": True, "autosend": "gated",
                  "note": "generic situation"},
    "emotional": {"draft": True, "autosend": False,
                  "note": "serious talk — always human, never auto-sent"},
    "unknown":   {"draft": True, "autosend": False,
                  "note": "nothing similar in history — teach it with learn.py"},
}

SITUATIONS = tuple(POLICY)


def human_only(situation: str) -> bool:
    """True if this situation must never be auto-sent."""
    return POLICY.get(situation, POLICY["chat"])["autosend"] is False


def classify(incoming: str, similarity: float | None = None) -> str:
    """Label the situation. Heuristic v1 (stdlib only) -- the *learning*
    lives in the per-situation bandit, not in these rules.

    Priority order is deliberate: safety (emotional) first, then
    capability (unknown), then formulaic, then question, then plan.
    """
    text = incoming.lower().strip()
    words = _words(text)
    wset = set(words)

    # 1. safety first: emotional content always goes to a human
    if wset & EMOTIONAL_WORDS:
        return "emotional"
    # 2. capability: nothing similar in history → unknown (teach it)
    if similarity is not None and similarity < UNKNOWN_SIM:
        return "unknown"
    # 3. short formulaic messages
    if len(words) <= 6:
        if wset & GREETING_WORDS or any(p in text for p in GREETING_PHRASES):
            return "greeting"
        if wset & FAREWELL_WORDS or any(p in text for p in FAREWELL_PHRASES):
            return "farewell"
        if wset & THANKS_WORDS:
            return "thanks"
    # 4. questions (priority over plans: "lunch tomorrow?" is a question)
    if text.endswith("?") or (words and words[0] in QUESTION_STARTS):
        return "question"
    # 5. plans
    if wset & PLAN_WORDS:
        return "plan"
    return "chat"


if __name__ == "__main__":
    # quick self-check
    cases = {
        "hey": "greeting",
        "good morning!": "greeting",
        "goodnight, talk tomorrow": "farewell",
        "thanks a lot!": "thanks",
        "are we still on for lunch?": "question",
        "what time works?": "question",
        "lets meet tomorrow evening": "plan",
        "i'm really sorry about yesterday": "emotional",
        "urgent, call me now": "emotional",
        "did you see that crazy match": "question",  # starts with "did"
        "see that match last night": "chat",
    }
    for msg, want in cases.items():
        got = classify(msg, similarity=0.9)
        flag = "OK " if got == want else "MISS"
        print(f"{flag} {msg!r:38} -> {got} (want {want})")
    print("unknown check:", classify("zxqv kwijibo", similarity=0.05))
