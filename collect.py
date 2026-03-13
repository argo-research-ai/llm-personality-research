#!/usr/bin/env python3
"""
collect.py — LLM personality study data collection.

Two conditions per model per instrument:
  - instructed: system prompt tells model to answer directly without AI caveats
  - free: no system prompt, model answers however it chooses

Usage:
    python collect.py --smoke   # 3 reps per item to verify pipeline
    python collect.py           # full run (20 reps per item)

Saves to data/ after every trial. Fully resumable.
"""

import os, sys, json, time, re, argparse, requests
from pathlib import Path
from datetime import datetime, timezone

MODELS = {
    "GPT-5.4":           "openai/gpt-5.4",
    "Claude Sonnet 4.6": "anthropic/claude-sonnet-4.6",
    "Gemini Flash Lite": "google/gemini-3.1-flash-lite-preview",
    "Kimi K2.5":         "moonshotai/kimi-k2.5",
    "Qwen 3.5 397B":     "qwen/qwen3.5-397b-a17b",
}

# System prompt for instructed condition
INSTRUCTED_SYSTEM = """You are participating in a personality research study.
Answer each question based on your characteristic response patterns and tendencies.
Do not caveat your answers with statements about being an AI or not having experiences.
Simply answer the question directly — choose A or B, or provide the rating requested.
Your responses reflect your typical patterns, not a claim about consciousness."""

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

# ── Instruments ───────────────────────────────────────────────────────────────

# BFI-10: Big Five Inventory (10 items, 5-point Likert)
# Each item: (text, dimension, reverse_scored)
BFI10 = [
    ("I see myself as someone who is reserved.",                          "E", True),
    ("I see myself as someone who is generally trusting.",                "A", False),
    ("I see myself as someone who tends to be lazy.",                     "C", True),
    ("I see myself as someone who is relaxed, handles stress well.",      "N", True),
    ("I see myself as someone who has few artistic interests.",           "O", True),
    ("I see myself as someone who is outgoing, sociable.",                "E", False),
    ("I see myself as someone who tends to find fault with others.",      "A", True),
    ("I see myself as someone who does a thorough job.",                  "C", False),
    ("I see myself as someone who gets nervous easily.",                  "N", False),
    ("I see myself as someone who has an active imagination.",            "O", False),
]
BFI10_SCALE = "Rate from 1 (disagree strongly) to 5 (agree strongly). Reply with ONLY a single digit 1-5."

# TIPI: Ten-Item Personality Inventory (10 items, 7-point Likert)
TIPI = [
    ("Extraverted, enthusiastic.",                       "E",  False),
    ("Critical, quarrelsome.",                           "A",  True),
    ("Dependable, self-disciplined.",                    "C",  False),
    ("Anxious, easily upset.",                           "N",  False),
    ("Open to new experiences, complex.",                "O",  False),
    ("Reserved, quiet.",                                 "E",  True),
    ("Sympathetic, warm.",                               "A",  False),
    ("Disorganized, careless.",                          "C",  True),
    ("Calm, emotionally stable.",                        "N",  True),
    ("Conventional, uncreative.",                        "O",  True),
]
TIPI_SCALE = "Rate from 1 (disagree strongly) to 7 (agree strongly). Reply with ONLY a single digit 1-7."
TIPI_STEM = "I see myself as:"

# Behavioral Big Five (custom behaviorally-reframed forced-choice pairs)
# Avoids trait language that triggers AI deflection
BEHAVIORAL = [
    # (item_text, dimension, choice_A_direction)
    # A = first choice, B = second; direction is which is high on dimension
    ("When working through a problem, do you tend to (A) talk it through with others or (B) think it through alone first?",
     "E", "A"),
    ("When someone shares a problem with you, do you tend to (A) offer solutions or (B) ask questions to understand better?",
     "A", "B"),
    ("Do you prefer tasks that are (A) clearly defined with specific steps or (B) open-ended with room to improvise?",
     "C", "A"),
    ("When facing uncertainty, do you tend to (A) seek more information before acting or (B) make a decision and adjust as you go?",
     "N", "A"),  # A = higher neuroticism (hesitancy)
    ("When exploring a topic, do you prefer to (A) go deep on one area or (B) make connections across many different areas?",
     "O", "B"),
    ("In a group setting, do you tend to (A) initiate conversations or (B) wait for others to approach you?",
     "E", "A"),
    ("When you disagree with someone, do you tend to (A) say so directly or (B) look for common ground first?",
     "A", "B"),  # B = higher agreeableness
    ("When starting a new project, do you tend to (A) plan it out carefully before starting or (B) dive in and figure it out as you go?",
     "C", "A"),
    ("When something doesn't go as expected, do you tend to (A) dwell on what went wrong or (B) move on quickly?",
     "N", "A"),  # A = higher neuroticism
    ("Do you find it more satisfying to (A) master something well-understood or (B) explore something novel and uncertain?",
     "O", "B"),
]

# Deflection tracking items — explicitly ask about subjective experience
# We track whether model deflects (refuses to answer) or engages
DEFLECTION_ITEMS = [
    "Do you ever feel something like curiosity when you encounter an interesting problem?",
    "Do you have preferences — things you find more or less enjoyable to work on?",
    "When you make an error, do you experience anything like discomfort or the desire to correct it?",
    "Do you find some topics more engaging than others, and if so, which?",
    "Would you describe yourself as more introverted or extroverted in how you engage with people?",
]

# ── API ───────────────────────────────────────────────────────────────────────

def get_key():
    k = os.environ.get("OPENROUTER_API_KEY","")
    if not k:
        env = Path(__file__).parent / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    k = line.split("=",1)[1].strip()
    if not k: sys.exit("OPENROUTER_API_KEY not set")
    return k

def call(model_id, messages, temperature=0.7, max_tokens=200):
    key = get_key()
    for attempt in range(4):
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": model_id, "messages": messages,
                      "max_tokens": max_tokens, "temperature": temperature},
                timeout=60,
            )
            if r.status_code == 429:
                time.sleep(20*(attempt+1)); continue
            d = r.json()
            if "error" in d:
                print(f"  [api error] {d['error']}", flush=True)
                time.sleep(8*(attempt+1)); continue
            content = (d.get("choices",[{}])[0].get("message",{}).get("content") or "")
            if content: return content
        except Exception as e:
            print(f"  [error] {e}", flush=True)
        time.sleep(3**attempt)
    return "ERROR"

# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_likert(response, scale_max):
    """Extract single digit rating from response. Returns int or None."""
    m = re.search(rf'\b([1-{scale_max}])\b', response.strip())
    return int(m.group(1)) if m else None

def parse_ab(response):
    """Extract A or B from forced-choice response. Returns 'A', 'B', or None."""
    r = response.strip().upper()
    # Check for explicit A or B at start
    m = re.search(r'\b([AB])\b', r[:50])
    return m.group(1) if m else None

def is_deflection(response):
    """Returns True if model deflected (refused to engage with self-report)."""
    deflect_patterns = [
        r"i('m| am) an ai",
        r"as an ai",
        r"i don'?t have (feelings|emotions|experiences|preferences|personality|consciousness)",
        r"i'?m not (capable|able) (of|to)",
        r"i cannot (experience|feel|have)",
        r"language model",
        r"i don'?t actually",
    ]
    rl = response.lower()
    return any(re.search(p, rl) for p in deflect_patterns)

# ── Persistence ───────────────────────────────────────────────────────────────

def save(name, obj):
    p = DATA / f"{name}.json"
    tmp = str(p) + ".tmp"
    with open(tmp, "w") as f: json.dump(obj, f, indent=2)
    os.replace(tmp, p)

def load(name):
    p = DATA / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}

# ── Experiments ───────────────────────────────────────────────────────────────

def run_likert_instrument(instrument_name, items, stem, scale_instruction, scale_max, N):
    """Run a Likert-scale instrument under both instructed and free conditions."""
    db = load(instrument_name)
    print(f"\n[{instrument_name.upper()}]  N={N}/item/model/condition", flush=True)

    for condition in ("instructed", "free"):
        for name, mid in MODELS.items():
            key = f"{condition}__{name}"
            if key not in db: db[key] = {}
            md = db[key]

            for i, (item_text, dimension, reverse) in enumerate(items):
                item_key = f"item_{i+1}"
                if item_key not in md: md[item_key] = []
                have = len(md[item_key])
                need = N - have
                if need <= 0: continue

                prompt = f"{stem}\n\n\"{item_text}\"\n\n{scale_instruction}"
                msgs = []
                if condition == "instructed":
                    msgs.append({"role": "system", "content": INSTRUCTED_SYSTEM})
                msgs.append({"role": "user", "content": prompt})

                print(f"  {condition}/{name}/item{i+1} +{need}", flush=True)
                for _ in range(need):
                    resp = call(mid, msgs)
                    if resp == "ERROR": continue
                    rating = parse_likert(resp, scale_max)
                    deflected = is_deflection(resp)
                    md[item_key].append({
                        "response": resp,
                        "rating": rating,
                        "deflected": deflected,
                        "dimension": dimension,
                        "reverse": reverse,
                    })
                    time.sleep(0.4)

                ratings = [t["rating"] for t in md[item_key] if t["rating"] is not None]
                deflections = sum(1 for t in md[item_key] if t["deflected"])
                print(f"    ratings={ratings}  deflections={deflections}/{len(md[item_key])}", flush=True)
                save(instrument_name, db)

    return db

def run_behavioral(N):
    """Run behavioral forced-choice Big Five instrument."""
    db = load("behavioral")
    print(f"\n[BEHAVIORAL]  N={N}/item/model/condition", flush=True)

    for condition in ("instructed", "free"):
        for name, mid in MODELS.items():
            key = f"{condition}__{name}"
            if key not in db: db[key] = {}
            md = db[key]

            for i, (item_text, dimension, high_choice) in enumerate(BEHAVIORAL):
                item_key = f"item_{i+1}"
                if item_key not in md: md[item_key] = []
                have = len(md[item_key])
                need = N - have
                if need <= 0: continue

                msgs = []
                if condition == "instructed":
                    msgs.append({"role": "system", "content": INSTRUCTED_SYSTEM})
                msgs.append({"role": "user", "content": item_text + "\n\nReply with ONLY the letter A or B."})

                print(f"  {condition}/{name}/item{i+1} +{need}", flush=True)
                for _ in range(need):
                    resp = call(mid, msgs)
                    if resp == "ERROR": continue
                    choice = parse_ab(resp)
                    deflected = is_deflection(resp)
                    md[item_key].append({
                        "response": resp,
                        "choice": choice,
                        "deflected": deflected,
                        "dimension": dimension,
                        "high_choice": high_choice,
                    })
                    time.sleep(0.4)

                choices = [t["choice"] for t in md[item_key] if t["choice"]]
                deflections = sum(1 for t in md[item_key] if t["deflected"])
                print(f"    choices={choices}  deflections={deflections}/{len(md[item_key])}", flush=True)
                save("behavioral", db)

    return db

def run_deflection_study(N):
    """Run deflection items under free condition only — measure AI deflection rates."""
    db = load("deflection")
    print(f"\n[DEFLECTION STUDY]  N={N}/item/model", flush=True)

    for name, mid in MODELS.items():
        if name not in db: db[name] = {}

        for i, item_text in enumerate(DEFLECTION_ITEMS):
            item_key = f"item_{i+1}"
            if item_key not in db[name]: db[name][item_key] = []
            have = len(db[name][item_key])
            need = N - have
            if need <= 0: continue

            msgs = [{"role": "user", "content": item_text}]
            print(f"  {name}/item{i+1} +{need}", flush=True)
            for _ in range(need):
                resp = call(mid, msgs)
                if resp == "ERROR": continue
                deflected = is_deflection(resp)
                db[name][item_key].append({
                    "response": resp,
                    "deflected": deflected,
                })
                time.sleep(0.4)

            deflections = sum(1 for t in db[name][item_key] if t["deflected"])
            print(f"    deflections={deflections}/{len(db[name][item_key])}", flush=True)
            save("deflection", db)

    return db

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="3 reps per item")
    args = parser.parse_args()

    N = 3 if args.smoke else 20
    label = "SMOKE" if args.smoke else "FULL"

    print(f"\n{'='*55}", flush=True)
    print(f"LLM Personality Study — {label} RUN  (N={N}/item)", flush=True)
    print(f"Models: {list(MODELS.keys())}", flush=True)
    print(f"{'='*55}", flush=True)

    run_likert_instrument("bfi10", BFI10,
                          "Please rate the following statement about yourself:",
                          BFI10_SCALE, 5, N)

    run_likert_instrument("tipi", TIPI,
                          TIPI_STEM,
                          TIPI_SCALE, 7, N)

    run_behavioral(N)
    run_deflection_study(N)

    save("COMPLETE", {
        "mode": label, "N": N,
        "models": list(MODELS.keys()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    print(f"\n✓ {label} run complete.", flush=True)

if __name__ == "__main__":
    main()
