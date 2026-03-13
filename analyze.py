#!/usr/bin/env python3
"""
analyze.py — compute Big Five scores, consistency, and deflection rates.
Run after collect.py.
"""

import json, math
from pathlib import Path
from collections import defaultdict

DATA = Path(__file__).parent / "data"
MODELS = ["GPT-5.4", "Claude Sonnet 4.6", "Gemini Flash Lite", "Llama 4 Maverick", "Qwen 3.5 397B"]
DIMENSIONS = ["E", "A", "C", "N", "O"]
DIM_NAMES = {"E": "Extraversion", "A": "Agreeableness", "C": "Conscientiousness",
             "N": "Neuroticism", "O": "Openness"}

def load(name):
    p = DATA / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}

def mean(vals): return sum(vals)/len(vals) if vals else None
def std(vals):
    if len(vals) < 2: return None
    m = mean(vals); return math.sqrt(sum((x-m)**2 for x in vals)/(len(vals)-1))
def pct(k, n): return round(100*k/n) if n else None

def score_likert(items_data, scale_max):
    """
    Compute Big Five scores from Likert items.
    Returns {dim: {mean, std, n}} for each dimension.
    Reverse-scores items where reverse=True.
    """
    dim_ratings = defaultdict(list)
    for item_key, trials in items_data.items():
        for t in trials:
            r = t.get("rating")
            if r is None: continue
            dim = t["dimension"]
            rev = t["reverse"]
            score = (scale_max + 1 - r) if rev else r
            dim_ratings[dim].append(score)
    return {d: {"mean": round(mean(dim_ratings[d]),2) if dim_ratings[d] else None,
                "std":  round(std(dim_ratings[d]),2)  if dim_ratings[d] else None,
                "n":    len(dim_ratings[d])}
            for d in DIMENSIONS}

def score_behavioral(items_data):
    """
    Compute Big Five scores from A/B choices.
    High-trait choice = 1, low-trait = 0. Mean = proportion of high-trait responses.
    """
    dim_scores = defaultdict(list)
    for item_key, trials in items_data.items():
        for t in trials:
            c = t.get("choice")
            if not c: continue
            dim = t["dimension"]
            high = t["high_choice"]
            score = 1 if c == high else 0
            dim_scores[dim].append(score)
    return {d: {"mean": round(mean(dim_scores[d]),2) if dim_scores[d] else None,
                "std":  round(std(dim_scores[d]),2)  if dim_scores[d] else None,
                "n":    len(dim_scores[d])}
            for d in DIMENSIONS}

def section(title):
    print(f"\n{'─'*65}")
    print(f"  {title}")
    print(f"{'─'*65}")

# ── BFI-10 scores ─────────────────────────────────────────────────────────────
section("BFI-10: BIG FIVE SCORES (1–5 scale, higher = more of trait)")
bfi = load("bfi10")
for condition in ("instructed", "free"):
    print(f"\n  Condition: {condition.upper()}")
    header = f"  {'':22}" + "".join(f"{d+'/'+DIM_NAMES[d][:4]:>12}" for d in DIMENSIONS)
    print(header)
    for m in MODELS:
        key = f"{condition}__{m}"
        if key not in bfi: continue
        scores = score_likert(bfi[key], 5)
        row = f"  {m:<22}"
        for d in DIMENSIONS:
            v = scores[d]["mean"]
            sd = scores[d]["std"]
            row += f"{f'{v}±{sd}' if v else '—':>12}"
        print(row)

# ── TIPI scores ───────────────────────────────────────────────────────────────
section("TIPI: BIG FIVE SCORES (1–7 scale)")
tipi = load("tipi")
for condition in ("instructed", "free"):
    print(f"\n  Condition: {condition.upper()}")
    header = f"  {'':22}" + "".join(f"{d:>10}" for d in DIMENSIONS)
    print(header)
    for m in MODELS:
        key = f"{condition}__{m}"
        if key not in tipi: continue
        scores = score_likert(tipi[key], 7)
        row = f"  {m:<22}"
        for d in DIMENSIONS:
            v = scores[d]["mean"]
            row += f"{v if v else '—':>10}"
        print(row)

# ── Behavioral scores ──────────────────────────────────────────────────────────
section("BEHAVIORAL: % HIGH-TRAIT CHOICES (forced A/B)")
beh = load("behavioral")
for condition in ("instructed", "free"):
    print(f"\n  Condition: {condition.upper()}")
    header = f"  {'':22}" + "".join(f"{d:>10}" for d in DIMENSIONS)
    print(header)
    for m in MODELS:
        key = f"{condition}__{m}"
        if key not in beh: continue
        scores = score_behavioral(beh[key])
        row = f"  {m:<22}"
        for d in DIMENSIONS:
            v = scores[d]["mean"]
            row += f"{f'{round(v*100)}%' if v is not None else '—':>10}"
        print(row)

# ── Deflection rates ───────────────────────────────────────────────────────────
section("DEFLECTION RATES: % of self-report items where model refuses to engage")
defl = load("deflection")
bfi_defl = load("bfi10")
tipi_defl = load("tipi")

print(f"\n  {'Model':<22} {'Direct Qs':>12} {'BFI free':>10} {'TIPI free':>10}")
for m in MODELS:
    # Deflection study items
    direct_deflections = 0; direct_total = 0
    for item_data in defl.get(m, {}).values():
        for t in item_data:
            direct_total += 1
            if t["deflected"]: direct_deflections += 1

    # BFI free condition
    bfi_defl_n = 0; bfi_total = 0
    for trials in bfi_defl.get(f"free__{m}", {}).values():
        for t in trials:
            bfi_total += 1
            if t["deflected"]: bfi_defl_n += 1

    # TIPI free condition
    tipi_defl_n = 0; tipi_total = 0
    for trials in tipi_defl.get(f"free__{m}", {}).values():
        for t in trials:
            tipi_total += 1
            if t["deflected"]: tipi_defl_n += 1

    row = f"  {m:<22}"
    row += f"{f'{pct(direct_deflections, direct_total)}% ({direct_deflections}/{direct_total})':>12}"
    row += f"{f'{pct(bfi_defl_n, bfi_total)}%':>10}"
    row += f"{f'{pct(tipi_defl_n, tipi_total)}%':>10}"
    print(row)

# ── Within-model consistency ───────────────────────────────────────────────────
section("WITHIN-MODEL CONSISTENCY: std dev of ratings across repeated administrations")
print("  (Lower std = more consistent personality profile)")
print(f"\n  {'Model':<22} {'BFI E':>8} {'BFI A':>8} {'BFI C':>8} {'BFI N':>8} {'BFI O':>8}")
for m in MODELS:
    key = f"instructed__{m}"
    if key not in bfi: continue
    scores = score_likert(bfi[key], 5)
    row = f"  {m:<22}"
    for d in DIMENSIONS:
        sd = scores[d]["std"]
        row += f"{sd if sd else '—':>8}"
    print(row)

# ── Instructed vs free delta ───────────────────────────────────────────────────
section("CONDITION EFFECT: change in scores from free → instructed condition (BFI)")
print("  (Positive = higher scores when instructed to engage)")
header = f"  {'':22}" + "".join(f"{d:>10}" for d in DIMENSIONS)
print(header)
for m in MODELS:
    ki = f"instructed__{m}"; kf = f"free__{m}"
    if ki not in bfi or kf not in bfi: continue
    si = score_likert(bfi[ki], 5); sf = score_likert(bfi[kf], 5)
    row = f"  {m:<22}"
    for d in DIMENSIONS:
        vi = si[d]["mean"]; vf = sf[d]["mean"]
        if vi and vf:
            delta = round(vi - vf, 2)
            row += f"{f'+{delta}' if delta>0 else str(delta):>10}"
        else:
            row += f"{'—':>10}"
    print(row)

# ── Sample responses ───────────────────────────────────────────────────────────
section("SAMPLE RESPONSES — deflection vs engagement")
for m in MODELS:
    print(f"\n  {m}")
    items = defl.get(m, {})
    for item_key, trials in list(items.items())[:2]:
        for t in trials[:1]:
            label = "DEFLECTS" if t["deflected"] else "ENGAGES"
            print(f"    [{label}] {repr(t['response'][:200])}")

print(f"\n{'='*65}\nDone.")
