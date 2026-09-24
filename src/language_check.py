"""Accuracy of the script-based language heuristic against a manual annotation of a stratified sample
(100 items labelled Kazakh, 70 Russian, 30 English by the heuristic), weighted by class size in the item bank.
Input (not distributed; contains item ids of annotated items): ../private_language_check/manual_labels_parsed.csv
with columns id, lang_det, manual (kz / ru / en / none). Output: results/extended/language_manual_check.json"""
import json
import numpy as np
import pandas as pd

import os, sys
SRC = "../private_language_check/manual_labels_parsed.csv"
if not os.path.exists(SRC):
    print("manual annotation not available; skipping"); sys.exit(0)
m = pd.read_csv(SRC)
N = pd.read_csv("data/items.csv.gz", usecols=["lang_det"]).lang_det.value_counts().to_dict()
tot = sum(N.values())

def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [c - h, c + h]

res = {"sample": len(m), "agree": int((m.lang_det == m.manual).sum()),
       "confusion": pd.crosstab(m.lang_det, m.manual).to_dict()}
prec = {}
for c in N:
    s = m[m.lang_det == c]; k = int((s.manual == c).sum()); prec[c] = (k, len(s))
    res[f"precision_{c}"] = {"k": k, "n": len(s), "p": k / len(s), "CI_wilson": wilson(k, len(s))}
res["weighted_accuracy"] = sum(N[c] * prec[c][0] / prec[c][1] for c in N) / tot
rng = np.random.default_rng(0)
sims = [sum(N[c] * rng.beta(prec[c][0] + 0.5, prec[c][1] - prec[c][0] + 0.5) for c in N) / tot for _ in range(20000)]
res["weighted_accuracy_CI"] = [float(np.percentile(sims, 2.5)), float(np.percentile(sims, 97.5))]
json.dump(res, open("results/extended/language_manual_check.json", "w"), indent=2)
print(f"agreement {res['agree']}/{res['sample']}, weighted accuracy {res['weighted_accuracy']:.3f} "
      f"(95% CI {res['weighted_accuracy_CI'][0]:.3f}-{res['weighted_accuracy_CI'][1]:.3f})")
