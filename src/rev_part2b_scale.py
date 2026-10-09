"""Revision, Part 2: sensitivity of the cold-start validation to the common-scale assumption.
The between-item 3-dimensional 2PL (one correlated trait per tournament series) fitted better than the
unidimensional 2PL, mainly because the regional-cup series correlated less with the other two. This script
repeats the cold-start correlations (i) with difficulties from the 3-dimensional model, (ii) without the cup
items and (iii) within each series. Output: results/revision/part2_scale_sensitivity.json"""
import glob, json, sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

sys.argv = sys.argv[:1]
src = open("src/rev_part2_irt.py", encoding="utf-8").read()
# reuse the data preparation, the IRT functions and the 3-dimensional model of rev_part2_irt.py (sections up to A)
exec(src.split("# ======================= A. design and connectivity")[0])
head_B = src.split("# ======================= B. common scale across tournament series =======================")[1].split("prm3, ll3 = fit3(R, M)")[0]
exec(head_B)
prm3, ll3 = fit3(R, M)
b3 = prm3["b"].detach().numpy(); a3 = np.exp(prm3["la"].detach().numpy())

clean = pd.read_csv("results/extended/coldstart_clean_items.csv")
ix = np.searchsorted(items, clean.id.values)
ser = np.array([SERIES[s] for s in item_stage[ix]])
TEXT = {"KazRoBERTa": "kaz-roberta-conversational", "XLM-R base": "xlm-roberta-base", "XLM-R large": "xlm-roberta-large",
        "mBERT": "bert-base-multilingual-cased"}
tp = {k: pd.concat([pd.read_csv(f).set_index("id").pred for f in sorted(glob.glob(f"results/finetune/{m}__text__s*__tour.csv"))], axis=1).mean(axis=1).loc[clean.id].values
      for k, m in TEXT.items()}
tp["TF-IDF"] = pd.read_csv("results/tfidf__text__tour.csv").set_index("id").pred.loc[clean.id].values
b2 = clean.b_2PL.values
rng = np.random.default_rng(31)
def r_ci(x, y_, nb=2000):
    bt = []
    for _ in range(nb):
        i = rng.integers(0, len(x), len(x)); bt.append(pearsonr(x[i], y_[i])[0])
    return [float(pearsonr(x, y_)[0]), float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))]
out = {"items_by_series": pd.Series(ser).value_counts().to_dict(),
       "r_b_1D_vs_b_3D": float(pearsonr(b2, b3[ix])[0]), "rho_b_1D_vs_b_3D": float(spearmanr(b2, b3[ix])[0]),
       "r_b_1D_vs_b_3D_by_series": {s: float(pearsonr(b2[ser == s], b3[ix][ser == s])[0]) for s in np.unique(ser)},
       "models": {}}
for k, p in tp.items():
    noc = ser != "Regional cup"
    out["models"][k] = {"b_2PL_all": r_ci(p, b2), "b_3D_all": r_ci(p, b3[ix]), "b_2PL_without_cup": r_ci(p[noc], b2[noc]),
                        "within_series_b_2PL": {s: r_ci(p[ser == s], b2[ser == s]) + [int((ser == s).sum())] for s in np.unique(ser)}}
json.dump(out, open("results/revision/part2_scale_sensitivity.json", "w"), indent=2)
print(json.dumps(out, indent=1))
