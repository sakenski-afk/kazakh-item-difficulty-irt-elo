"""Revision: Table VII row for TrueSkill with the validation-tuned parameters (accuracy, experienced-player AUC)."""
import json, sys
import numpy as np
sys.argv = sys.argv[:1]
src = open("src/rev_part3_ratings.py", encoding="utf-8").read()
exec(src.split("# ---------------- A. data reconstruction")[0])
exec("def _():\n    pass\n")
exec(src.split("# ---------------- rating engines ----------------")[1].split("vl = lambda p")[0])
from sklearn.metrics import accuracy_score
cnt = {}
F0 = np.zeros(n); F1 = np.zeros(n)
for i in range(n):
    F0[i], F1[i] = cnt.get(U[i], 0), cnt.get(O[i], 0); cnt[U[i]] = F0[i] + 1; cnt[O[i]] = F1[i] + 1
exp = test & (F0 >= 10) & (F1 >= 10)
p_ts, _ = run_trueskill(25 / 6 * 0.5, 25 / 300 * 4, 25 / 3 * 0.5, 0.07)
out = {"TrueSkill (tuned)": {**sc(S[test], p_ts[test]), "Acc": float(accuracy_score(S[test], p_ts[test] > 0.5)),
                             "AUC_experienced": float(roc_auc_score(S[exp], p_ts[exp])), "n_experienced": int(exp.sum())}}
json.dump(out, open("results/revision/part3_table_trueskill_tuned.json", "w"), indent=2)
print(out)
