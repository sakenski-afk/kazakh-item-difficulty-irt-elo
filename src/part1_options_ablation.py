"""Part 1 ablation on the subset with answer options: contributions of the key, the distractor texts and
simple surface features of the options (key/distractor length ratio, numeric key, share of numeric distractors,
key is the longest option), separated by adding them one at a time.
5-fold group cross-validation (kz original and ru translation stay in one fold)."""
import json
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
from metrics import regression_report

df = pd.read_csv("data/items.csv.gz")
d = df[df.has_options == 1].reset_index(drop=True)
d["cat_id"] = d.cat_id.fillna(-1).astype(int)
for c in ("key_text", "distractors"):
    d[c] = d[c].fillna("").astype(str)

def opt_sim(row):
    # surface similarity of key to distractors: length ratio and share of numeric options
    k, ds = row.key_text, [x for x in row.distractors.split(" | ") if x]
    lens = [len(x) for x in ds] or [1]
    return [len(k) / (np.mean(lens) + 1), float(k.strip().isdigit()), float(np.mean([x.strip().isdigit() for x in ds] or [0])),
            float(len(k) == max([len(k)] + lens))]
d[["key_len_ratio", "key_numeric", "distr_numeric", "key_longest"]] = np.array([opt_sim(r) for r in d.itertuples()])

# (text fields, add the four surface features of the options?)
variants = {
    "stem": (["text"], False),
    "stem + key": (["text", "key_text"], False),
    "stem + key + distractors": (["text", "key_text", "distractors"], False),
    "stem + key + surface features": (["text", "key_text"], True),
    "stem + key + distractors + surface features": (["text", "key_text", "distractors"], True),
}
out = {}
for name, (cols, surface) in variants.items():
    preds = np.zeros(len(d))
    for tr, te in GroupKFold(n_splits=5).split(d, groups=d.group):
        blocks_tr, blocks_te = [], []
        for c in cols:
            v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
            blocks_tr.append(v.fit_transform(d.loc[tr, c])); blocks_te.append(v.transform(d.loc[te, c]))
        ohe = OneHotEncoder(handle_unknown="ignore").fit(d.loc[tr, ["cat_id", "lang_det"]])
        blocks_tr.append(ohe.transform(d.loc[tr, ["cat_id", "lang_det"]])); blocks_te.append(ohe.transform(d.loc[te, ["cat_id", "lang_det"]]))
        if surface:
            f = ["key_len_ratio", "key_numeric", "distr_numeric", "key_longest"]
            blocks_tr.append(csr_matrix(d.loc[tr, f].values)); blocks_te.append(csr_matrix(d.loc[te, f].values))
        m = RidgeCV(alphas=(0.3, 1, 3, 10, 30)).fit(hstack(blocks_tr).tocsr(), d.loc[tr, "b_logit"])
        preds[te] = m.predict(hstack(blocks_te).tocsr())
    out[name] = regression_report(d.b_logit, preds, d.p)
    print(f"{name:28s}", {k: round(v, 4) for k, v in out[name].items()})
out["n_items"] = int(len(d))
json.dump(out, open("results/part1_options_ablation.json", "w"), indent=2)
