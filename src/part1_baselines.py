"""Part 1 baselines: mean, category mean, TF-IDF + Ridge, LightGBM on metadata, TF-IDF + metadata Ridge."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from metrics import regression_report

df = pd.read_csv("data/items.csv.gz")
df["cat_id"] = df.cat_id.fillna(-1).astype(int)
tr, va, te = (df[df.split == s].copy() for s in ("train", "val", "test"))
y = "b_logit"
results, preds = {}, {"id": te.id.values, "b_true": te[y].values}

def record(name, pred):
    results[name] = regression_report(te[y], pred, te.p)
    preds[name] = pred
    print(f"{name:28s}", {k: round(v, 4) for k, v in results[name].items()})

# 1. global mean
record("Mean", np.full(len(te), tr[y].mean()))
# 2. category mean
cm = tr.groupby("cat_id")[y].mean()
record("Category mean", te.cat_id.map(cm).fillna(tr[y].mean()).values)

# text features
word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=200000)
Xw_tr, Xc_tr = word.fit_transform(tr.text), char.fit_transform(tr.text)
T = lambda d: hstack([word.transform(d.text), char.transform(d.text)]).tocsr()
Xt_tr, Xt_va, Xt_te = hstack([Xw_tr, Xc_tr]).tocsr(), T(va), T(te)

num_cols = ["n_chars", "n_words", "has_digit", "has_qmark", "has_image", "is_translation", "user_submitted"]
ohe = OneHotEncoder(handle_unknown="ignore").fit(tr[["cat_id", "lang_det"]])
sc = StandardScaler().fit(tr[num_cols])
M = lambda d: hstack([ohe.transform(d[["cat_id", "lang_det"]]), csr_matrix(sc.transform(d[num_cols]))]).tocsr()

def tune_ridge(Xtr, Xva):
    best = None
    for a in (0.3, 1, 3, 10, 30):
        m = Ridge(alpha=a).fit(Xtr, tr[y])
        r = np.sqrt(np.mean((m.predict(Xva) - va[y]) ** 2))
        if best is None or r < best[0]:
            best = (r, a, m)
    print("   ridge alpha", best[1])
    return best[2]

# 3. TF-IDF ridge (text only)
m = tune_ridge(Xt_tr, Xt_va)
record("TF-IDF + Ridge", m.predict(Xt_te))
# text-only model applied to cold-start tournament items (Part 2 validation)
tour = pd.read_csv("data/tournament_items.csv")
pd.DataFrame({"id": tour.id, "pred": m.predict(T(tour))}).to_csv("results/tfidf__text__tour.csv", index=False)

# 4. LightGBM on metadata only
feat = num_cols + ["cat_id", "lang_det"]
def lgb_frame(d):
    f = d[feat].copy()
    f["cat_id"] = f.cat_id.astype("category")
    f["lang_det"] = f.lang_det.astype("category")
    return f
cats_all = pd.Categorical(df.cat_id).categories
langs_all = pd.Categorical(df.lang_det).categories
def lgb_fix(f):
    f["cat_id"] = pd.Categorical(f.cat_id.astype(int), categories=cats_all)
    f["lang_det"] = pd.Categorical(f.lang_det.astype(str), categories=langs_all)
    return f
L_tr, L_va, L_te = (lgb_fix(lgb_frame(d)) for d in (tr, va, te))
gbm = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=31, min_child_samples=30,
                        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
gbm.fit(L_tr, tr[y], eval_X=L_va, eval_y=va[y], callbacks=[lgb.early_stopping(100, verbose=False)])
record("LightGBM (metadata)", gbm.predict(L_te))

# 5. TF-IDF + metadata ridge
Xa_tr, Xa_va, Xa_te = (hstack([a, M(d)]).tocsr() for a, d in ((Xt_tr, tr), (Xt_va, va), (Xt_te, te)))
m = tune_ridge(Xa_tr, Xa_va)
record("TF-IDF + metadata Ridge", m.predict(Xa_te))

json.dump(results, open("results/part1_baselines.json", "w"), indent=2)
pd.DataFrame(preds).to_csv("results/part1_baseline_preds.csv", index=False)
