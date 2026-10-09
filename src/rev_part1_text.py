"""Revision analyses for Part 1 (text-based difficulty) and the cross-lingual comparison.
  A. near-duplicate audit across the D1 partitions (character TF-IDF cosine), test metrics without near duplicates
  B. token truncation rates at the maximum sequence length of 64 tokens (all four tokenizers, both inputs)
  C. answer-option ablation with group-bootstrap CIs and paired differences between inputs
  D. Table I: seed ensembles of all fine-tuned configurations; screening of the easiest / hardest deciles
  E. cross-lingual comparison: category-cluster bootstrap and Wald tests under design-effect inflation
Outputs: results/revision/part1_*.json"""
import glob, json, os, re
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from scipy.stats import pearsonr, spearmanr, norm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder

OUT = "results/revision/"; os.makedirs(OUT, exist_ok=True)
B = 2000
it = pd.read_csv("data/items.csv.gz")
tour = pd.read_csv("data/tournament_items.csv")
res = {}

def ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

def ens(files):
    return pd.concat([pd.read_csv(f).set_index("id").pred for f in files], axis=1).mean(axis=1)

def group_boot_idx(groups, rng, nb=B):
    ug = np.unique(groups); pos = {u: np.flatnonzero(groups == u) for u in ug}
    return [np.concatenate([pos[u] for u in rng.choice(ug, len(ug))]) for _ in range(nb)]

te = it[it.split == "test"].set_index("id")
yy = te.b_logit.values
base = pd.read_csv("results/part1_baseline_preds.csv").set_index("id")
bas = base["TF-IDF + metadata Ridge"].loc[te.index].values
sel = ens(sorted(glob.glob("results/finetune/xlm-roberta-large__cat_text__s*__test.csv"))).loc[te.index].values
gidx = group_boot_idx(te.group.values, np.random.default_rng(0))

# ---------------- A. near-duplicate audit ----------------
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True).fit(it.text)
X = vec.transform(it.text)
tr_mask, va_mask, te_mask = (it.split == "train").values, (it.split == "val").values, (it.split == "test").values
def max_sim(src, dst):
    out = np.zeros(src.sum())
    S_, D_ = X[src], X[dst].T.tocsc()
    for s in range(0, S_.shape[0], 2000):
        out[s:s + 2000] = (S_[s:s + 2000] @ D_).max(axis=1).toarray().ravel()
    return out
sim_te = max_sim(te_mask, tr_mask | va_mask)
sim_va = max_sim(va_mask, tr_mask)
A = {"vectoriser": "character 3-5-gram TF-IDF (char_wb), cosine similarity to the most similar item in the other partitions"}
for thr in (0.8, 0.9, 0.95):
    A[f"test_items_with_train_or_val_neighbour_cos_ge_{thr}"] = int((sim_te >= thr).sum())
    A[f"val_items_with_train_neighbour_cos_ge_{thr}"] = int((sim_va >= thr).sum())
A["test_items"] = int(te_mask.sum()); A["median_max_cosine_test"] = float(np.median(sim_te))
keep = sim_te < 0.8
te_sub = te[keep]; yk = yy[keep]; bk = bas[keep]; sk = sel[keep]
gk = group_boot_idx(te_sub.group.values, np.random.default_rng(1))
A["test_without_near_duplicates"] = {
    "n": int(keep.sum()),
    "r_selected": float(pearsonr(yk, sk)[0]), "r_selected_CI": ci([pearsonr(yk[i], sk[i])[0] for i in gk]),
    "r_tfidf_meta": float(pearsonr(yk, bk)[0]), "r_tfidf_meta_CI": ci([pearsonr(yk[i], bk[i])[0] for i in gk]),
    "delta_r": float(pearsonr(yk, sk)[0] - pearsonr(yk, bk)[0]),
    "delta_r_CI": ci([pearsonr(yk[i], sk[i])[0] - pearsonr(yk[i], bk[i])[0] for i in gk]),
    "RMSE_selected": float(np.sqrt(np.mean((yk - sk) ** 2))), "RMSE_tfidf_meta": float(np.sqrt(np.mean((yk - bk) ** 2)))}
# near duplicates whose difficulty differs (the label noise argument)
nd = ~keep
A["near_duplicate_test_items_abs_b_diff_to_neighbour_median"] = None
if nd.any():
    S_ = X[np.flatnonzero(te_mask)[nd]] @ X[tr_mask | va_mask].T
    nb = np.asarray(S_.argmax(axis=1)).ravel()
    other = it[tr_mask | va_mask].b_logit.values
    A["near_duplicate_test_items_abs_b_diff_to_neighbour_median"] = float(np.median(np.abs(yy[nd] - other[nb])))
res["A_near_duplicates"] = A
print("A:", A, flush=True)

# ---------------- B. truncation rates ----------------
from transformers import AutoTokenizer
TOK = {"mBERT": "google-bert/bert-base-multilingual-cased", "KazRoBERTa": "kz-transformers/kaz-roberta-conversational",
       "XLM-R (base and large)": "FacebookAI/xlm-roberta-base"}
inputs = {"text": it.text.astype(str).tolist(), "cat_text": (it.cat_name.fillna("").astype(str) + " : " + it.text.astype(str)).tolist()}
Bt = {}
for nm, mid in TOK.items():
    tok = AutoTokenizer.from_pretrained(mid)
    Bt[nm] = {}
    for inp, texts in inputs.items():
        L = np.array([len(x) for x in tok(texts, truncation=False, add_special_tokens=True)["input_ids"]])
        Bt[nm][inp] = {"share_truncated_gt64": float(np.mean(L > 64)), "n_truncated": int((L > 64).sum()),
                       "median_tokens": float(np.median(L)), "p95_tokens": float(np.percentile(L, 95)), "max_tokens": int(L.max()),
                       "share_truncated_by_language": {lg: float(np.mean(L[(it.lang_det == lg).values] > 64)) for lg in ("kz", "ru", "en")}}
    Lt = np.array([len(x) for x in tok(tour.text.astype(str).tolist(), truncation=False)["input_ids"]])
    Bt[nm]["tournament_text"] = {"share_truncated_gt64": float(np.mean(Lt > 64)), "median_tokens": float(np.median(Lt))}
res["B_truncation"] = Bt
print("B:", json.dumps(Bt, indent=1), flush=True)

# ---------------- C. answer-option ablation with uncertainty ----------------
d = it[it.has_options == 1].reset_index(drop=True)
d["cat_id"] = d.cat_id.fillna(-1).astype(int)
for c in ("key_text", "distractors"):
    d[c] = d[c].fillna("").astype(str)
def opt_sim(row):
    k, ds = row.key_text, [x for x in row.distractors.split(" | ") if x]
    lens = [len(x) for x in ds] or [1]
    return [len(k) / (np.mean(lens) + 1), float(k.strip().isdigit()), float(np.mean([x.strip().isdigit() for x in ds] or [0])),
            float(len(k) == max([len(k)] + lens))]
d[["key_len_ratio", "key_numeric", "distr_numeric", "key_longest"]] = np.array([opt_sim(r) for r in d.itertuples()])
variants = {"stem": (["text"], False), "stem + key": (["text", "key_text"], False),
            "stem + key + distractors": (["text", "key_text", "distractors"], False),
            "stem + key + surface features": (["text", "key_text"], True),
            "stem + key + distractors + surface features": (["text", "key_text", "distractors"], True)}
oof, fold_r = {}, {}
folds = list(GroupKFold(n_splits=5).split(d, groups=d.group))
for name, (cols, surface) in variants.items():
    preds = np.zeros(len(d)); fr = []
    for tr, tt in folds:
        btr, bte = [], []
        for c in cols:
            v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
            btr.append(v.fit_transform(d.loc[tr, c])); bte.append(v.transform(d.loc[tt, c]))
        ohe = OneHotEncoder(handle_unknown="ignore").fit(d.loc[tr, ["cat_id", "lang_det"]])
        btr.append(ohe.transform(d.loc[tr, ["cat_id", "lang_det"]])); bte.append(ohe.transform(d.loc[tt, ["cat_id", "lang_det"]]))
        if surface:
            f = ["key_len_ratio", "key_numeric", "distr_numeric", "key_longest"]
            btr.append(csr_matrix(d.loc[tr, f].values)); bte.append(csr_matrix(d.loc[tt, f].values))
        mdl = RidgeCV(alphas=(0.3, 1, 3, 10, 30)).fit(hstack(btr).tocsr(), d.loc[tr, "b_logit"])
        preds[tt] = mdl.predict(hstack(bte).tocsr()); fr.append(pearsonr(d.b_logit.values[tt], preds[tt])[0])
    oof[name] = preds; fold_r[name] = fr
yo = d.b_logit.values; go = group_boot_idx(d.group.values, np.random.default_rng(2))
C = {"n_items": int(len(d)), "groups": int(d.group.nunique()), "variants": {}, "paired_differences": {}}
for name, p in oof.items():
    C["variants"][name] = {"r": float(pearsonr(yo, p)[0]), "r_CI": ci([pearsonr(yo[i], p[i])[0] for i in go]),
                           "RMSE": float(np.sqrt(np.mean((yo - p) ** 2))), "fold_r_mean": float(np.mean(fold_r[name])),
                           "fold_r_sd": float(np.std(fold_r[name], ddof=1))}
for a_, b_ in (("stem + key", "stem"), ("stem + key + distractors", "stem + key"), ("stem + key + surface features", "stem + key"),
               ("stem + key + distractors + surface features", "stem + key"), ("stem + key + distractors + surface features", "stem")):
    pa, pb = oof[a_], oof[b_]
    C["paired_differences"][f"{a_} minus {b_}"] = {
        "delta_r": float(pearsonr(yo, pa)[0] - pearsonr(yo, pb)[0]), "CI": ci([pearsonr(yo[i], pa[i])[0] - pearsonr(yo[i], pb[i])[0] for i in go]),
        "delta_RMSE": float(np.sqrt(np.mean((yo - pa) ** 2)) - np.sqrt(np.mean((yo - pb) ** 2))),
        "delta_RMSE_CI": ci([np.sqrt(np.mean((yo[i] - pa[i]) ** 2)) - np.sqrt(np.mean((yo[i] - pb[i]) ** 2)) for i in go])}
res["C_option_ablation"] = C
print("C:", json.dumps(C, indent=1), flush=True)

# ---------------- D. Table I ensembles and screening ----------------
NAMES = {"bert-base-multilingual-cased": "mBERT", "kaz-roberta-conversational": "KazRoBERTa",
         "xlm-roberta-base": "XLM-R base", "xlm-roberta-large": "XLM-R large"}
sig = lambda x: 1 / (1 + np.exp(x))
D = {"ensembles": {}}
for m, nm in NAMES.items():
    for inp in ("cat_text", "text"):
        p = ens(sorted(glob.glob(f"results/finetune/{m}__{inp}__s*__test.csv"))).loc[te.index].values
        D["ensembles"][f"{nm} | {inp}"] = {"RMSE_b": float(np.sqrt(np.mean((yy - p) ** 2))), "RMSE_p": float(np.sqrt(np.mean((te.p.values - sig(p)) ** 2))),
                                         "Pearson": float(pearsonr(yy, p)[0]), "Spearman": float(spearmanr(yy, p)[0]),
                                         "Pearson_CI_group": ci([pearsonr(yy[i], p[i])[0] for i in gidx[:1000]])}
def screen(t, p_, idxs):
    def stats(ix):
        tt, pp = t[ix], p_[ix]; kk = int(round(0.1 * len(ix)))
        ht = tt >= np.sort(tt)[-kk]; et = tt <= np.sort(tt)[kk - 1]
        hp = np.argsort(-pp)[:kk]; ep = np.argsort(pp)[:kk]; med = tt <= np.median(tt)
        return {"AUC_hardest_decile": roc_auc_score(ht, pp), "AUC_easiest_decile": roc_auc_score(et, -pp),
                "precision_at_10pct_hard": float(ht[hp].mean()), "precision_at_10pct_easy": float(et[ep].mean()),
                "predicted_hardest_decile_truly_in_easier_half": float(med[hp].mean()),
                "predicted_easiest_decile_truly_in_harder_half": float((~med)[ep].mean())}
    pt = stats(np.arange(len(t))); bt = [stats(i) for i in idxs]
    return {k: {"value": float(v), "CI": ci([b[k] for b in bt])} for k, v in pt.items()}
D["screening_test_set"] = {"selected XLM-R large ensemble": screen(yy, sel, gidx[:1000]),
                           "TF-IDF + metadata": screen(yy, bas, gidx[:1000])}
D["screening_difference_AUC_hardest"] = ci([roc_auc_score(yy[i] >= np.sort(yy[i])[-int(round(0.1 * len(i)))], sel[i]) -
                                            roc_auc_score(yy[i] >= np.sort(yy[i])[-int(round(0.1 * len(i)))], bas[i]) for i in gidx[:1000]])
# proportion-correct scale: share of items with |p_hat - p| < 0.10 and < 0.20
D["abs_error_p_scale"] = {nm: {"share_lt_0.10": float(np.mean(np.abs(te.p.values - sig(p)) < 0.10)),
                               "share_lt_0.20": float(np.mean(np.abs(te.p.values - sig(p)) < 0.20)),
                               "MAE_p": float(np.mean(np.abs(te.p.values - sig(p))))}
                          for nm, p in (("selected", sel), ("TF-IDF + metadata", bas), ("category mean", base["Category mean"].loc[te.index].values))}
res["D_table1_screening"] = D
print("D:", json.dumps(D, indent=1), flush=True)

# ---------------- E. cross-lingual comparison ----------------
idx_it = it.set_index("id")
pr = it[(it.is_translation == 1) & it.main_question_id.isin(idx_it.index)].reset_index(drop=True)
kz = idx_it.loc[pr.main_question_id]
dlt = pr.b_logit.values - kz.b_logit.values
def var_logit(c, n_):
    p = (c + 0.5) / (n_ + 1); return 1 / (n_ * p * (1 - p))
se = np.sqrt(var_logit(pr.correct.values, pr.n.values) + var_logit(kz.correct.values, kz.n.values))
cat = pr.cat_id.values; uc = np.unique(cat); cpos = {c: np.flatnonzero(cat == c) for c in uc}
rc = np.random.default_rng(4)
cidx = [np.concatenate([cpos[c] for c in rc.choice(uc, len(uc))]) for _ in range(B)]
xb, yb = kz.b_logit.values, pr.b_logit.values
E = {"pairs": int(len(dlt)), "categories": int(len(uc)),
     "pearson": float(pearsonr(xb, yb)[0]), "pearson_CI_category_cluster": ci([pearsonr(xb[i], yb[i])[0] for i in cidx]),
     "mean_delta_b": float(dlt.mean()), "mean_delta_b_CI_category_cluster": ci([dlt[i].mean() for i in cidx]),
     "share_abs_delta_gt_0.5": float(np.mean(np.abs(dlt) > 0.5)), "share_abs_delta_gt_0.5_CI_category_cluster": ci([np.mean(np.abs(dlt[i]) > 0.5) for i in cidx]),
     "share_ru_harder": float(np.mean(dlt > 0)), "categories_with_positive_mean_shift": int(sum(dlt[cpos[c]].mean() > 0 for c in uc)),
     "wald_design_effect_sensitivity": {}}
def holm_share(pv):
    m_ = len(pv); o = np.argsort(pv); h = np.zeros(m_, bool)
    for r_, j in enumerate(o):
        if pv[j] <= 0.05 / (m_ - r_):
            h[j] = True
        else:
            break
    return h
for deff in (1, 2, 5, 10, 20, 50):
    z = dlt / (se * np.sqrt(deff)); pv = 2 * norm.sf(np.abs(z)); h = holm_share(pv)
    E["wald_design_effect_sensitivity"][str(deff)] = {"share_holm_0.05": float(h.mean()), "share_holm_and_abs_gt_0.5": float((h & (np.abs(dlt) > 0.5)).mean())}
E["note"] = ("design effect = factor by which repeated attempts inflate the sampling variance of each logit; "
             "its true value is unknown because the counters do not record who answered")
res["E_crosslingual"] = E
print("E:", json.dumps(E, indent=1), flush=True)
json.dump(res, open(OUT + "part1_text.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
print("saved")
