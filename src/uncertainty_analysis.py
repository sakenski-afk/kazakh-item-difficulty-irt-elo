"""Uncertainty and overlap analyses.
  1. clean cold-start set: tournament items without exact / near duplicates (char TF-IDF cosine >= 0.8) or
     translation links to any item-bank item; Table V recomputed with 95% CIs and paired model differences
  2. item-bank test metrics with GROUP bootstrap (translation/duplicate groups), incl. language breakdown
  3. sensitivity of fine-tuning results to the diverged XLM-R large run
  4. Part 3: learner-clustered bootstrap of AUC differences (responses); game bootstrap (duels)
  5. cross-lingual Wald tests: standard error, Holm and Benjamini-Hochberg adjustment
  6. language heuristic vs the platform's own language label; sample for manual annotation
Outputs: results/extended/*.json|csv."""
import glob, json, os, re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, norm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

OUT = "results/extended/"; os.makedirs(OUT, exist_ok=True)
B = 2000
rng = np.random.default_rng(0)
it = pd.read_csv("data/items.csv.gz")
q_raw = pd.read_csv("data/raw/questions.csv.gz", low_memory=False)
tour = pd.read_csv("data/tournament_items.csv")
res = {}

def ens(files):
    return pd.concat([pd.read_csv(f).set_index("id").pred for f in files], axis=1).mean(axis=1)

def ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

# ---------------- 1. clean cold-start set ----------------
norm_txt = lambda s: re.sub(r"[^\w]+", " ", str(s).lower()).strip()
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(pd.concat([it.text, tour.text]))
sim = vec.transform(tour.text) @ vec.transform(it.text).T
best = np.asarray(sim.max(axis=1).todense()).ravel()
exact = tour.text.map(norm_txt).isin(set(it.text.map(norm_txt))).values
mq = q_raw.set_index("id").main_question_id
linked = tour.id.map(mq).isin(it.id).values | tour.id.isin(it.main_question_id.dropna().astype(int)).values
excl = (best >= 0.8) | exact | linked
tour["max_cosine_to_item_bank"] = best
tour["excluded"] = excl
tour[~excl][["id", "b_1PL", "b_2PL", "b_3PL", "p_value", "n_resp"]].to_csv(OUT + "coldstart_clean_items.csv", index=False)
res["coldstart_overlap"] = {"items_with_text": int(len(tour)), "exact_normalised_duplicates": int(exact.sum()),
                            "near_duplicates_cos_ge_0.8": int((best >= 0.8).sum()), "translation_links": int(linked.sum()),
                            "excluded_total": int(excl.sum()), "clean_items": int((~excl).sum()),
                            "tournament_categories": tour.cat_name.value_counts().to_dict(),
                            "note": "'Танымал тұлғалар' also exists as an item-bank category; 'Қазақстан тарихы' overlaps in subject with 'Ел тарихы'."}
SRC = {"TF-IDF + Ridge": ["results/tfidf__text__tour.csv"]}
for f in sorted(glob.glob("results/embeddings/*__text__tour.csv")):
    SRC[os.path.basename(f).split("__")[0] + " + Ridge"] = [f]
NAMES = {"kaz-roberta-conversational": "KazRoBERTa", "xlm-roberta-base": "XLM-R base",
         "xlm-roberta-large": "XLM-R large", "bert-base-multilingual-cased": "mBERT"}
for m, nm in NAMES.items():
    SRC[nm + " (fine-tuned)"] = sorted(glob.glob(f"results/finetune/{m}__text__s*__tour.csv"))
def table_v(sub, label):
    ids = sub.id.values; yb = sub.set_index("id").b_2PL.loc[ids].values
    P_ = {k: ens(f).loc[ids].values for k, f in SRC.items()}
    idx = [rng.integers(0, len(ids), len(ids)) for _ in range(B)]
    bt = {k: np.array([pearsonr(yb[i], p[i])[0] for i in idx]) for k, p in P_.items()}
    rows = {}
    for k, p in P_.items():
        row = {"n": int(len(ids))}
        for t in ("b_1PL", "b_2PL", "b_3PL"):
            row[f"r_{t}"] = float(pearsonr(sub.set_index("id")[t].loc[ids].values, p)[0])
        row["rho_b_2PL"] = float(spearmanr(yb, p)[0]); row["r_b_2PL_CI"] = ci(bt[k])
        rows[k] = row
    ref = max(rows, key=lambda k: rows[k]["r_b_2PL"])
    diffs = {f"{ref} minus {k}": {"delta_r": rows[ref]["r_b_2PL"] - rows[k]["r_b_2PL"], "CI": ci(bt[ref] - bt[k])}
             for k in P_ if k != ref}
    ft = [k for k in P_ if "fine-tuned" in k]
    diffs["mean of fine-tuned minus TF-IDF"] = {"delta_r": float(np.mean([rows[k]["r_b_2PL"] for k in ft]) - rows["TF-IDF + Ridge"]["r_b_2PL"]),
                                               "CI": ci(np.mean([bt[k] for k in ft], axis=0) - bt["TF-IDF + Ridge"])}
    return {"label": label, "models": rows, "highest_observed": ref, "paired_differences": diffs}
res["table_v_clean"] = table_v(tour[~excl], "clean cold-start items")
res["table_v_all_347"] = table_v(tour, "all tournament items with text (sensitivity analysis, duplicates included)")
print("Table V clean:", {k: round(v["r_b_2PL"], 3) for k, v in res["table_v_clean"]["models"].items()})

# ---------------- 2. item-bank test metrics with group bootstrap ----------------
te = it[it.split == "test"].set_index("id")
yy, grp, lang = te.b_logit.values, te.group.values, te.lang_det.values
ug = np.unique(grp); pos = {u: np.flatnonzero(grp == u) for u in ug}
gidx = [np.concatenate([pos[u] for u in rng.choice(ug, len(ug))]) for _ in range(B)]
base = pd.read_csv("results/part1_baseline_preds.csv").set_index("id")
MODELS = {f"baseline | {c}": base[c].loc[te.index].values for c in base.columns if c not in ("b_true", "Mean")}
for f in sorted(glob.glob("results/embeddings/*__test.csv")):
    n_, v_ = os.path.basename(f).split("__")[:2]
    MODELS[f"embeddings | {n_} | {v_}"] = pd.read_csv(f).set_index("id").pred.loc[te.index].values
for m, nm in NAMES.items():
    for inp in ("cat_text", "text"):
        MODELS[f"fine-tuned ensemble | {nm} | {inp}"] = ens(sorted(glob.glob(f"results/finetune/{m}__{inp}__s*__test.csv"))).loc[te.index].values
G = {}
for k, p in MODELS.items():
    bt = np.array([pearsonr(yy[i], p[i])[0] for i in gidx])
    G[k] = {"r": float(pearsonr(yy, p)[0]), "r_CI_group": ci(bt), "RMSE": float(np.sqrt(np.mean((yy - p) ** 2)))}
sel, bas = MODELS["fine-tuned ensemble | XLM-R large | cat_text"], MODELS["baseline | TF-IDF + metadata Ridge"]
d = np.array([pearsonr(yy[i], sel[i])[0] - pearsonr(yy[i], bas[i])[0] for i in gidx])
dr = np.array([np.sqrt(np.mean((yy[i] - bas[i]) ** 2)) - np.sqrt(np.mean((yy[i] - sel[i]) ** 2)) for i in gidx])
res["item_bank_group_bootstrap"] = {"groups": int(len(ug)), "items": int(len(yy)), "models": G,
    "selected_minus_tfidf_meta": {"delta_r": float(pearsonr(yy, sel)[0] - pearsonr(yy, bas)[0]), "CI": ci(d),
                                  "rmse_reduction": float(np.sqrt(np.mean((yy - bas) ** 2)) - np.sqrt(np.mean((yy - sel) ** 2))), "rmse_reduction_CI": ci(dr)}}
L = {}
for lg in ("kz", "ru", "en"):
    m_ = lang == lg; ul = np.unique(grp[m_]); pl = {u: np.flatnonzero((grp == u) & m_) for u in ul}
    li = [np.concatenate([pl[u] for u in rng.choice(ul, len(ul))]) for _ in range(B)]
    L[lg] = {"n": int(m_.sum()),
             "baseline_r": float(pearsonr(yy[m_], bas[m_])[0]), "baseline_CI": ci([pearsonr(yy[i], bas[i])[0] for i in li]),
             "selected_r": float(pearsonr(yy[m_], sel[m_])[0]), "selected_CI": ci([pearsonr(yy[i], sel[i])[0] for i in li]),
             "delta_CI": ci([pearsonr(yy[i], sel[i])[0] - pearsonr(yy[i], bas[i])[0] for i in li])}
res["item_bank_group_bootstrap"]["by_language"] = L
print("Group bootstrap delta r:", res["item_bank_group_bootstrap"]["selected_minus_tfidf_meta"])

# ---------------- 3. sensitivity to the diverged run ----------------
def seedset(files):
    per = [pearsonr(yy, pd.read_csv(f).set_index("id").pred.loc[te.index].values)[0] for f in files]
    return {"mean_r": float(np.mean(per)), "sd_r": float(np.std(per, ddof=1)), "ensemble_r": float(pearsonr(yy, ens(files).loc[te.index].values)[0])}
rep = sorted(glob.glob("results/finetune/xlm-roberta-large__text__s*__test.csv"))
div = sorted(glob.glob("results/finetune_diverged/xlm-roberta-large__text__s3__test.csv"))
s12 = [f for f in rep if "__s4__" not in f]
res["diverged_run_sensitivity"] = {"reported_seeds_1_2_4": seedset(rep), "seeds_1_2_3_including_diverged": seedset(s12 + div),
                                  "all_four_seeds": seedset(rep + div),
                                  "note": "The diverged run concerns the text-only XLM-R large configuration, which is not the selected model."}

# ---------------- 4. Part 3 uncertainty ----------------
r3 = pd.read_csv("results/part3_response_test_predictions.csv")
pl = r3.player.values; up = np.unique(pl); pp = {u: np.flatnonzero(pl == u) for u in up}
cidx = [np.concatenate([pp[u] for u in rng.choice(up, len(up))]) for _ in range(1000)]
def auc_diff(a, b):
    yv = r3.y.values; A_, B_ = r3[a].values, r3[b].values
    bt = [roc_auc_score(yv[i], A_[i]) - roc_auc_score(yv[i], B_[i]) for i in cidx]
    return {"delta_AUC": float(roc_auc_score(yv, A_) - roc_auc_score(yv, B_)), "CI_learner_cluster": ci(bt)}
res["part3_response_auc_differences"] = {"test_responses": int(len(r3)), "test_learners": int(len(up)),
    "elo_uc_minus_elo": auc_diff("elo_uc", "elo"), "elo_minus_player_mean": auc_diff("elo", "player_mean")}
g3 = pd.read_csv("results/part3_game_test_predictions.csv")
gi = [rng.integers(0, len(g3), len(g3)) for _ in range(1000)]
res["part3_game_auc_differences"] = {k: {"delta_AUC": float(roc_auc_score(g3.y, g3[a]) - roc_auc_score(g3.y, g3[b])),
    "CI_game_bootstrap": ci([roc_auc_score(g3.y.values[i], g3[a].values[i]) - roc_auc_score(g3.y.values[i], g3[b].values[i]) for i in gi])}
    for k, (a, b) in {"elo_minus_glicko2": ("elo", "glicko2"), "elo_minus_trueskill": ("elo", "trueskill"),
                      "lightgbm_minus_elo": ("lightgbm", "elo"), "elo_minus_logreg": ("elo", "logreg")}.items()}
print("Part 3:", res["part3_response_auc_differences"]["elo_uc_minus_elo"], res["part3_game_auc_differences"]["lightgbm_minus_elo"])

# ---------------- 4b. category-level link: leave-one-category-out ----------------
lk = pd.read_csv("results/table_elo_category_linkage.csv")
xb, ye = lk.bank_b.values, lk.elo_difficulty.values
loo = [pearsonr(np.delete(xb, i), np.delete(ye, i)) for i in range(len(xb))]
few = ~lk.cat_id.isin(lk.sort_values("n_resp").cat_id.head(2))   # two categories with fewest duel responses
res["category_link_sensitivity"] = {
    "n_categories": int(len(lk)), "r_all": float(pearsonr(xb, ye)[0]),
    "loo_r_range": [float(min(a[0] for a in loo)), float(max(a[0] for a in loo))],
    "loo_p_range": [float(min(a[1] for a in loo)), float(max(a[1] for a in loo))],
    "loo_min_dropped_category": int(lk.cat_id.values[int(np.argmin([a[0] for a in loo]))]),
    "without_two_low_response": {"n": int(few.sum()), "r": float(pearsonr(xb[few], ye[few])[0]), "p": float(pearsonr(xb[few], ye[few])[1]),
                                 "rho": float(spearmanr(xb[few], ye[few])[0]), "rho_p": float(spearmanr(xb[few], ye[few])[1])}}
print("Category link sensitivity:", res["category_link_sensitivity"])

# ---------------- 5. cross-lingual Wald tests ----------------
idx_it = it.set_index("id")
pr = it[(it.is_translation == 1) & it.main_question_id.isin(idx_it.index)]
kz = idx_it.loc[pr.main_question_id]
def var_logit(c, n):
    p = (c + 0.5) / (n + 1); return 1 / (n * p * (1 - p))
dlt = pr.b_logit.values - kz.b_logit.values
se = np.sqrt(var_logit(pr.correct.values, pr.n.values) + var_logit(kz.correct.values, kz.n.values))
pv = 2 * norm.sf(np.abs(dlt / se)); m = len(pv); o = np.argsort(pv)
holm = np.zeros(m, bool); k_ = 0
for r_, j in enumerate(o):
    if pv[j] <= 0.05 / (m - r_):
        holm[j] = True
    else:
        break
bh_thr = pv[o] <= 0.05 * np.arange(1, m + 1) / m
bh = np.zeros(m, bool); bh[o[: (np.max(np.flatnonzero(bh_thr)) + 1) if bh_thr.any() else 0]] = True
res["crosslingual_wald"] = {"pairs": int(m), "se_formula": "SE(delta b) = sqrt(1/(n_ru p_ru (1-p_ru)) + 1/(n_kz p_kz (1-p_kz)))",
    "share_abs_z_gt_3.29": float(np.mean(np.abs(dlt / se) > 3.29)), "share_holm_0.05": float(holm.mean()),
    "share_bh_q0.05": float(bh.mean()), "share_abs_delta_gt_0.5": float(np.mean(np.abs(dlt) > 0.5)),
    "share_abs_delta_gt_0.5_and_holm": float(np.mean((np.abs(dlt) > 0.5) & holm)),
    "median_se": float(np.median(se))}
print("Wald:", {k: round(v, 3) if isinstance(v, float) else v for k, v in res["crosslingual_wald"].items()})

# ---------------- 6. language heuristic ----------------
lab = q_raw.set_index("id").lang
it["platform_lang"] = it.id.map(lab)
ru_lab = it[it.platform_lang == "ru"]
res["language_heuristic"] = {"items_with_platform_label_ru": int(len(ru_lab)),
    "heuristic_agrees_ru": float((ru_lab.lang_det == "ru").mean()),
    "heuristic_distribution_on_platform_ru": ru_lab.lang_det.value_counts().to_dict(),
    "note": "The platform labels only Russian translations; Kazakh and Latin-script labels require manual annotation."}
samp = pd.concat([it[it.lang_det == l].sample(n, random_state=7) for l, n in (("kz", 100), ("ru", 70), ("en", 30))])
os.makedirs("../private_language_check", exist_ok=True)
samp[["id", "text", "lang_det"]].assign(manual_label="").sample(frac=1, random_state=8).to_csv(
    "../private_language_check/language_annotation_sample.csv", index=False, encoding="utf-8-sig")
print("Language:", res["language_heuristic"])
json.dump(res, open(OUT + "uncertainty.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
print("saved", OUT + "uncertainty.json")
