"""Aggregate results for the paper: Part 1 model table (mean +- sd over seeds, bootstrap CI), per-language results,
cross-lingual difficulty (kz original vs ru translation), cold-start validation against IRT (Part 1 -> Part 2),
category-level linkage with Elo (Part 1 <-> Part 3), and figures."""
import glob, json, os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results/figures", exist_ok=True)
rng = np.random.default_rng(0)
items = pd.read_csv("data/items.csv.gz")
test = items[items.split == "test"].set_index("id")
tour = pd.read_csv("data/tournament_items.csv").set_index("id")
out = {}

def boot_ci(y, p, B=2000):
    n = len(y); rs = []
    for _ in range(B):
        i = rng.integers(0, n, n); rs.append(pearsonr(y[i], p[i])[0])
    return np.percentile(rs, [2.5, 97.5])

# ---------- Part 1 main table ----------
rows = []
base = json.load(open("results/part1_baselines.json"))
bpred = pd.read_csv("results/part1_baseline_preds.csv").set_index("id")
for k, v in base.items():
    rows.append({"family": "Baseline", "model": k, "input": "-", "seeds": 1, **{m: v[m] for m in ("RMSE_b", "MAE_b", "RMSE_p", "Pearson", "Spearman")}})
emb = json.load(open("results/part1_embeddings.json"))
for k, v in emb.items():
    rows.append({"family": "Embeddings + Ridge", "model": k.split(" + ")[0], "input": k.split("(")[1].rstrip(")"),
                 "seeds": 1, **{m: v[m] for m in ("RMSE_b", "MAE_b", "RMSE_p", "Pearson", "Spearman")}})
ft = [json.load(open(f)) for f in glob.glob("results/finetune/*.json")]
ftdf = pd.DataFrame([{"model": d["tag"].split("__")[0], "input": d["tag"].split("__")[1], "seed": d["args"]["seed"], **d["test"]} for d in ft])
ens_preds = {}
if len(ftdf):
    for (mdl, inp), g in ftdf.groupby(["model", "input"]):
        r = {"family": "Fine-tuned", "model": mdl, "input": inp, "seeds": len(g)}
        for m in ("RMSE_b", "MAE_b", "RMSE_p", "Pearson", "Spearman"):
            r[m] = g[m].mean(); r[m + "_sd"] = g[m].std() if len(g) > 1 else np.nan
        rows.append(r)
        # seed ensemble
        ps = [pd.read_csv(f).set_index("id").pred for f in glob.glob(f"results/finetune/{mdl}__{inp}__s*__test.csv")]
        ens_preds[(mdl, inp)] = pd.concat(ps, axis=1).mean(1)
table = pd.DataFrame(rows)
table.to_csv("results/table_part1_models.csv", index=False)
print(table.round(4).to_string())

# bootstrap CI for best baseline vs best fine-tuned (seed ensemble)
y = test.b_logit
best_base = "TF-IDF + metadata Ridge"
ci_rows = [{"model": best_base, "Pearson": pearsonr(y, bpred.loc[y.index, best_base])[0],
            "CI95": boot_ci(y.values, bpred.loc[y.index, best_base].values).tolist()}]
for (mdl, inp), p in ens_preds.items():
    p = p.loc[y.index]
    ci_rows.append({"model": f"{mdl} ({inp}, seed-ensemble)", "Pearson": pearsonr(y, p)[0], "CI95": boot_ci(y.values, p.values).tolist(),
                    "RMSE_b": float(np.sqrt(np.mean((y - p) ** 2)))})
out["part1_ci"] = ci_rows
if ens_preds:
    # model selection on the validation set only (mean validation RMSE over seeds)
    val_rmse = pd.DataFrame([{"k": tuple(d["tag"].split("__")[:2]), "v": d["val_rmse"]} for d in ft]).groupby("k").v.mean()
    best_ft = val_rmse.idxmin()
    out["part1_selection_by_val_rmse"] = {f"{k[0]} ({k[1]})": float(v) for k, v in val_rmse.sort_values().items()}
    pb, pf = bpred.loc[y.index, best_base].values, ens_preds[best_ft].loc[y.index].values
    # paired bootstrap of Pearson difference
    diffs = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y)); diffs.append(pearsonr(y.values[i], pf[i])[0] - pearsonr(y.values[i], pb[i])[0])
    out["part1_best_vs_baseline"] = {"best_finetuned": f"{best_ft[0]} ({best_ft[1]})", "delta_r": float(np.mean(diffs)),
                                     "CI95": np.percentile(diffs, [2.5, 97.5]).tolist(),
                                     "abs_err_wilcoxon_p": float(wilcoxon(np.abs(y.values - pf), np.abs(y.values - pb)).pvalue)}
    print("best vs baseline", out["part1_best_vs_baseline"])
    # per-language
    lang = []
    for lg, idx in test.groupby("lang_det").groups.items():
        lang.append({"lang": lg, "n": len(idx), "baseline_r": pearsonr(y[idx], bpred.loc[idx, best_base])[0],
                     "finetuned_r": pearsonr(y[idx], ens_preds[best_ft].loc[idx])[0],
                     "baseline_RMSE": float(np.sqrt(np.mean((y[idx] - bpred.loc[idx, best_base]) ** 2))),
                     "finetuned_RMSE": float(np.sqrt(np.mean((y[idx] - ens_preds[best_ft].loc[idx]) ** 2)))})
    out["part1_by_language"] = lang
    print(pd.DataFrame(lang).round(4).to_string())

# ---------- cross-lingual difficulty: kz original vs ru translation ----------
it = items.set_index("id")
pairs = items[items.is_translation == 1][["id", "main_question_id", "b_logit", "p", "n", "cat_name"]].rename(
    columns={"id": "ru_id", "b_logit": "b_ru", "p": "p_ru", "n": "n_ru"})
pairs = pairs[pairs.main_question_id.isin(it.index)]
pairs["b_kz"] = it.loc[pairs.main_question_id, "b_logit"].values
pairs["p_kz"] = it.loc[pairs.main_question_id, "p"].values
pairs["n_kz"] = it.loc[pairs.main_question_id, "n"].values
d = pairs.b_ru - pairs.b_kz
se = np.sqrt(1 / (pairs.p_ru * (1 - pairs.p_ru) * pairs.n_ru).clip(lower=1e-9) + 1 / (pairs.p_kz * (1 - pairs.p_kz) * pairs.n_kz).clip(lower=1e-9))
z = d / se
out["cross_lingual"] = {"pairs": int(len(pairs)), "pearson_b": float(pearsonr(pairs.b_kz, pairs.b_ru)[0]),
                        "spearman_b": float(spearmanr(pairs.b_kz, pairs.b_ru)[0]),
                        "mean_p_kz": float(pairs.p_kz.mean()), "mean_p_ru": float(pairs.p_ru.mean()),
                        "mean_delta_b_ru_minus_kz": float(d.mean()), "paired_t_p": float(ttest_rel(pairs.b_ru, pairs.b_kz).pvalue),
                        "share_abs_delta_gt_0.5": float((d.abs() > 0.5).mean()),
                        "share_significant_DIF_|z|>3.29": float((z.abs() > 3.29).mean())}
by_cat = pairs.assign(d=d).groupby("cat_name").d.agg(["mean", "size"]).query("size >= 50").sort_values("mean")
out["cross_lingual_by_category"] = by_cat.round(3).reset_index().to_dict("records")
print("cross-lingual", out["cross_lingual"])
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.scatter(pairs.b_kz, pairs.b_ru, s=4, alpha=0.25, color="#1f5f8b")
lim = [-4, 4]; ax.plot(lim, lim, color="#999", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Difficulty, Kazakh original (logit)"); ax.set_ylabel("Difficulty, Russian translation (logit)")
ax.set_title(f"n = {len(pairs)}, r = {out['cross_lingual']['pearson_b']:.2f}", fontsize=10)
fig.tight_layout(); fig.savefig("results/figures/fig_crosslingual.png", dpi=200); plt.close(fig)

# ---------- cold-start validation: text-only predictions vs IRT on tournament items ----------
cs = []
cands = {"TF-IDF + Ridge": ["results/tfidf__text__tour.csv"]}
for f in glob.glob("results/finetune/*__text__s*__tour.csv"):
    mdl = os.path.basename(f).split("__")[0]
    cands.setdefault(mdl, []).append(f)
for f in glob.glob("results/embeddings/*__text__tour.csv"):
    cands[os.path.basename(f).split("__")[0] + " + Ridge"] = [f]
for name, files in cands.items():
    if not files:
        continue
    p = pd.concat([pd.read_csv(f).set_index("id").pred for f in files], axis=1).mean(1).loc[tour.index]
    row = {"model": name, "n_items": len(tour)}
    for tgt in ("b_1PL", "b_2PL", "b_3PL"):
        row[f"r_{tgt}"] = pearsonr(p, tour[tgt])[0]; row[f"rho_{tgt}"] = spearmanr(p, tour[tgt])[0]
    row["r_logit_pvalue"] = pearsonr(p, -np.log(tour.p_value.clip(.01, .99) / (1 - tour.p_value.clip(.01, .99))))[0]
    row["p_value_of_r_b2PL"] = pearsonr(p, tour.b_2PL)[1]
    cs.append(row)
cs = pd.DataFrame(cs).sort_values("r_b_2PL", ascending=False)
cs.to_csv("results/table_coldstart_irt.csv", index=False)
out["coldstart_irt"] = cs.round(4).to_dict("records")
print(cs.round(3).to_string())

# ---------- category-level linkage with Elo ----------
cd = pd.read_csv("results/part3_category_difficulty.csv")
agg = items.groupby("cat_id").agg(bank_b=("b_logit", "mean"), n_items=("id", "size"), cat_name=("cat_name", "first"))
link = cd.merge(agg, left_on="cat_id", right_index=True, how="inner")
if ens_preds:
    allp = pd.concat([pd.read_csv(f).set_index("id").pred for f in glob.glob(f"results/finetune/{best_ft[0]}__{best_ft[1]}__s*__test.csv")], axis=1).mean(1)
    tp = test.assign(pred=allp.loc[test.index]).groupby("cat_id").pred.mean()
    link["model_b_test"] = link.cat_id.map(tp)
out["elo_linkage"] = {"n_categories": int(len(link)),
                      "pearson_elo_vs_bank": float(pearsonr(link.elo_difficulty, link.bank_b)[0]),
                      "pearson_p": float(pearsonr(link.elo_difficulty, link.bank_b)[1]),
                      "spearman_elo_vs_bank": float(spearmanr(link.elo_difficulty, link.bank_b)[0]),
                      "spearman_p": float(spearmanr(link.elo_difficulty, link.bank_b)[1])}
if "model_b_test" in link:
    l2 = link.dropna(subset=["model_b_test"])
    out["elo_linkage"].update({"pearson_elo_vs_model": float(pearsonr(l2.elo_difficulty, l2.model_b_test)[0]),
                               "pearson_elo_vs_model_p": float(pearsonr(l2.elo_difficulty, l2.model_b_test)[1])})
link.to_csv("results/table_elo_category_linkage.csv", index=False)
print("elo linkage", out["elo_linkage"])
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.scatter(link.bank_b, link.elo_difficulty, color="#1f5f8b")
for r in link.itertuples():
    ax.annotate(str(int(r.cat_id)), (r.bank_b, r.elo_difficulty), fontsize=7, xytext=(3, 2), textcoords="offset points")
ax.set_xlabel("Mean item difficulty in bank (logit)"); ax.set_ylabel("Elo category difficulty")
ax.set_title(f"{len(link)} categories, r = {out['elo_linkage']['pearson_elo_vs_bank']:.2f}", fontsize=10)
fig.tight_layout(); fig.savefig("results/figures/fig_elo_categories.png", dpi=200); plt.close(fig)

# ---------- difficulty distribution figure ----------
fig, ax = plt.subplots(figsize=(5, 3))
ax.hist(items.p, bins=40, color="#1f5f8b"); ax.set_xlabel("Proportion correct"); ax.set_ylabel("Items")
fig.tight_layout(); fig.savefig("results/figures/fig_difficulty_hist.png", dpi=200); plt.close(fig)

# ---------- IRT figure: model comparison ----------
irt = json.load(open("results/part2_irt.json"))
out["irt"] = irt
if len(cs):
    top = cs.iloc[0]
    files = cands[top.model]
    p = pd.concat([pd.read_csv(f).set_index("id").pred for f in files], axis=1).mean(1).loc[tour.index]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(p, tour.b_2PL, s=10, alpha=0.6, color="#1f5f8b")
    ax.set_xlabel(f"Predicted difficulty from text ({top.model})"); ax.set_ylabel("IRT 2PL difficulty b")
    ax.set_title(f"Cold-start items n = {len(tour)}, r = {top.r_b_2PL:.2f}", fontsize=10)
    fig.tight_layout(); fig.savefig("results/figures/fig_coldstart_irt.png", dpi=200); plt.close(fig)

json.dump(out, open("results/summary.json", "w"), indent=2, default=float)
print("saved results/summary.json")
