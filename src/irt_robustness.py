"""Robustness analyses and calibration experiment for Part 2.
  A. new-learner hold-out: item parameters from 80% of learners; for the other 20%, ability from half of their
     responses, prediction of the other half
  B. maximum-likelihood (no prior) 1PL/2PL fits, likelihood-ratio test, AIC/BIC with N = learners and N = responses
  C. standard errors of 2PL discriminations (observed information) and heterogeneity beyond estimation noise
  D. sensitivity of the 3PL to the prior on the pseudo-guessing parameter
  E. item fit (infit / outfit mean squares) and local dependence (Yen's Q3) for the reported 2PL model
  F. calibration experiment: difficulty of a new item estimated from k responses with (i) a weak prior,
     (ii) a population prior, (iii) a prior from the text model; RMSE against the all-response estimate
Outputs: results/extended/irt_robustness.json, results/extended/calibration_experiment.json, figures."""
import glob, json, os
import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

torch.manual_seed(0)
RAW, OUT = "data/raw/", "results/extended/"
os.makedirs(OUT, exist_ok=True)
MIN_ITEM, MIN_PERSON = 30, 10

# ---------------- responses (identical preparation to part2_irt.py) ----------------
gq = pd.read_csv(RAW + "group_questions.csv.gz").dropna(subset=["question_id"])
ca = pd.read_csv(RAW + "correct_answers.csv.gz")
keys = ca.groupby("question_id").answer_id.apply(set)
gq = gq[gq.question_id.isin(keys.index)].copy()
gq["created_at"] = pd.to_datetime(gq.created_at)
gq = gq.sort_values("created_at").drop_duplicates(["user_id", "question_id"], keep="first")
gq["y"] = [int(pd.notna(a) and a in keys[q]) for q, a in zip(gq.question_id, gq.answer_id)]
for _ in range(3):
    gq = gq[gq.groupby("question_id").user_id.transform("size") >= MIN_ITEM]
    gq = gq[gq.groupby("user_id").question_id.transform("size") >= MIN_PERSON]
persons = np.sort(gq.user_id.unique()); items = np.sort(gq.question_id.unique())
pi = np.searchsorted(persons, gq.user_id.values); ii = np.searchsorted(items, gq.question_id.values)
y = gq.y.values.astype(np.float64)
P, I, N = len(persons), len(items), len(y)
nodes_np, w = np.polynomial.hermite_e.hermegauss(41)
nodes = torch.tensor(nodes_np, dtype=torch.float64); logw = torch.log(torch.tensor(w / w.sum(), dtype=torch.float64))

def build(mask):
    R = torch.zeros(P, I, dtype=torch.float64); M = torch.zeros(P, I, dtype=torch.float64)
    R[pi[mask], ii[mask]] = torch.tensor(y[mask]); M[pi[mask], ii[mask]] = 1.0
    return R, M

def probs(prm, model):
    a = torch.exp(prm["la"]).expand(I) if model == "1PL" else torch.exp(prm["la"])
    p = torch.sigmoid(a[None, :] * (nodes[:, None] - prm["b"][None, :]))
    if model == "3PL":
        c = torch.sigmoid(prm["lc"]); p = c[None, :] + (1 - c[None, :]) * p
    return p.clamp(1e-9, 1 - 1e-9)

def logprior(prm, model, use_prior, cprior):
    if not use_prior:
        return torch.zeros((), dtype=torch.float64)
    lp = -0.5 * (prm["b"] / 3.0).pow(2).sum()
    if model != "1PL":
        lp = lp - 0.5 * (prm["la"] / 0.5).pow(2).sum()
    if model == "3PL":
        c = torch.sigmoid(prm["lc"]); a0, b0 = cprior
        lp = lp + ((a0 - 1) * torch.log(c) + (b0 - 1) * torch.log(1 - c)).sum()
    return lp

def pll(prm, model, R, M):
    p = probs(prm, model)
    return R @ torch.log(p).T + (M - R) @ torch.log(1 - p).T  # P x Q

def mll(prm, model, R, M):
    return torch.logsumexp(pll(prm, model, R, M) + logw[None, :], dim=1).sum()

def fit(model, R, M, use_prior=True, cprior=(5, 17)):
    prm = {"b": torch.zeros(I, dtype=torch.float64, requires_grad=True),
           "la": torch.zeros(1 if model == "1PL" else I, dtype=torch.float64, requires_grad=True)}
    if model == "3PL":
        prm["lc"] = torch.full((I,), -1.2, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS(list(prm.values()), lr=1, max_iter=500, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-8, tolerance_change=1e-12)
    def closure():
        opt.zero_grad()
        loss = -(mll(prm, model, R, M) + logprior(prm, model, use_prior, cprior)); loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad():
        ll = mll(prm, model, R, M).item()
    return prm, ll, sum(v.numel() for v in prm.values())

def posterior(prm, model, R, M):
    with torch.no_grad():
        return torch.softmax(pll(prm, model, R, M) + logw[None, :], dim=1)

def pred(prm, model, post, pp, iii):
    with torch.no_grad():
        return (post[pp] * probs(prm, model)[:, iii].T).sum(1).numpy()

def sc(yt, pr):
    pr = np.clip(pr, 1e-6, 1 - 1e-6)
    return {"AUC": roc_auc_score(yt, pr), "LogLoss": log_loss(yt, pr), "Brier": brier_score_loss(yt, pr)}

res = {"data": {"responses": int(N), "learners": int(P), "items": int(I)}}
rng = np.random.default_rng(0)

# ---------------- A. new-learner hold-out ----------------
test_persons = rng.random(P) < 0.20
tp = test_persons[pi]
half = rng.random(N) < 0.5
obs, tgt = tp & half, tp & ~half           # test learners: observed half -> ability, target half -> evaluation
R_tr, M_tr = build(~tp); R_obs, M_obs = build(obs)
A = {}
itm = pd.Series(y[~tp]).groupby(ii[~tp]).mean()
b_item = itm.reindex(ii[tgt]).values
pm = pd.Series(y[obs]).groupby(pi[obs]).mean().reindex(pi[tgt]).fillna(y[~tp].mean()).values
lg = lambda v: np.log(np.clip(v, 1e-3, 1 - 1e-3) / (1 - np.clip(v, 1e-3, 1 - 1e-3)))
A["Item p-value"] = sc(y[tgt], b_item)
A["Item + person logit (additive)"] = sc(y[tgt], 1 / (1 + np.exp(-(lg(b_item) + lg(pm) - lg(y[~tp].mean())))))
for model in ("1PL", "2PL", "3PL"):
    prm, _, _ = fit(model, R_tr, M_tr)
    A[model] = sc(y[tgt], pred(prm, model, posterior(prm, model, R_obs, M_obs), pi[tgt], ii[tgt]))
res["A_new_learner_holdout"] = {"test_learners": int(test_persons.sum()), "target_responses": int(tgt.sum()),
                                "observed_responses": int(obs.sum()), "results": A}
print("A new learners:", {k: round(v["AUC"], 4) for k, v in A.items()})

# ---------------- B. maximum likelihood, LR test, information criteria ----------------
R, M = build(np.ones(N, bool))
B = {}
fits_ml = {}
for model in ("1PL", "2PL"):
    prm, ll, k = fit(model, R, M, use_prior=False)
    fits_ml[model] = prm
    B[model + " (ML)"] = {"logLik": ll, "k": k, "AIC": -2 * ll + 2 * k, "BIC_N_learners": -2 * ll + k * np.log(P),
                         "BIC_N_responses": -2 * ll + k * np.log(N)}
fits_map = {}
for model in ("1PL", "2PL", "3PL"):
    prm, ll, k = fit(model, R, M, use_prior=True)
    fits_map[model] = prm
    B[model + " (MAP, log-likelihood at MAP estimate)"] = {"logLik": ll, "k": k, "AIC": -2 * ll + 2 * k,
        "BIC_N_learners": -2 * ll + k * np.log(P), "BIC_N_responses": -2 * ll + k * np.log(N)}
lr = 2 * (B["2PL (ML)"]["logLik"] - B["1PL (ML)"]["logLik"]); df = B["2PL (ML)"]["k"] - B["1PL (ML)"]["k"]
B["LR_test_1PL_vs_2PL_ML"] = {"LR": lr, "df": df, "p": float(chi2.sf(lr, df)),
                               "caveat": "ML 2PL has non-finite discriminations for some items (see C); LR test not reported in the paper"}
res["B_information_criteria"] = B
print("B LR test:", B["LR_test_1PL_vs_2PL_ML"])

# ---------------- C. uncertainty of 2PL discriminations ----------------
# ML estimation of the 2PL does not yield finite discriminations for all items; count them
a_ml = torch.exp(fits_ml["2PL"]["la"]).detach().numpy()
C = {"ML_2PL_items_a_below_0.05_or_above_20": int(np.sum((a_ml < 0.05) | (a_ml > 20))),
     "ML_2PL_a_range": [float(a_ml.min()), float(a_ml.max())]}
# posterior standard errors at the reported MAP 2PL estimate (curvature of the negative log posterior)
prm_m = fits_map["2PL"]
b_m, la_m = prm_m["b"].detach(), prm_m["la"].detach()
def neglp(v):
    q = {"b": v[:I], "la": v[I:]}
    return -(mll(q, "2PL", R, M) + logprior(q, "2PL", True, (5, 17)))
H = torch.autograd.functional.hessian(neglp, torch.cat([b_m, la_m]))
cov = torch.linalg.inv(H)
se_la = torch.sqrt(torch.clamp(torch.diag(cov)[I:], min=0)).numpy()
la = la_m.numpy(); a1 = float(torch.exp(fits_map["1PL"]["la"]).item())
lo, hi = np.exp(la - 1.96 * se_la), np.exp(la + 1.96 * se_la)
C.update({"common_a_1PL_MAP": a1, "a_2PL_MAP_range": [float(np.exp(la).min()), float(np.exp(la).max())],
          "median_posterior_SE_log_a": float(np.median(se_la)),
          "n_items_95CI_excludes_common_a": int(np.sum((lo > a1) | (hi < a1))),
          "share_items_95CI_excludes_common_a": float(np.mean((lo > a1) | (hi < a1))),
          "n_items_a_below_0.5": int(np.sum(np.exp(la) < 0.5)),
          "n_items_upper_95CI_below_0.5": int(np.sum(hi < 0.5)),
          "sd_log_a": float(np.std(la)), "rms_posterior_SE_log_a": float(np.sqrt(np.mean(se_la ** 2)))})
res["C_discrimination_uncertainty"] = C
print("C:", C)

# ---------------- D. 3PL prior sensitivity ----------------
split = np.random.default_rng(0).random(N) < 0.10   # the same 10% hold-out as part2_irt.py
R_h, M_h = build(~split)
D = {}
for name, cp in {"Beta(5,17) [reported]": (5, 17), "Beta(2.5,7.5)": (2.5, 7.5), "Beta(1,3)": (1, 3),
                 "Beta(2,18)": (2, 18)}.items():
    prm, ll, _ = fit("3PL", R, M, cprior=cp)
    c = torch.sigmoid(prm["lc"]).detach().numpy()
    prm_h, _, _ = fit("3PL", R_h, M_h, cprior=cp)
    s = sc(y[split], pred(prm_h, "3PL", posterior(prm_h, "3PL", R_h, M_h), pi[split], ii[split]))
    D[name] = {"prior_mean": cp[0] / sum(cp), "mean_c": float(c.mean()), "sd_c": float(c.std()),
               "logLik": ll, "heldout_AUC": s["AUC"], "heldout_LogLoss": s["LogLoss"]}
res["D_3PL_prior_sensitivity"] = D
print("D:", {k: (round(v["mean_c"], 3), round(v["heldout_AUC"], 4)) for k, v in D.items()})

# ---------------- E. item fit and local dependence (reported 2PL, MAP) ----------------
prm2 = fits_map["2PL"]
post = posterior(prm2, "2PL", R, M)
theta = (post * nodes[None, :]).sum(1).numpy()
a2 = torch.exp(prm2["la"]).detach().numpy(); b2 = prm2["b"].detach().numpy()
Pexp = 1 / (1 + np.exp(-a2[ii] * (theta[pi] - b2[ii])))
resid = y - Pexp; var = Pexp * (1 - Pexp)
df_ = pd.DataFrame({"i": ii, "p": pi, "r": resid, "v": var, "z2": resid ** 2 / var})
infit = df_.groupby("i").apply(lambda d: (d.r ** 2).sum() / d.v.sum(), include_groups=False)
outfit = df_.groupby("i").z2.mean()
E = {"infit_share_in_0.7_1.3": float(infit.between(0.7, 1.3).mean()),
     "outfit_share_in_0.7_1.3": float(outfit.between(0.7, 1.3).mean()),
     "infit_median": float(infit.median()), "outfit_median": float(outfit.median())}
# Yen's Q3 for item pairs with >= 50 common learners
Rm = np.full((P, I), np.nan); Rm[pi, ii] = resid
cats = pd.read_csv(RAW + "questions.csv.gz", low_memory=False, usecols=["id", "cat_id"]).set_index("id").cat_id
icat = pd.Series(items).map(cats).values
q3, same = [], []
obsmask = ~np.isnan(Rm)
common = obsmask.T.astype(float) @ obsmask.astype(float)
for a_ in range(I):
    for b_ in range(a_ + 1, I):
        if common[a_, b_] >= 50:
            m_ = obsmask[:, a_] & obsmask[:, b_]
            q3.append(np.corrcoef(Rm[m_, a_], Rm[m_, b_])[0, 1]); same.append(icat[a_] == icat[b_])
q3 = np.array(q3); same = np.array(same); q3s = q3 - q3.mean()
E.update({"Q3_pairs": int(len(q3)), "Q3_mean": float(q3.mean()), "Q3star_share_abs_gt_0.2": float(np.mean(np.abs(q3s) > 0.2)),
          "Q3_mean_same_category": float(q3[same].mean()) if same.any() else None,
          "Q3_mean_different_category": float(q3[~same].mean()) if (~same).any() else None,
          "pairs_same_category": int(same.sum())})
res["E_item_fit_local_dependence"] = E
print("E:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in E.items()})
json.dump(res, open(OUT + "irt_robustness.json", "w"), indent=2, default=float)

# ---------------- F. calibration experiment ----------------
tour = pd.read_csv("results/extended/coldstart_clean_items.csv")          # from uncertainty_analysis.py
clean_ids = set(tour.id)
LL = pll(prm2, "2PL", R, M).detach().numpy()                             # P x Q
p2 = probs(prm2, "2PL").detach().numpy()                                 # Q x I
Rn, Mn = R.numpy(), M.numpy()
TEXT = {"XLM-R large (pre-specified)": "xlm-roberta-large", "KazRoBERTa": "kaz-roberta-conversational"}
preds = {k: pd.concat([pd.read_csv(f).set_index("id").pred for f in sorted(glob.glob(f"results/finetune/{m}__text__s*__tour.csv"))],
                      axis=1).mean(axis=1) for k, m in TEXT.items()}
preds["TF-IDF"] = pd.read_csv("results/tfidf__text__tour.csv").set_index("id").pred
KS = [5, 10, 20, 40, 60, 80]; DRAWS = 200

def est_b(th, yy, mu, s2):
    """MAP of b for a new item with a = 1 fixed and abilities given; damped Newton, vectorised over rows."""
    b = np.full(th.shape[0], float(mu)) if np.isscalar(mu) else np.asarray(mu, dtype=float).copy()
    for _ in range(60):
        pz = 1 / (1 + np.exp(-(th - b[:, None])))
        g = (yy - pz).sum(1) + (b - mu) / s2          # gradient of the negative log posterior
        h = (pz * (1 - pz)).sum(1) + 1 / s2
        b = b - np.clip(g / h, -2.0, 2.0)
    return b

item_data = {}
for j, qid in enumerate(items):
    if qid not in clean_ids:
        continue
    resp = np.flatnonzero(Mn[:, j] > 0)
    llj = LL - (Rn[:, [j]] * np.log(p2[:, j])[None, :] + (Mn[:, [j]] - Rn[:, [j]]) * np.log(1 - p2[:, j])[None, :])
    postj = np.exp(llj[resp] + logw.numpy()[None, :] - (llj[resp] + logw.numpy()[None, :]).max(1, keepdims=True))
    postj /= postj.sum(1, keepdims=True)
    th = postj @ nodes_np                                                 # leave-item-out EAP abilities
    yy = Rn[resp, j]
    truth = est_b(th[None, :], yy[None, :], 0.0, 9.0)[0]
    item_data[qid] = (th, yy, truth)
qids = np.array(sorted(item_data)); truth = np.array([item_data[q][2] for q in qids])
folds = np.random.default_rng(1).integers(0, 5, len(qids))
priors = {"weak N(0,3^2)": (np.zeros(len(qids)), np.full(len(qids), 9.0))}
mu_pop = np.zeros(len(qids)); s2_pop = np.zeros(len(qids))
for f in range(5):
    trn = folds != f
    mu_pop[~trn] = truth[trn].mean(); s2_pop[~trn] = truth[trn].var()
priors["population N(mean, var)"] = (mu_pop, s2_pop)
for name, pser in preds.items():
    x = pser.loc[qids].values; mu = np.zeros(len(qids)); s2 = np.zeros(len(qids))
    for f in range(5):
        trn = folds != f
        coef = np.polyfit(x[trn], truth[trn], 1)
        mu[~trn] = np.polyval(coef, x[~trn]); s2[~trn] = np.var(truth[trn] - np.polyval(coef, x[trn]))
    priors[f"text: {name}"] = (mu, s2)

rngc = np.random.default_rng(2)
sq = {pn: {k: np.zeros(len(qids)) for k in [0] + KS} for pn in priors}
for n_, q in enumerate(qids):
    th, yy, _ = item_data[q]
    for k in KS:
        idx = np.array([rngc.choice(len(yy), k, replace=False) for _ in range(DRAWS)])
        TH, YY = th[idx], yy[idx]
        for pn, (mu, s2) in priors.items():
            bh = est_b(TH, YY, np.full(DRAWS, mu[n_]), s2[n_])
            sq[pn][k][n_] = np.mean((bh - truth[n_]) ** 2)
    for pn, (mu, s2) in priors.items():
        sq[pn][0][n_] = (mu[n_] - truth[n_]) ** 2
boot_items = [np.random.default_rng(3 + b).integers(0, len(qids), len(qids)) for b in range(1000)]
F = {"items": int(len(qids)), "draws_per_item_and_k": DRAWS, "k": [0] + KS, "rmse": {}, "rmse_CI": {}}
for pn in priors:
    F["rmse"][pn] = {k: float(np.sqrt(sq[pn][k].mean())) for k in [0] + KS}
    F["rmse_CI"][pn] = {k: [float(np.percentile([np.sqrt(sq[pn][k][b].mean()) for b in boot_items], q)) for q in (2.5, 97.5)] for k in [0] + KS}
# equivalent number of responses without text information (interpolated on the weak-prior curve)
wk = np.array([F["rmse"]["weak N(0,3^2)"][k] for k in KS])
def k_equiv(r):
    if r >= wk[0]:
        return float(KS[0] * (wk[0] / r) ** 2)   # extrapolate with RMSE ~ 1/sqrt(k)
    return float(np.exp(np.interp(-r, -wk, np.log(KS))))
F["equivalent_responses_under_weak_prior"] = {pn: {k: k_equiv(F["rmse"][pn][k]) for k in KS} for pn in priors}
# paired comparison text (pre-specified) vs population prior, per k
tp_ = "text: XLM-R large (pre-specified)"
F["text_minus_population_rmse_CI"] = {k: [float(np.percentile([np.sqrt(sq[tp_][k][b].mean()) - np.sqrt(sq["population N(mean, var)"][k][b].mean()) for b in boot_items], q)) for q in (2.5, 97.5)] for k in [0] + KS}
json.dump(F, open(OUT + "calibration_experiment.json", "w"), indent=2)
print("F RMSE:", {pn: {k: round(v, 3) for k, v in F["rmse"][pn].items()} for pn in priors})

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(3.4, 2.7))
style = {"weak N(0,3^2)": ("#8a96a3", "o", "Weak prior"), "population N(mean, var)": ("#9a5a00", "s", "Population prior"),
         tp_: ("#1f5f8b", "^", "Text prior (XLM-R large)")}
for pn, (col, mk, lab) in style.items():
    ks = KS; r = [F["rmse"][pn][k] for k in ks]
    lo_ = [F["rmse_CI"][pn][k][0] for k in ks]; hi_ = [F["rmse_CI"][pn][k][1] for k in ks]
    ax.plot(ks, r, marker=mk, color=col, lw=1.2, ms=4, label=lab); ax.fill_between(ks, lo_, hi_, color=col, alpha=0.15, lw=0)
ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels(KS)
ax.set_xlabel("Responses to the new item (k)"); ax.set_ylabel("RMSE of difficulty (logit)")
ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(OUT + "fig_calibration.png", dpi=300); plt.close(fig)
