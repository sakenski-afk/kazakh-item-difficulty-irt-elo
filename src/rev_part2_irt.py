"""Revision analyses for Part 2 (IRT), in response to the reviewers.
  A. design and connectivity of the tournament response matrix (tournament series, rounds, learner-item graph)
  B. common ability scale: confirmatory between-item 3-dimensional 2PL (one trait per tournament series, correlated)
     against the unidimensional 2PL; per-series EAP abilities of learners who took part in two series
  C. local dependence: Yen's Q3 against a parametric-bootstrap null distribution from the fitted 2PL
  D. uncertainty of the small predictive differences between IRT models (learner-clustered bootstrap)
  E. does the choice of IRT model change the cold-start reference? (b under 1PL / 2PL / 3PL)
  F. cross-fitted calibration experiment: anchor item parameters and learner abilities estimated WITHOUT the
     held-out items (5 item folds); reference from a disjoint half of the responses; sensitivity to the
     treatment of discrimination (a = 1, a = mean anchor discrimination, joint 2PL estimation of a and b)
  G. screening task: identification of the hardest and easiest deciles of new items from text predictions
Outputs: results/revision/part2_*.json, figures."""
import glob, json, os
import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, log_loss

torch.manual_seed(0)
RAW, OUT = "data/raw/", "results/revision/"
os.makedirs(OUT, exist_ok=True)
MIN_ITEM, MIN_PERSON = 30, 10
B_BOOT = 2000
ONLY_CAL = os.environ.get("ONLY_CAL") == "1"   # re-run only F and G, keep the saved A-E results

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
groups = pd.read_csv(RAW + "groups.csv.gz").set_index("id")
gq["stage"] = gq.group_id.map(groups.stage_id)
SERIES = {63: "Regional cup", 64: "Famous People", 65: "History of Kazakhstan"}
item_stage = gq.groupby("question_id").stage.agg(lambda s: s.mode().iloc[0]).reindex(items).values
assert gq.groupby("question_id").stage.nunique().max() == 1, "an item belongs to more than one series"
res = {}

nodes_np, w = np.polynomial.hermite_e.hermegauss(41)
nodes = torch.tensor(nodes_np, dtype=torch.float64); logw = torch.log(torch.tensor(w / w.sum(), dtype=torch.float64))

def build(mask, yy=None):
    yy = y if yy is None else yy
    R = torch.zeros(P, I, dtype=torch.float64); M = torch.zeros(P, I, dtype=torch.float64)
    R[pi[mask], ii[mask]] = torch.tensor(yy[mask]); M[pi[mask], ii[mask]] = 1.0
    return R, M

def probs(prm, model):
    a = torch.exp(prm["la"]).expand(I) if model == "1PL" else torch.exp(prm["la"])
    p = torch.sigmoid(a[None, :] * (nodes[:, None] - prm["b"][None, :]))
    if model == "3PL":
        c = torch.sigmoid(prm["lc"]); p = c[None, :] + (1 - c[None, :]) * p
    return p.clamp(1e-9, 1 - 1e-9)

def logprior(prm, model):
    lp = -0.5 * (prm["b"] / 3.0).pow(2).sum()
    if model != "1PL":
        lp = lp - 0.5 * (prm["la"] / 0.5).pow(2).sum()
    if model == "3PL":
        c = torch.sigmoid(prm["lc"]); lp = lp + (4 * torch.log(c) + 16 * torch.log(1 - c)).sum()
    return lp

def pll(prm, model, R, M):
    p = probs(prm, model)
    return R @ torch.log(p).T + (M - R) @ torch.log(1 - p).T

def mll(prm, model, R, M):
    return torch.logsumexp(pll(prm, model, R, M) + logw[None, :], dim=1).sum()

def fit(model, R, M, iters=500):
    prm = {"b": torch.zeros(I, dtype=torch.float64, requires_grad=True),
           "la": torch.zeros(1 if model == "1PL" else I, dtype=torch.float64, requires_grad=True)}
    if model == "3PL":
        prm["lc"] = torch.full((I,), -1.2, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS(list(prm.values()), lr=1, max_iter=iters, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-7, tolerance_change=1e-10)
    def closure():
        opt.zero_grad(); loss = -(mll(prm, model, R, M) + logprior(prm, model)); loss.backward(); return loss
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

def ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

R, M = build(np.ones(N, bool))
fits = {m: fit(m, R, M)[0] for m in ("1PL", "2PL", "3PL")}
prm2 = fits["2PL"]
a2 = torch.exp(prm2["la"]).detach().numpy(); b2 = prm2["b"].detach().numpy()
post2 = posterior(prm2, "2PL", R, M)
theta = (post2 * nodes[None, :]).sum(1).numpy()

if not ONLY_CAL:
    exec(compile('# ======================= A. design and connectivity =======================\ng = coo_matrix((np.ones(N), (pi, P + ii)), shape=(P + I, P + I))\nncomp, lab = connected_components(g, directed=False)\nperson_series = gq.groupby("user_id").stage.nunique().reindex(persons).values\ncommon = (M.T @ M).numpy()\niu = np.triu_indices(I, 1)\nA = {"responses": int(N), "learners": int(P), "items": int(I), "density": float(N / (P * I)),\n     "tournament_series": {SERIES[s]: {"items": int((item_stage == s).sum()),\n                                       "rounds": int(gq[gq.stage == s].group_id.nunique()),\n                                       "learners": int(gq[gq.stage == s].user_id.nunique()),\n                                       "responses": int((gq.stage == s).sum()),\n                                       "dates": [str(gq[gq.stage == s].created_at.min().date()), str(gq[gq.stage == s].created_at.max().date())]}\n                           for s in SERIES},\n     "rounds_total": int(gq.group_id.nunique()),\n     "learners_in_more_than_one_round": int((gq.groupby("user_id").group_id.nunique() > 1).sum()),\n     "learners_by_number_of_series": {int(k): int(v) for k, v in pd.Series(person_series).value_counts().sort_index().items()},\n     "learner_item_graph_connected_components": int(ncomp),\n     "item_pairs_with_common_learners_share": float(np.mean(common[iu] > 0)),\n     "item_pairs_with_ge50_common_learners_share": float(np.mean(common[iu] >= 50)),\n     "cross_series_item_pairs_with_common_learners_share": float(np.mean((common[iu] > 0)[item_stage[iu[0]] != item_stage[iu[1]]])),\n     "median_responses_per_learner": float(np.median(np.bincount(pi))),\n     "median_responses_per_item": float(np.median(np.bincount(ii)))}\nfor s in SERIES:\n    for t in SERIES:\n        if s < t:\n            A[f"learners_in_both_{SERIES[s]}_and_{SERIES[t]}"] = int(len(set(gq[gq.stage == s].user_id) & set(gq[gq.stage == t].user_id)))\nres["A_design_connectivity"] = A\nprint("A:", json.dumps(A, indent=1, ensure_ascii=False))\n\n# ======================= B. common scale across tournament series =======================\n# confirmatory between-item 3-dimensional 2PL with correlated traits (one per series), 15^3 quadrature\nS_ids = sorted(SERIES); dim = np.array([S_ids.index(s) for s in item_stage])\ngh, gw = np.polynomial.hermite_e.hermegauss(15)\nZ = np.array(np.meshgrid(gh, gh, gh, indexing="ij")).reshape(3, -1).T\nZt = torch.tensor(Z, dtype=torch.float64)\nlogw3 = torch.log(torch.tensor(np.prod(np.array(np.meshgrid(gw, gw, gw, indexing="ij")).reshape(3, -1).T, 1) / gw.sum() ** 3,\n                               dtype=torch.float64))\ndim_t = torch.tensor(dim)\n\ndef chol(raw):\n    # unit-variance Cholesky factor from 3 unconstrained parameters (correlations via tanh)\n    r = torch.tanh(raw)\n    L = torch.zeros(3, 3, dtype=torch.float64)\n    L[0, 0] = 1.0\n    L[1, 0] = r[0]; L[1, 1] = torch.sqrt(1 - r[0] ** 2)\n    L[2, 0] = r[1]; v = torch.sqrt(1 - r[1] ** 2)\n    L[2, 1] = v * r[2]; L[2, 2] = v * torch.sqrt(1 - r[2] ** 2)\n    return L\n\ndef pll3(prm, R_, M_):\n    L = chol(prm["rho"]); TH = Zt @ L.T                   # Q x 3\n    a = torch.exp(prm["la"]); p = torch.sigmoid(a[None, :] * (TH[:, dim_t] - prm["b"][None, :])).clamp(1e-9, 1 - 1e-9)\n    return R_ @ torch.log(p).T + (M_ - R_) @ torch.log(1 - p).T\n\ndef fit3(R_, M_):\n    prm = {"b": prm2["b"].detach().clone().requires_grad_(True), "la": prm2["la"].detach().clone().requires_grad_(True),\n           "rho": torch.full((3,), 1.0, dtype=torch.float64, requires_grad=True)}\n    opt = torch.optim.LBFGS(list(prm.values()), lr=1, max_iter=400, line_search_fn="strong_wolfe",\n                            tolerance_grad=1e-7, tolerance_change=1e-10)\n    def closure():\n        opt.zero_grad()\n        loss = -(torch.logsumexp(pll3(prm, R_, M_) + logw3[None, :], 1).sum()\n                 - 0.5 * (prm["b"] / 3.0).pow(2).sum() - 0.5 * (prm["la"] / 0.5).pow(2).sum())\n        loss.backward(); return loss\n    opt.step(closure)\n    with torch.no_grad():\n        ll = torch.logsumexp(pll3(prm, R_, M_) + logw3[None, :], 1).sum().item()\n    return prm, ll\n\nprm3, ll3 = fit3(R, M)\nll1 = mll(prm2, "2PL", R, M).item()\nwith torch.no_grad():\n    Lh = chol(prm3["rho"]); Cor = (Lh @ Lh.T).numpy()\n# held-out responses (the same 10% split as part2_irt.py)\ntest = np.random.default_rng(0).random(N) < 0.10\nR_tr, M_tr = build(~test)\np2_tr, _, _ = fit("2PL", R_tr, M_tr)\npr_1d = pred(p2_tr, "2PL", posterior(p2_tr, "2PL", R_tr, M_tr), pi[test], ii[test])\np3_tr, _ = fit3(R_tr, M_tr)\nwith torch.no_grad():\n    post3 = torch.softmax(pll3(p3_tr, R_tr, M_tr) + logw3[None, :], 1)\n    L3 = chol(p3_tr["rho"]); TH3 = Zt @ L3.T\n    pmat = torch.sigmoid(torch.exp(p3_tr["la"])[None, :] * (TH3[:, dim_t] - p3_tr["b"][None, :]))\n    pr_3d = (post3[pi[test]] * pmat[:, ii[test]].T).sum(1).numpy()\nk1, k3 = 2 * I, 2 * I + 3\n# per-series EAP abilities (2PL item parameters fixed at the full-data estimates) for learners in two series\nBser = {}\nfor s in S_ids:\n    for t in S_ids:\n        if s >= t:\n            continue\n        both = np.intersect1d(np.unique(pi[item_stage[ii] == s]), np.unique(pi[item_stage[ii] == t]))\n        out = {}\n        for lab_, ser in (("s", s), ("t", t)):\n            mk = item_stage[ii] == ser\n            Rs, Ms = build(mk)\n            ps = posterior(prm2, "2PL", Rs, Ms)\n            eap = (ps * nodes[None, :]).sum(1).numpy(); psd2 = (ps * nodes[None, :] ** 2).sum(1).numpy() - eap ** 2\n            out[lab_] = (eap[both], psd2[both])\n        r_ = pearsonr(out["s"][0], out["t"][0])[0]\n        rel = [float(np.var(e) / (np.var(e) + v.mean())) for e, v in (out["s"], out["t"])]\n        Bser[f"{SERIES[s]} vs {SERIES[t]}"] = {"learners": int(len(both)), "r_EAP": float(r_), "reliabilities": rel,\n                                               "r_disattenuated": float(r_ / np.sqrt(rel[0] * rel[1]))}\nres["B_common_scale"] = {\n    "model": "between-item 3-dimensional 2PL, one correlated trait per tournament series, 15^3 Gauss-Hermite nodes",\n    "latent_correlations": {f"{SERIES[S_ids[i]]} - {SERIES[S_ids[j]]}": float(Cor[i, j]) for i in range(3) for j in range(3) if i < j},\n    "logLik_1D_2PL": ll1, "logLik_3D_2PL": ll3, "LR": 2 * (ll3 - ll1), "extra_parameters": 3,\n    "AIC_1D": -2 * ll1 + 2 * k1, "AIC_3D": -2 * ll3 + 2 * k3, "BIC_1D": -2 * ll1 + k1 * np.log(P), "BIC_3D": -2 * ll3 + k3 * np.log(P),\n    "heldout_1D": {"AUC": roc_auc_score(y[test], pr_1d), "LogLoss": log_loss(y[test], pr_1d)},\n    "heldout_3D": {"AUC": roc_auc_score(y[test], pr_3d), "LogLoss": log_loss(y[test], pr_3d)},\n    "per_series_EAP_correlations": Bser}\nprint("B:", json.dumps(res["B_common_scale"], indent=1))\n\n# ======================= C. Q3 against a parametric-bootstrap null =======================\ndef q3_matrix(resid_full, obs):\n    Zr = np.where(obs, resid_full, 0.0); O = obs.astype(float)\n    n = O.T @ O\n    with np.errstate(invalid="ignore", divide="ignore"):\n        Sx = Zr.T @ O; Sy = Sx.T; Sxx = (Zr ** 2).T @ O; Syy = Sxx.T; Sxy = Zr.T @ Zr\n        cov = Sxy / n - (Sx / n) * (Sy / n)\n        vx = Sxx / n - (Sx / n) ** 2; vy = Syy / n - (Sy / n) ** 2\n        q = cov / np.sqrt(vx * vy)\n    return q, n\n\ndef q3_stats(prm, R_, M_, yy):\n    ps = posterior(prm, "2PL", R_, M_); th = (ps * nodes[None, :]).sum(1).numpy()\n    a_ = torch.exp(prm["la"]).detach().numpy(); b_ = prm["b"].detach().numpy()\n    pe = 1 / (1 + np.exp(-a_[ii] * (th[pi] - b_[ii])))\n    full = np.zeros((P, I)); full[pi, ii] = yy - pe\n    obs = M_.numpy() > 0\n    q, n = q3_matrix(full, obs)\n    sel = (n[iu] >= 50)\n    qv = q[iu][sel]; qs = qv - qv.mean()\n    same = (item_stage[iu[0]] == item_stage[iu[1]])[sel]\n    return {"mean": float(qv.mean()), "share_abs_q3star_gt_0.2": float(np.mean(np.abs(qs) > 0.2)),\n            "q99": float(np.percentile(qs, 99)), "q999": float(np.percentile(qs, 99.9)), "max": float(qs.max()),\n            "pairs": int(sel.sum())}, qs, same\n\nobs_stats, q_obs, same_obs = q3_stats(prm2, R, M, y)\nnull = []\nrng_n = np.random.default_rng(11)\npost_np = post2.numpy()\nNREP = 30\nfor rep in range(NREP):\n    # abilities drawn from each learner\'s posterior; responses simulated from the fitted 2PL with the observed pattern\n    th_s = nodes_np[[rng_n.choice(len(nodes_np), p=post_np[p_] / post_np[p_].sum()) for p_ in range(P)]]\n    ys = (rng_n.random(N) < 1 / (1 + np.exp(-a2[ii] * (th_s[pi] - b2[ii])))).astype(float)\n    Rs, Ms = build(np.ones(N, bool), ys)\n    prm_s, _, _ = fit("2PL", Rs, Ms, iters=300)\n    st, qs_null, _ = q3_stats(prm_s, Rs, Ms, ys)\n    null.append(st)\n    if rep == 0:\n        pooled_null = qs_null\n    else:\n        pooled_null = np.concatenate([pooled_null, qs_null])\n    print(f"Q3 null rep {rep + 1}/{NREP}: share>0.2 = {st[\'share_abs_q3star_gt_0.2\']:.3f}", flush=True)\nthr = float(np.percentile(np.abs(pooled_null), 99.9))\nres["C_local_dependence"] = {\n    "observed": obs_stats, "null_replications": NREP,\n    "null_share_abs_q3star_gt_0.2": {"mean": float(np.mean([s["share_abs_q3star_gt_0.2"] for s in null])),\n                                     "range": [float(min(s["share_abs_q3star_gt_0.2"] for s in null)), float(max(s["share_abs_q3star_gt_0.2"] for s in null))]},\n    "null_q99_mean": float(np.mean([s["q99"] for s in null])), "null_max_range": [float(min(s["max"] for s in null)), float(max(s["max"] for s in null))],\n    "pooled_null_abs_q3star_99.9th_percentile": thr,\n    "observed_pairs_above_null_99.9": int(np.sum(np.abs(q_obs) > thr)),\n    "expected_pairs_above_null_99.9": float(0.001 * len(q_obs)),\n    "observed_pairs_above_null_99.9_same_series": int(np.sum((np.abs(q_obs) > thr) & same_obs)),\n    "observed_mean_q3star_same_series": float(q_obs[same_obs].mean()) if same_obs.any() else None,\n    "observed_mean_q3star_different_series": float(q_obs[~same_obs].mean()) if (~same_obs).any() else None}\nprint("C:", res["C_local_dependence"])\n\n# ======================= D. uncertainty of IRT model differences =======================\npreds_te = {}\nitem_mean = pd.Series(y[~test]).groupby(ii[~test]).mean(); person_mean = pd.Series(y[~test]).groupby(pi[~test]).mean()\nbi = item_mean.reindex(ii[test]).fillna(y[~test].mean()).values\nlg = lambda v: np.log(np.clip(v, 1e-3, 1 - 1e-3) / (1 - np.clip(v, 1e-3, 1 - 1e-3)))\npreds_te["additive"] = 1 / (1 + np.exp(-(lg(bi) + lg(person_mean.reindex(pi[test]).fillna(y[~test].mean()).values) - lg(y[~test].mean()))))\nfor m in ("1PL", "2PL", "3PL"):\n    pm, _, _ = fit(m, R_tr, M_tr)\n    preds_te[m] = pred(pm, m, posterior(pm, m, R_tr, M_tr), pi[test], ii[test])\nyt = y[test]; pt = pi[test]; up = np.unique(pt); pos = {u: np.flatnonzero(pt == u) for u in up}\nrng_b = np.random.default_rng(5)\nbidx = [np.concatenate([pos[u] for u in rng_b.choice(up, len(up))]) for _ in range(B_BOOT)]\nD = {}\nfor a_, b_ in (("2PL", "1PL"), ("3PL", "2PL"), ("3PL", "1PL"), ("1PL", "additive")):\n    dA = [roc_auc_score(yt[i], preds_te[a_][i]) - roc_auc_score(yt[i], preds_te[b_][i]) for i in bidx]\n    dL = [log_loss(yt[i], preds_te[a_][i]) - log_loss(yt[i], preds_te[b_][i]) for i in bidx]\n    D[f"{a_} minus {b_}"] = {"delta_AUC": float(roc_auc_score(yt, preds_te[a_]) - roc_auc_score(yt, preds_te[b_])), "AUC_CI": ci(dA),\n                             "delta_LogLoss": float(log_loss(yt, preds_te[a_]) - log_loss(yt, preds_te[b_])), "LogLoss_CI": ci(dL)}\nres["D_irt_model_differences_learner_cluster_bootstrap"] = {"test_responses": int(test.sum()), "learners": int(len(up)), "B": B_BOOT, "results": D}\nprint("D:", D)\n\n', 'sections_A_D', 'exec'))
else:
    res = json.load(open(OUT + 'part2_irt_assumptions.json', encoding='utf-8'))
    test = np.random.default_rng(0).random(N) < 0.10
# ======================= E. IRT model choice and the cold-start reference =======================
clean = pd.read_csv("results/extended/coldstart_clean_items.csv")
ix = np.searchsorted(items, clean.id.values)
bm = {m: fits[m]["b"].detach().numpy()[ix] for m in fits}
pv = np.bincount(ii, weights=y, minlength=I)[ix] / np.bincount(ii, minlength=I)[ix]
nres = np.bincount(ii, minlength=I)[ix]
lgp = -np.log(((pv * nres) + 0.5) / (nres + 1) / (1 - ((pv * nres) + 0.5) / (nres + 1)))
E = {"pearson_b": {f"{a_}-{b_}": float(pearsonr(bm[a_], bm[b_])[0]) for a_, b_ in (("1PL", "2PL"), ("1PL", "3PL"), ("2PL", "3PL"))},
     "spearman_b": {f"{a_}-{b_}": float(spearmanr(bm[a_], bm[b_])[0]) for a_, b_ in (("1PL", "2PL"), ("1PL", "3PL"), ("2PL", "3PL"))},
     "pearson_b_vs_observed_logit_difficulty": {m: float(pearsonr(bm[m], lgp)[0]) for m in bm},
     "spearman_b_vs_observed_logit_difficulty": {m: float(spearmanr(bm[m], lgp)[0]) for m in bm},
     "share_items_p_gt_0.9": float(np.mean(pv > 0.9))}
TEXT = {"XLM-R large": "xlm-roberta-large", "KazRoBERTa": "kaz-roberta-conversational", "XLM-R base": "xlm-roberta-base",
        "mBERT": "bert-base-multilingual-cased"}
tpred = {k: pd.concat([pd.read_csv(f).set_index("id").pred for f in sorted(glob.glob(f"results/finetune/{m}__text__s*__tour.csv"))], axis=1).mean(axis=1)
         for k, m in TEXT.items()}
tpred["TF-IDF"] = pd.read_csv("results/tfidf__text__tour.csv").set_index("id").pred
easy = pv > 0.9
E["text_r_by_reference"] = {k: {"b_1PL": float(pearsonr(v.loc[clean.id].values, bm["1PL"])[0]),
                                "b_2PL": float(pearsonr(v.loc[clean.id].values, bm["2PL"])[0]),
                                "b_3PL": float(pearsonr(v.loc[clean.id].values, bm["3PL"])[0]),
                                "observed_logit": float(pearsonr(v.loc[clean.id].values, lgp)[0]),
                                "b_2PL_without_items_p_gt_0.9": float(pearsonr(v.loc[clean.id].values[~easy], bm["2PL"][~easy])[0])}
                            for k, v in tpred.items()}
E["n_items_without_p_gt_0.9"] = int((~easy).sum())
res["E_model_choice_reference"] = E
print("E:", json.dumps(E, indent=1))
json.dump(res, open(OUT + "part2_irt_assumptions.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)

# ======================= F. cross-fitted calibration experiment =======================
qids = clean.id.values; cj = np.searchsorted(items, qids)
folds = np.random.default_rng(1).integers(0, 5, len(qids))
Rn, Mn = R.numpy(), M.numpy()

def newton_b(th, yy, a, mu, s2, iters=60):
    """MAP of b given abilities and a fixed discrimination a (rows = replicates)."""
    b = np.array(mu, dtype=float).copy()
    for _ in range(iters):
        pz = 1 / (1 + np.exp(-a[:, None] * (th - b[:, None])))
        gr = a * (yy - pz).sum(1) + (b - mu) / s2          # gradient of the negative log posterior
        h = a ** 2 * (pz * (1 - pz)).sum(1) + 1 / s2
        b = b - np.clip(gr / h, -2.0, 2.0)
    return b

def newton_ab(th, yy, mu, s2, l0, sl2=0.25, iters=80):
    """joint MAP of (b, log a) given abilities; Fisher scoring, rows = replicates."""
    b = np.array(mu, dtype=float).copy(); l = np.full(len(b), float(l0))
    for _ in range(iters):
        a = np.exp(l); z = a[:, None] * (th - b[:, None]); pz = 1 / (1 + np.exp(-z)); w_ = pz * (1 - pz); e = pz - yy
        gb = (e * (-a[:, None])).sum(1) + (b - mu) / s2
        gl = (e * z).sum(1) + (l - l0) / sl2
        hbb = (w_ * a[:, None] ** 2).sum(1) + 1 / s2; hll = (w_ * z ** 2).sum(1) + 1 / sl2; hbl = (w_ * (-a[:, None]) * z).sum(1)
        det = hbb * hll - hbl ** 2
        db = (hll * gb - hbl * gl) / det; dl = (hbb * gl - hbl * gb) / det
        b = b - np.clip(db, -1.0, 1.0); l = l - np.clip(dl, -0.5, 0.5)
    return b, np.exp(l)

KS = [5, 10, 20, 40]; DRAWS = 200
anchor_info = {}
item_data = {}
for f in range(5):
    hold = cj[folds == f]
    keep = ~np.isin(ii, hold)
    R_a, M_a = build(keep)
    prm_a, _, _ = fit("2PL", R_a, M_a)
    post_a = posterior(prm_a, "2PL", R_a, M_a)
    th_a = (post_a * nodes[None, :]).sum(1).numpy()
    a_anchor = np.exp(prm_a["la"].detach().numpy()); anch = np.setdiff1d(np.arange(I), hold)
    abar = float(np.exp(np.mean(np.log(a_anchor[anch]))))
    anchor_info[f] = {"held_out_items": int(len(hold)), "anchor_items": int(len(anch)), "geometric_mean_anchor_a": abar}
    rng_s = np.random.default_rng(100 + f)
    for j in hold:
        resp = np.flatnonzero(Mn[:, j] > 0)
        perm = rng_s.permutation(len(resp)); h = len(resp) // 2
        ref_idx, cal_idx = resp[perm[:h]], resp[perm[h:]]
        item_data[items[j]] = {"th_all": th_a[resp], "y_all": Rn[resp, j], "th_ref": th_a[ref_idx], "y_ref": Rn[ref_idx, j],
                               "th_cal": th_a[cal_idx], "y_cal": Rn[cal_idx, j], "abar": abar, "fold": f}
    print(f"fold {f}: anchors refitted without {len(hold)} items", flush=True)

def ref_value(d, mode, which):
    th, yy = (d["th_ref"], d["y_ref"]) if which == "half" else (d["th_all"], d["y_all"])
    if mode == "a=1":
        return newton_b(th[None], yy[None], np.array([1.0]), np.array([0.0]), 9.0)[0], 1.0
    if mode == "a=mean anchor":
        return newton_b(th[None], yy[None], np.array([d["abar"]]), np.array([0.0]), 9.0)[0], d["abar"]
    bb, aa = newton_ab(th[None], yy[None], np.array([0.0]), 9.0, np.log(d["abar"]))
    return bb[0], aa[0]

def run_design(mode, which):
    """mode: discrimination treatment; which: 'half' = disjoint reference half, 'all' = all responses (original)."""
    truth = np.array([ref_value(item_data[q], mode, which)[0] for q in qids])
    priors = {"weak": (np.zeros(len(qids)), np.full(len(qids), 9.0))}
    mu_pop = np.zeros(len(qids)); s2_pop = np.zeros(len(qids))
    for f in range(5):
        trn = folds != f; mu_pop[~trn] = truth[trn].mean(); s2_pop[~trn] = truth[trn].var()
    priors["population"] = (mu_pop, s2_pop)
    for name in ("XLM-R large", "KazRoBERTa", "TF-IDF"):
        x = tpred[name].loc[qids].values; mu = np.zeros(len(qids)); s2 = np.zeros(len(qids))
        for f in range(5):
            trn = folds != f; coef = np.polyfit(x[trn], truth[trn], 1)
            mu[~trn] = np.polyval(coef, x[~trn]); s2[~trn] = np.var(truth[trn] - np.polyval(coef, x[trn]))
        priors[f"text: {name}"] = (mu, s2)
    rngc = np.random.default_rng(2)
    sq = {pn: {k: np.zeros(len(qids)) for k in [0] + KS} for pn in priors}
    for n_, q in enumerate(qids):
        d = item_data[q]
        th, yy = (d["th_cal"], d["y_cal"]) if which == "half" else (d["th_all"], d["y_all"])
        for k in KS:
            idx = np.array([rngc.choice(len(yy), k, replace=False) for _ in range(DRAWS)])
            TH, YY = th[idx], yy[idx]
            for pn, (mu, s2) in priors.items():
                if mode == "a=1":
                    bh = newton_b(TH, YY, np.ones(DRAWS), np.full(DRAWS, mu[n_]), s2[n_])
                elif mode == "a=mean anchor":
                    bh = newton_b(TH, YY, np.full(DRAWS, d["abar"]), np.full(DRAWS, mu[n_]), s2[n_])
                else:
                    bh, _ = newton_ab(TH, YY, np.full(DRAWS, mu[n_]), s2[n_], np.log(d["abar"]))
                sq[pn][k][n_] = np.mean((bh - truth[n_]) ** 2)
        for pn, (mu, s2) in priors.items():
            sq[pn][0][n_] = (mu[n_] - truth[n_]) ** 2
    boot = [np.random.default_rng(3 + b).integers(0, len(qids), len(qids)) for b in range(1000)]
    out = {"rmse": {}, "rmse_CI": {}, "text_minus_population": {}}
    for pn in priors:
        out["rmse"][pn] = {k: float(np.sqrt(sq[pn][k].mean())) for k in [0] + KS}
        out["rmse_CI"][pn] = {k: ci([np.sqrt(sq[pn][k][b].mean()) for b in boot]) for k in [0] + KS}
    for name in ("XLM-R large", "KazRoBERTa", "TF-IDF"):
        tp = f"text: {name}"
        out["text_minus_population"][name] = {k: {"delta": out["rmse"][tp][k] - out["rmse"]["population"][k],
            "CI": ci([np.sqrt(sq[tp][k][b].mean()) - np.sqrt(sq["population"][k][b].mean()) for b in boot])} for k in [0] + KS}
    out["population_minus_weak"] = {k: {"delta": out["rmse"]["population"][k] - out["rmse"]["weak"][k],
        "CI": ci([np.sqrt(sq["population"][k][b].mean()) - np.sqrt(sq["weak"][k][b].mean()) for b in boot])} for k in KS}
    out["reference_sd"] = float(truth.std())
    out["text_prior_r_with_reference"] = {name: float(pearsonr(tpred[name].loc[qids].values, truth)[0]) for name in ("XLM-R large", "KazRoBERTa", "TF-IDF")}
    return out, truth

F = {"items": int(len(qids)), "folds": 5, "k": [0] + KS, "draws": DRAWS, "anchor_fits": anchor_info, "designs": {}}
truths = {}
for mode in ("a=1", "a=mean anchor", "2PL (a and b estimated)"):
    for which in ("half", "all"):
        key = f"{mode} | reference: {'disjoint half' if which == 'half' else 'all responses (overlapping)'}"
        F["designs"][key], truths[key] = run_design(mode, which)
        r_ = F["designs"][key]["rmse"]
        print(key, {pn: [round(r_[pn][k], 3) for k in [0] + KS] for pn in r_}, flush=True)
orig = json.load(open("results/extended/calibration_experiment.json"))
F["original_non_crossfitted_rmse"] = orig["rmse"]
k_main = "a=1 | reference: disjoint half"
F["reference_agreement"] = {k: float(pearsonr(truths[k_main], v)[0]) for k, v in truths.items()}
F["crossfitted_reference_vs_full_2PL_b"] = float(pearsonr(truths["a=1 | reference: all responses (overlapping)"], bm["2PL"])[0])
json.dump(F, open(OUT + "part2_calibration_crossfit.json", "w"), indent=2)

# ======================= G. screening task =======================
def screen(truth_v, pred_v, rng_):
    n = len(truth_v); k = int(round(0.1 * n))
    hard_t = truth_v >= np.sort(truth_v)[-k]; easy_t = truth_v <= np.sort(truth_v)[k - 1]
    def stats(ix_):
        t, p_ = truth_v[ix_], pred_v[ix_]
        kk = int(round(0.1 * len(ix_)))
        ht = t >= np.sort(t)[-kk]; et = t <= np.sort(t)[kk - 1]
        hp = np.argsort(-p_)[:kk]; ep = np.argsort(p_)[:kk]
        below_median = t <= np.median(t)
        return {"AUC_hardest_decile": roc_auc_score(ht, p_), "AUC_easiest_decile": roc_auc_score(et, -p_),
                "precision_at_10pct_hard": float(ht[hp].mean()), "precision_at_10pct_easy": float(et[ep].mean()),
                "predicted_hardest_decile_truly_in_easier_half": float(below_median[hp].mean()),
                "predicted_easiest_decile_truly_in_harder_half": float((~below_median)[ep].mean()),
                "spearman": spearmanr(t, p_)[0]}
    point = stats(np.arange(n))
    bt = [stats(rng_.integers(0, n, n)) for _ in range(B_BOOT)]
    return {k_: {"value": float(v), "CI": ci([b[k_] for b in bt])} for k_, v in point.items()}

G = {"truth": "2PL difficulty from all tournament responses (and the cross-fitted disjoint-half reference)",
     "chance_level": {"AUC": 0.5, "precision_at_10pct": 0.10, "wrong_half": 0.5}, "items": int(len(qids)), "models": {}}
for name in ("XLM-R large", "KazRoBERTa", "XLM-R base", "mBERT", "TF-IDF"):
    G["models"][name] = {"vs_b_2PL": screen(bm["2PL"], tpred[name].loc[qids].values, np.random.default_rng(21)),
                         "vs_crossfitted_reference": screen(truths[k_main], tpred[name].loc[qids].values, np.random.default_rng(22))}
json.dump(G, open(OUT + "part2_screening.json", "w"), indent=2)
print("G:", {k: {m: round(v["vs_b_2PL"][m]["value"], 3) for m in ("AUC_hardest_decile", "AUC_easiest_decile", "precision_at_10pct_hard", "precision_at_10pct_easy")} for k, v in G["models"].items()})

# ---------------- figure: cross-fitted calibration experiment ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(3.4, 2.7))
dm = F["designs"][k_main]
style = {"weak": ("#8a96a3", "o", "Weak prior"), "population": ("#9a5a00", "s", "Population prior"),
         "text: XLM-R large": ("#1f5f8b", "^", "Text prior (XLM-R large)")}
xs = [0] + KS; xpos = [2.5] + KS
for pn, (col, mk, lab_) in style.items():
    if pn == "weak":
        xx, kk = KS, KS
    else:
        xx, kk = xpos, xs
    r_ = [dm["rmse"][pn][k] for k in kk]; lo_ = [dm["rmse_CI"][pn][k][0] for k in kk]; hi_ = [dm["rmse_CI"][pn][k][1] for k in kk]
    ax.errorbar(xx, r_, yerr=[np.array(r_) - lo_, np.array(hi_) - r_], marker=mk, color=col, lw=1.1, ms=4, capsize=2, label=lab_)
ax.set_xscale("log"); ax.set_xticks(xpos); ax.set_xticklabels(["0"] + [str(k) for k in KS])
ax.set_xlabel("Responses to the new item (k)"); ax.set_ylabel("RMSE of difficulty (logit)")
ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(OUT + "fig_calibration_crossfit.png", dpi=300); plt.close(fig)
print("saved")
