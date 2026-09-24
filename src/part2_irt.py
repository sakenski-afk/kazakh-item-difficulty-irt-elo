"""Part 2: 1PL / 2PL / 3PL IRT by marginal maximum a posteriori (Gauss-Hermite quadrature, theta ~ N(0,1))
on tournament responses (group_questions). Model comparison by AIC/BIC and held-out response prediction."""
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

torch.manual_seed(0)
RAW = "data/raw/"
MIN_ITEM, MIN_PERSON = 30, 10

# ---------- responses ----------
gq = pd.read_csv(RAW + "group_questions.csv.gz").dropna(subset=["question_id"])
ca = pd.read_csv(RAW + "correct_answers.csv.gz")
keys = ca.groupby("question_id").answer_id.apply(set)
gq = gq[gq.question_id.isin(keys.index)].copy()
gq["created_at"] = pd.to_datetime(gq.created_at)
gq = gq.sort_values("created_at").drop_duplicates(["user_id", "question_id"], keep="first")
# unanswered (timeout) counts as incorrect
gq["y"] = [int(pd.notna(a) and a in keys[q]) for q, a in zip(gq.question_id, gq.answer_id)]
for _ in range(3):  # iterative filtering
    gq = gq[gq.groupby("question_id").user_id.transform("size") >= MIN_ITEM]
    gq = gq[gq.groupby("user_id").question_id.transform("size") >= MIN_PERSON]
persons = np.sort(gq.user_id.unique()); items = np.sort(gq.question_id.unique())
pi = np.searchsorted(persons, gq.user_id.values); ii = np.searchsorted(items, gq.question_id.values)
y = gq.y.values.astype(np.float32)
P, I, N = len(persons), len(items), len(y)
print(f"responses {N}, persons {P}, items {I}, accuracy {y.mean():.3f}, "
      f"timeouts {(gq.answer_id.isna()).sum()}, density {N / (P * I):.3f}")

nodes, w = np.polynomial.hermite_e.hermegauss(41)
nodes = torch.tensor(nodes, dtype=torch.float64); logw = torch.log(torch.tensor(w / w.sum(), dtype=torch.float64))


def build(mask):
    R = torch.zeros(P, I, dtype=torch.float64); M = torch.zeros(P, I, dtype=torch.float64)
    R[pi[mask], ii[mask]] = torch.tensor(y[mask], dtype=torch.float64); M[pi[mask], ii[mask]] = 1.0
    return R, M


def item_probs(params, model):
    a = torch.exp(params["la"]) if model != "1PL" else torch.exp(params["la"]).expand(I)
    b = params["b"]
    p = torch.sigmoid(a[None, :] * (nodes[:, None] - b[None, :]))  # Q x I
    if model == "3PL":
        c = torch.sigmoid(params["lc"])
        p = c[None, :] + (1 - c[None, :]) * p
    return p.clamp(1e-9, 1 - 1e-9)


def log_prior(params, model):
    lp = -0.5 * (params["b"] / 3.0).pow(2).sum()  # weak N(0,3) on b
    if model != "1PL":
        lp = lp - 0.5 * (params["la"] / 0.5).pow(2).sum()  # lognormal(0, .5) on a
    if model == "3PL":
        c = torch.sigmoid(params["lc"])
        lp = lp + (4 * torch.log(c) + 16 * torch.log(1 - c)).sum()  # Beta(5,17) on c
    return lp


def person_loglik(params, model, R, M):
    p = item_probs(params, model)  # Q x I
    ll = R @ torch.log(p).T + (M - R) @ torch.log(1 - p).T  # P x Q
    return ll


def fit(model, R, M, iters=400):
    params = {"b": torch.zeros(I, dtype=torch.float64, requires_grad=True),
              "la": torch.zeros(1 if model == "1PL" else I, dtype=torch.float64, requires_grad=True)}
    if model == "3PL":
        params["lc"] = torch.full((I,), -1.2, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS(list(params.values()), lr=1, max_iter=iters, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-7, tolerance_change=1e-10)
    def closure():
        opt.zero_grad()
        mll = torch.logsumexp(person_loglik(params, model, R, M) + logw[None, :], dim=1).sum()
        loss = -(mll + log_prior(params, model))
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        mll = torch.logsumexp(person_loglik(params, model, R, M) + logw[None, :], dim=1).sum().item()
    k = sum(v.numel() for v in params.values())
    return params, mll, k


def predict(params, model, R, M, pi_t, ii_t):
    with torch.no_grad():
        post = person_loglik(params, model, R, M) + logw[None, :]
        post = torch.softmax(post, dim=1)  # P x Q posterior over theta
        p = item_probs(params, model)  # Q x I
        pred = (post[pi_t] * p[:, ii_t].T).sum(1).numpy()
        theta = (post * nodes[None, :]).sum(1).numpy()
    return pred, theta


# ---------- held-out evaluation (10% of responses) ----------
rng = np.random.default_rng(0)
test = rng.random(N) < 0.10
R_tr, M_tr = build(~test)
yt = y[test]
res = {}
# baselines
item_mean = pd.Series(y[~test]).groupby(ii[~test]).mean()
person_mean = pd.Series(y[~test]).groupby(pi[~test]).mean()
b1 = item_mean.reindex(ii[test]).fillna(y[~test].mean()).values
lg = lambda v: np.log(np.clip(v, 1e-3, 1 - 1e-3) / (1 - np.clip(v, 1e-3, 1 - 1e-3)))
b2 = 1 / (1 + np.exp(-(lg(b1) + lg(person_mean.reindex(pi[test]).fillna(y[~test].mean()).values) - lg(y[~test].mean()))))
for name, pr in (("Item p-value", b1), ("Item + person logit (additive)", b2)):
    res[name] = {"AUC": roc_auc_score(yt, pr), "LogLoss": log_loss(yt, pr), "Brier": brier_score_loss(yt, pr)}
for model in ("1PL", "2PL", "3PL"):
    params, _, _ = fit(model, R_tr, M_tr)
    pr, _ = predict(params, model, R_tr, M_tr, pi[test], ii[test])
    res[model] = {"AUC": roc_auc_score(yt, pr), "LogLoss": log_loss(yt, pr), "Brier": brier_score_loss(yt, pr)}
for k, v in res.items():
    print(f"held-out {k:32s}", {m: round(x, 4) for m, x in v.items()})

# ---------- full-data fit, information criteria, item parameters ----------
R, M = build(np.ones(N, bool))
fits, table = {}, {}
for model in ("1PL", "2PL", "3PL"):
    params, mll, k = fit(model, R, M)
    fits[model] = params
    table[model] = {"logLik": mll, "k": k, "AIC": -2 * mll + 2 * k, "BIC": -2 * mll + k * np.log(P)}
    print(model, {m: round(x, 1) for m, x in table[model].items()})

best = min(table, key=lambda m: table[m]["BIC"])
print("best by BIC:", best)
out = pd.DataFrame({"question_id": items.astype(int), "n_resp": np.bincount(ii, minlength=I),
                    "p_value": np.bincount(ii, weights=y, minlength=I) / np.bincount(ii, minlength=I)})
for model, prm in fits.items():
    out[f"b_{model}"] = prm["b"].detach().numpy()
    out[f"a_{model}"] = (torch.exp(prm["la"]).expand(I) if model == "1PL" else torch.exp(prm["la"])).detach().numpy()
out["c_3PL"] = torch.sigmoid(fits["3PL"]["lc"]).detach().numpy()
with torch.no_grad():
    post = torch.softmax(person_loglik(fits[best], best, R, M) + logw[None, :], dim=1)
    theta = (post * nodes[None, :]).sum(1).numpy()
    psd2 = (post * nodes[None, :] ** 2).sum(1).numpy() - theta ** 2
# marginal (EAP) reliability
rel = float(np.var(theta) / (np.var(theta) + psd2.mean()))
print("EAP reliability", round(rel, 3))
out.to_csv("results/part2_item_params.csv", index=False)
pd.DataFrame({"user_id": persons, "theta_eap": theta}).to_csv("results/part2_person_theta.csv", index=False)
print("item params summary\n", out.describe().round(3).T[["mean", "std", "min", "max"]])
json.dump({"data": {"responses": int(N), "persons": int(P), "items": int(I), "accuracy": float(y.mean())},
           "heldout": res, "fit": table, "best_by_BIC": best, "eap_var": float(np.var(theta)), "eap_reliability": rel},
          open("results/part2_irt.json", "w"), indent=2)
