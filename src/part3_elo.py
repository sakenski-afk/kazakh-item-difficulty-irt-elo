"""Part 3: rating models of learner proficiency.
3a  game outcome prediction (Elo / Glicko-2 / TrueSkill vs logistic regression and LightGBM), online, chronological.
3b  per-response correctness with player-vs-category Elo (category difficulty estimates) vs running-mean baselines."""
import json, math
from collections import defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb
import trueskill
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, accuracy_score

RAW = "data/raw/"
res = {}

def scores(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"AUC": roc_auc_score(y, p), "LogLoss": log_loss(y, p), "Brier": brier_score_loss(y, p),
            "Acc": accuracy_score(y, p > 0.5)}

# ================= 3a. game outcomes =================
g = pd.read_csv(RAW + "games.csv.gz")
g = g[g.user_id != g.opponent_id].copy()
g["created_at"] = pd.to_datetime(g.created_at)
g = g.sort_values(["created_at", "id"]).reset_index(drop=True)
g["s"] = np.where(g.winner_id.isna(), 0.5, (g.winner_id == g.user_id).astype(float))  # draw = 0.5
n = len(g)
g["period"] = np.where(np.arange(n) < 0.6 * n, "train", np.where(np.arange(n) < 0.8 * n, "val", "test"))
U, O, S = g.user_id.values, g.opponent_id.values, g.s.values
print(f"games {n}, players {len(set(U) | set(O))}, draws {(S == 0.5).mean():.3f}, initiator wins {(S == 1).mean():.3f}")

def run_elo(K, home=0.0):
    r = defaultdict(float); pred = np.empty(n)
    for i in range(n):
        u, o = U[i], O[i]
        p = 1 / (1 + 10 ** ((r[o] - r[u] - home) / 400))
        pred[i] = p
        r[u] += K * (S[i] - p); r[o] -= K * (S[i] - p)
    return pred

def run_glicko2(tau=0.5):
    # Glicko-2 with one game per rating period (standard online approximation)
    st = defaultdict(lambda: [0.0, 350 / 173.7178, 0.06])  # mu, phi, sigma
    gfun = lambda phi: 1 / math.sqrt(1 + 3 * phi ** 2 / math.pi ** 2)
    pred = np.empty(n)
    for i in range(n):
        a, b = st[U[i]], st[O[i]]
        pred[i] = 1 / (1 + math.exp(-gfun(math.sqrt(a[1] ** 2 + b[1] ** 2)) * (a[0] - b[0])))
        for me, op, s in ((a, b, S[i]), (b, a, 1 - S[i])):
            mu, phi, sig = me
            gg = gfun(op[1]); E = 1 / (1 + math.exp(-gg * (mu - op[0])))
            v = 1 / (gg ** 2 * E * (1 - E)); delta = v * gg * (s - E)
            A = math.log(sig ** 2); f = lambda x: (math.exp(x) * (delta ** 2 - phi ** 2 - v - math.exp(x))) / (
                2 * (phi ** 2 + v + math.exp(x)) ** 2) - (x - A) / tau ** 2
            lo = A
            if delta ** 2 > phi ** 2 + v:
                hi = math.log(delta ** 2 - phi ** 2 - v)
            else:
                k = 1
                while f(A - k * tau) < 0:
                    k += 1
                hi = A - k * tau
            flo, fhi = f(lo), f(hi)
            for _ in range(50):
                if abs(hi - lo) < 1e-6:
                    break
                c = lo + (lo - hi) * flo / (fhi - flo); fc = f(c)
                if fc * fhi <= 0:
                    lo, flo = hi, fhi
                else:
                    flo /= 2
                hi, fhi = c, fc
            sig_new = math.exp(lo / 2)
            phi_star = math.sqrt(phi ** 2 + sig_new ** 2)
            phi_new = 1 / math.sqrt(1 / phi_star ** 2 + 1 / v)
            me[3:] = [mu + phi_new ** 2 * gg * (s - E), min(phi_new, 350 / 173.7178), sig_new]
        for me in (a, b):
            me[0], me[1], me[2] = me[3], me[4], me[5]; del me[3:]
    return pred

def run_trueskill():
    env = trueskill.TrueSkill(draw_probability=0.07)
    r = defaultdict(env.create_rating); pred = np.empty(n)
    for i in range(n):
        a, b = r[U[i]], r[O[i]]
        pred[i] = env.cdf((a.mu - b.mu) / math.sqrt(2 * env.beta ** 2 + a.sigma ** 2 + b.sigma ** 2))
        if S[i] == 0.5:
            r[U[i]], r[O[i]] = env.rate_1vs1(a, b, drawn=True)
        elif S[i] == 1:
            r[U[i]], r[O[i]] = env.rate_1vs1(a, b)
        else:
            r[O[i]], r[U[i]] = env.rate_1vs1(b, a)
    return pred

dec = S != 0.5
val, test = (g.period == "val").values & dec, (g.period == "test").values & dec
ytest = S[test]

# tune Elo K and initiator advantage on validation log loss
best = None
for K in (16, 32, 48, 64, 96, 128, 192):
    for home in (0, -25, 25):
        p = run_elo(K, home); ll = log_loss(S[val], np.clip(p[val], 1e-6, 1 - 1e-6))
        if best is None or ll < best[0]:
            best = (ll, K, home, p)
print("Elo tuned K", best[1], "initiator adv", best[2])
p_elo = best[3]
best_tau = min((0.2, 0.3, 0.5, 0.9, 1.2), key=lambda t: log_loss(S[val], np.clip(run_glicko2(t)[val], 1e-6, 1 - 1e-6)))
p_gl = run_glicko2(best_tau); print("Glicko-2 tau", best_tau)
p_ts = run_trueskill()

# history features (known before each game)
cnt, wins = defaultdict(int), defaultdict(float)
F = np.zeros((n, 6))
for i in range(n):
    u, o = U[i], O[i]
    wr_u = (wins[u] + 1) / (cnt[u] + 2); wr_o = (wins[o] + 1) / (cnt[o] + 2)
    F[i] = [np.log1p(cnt[u]), np.log1p(cnt[o]), wr_u, wr_o, wr_u - wr_o, 0]
    cnt[u] += 1; cnt[o] += 1; wins[u] += S[i]; wins[o] += 1 - S[i]
F[:, 5] = np.log(p_elo / (1 - p_elo))  # Elo logit as a feature for the GBM
trn = (g.period == "train").values & dec
lr = LogisticRegression(max_iter=1000).fit(F[trn, :5], S[trn])
p_lr = lr.predict_proba(F[:, :5])[:, 1]
gbm = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=31, min_child_samples=50, verbose=-1)
gbm.fit(F[trn], S[trn], eval_X=F[val], eval_y=S[val], callbacks=[lgb.early_stopping(100, verbose=False)])
p_gbm = gbm.predict_proba(F)[:, 1]

res["3a_game_outcome"] = {
    "Initiator base rate": scores(ytest, np.full(test.sum(), S[trn].mean())),
    "Logistic regression (win-rate history)": scores(ytest, p_lr[test]),
    "Elo": scores(ytest, p_elo[test]),
    "Glicko-2": scores(ytest, p_gl[test]),
    "TrueSkill": scores(ytest, p_ts[test]),
    "LightGBM (history + Elo)": scores(ytest, p_gbm[test]),
}
# experienced-player subset: both players have >= 10 prior games
exp = test & (F[:, 0] >= np.log1p(10)) & (F[:, 1] >= np.log1p(10))
res["3a_game_outcome_experienced"] = {k: scores(S[exp], p[exp]) for k, p in (
    ("Logistic regression (win-rate history)", p_lr), ("Elo", p_elo), ("Glicko-2", p_gl), ("TrueSkill", p_ts),
    ("LightGBM (history + Elo)", p_gbm))}
res["3a_meta"] = {"games": int(n), "test_decisive": int(test.sum()), "experienced_test": int(exp.sum()),
                  "elo_K": best[1], "elo_initiator_adv": best[2], "glicko_tau": best_tau}
# per-game test predictions (for bootstrap comparisons)
pd.DataFrame({"game_id": g.id.values[test], "user_id": U[test], "opponent_id": O[test], "y": S[test],
              "elo": p_elo[test], "glicko2": p_gl[test], "trueskill": p_ts[test], "logreg": p_lr[test],
              "lightgbm": p_gbm[test]}).to_csv("results/part3_game_test_predictions.csv", index=False)
for k, v in res["3a_game_outcome"].items():
    print(f"3a {k:40s}", {m: round(x, 4) for m, x in v.items()})
print("experienced subset n =", exp.sum())
for k, v in res["3a_game_outcome_experienced"].items():
    print(f"3a* {k:39s}", {m: round(x, 4) for m, x in v.items()})

# ================= 3b. per-response Elo: player vs category =================
m = pd.read_csv(RAW + "matches.csv.gz").dropna(subset=["user_correct_count", "cat_id"])
m = m.merge(g[["id", "user_id", "opponent_id"]], left_on="game_id", right_on="id", suffixes=("", "_g"))
m["created_at"] = pd.to_datetime(m.created_at)
m = m.sort_values(["created_at", "id"])
rows = []
for r in m.itertuples(index=False):
    for who, cols in (("user_id", ("user_first_question", "user_second_question", "user_thirty_question")),
                      ("opponent_id", ("opponent_first_question", "opponent_second_question", "opponent_thirty_question"))):
        for c in cols:
            v = getattr(r, c)
            if not pd.isna(v):
                rows.append((getattr(r, who), int(r.cat_id), int(v)))
R = pd.DataFrame(rows, columns=["player", "cat", "y"])
nR = len(R); Pp, Cc, Yy = R.player.values, R.cat.values, R.y.values
test_r = np.arange(nR) >= 0.8 * nR
val_r = (np.arange(nR) >= 0.6 * nR) & ~test_r
print(f"3b responses {nR}, players {R.player.nunique()}, categories {R.cat.nunique()}, accuracy {Yy.mean():.3f}")

def run_pc(a, b, per_cat):
    th, d, thc = defaultdict(float), defaultdict(float), defaultdict(float)
    nu, nc, nuc = defaultdict(int), defaultdict(int), defaultdict(int)
    K = lambda k: a / (1 + b * k)  # uncertainty function (Pelanek 2016)
    pred = np.empty(nR)
    for i in range(nR):
        u, c, y = Pp[i], Cc[i], Yy[i]
        z = th[u] + (thc[(u, c)] if per_cat else 0) - d[c]
        p = 1 / (1 + math.exp(-z)); pred[i] = p; e = y - p
        th[u] += K(nu[u]) * e; d[c] -= K(nc[c]) * e
        if per_cat:
            thc[(u, c)] += K(nuc[(u, c)]) * e; nuc[(u, c)] += 1
        nu[u] += 1; nc[c] += 1
    return pred, dict(d)

def tune(per_cat):
    best = None
    for a in (0.03, 0.05, 0.1, 0.2, 0.4):
        for b in (0.001, 0.005, 0.02, 0.05):
            p, d = run_pc(a, b, per_cat)
            ll = log_loss(Yy[val_r], p[val_r])
            if best is None or ll < best[0]:
                best = (ll, a, b, p, d)
    return best

# running-mean baselines
gm, cm_s, cm_n, pm_s, pm_n = 0.0, defaultdict(float), defaultdict(int), defaultdict(float), defaultdict(int)
b_cat, b_pl = np.empty(nR), np.empty(nR); tot = 0
for i in range(nR):
    u, c, y = Pp[i], Cc[i], Yy[i]
    prior = gm / tot if tot else 0.5
    b_cat[i] = (cm_s[c] + 5 * prior) / (cm_n[c] + 5)
    b_pl[i] = (pm_s[u] + 5 * prior) / (pm_n[u] + 5)
    cm_s[c] += y; cm_n[c] += 1; pm_s[u] += y; pm_n[u] += 1; gm += y; tot += 1
e1, e2 = tune(False), tune(True)
print("Elo player-vs-category a,b =", e1[1], e1[2], "| + player-category skill a,b =", e2[1], e2[2])
res["3b_response"] = {
    "Category running mean": scores(Yy[test_r], b_cat[test_r]),
    "Player running mean": scores(Yy[test_r], b_pl[test_r]),
    "Elo (player skill - category difficulty)": scores(Yy[test_r], e1[3][test_r]),
    "Elo + player-category skill": scores(Yy[test_r], e2[3][test_r]),
}
# per-response test predictions (for learner-clustered bootstrap)
pd.DataFrame({"player": Pp[test_r], "cat": Cc[test_r], "y": Yy[test_r], "cat_mean": b_cat[test_r],
              "player_mean": b_pl[test_r], "elo": e1[3][test_r], "elo_uc": e2[3][test_r]}).to_csv(
    "results/part3_response_test_predictions.csv", index=False)
res["3b_meta"] = {"responses": int(nR), "train": int((~val_r & ~test_r).sum()), "val": int(val_r.sum()),
                  "test": int(test_r.sum()), "alpha_beta_elo": [e1[1], e1[2]], "alpha_beta_elo_uc": [e2[1], e2[2]], "players": int(R.player.nunique()), "categories": int(R.cat.nunique())}
for k, v in res["3b_response"].items():
    print(f"3b {k:40s}", {m_: round(x, 4) for m_, x in v.items()})

# category difficulty from Elo (higher = harder) + empirical accuracy, for linkage with Part 1
cd = pd.DataFrame({"cat_id": list(e1[4].keys()), "elo_difficulty": list(e1[4].values())})
acc = R.groupby("cat").y.agg(["mean", "size"]).rename(columns={"mean": "round_accuracy", "size": "n_resp"})
cd = cd.merge(acc, left_on="cat_id", right_index=True)
cd.to_csv("results/part3_category_difficulty.csv", index=False)
json.dump(res, open("results/part3_elo.json", "w"), indent=2)
