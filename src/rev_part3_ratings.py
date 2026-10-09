"""Revision analyses for Part 3 (rating models), in response to the reviewers.
  A. data reconstruction: coverage of the truncated export, round records -> per-question responses, ordering
  B. sensitivity of the rating models: TrueSkill tuned on validation (beta, tau, initial sigma, draw probability);
     Glicko-2 with rating periods of one game, one hour, one day and one week; Elo K grid
  C. treatment of draws: win probabilities with and without the TrueSkill draw margin; scores on all games
  D. dependence-aware uncertainty: two-way (initiator x opponent) player bootstrap with product Poisson weights,
     and a calendar-day block bootstrap, for AUC differences between models
  E. per-question responses: ordering by round completion time instead of game creation time; exact AUC values
Outputs: results/revision/part3_*.json"""
import json, math, os
from collections import defaultdict
import numpy as np
import pandas as pd
import trueskill
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

RAW, OUT = "data/raw/", "results/revision/"
os.makedirs(OUT, exist_ok=True)
res = {}

def ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

def sc(yv, p, w=None):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"AUC": roc_auc_score(yv, p, sample_weight=w), "LogLoss": log_loss(yv, p, sample_weight=w),
            "Brier": brier_score_loss(yv, p, sample_weight=w)}

# ================= games (identical preparation to part3_elo.py) =================
g = pd.read_csv(RAW + "games.csv.gz")
g = g[g.user_id != g.opponent_id].copy()
g["created_at"] = pd.to_datetime(g.created_at)
g = g.sort_values(["created_at", "id"]).reset_index(drop=True)
g["s"] = np.where(g.winner_id.isna(), 0.5, (g.winner_id == g.user_id).astype(float))
n = len(g)
g["period"] = np.where(np.arange(n) < 0.6 * n, "train", np.where(np.arange(n) < 0.8 * n, "val", "test"))
U, O, S = g.user_id.values, g.opponent_id.values, g.s.values
dec = S != 0.5
val, test = (g.period == "val").values & dec, (g.period == "test").values & dec

# ---------------- A. data reconstruction ----------------
m = pd.read_csv(RAW + "matches.csv.gz")
games_all = pd.read_csv(RAW + "games.csv.gz")
A = {"games_rows": int(len(games_all)), "games_id_range": [int(games_all.id.min()), int(games_all.id.max())],
     "games_window": [str(pd.to_datetime(games_all.created_at).min()), str(pd.to_datetime(games_all.created_at).max())],
     "games_self_play_removed": int((games_all.user_id == games_all.opponent_id).sum()),
     "games_status": games_all.status.value_counts().to_dict(),
     "players": int(len(set(U) | set(O))), "draw_share": float(np.mean(S == 0.5)),
     "matches_rows": int(len(m)), "matches_games": int(m.game_id.nunique()),
     "matches_games_found_in_games_table": int(m.game_id.drop_duplicates().isin(games_all.id).sum()),
     "matches_window": [str(pd.to_datetime(m.created_at).min()), str(pd.to_datetime(m.created_at).max())],
     "rounds_per_game": m.groupby("game_id").size().value_counts().to_dict(),
     "rounds_with_identical_created_at_within_game_share": float(m.groupby("game_id").created_at.nunique().eq(1).mean()),
     "rounds_missing_category": int(m.cat_id.isna().sum()),
     "rounds_missing_user_answers": int(m.user_first_question.isna().sum()),
     "rounds_missing_opponent_answers": int(m.opponent_first_question.isna().sum()),
     "rounds_with_category_and_at_least_one_player_answers": int((m.cat_id.notna() & (m.user_first_question.notna() | m.opponent_first_question.notna())).sum())}
res["A_reconstruction"] = A
print("A:", A)

# ---------------- rating engines ----------------
def run_elo(K, home=0.0):
    r = defaultdict(float); pred = np.empty(n)
    for i in range(n):
        u, o = U[i], O[i]
        p = 1 / (1 + 10 ** ((r[o] - r[u] - home) / 400)); pred[i] = p
        r[u] += K * (S[i] - p); r[o] -= K * (S[i] - p)
    return pred

SC = 173.7178
def g_(phi):
    return 1 / math.sqrt(1 + 3 * phi ** 2 / math.pi ** 2)

def vol_update(sig, phi, v, delta, tau):
    A_ = math.log(sig ** 2)
    f = lambda x: (math.exp(x) * (delta ** 2 - phi ** 2 - v - math.exp(x))) / (2 * (phi ** 2 + v + math.exp(x)) ** 2) - (x - A_) / tau ** 2
    lo = A_
    if delta ** 2 > phi ** 2 + v:
        hi = math.log(delta ** 2 - phi ** 2 - v)
    else:
        k = 1
        while f(A_ - k * tau) < 0:
            k += 1
        hi = A_ - k * tau
    flo, fhi = f(lo), f(hi)
    for _ in range(60):
        if abs(hi - lo) < 1e-6:
            break
        c = lo + (lo - hi) * flo / (fhi - flo); fc = f(c)
        if fc * fhi <= 0:
            lo, flo = hi, fhi
        else:
            flo /= 2
        hi, fhi = c, fc
    return math.exp(lo / 2)

def run_glicko2(tau, period_key, phi0=350 / SC, sig0=0.06):
    """Glicko-2 with rating periods defined by period_key (array, one key per game, non-decreasing in time).
    Predictions within a period use the ratings at the start of the period (plus RD inflation for inactivity is
    applied once per period in which a player plays, as in Glickman's algorithm for players who compete)."""
    st = {}
    def get(p_):
        if p_ not in st:
            st[p_] = [0.0, phi0, sig0]
        return st[p_]
    pred = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j < n and period_key[j] == period_key[i]:
            j += 1
        games_by_player = defaultdict(list)
        for t in range(i, j):
            a, b = get(U[t]), get(O[t])
            pred[t] = 1 / (1 + math.exp(-g_(math.sqrt(a[1] ** 2 + b[1] ** 2)) * (a[0] - b[0])))
            games_by_player[U[t]].append((O[t], S[t])); games_by_player[O[t]].append((U[t], 1 - S[t]))
        snap = {p_: tuple(get(p_)) for p_ in games_by_player}
        for p_, lst in games_by_player.items():
            mu, phi, sig = snap[p_]
            vinv, dsum = 0.0, 0.0
            for o_, s_ in lst:
                mo, po, _ = snap[o_]
                gg = g_(po); E = 1 / (1 + math.exp(-gg * (mu - mo)))
                vinv += gg ** 2 * E * (1 - E); dsum += gg * (s_ - E)
            v = 1 / vinv; delta = v * dsum
            sig_n = vol_update(sig, phi, v, delta, tau)
            phi_star = math.sqrt(phi ** 2 + sig_n ** 2)
            phi_n = 1 / math.sqrt(1 / phi_star ** 2 + 1 / v)
            st[p_] = [mu + phi_n ** 2 * dsum, min(phi_n, phi0), sig_n]
        i = j
    return pred

def run_trueskill(beta, tau, sigma0, draw_p, margin=False):
    env = trueskill.TrueSkill(mu=25.0, sigma=sigma0, beta=beta, tau=tau, draw_probability=draw_p)
    eps = trueskill.calc_draw_margin(draw_p, 2, env)
    r = defaultdict(env.create_rating); pred = np.empty(n); pdraw = np.empty(n)
    for i in range(n):
        a, b = r[U[i]], r[O[i]]
        c = math.sqrt(2 * beta ** 2 + a.sigma ** 2 + b.sigma ** 2); d = a.mu - b.mu
        pw, pl = norm.cdf((d - eps) / c), norm.cdf((-d - eps) / c)
        pred[i] = pw / (pw + pl) if margin else norm.cdf(d / c)
        pdraw[i] = 1 - pw - pl
        if S[i] == 0.5:
            r[U[i]], r[O[i]] = env.rate_1vs1(a, b, drawn=True)
        elif S[i] == 1:
            r[U[i]], r[O[i]] = env.rate_1vs1(a, b)
        else:
            r[O[i]], r[U[i]] = env.rate_1vs1(b, a)
    return pred, pdraw

vl = lambda p: log_loss(S[val], np.clip(p[val], 1e-6, 1 - 1e-6))
B = {}
# Elo grid (as in part3_elo.py, finer)
elo_grid = {K: vl(run_elo(K)) for K in (16, 32, 48, 64, 80, 96, 112, 128, 160, 192)}
K_best = min(elo_grid, key=elo_grid.get); p_elo = run_elo(K_best)
B["Elo"] = {"validation_logloss_by_K": elo_grid, "K_selected": K_best, "test": sc(S[test], p_elo[test])}
print("Elo", K_best, B["Elo"]["test"], flush=True)

# Glicko-2 with different rating periods
t_ = g.created_at
periods = {"one game": np.arange(n), "one hour": t_.dt.floor("h").values.astype("int64"),
           "one day": t_.dt.floor("D").values.astype("int64"), "one week": t_.dt.to_period("W").dt.start_time.values.astype("int64")}
B["Glicko-2"] = {}
gl_preds = {}
for pname, key in periods.items():
    grid = {}
    for tau in (0.2, 0.3, 0.5, 0.9, 1.2):
        p_ = run_glicko2(tau, key); grid[tau] = (vl(p_), p_)
    tb = min(grid, key=lambda k: grid[k][0]); gl_preds[pname] = grid[tb][1]
    B["Glicko-2"][pname] = {"tau_selected": tb, "validation_logloss": grid[tb][0], "test": sc(S[test], grid[tb][1][test]),
                            "periods": int(len(np.unique(key)))}
    print("Glicko-2", pname, tb, B["Glicko-2"][pname]["test"], flush=True)
gl_best = min(B["Glicko-2"], key=lambda k: B["Glicko-2"][k]["validation_logloss"])
B["Glicko-2_selected_by_validation"] = gl_best

# TrueSkill grid
beta0, tau0, sig0 = 25 / 6, 25 / 300, 25 / 3
ts_grid = {}
for bf in (0.5, 1.0, 2.0):
    for tf in (0.25, 1.0, 4.0):
        for sf in (0.5, 1.0):
            for dp in (0.07, 0.15):
                p_, _ = run_trueskill(beta0 * bf, tau0 * tf, sig0 * sf, dp)
                ts_grid[(bf, tf, sf, dp)] = (vl(p_), p_)
tsb = min(ts_grid, key=lambda k: ts_grid[k][0])
p_ts_default, pdraw_default = run_trueskill(beta0, tau0, sig0, 0.07)
p_ts_tuned = ts_grid[tsb][1]
p_ts_margin, pdraw_tuned = run_trueskill(beta0 * tsb[0], tau0 * tsb[1], sig0 * tsb[2], tsb[3], margin=True)
B["TrueSkill"] = {"default": {"params": {"beta": beta0, "tau": tau0, "sigma0": sig0, "draw_probability": 0.07},
                              "validation_logloss": vl(p_ts_default), "test": sc(S[test], p_ts_default[test])},
                  "tuned": {"params": {"beta": beta0 * tsb[0], "tau": tau0 * tsb[1], "sigma0": sig0 * tsb[2], "draw_probability": tsb[3]},
                            "validation_logloss": ts_grid[tsb][0], "test": sc(S[test], p_ts_tuned[test])},
                  "tuned_with_draw_margin_conditional": {"test": sc(S[test], p_ts_margin[test])},
                  "grid_size": len(ts_grid)}
print("TrueSkill", B["TrueSkill"], flush=True)
res["B_rating_sensitivity"] = B

# ---------------- C. draws ----------------
te_all = (g.period == "test").values
C = {"test_games_all": int(te_all.sum()), "test_draws": int((te_all & ~dec).sum()),
     "note": "Elo and Glicko-2 give an expected score E = P(win) + 0.5 P(draw); on decisive games E is used as P(win | decisive).",
     "expected_score_Brier_all_test_games_incl_draws": {
         "Elo": float(np.mean((S[te_all] - p_elo[te_all]) ** 2)),
         f"Glicko-2 ({gl_best})": float(np.mean((S[te_all] - gl_preds[gl_best][te_all]) ** 2)),
         "TrueSkill (tuned)": float(np.mean((S[te_all] - p_ts_tuned[te_all]) ** 2))},
     "trueskill_mean_predicted_draw_probability_test": float(pdraw_tuned[te_all].mean()),
     "observed_draw_share_test": float(np.mean(~dec[te_all])),
     "AUC_draw_vs_decisive_from_trueskill_pdraw": float(roc_auc_score(~dec[te_all], pdraw_tuned[te_all]))}
res["C_draws"] = C
print("C:", C)

# ---------------- D. dependence-aware bootstrap for the duel comparisons ----------------
models = {"Elo": p_elo, f"Glicko-2 ({gl_best})": gl_preds[gl_best], "Glicko-2 (one game)": gl_preds["one game"],
          "TrueSkill (default)": p_ts_default, "TrueSkill (tuned)": p_ts_tuned}
lgb_pred = pd.read_csv("results/part3_game_test_predictions.csv")
assert np.array_equal(lgb_pred.game_id.values, g.id.values[test])
models["LightGBM (history + Elo)"] = np.full(n, np.nan); models["LightGBM (history + Elo)"][test] = lgb_pred.lightgbm.values
models["Logistic regression"] = np.full(n, np.nan); models["Logistic regression"][test] = lgb_pred.logreg.values
yt = S[test]; ut, ot = U[test], O[test]
players = np.unique(np.concatenate([ut, ot])); pidx = {p_: k for k, p_ in enumerate(players)}
ui = np.array([pidx[p_] for p_ in ut]); oi = np.array([pidx[p_] for p_ in ot])
days = g.created_at.dt.floor("D").values[test]; ud = np.unique(days); dpos = {d: np.flatnonzero(days == d) for d in ud}
rng = np.random.default_rng(7); NB = 1000
W = [rng.poisson(1.0, len(players)) for _ in range(NB)]
DB = [np.concatenate([dpos[d] for d in rng.choice(ud, len(ud))]) for _ in range(NB)]
pairs = [("Elo", f"Glicko-2 ({gl_best})"), ("Elo", "Glicko-2 (one game)"), ("Elo", "TrueSkill (default)"),
         ("Elo", "TrueSkill (tuned)"), ("LightGBM (history + Elo)", "Elo"), ("Elo", "Logistic regression")]
D = {"test_decisive_games": int(test.sum()), "players_in_test": int(len(players)), "days_in_test": int(len(ud)), "B": NB,
     "method": "two-way player bootstrap: each player receives a Poisson(1) weight, a game the product of its two players' weights; day-block bootstrap resamples calendar days",
     "test_AUC": {k: float(roc_auc_score(yt, v[test])) for k, v in models.items()}, "differences": {}}
for a_, b_ in pairs:
    pa, pb = models[a_][test], models[b_][test]
    tw, dd = [], []
    for k in range(NB):
        w_ = W[k][ui] * W[k][oi]; msk = w_ > 0
        tw.append(roc_auc_score(yt[msk], pa[msk], sample_weight=w_[msk]) - roc_auc_score(yt[msk], pb[msk], sample_weight=w_[msk]))
        ix = DB[k]; dd.append(roc_auc_score(yt[ix], pa[ix]) - roc_auc_score(yt[ix], pb[ix]))
    gi = [rng.integers(0, len(yt), len(yt)) for _ in range(300)]
    D["differences"][f"{a_} minus {b_}"] = {"delta_AUC": float(roc_auc_score(yt, pa) - roc_auc_score(yt, pb)),
                                           "CI_two_way_player": ci(tw), "CI_day_block": ci(dd),
                                           "CI_naive_game": ci([roc_auc_score(yt[i], pa[i]) - roc_auc_score(yt[i], pb[i]) for i in gi])}
    print(a_, b_, D["differences"][f"{a_} minus {b_}"], flush=True)
res["D_duel_uncertainty"] = D
json.dump(res, open(OUT + "part3_ratings.json", "w"), indent=2, default=float)

# ---------------- E. per-question responses: ordering sensitivity and exact values ----------------
m = m.dropna(subset=["user_correct_count", "cat_id"])
m = m.merge(g[["id", "user_id", "opponent_id"]], left_on="game_id", right_on="id", suffixes=("", "_g"))
m["created_at"] = pd.to_datetime(m.created_at); m["updated_at"] = pd.to_datetime(m.updated_at)

def build_resp(order_cols):
    mm = m.sort_values(order_cols)
    rows = []
    for r in mm.itertuples(index=False):
        for who, cols in (("user_id", ("user_first_question", "user_second_question", "user_thirty_question")),
                          ("opponent_id", ("opponent_first_question", "opponent_second_question", "opponent_thirty_question"))):
            for c in cols:
                v = getattr(r, c)
                if not pd.isna(v):
                    rows.append((getattr(r, who), int(r.cat_id), int(v)))
    return pd.DataFrame(rows, columns=["player", "cat", "y"])

def run_pc(Rd, a, b, per_cat):
    Pp, Cc, Yy = Rd.player.values, Rd.cat.values, Rd.y.values
    th, d, thc = defaultdict(float), defaultdict(float), defaultdict(float)
    nu, nc, nuc = defaultdict(int), defaultdict(int), defaultdict(int)
    K = lambda k: a / (1 + b * k); pred = np.empty(len(Rd))
    for i in range(len(Rd)):
        u, c, yv = Pp[i], Cc[i], Yy[i]
        z = th[u] + (thc[(u, c)] if per_cat else 0) - d[c]
        p = 1 / (1 + math.exp(-z)); pred[i] = p; e = yv - p
        th[u] += K(nu[u]) * e; d[c] -= K(nc[c]) * e
        if per_cat:
            thc[(u, c)] += K(nuc[(u, c)]) * e; nuc[(u, c)] += 1
        nu[u] += 1; nc[c] += 1
    return pred

E = {}
for lab, cols in (("game creation time, round (as reported)", ["created_at", "id"]),
                  ("round completion time (updated_at)", ["updated_at", "id"])):
    Rd = build_resp(cols); nR = len(Rd); te_r = np.arange(nR) >= 0.8 * nR; va_r = (np.arange(nR) >= 0.6 * nR) & ~te_r
    out = {"responses": int(nR)}
    for per_cat, name in ((False, "Elo"), (True, "Elo + learner x category")):
        best = None
        for a in (0.03, 0.05, 0.1, 0.2, 0.4):
            for b in (0.001, 0.005, 0.02, 0.05):
                p_ = run_pc(Rd, a, b, per_cat); ll = log_loss(Rd.y.values[va_r], p_[va_r])
                if best is None or ll < best[0]:
                    best = (ll, a, b, p_)
        out[name] = {"alpha_beta": [best[1], best[2]], "AUC": float(roc_auc_score(Rd.y.values[te_r], best[3][te_r])),
                     "LogLoss": float(log_loss(Rd.y.values[te_r], best[3][te_r]))}
        out[name + "_pred"] = best[3][te_r]
    yy_ = Rd.y.values[te_r]; pl = Rd.player.values[te_r]; up = np.unique(pl); pos = {u: np.flatnonzero(pl == u) for u in up}
    rb = np.random.default_rng(9)
    bt = []
    for _ in range(1000):
        ix = np.concatenate([pos[u] for u in rb.choice(up, len(up))])
        bt.append(roc_auc_score(yy_[ix], out["Elo + learner x category_pred"][ix]) - roc_auc_score(yy_[ix], out["Elo_pred"][ix]))
    out["delta_AUC"] = out["Elo + learner x category"]["AUC"] - out["Elo"]["AUC"]
    out["delta_AUC_CI_learner_cluster"] = ci(bt)
    out.pop("Elo_pred"); out.pop("Elo + learner x category_pred")
    E[lab] = out
    print("E", lab, out, flush=True)
res["E_response_ordering"] = E
json.dump(res, open(OUT + "part3_ratings.json", "w"), indent=2, default=float)
print("saved")
