"""Figures: cold-start validation on items without item-bank duplicates (Fig. 3) and the calibration experiment (Fig. 4)."""
import glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, FixedLocator
from scipy.stats import pearsonr

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
C, OUT = "#1f5f8b", "results/extended/"

# Fig. 3: cold-start validation on items without duplicates in the item bank
cl = pd.read_csv(OUT + "coldstart_clean_items.csv").set_index("id")
files = sorted(glob.glob("results/finetune/kaz-roberta-conversational__text__s*__tour.csv"))
p = pd.concat([pd.read_csv(f).set_index("id").pred for f in files], axis=1).mean(axis=1).loc[cl.index]
fig, ax = plt.subplots(figsize=(3.4, 2.9))
ax.scatter(p, cl.b_2PL, s=7, alpha=0.55, color=C, lw=0)
k, b0 = np.polyfit(p, cl.b_2PL, 1); xs = np.linspace(p.min(), p.max(), 10)
ax.plot(xs, k * xs + b0, color="#9a5a00", lw=1.2)
ax.set_xlabel("Difficulty predicted from text (KazRoBERTa, logit)"); ax.set_ylabel("IRT 2PL difficulty b")
ax.text(0.03, 0.95, f"n = {len(p)}, r = {pearsonr(p, cl.b_2PL)[0]:.2f}", transform=ax.transAxes, va="top", fontsize=7.5)
fig.tight_layout(); fig.savefig(OUT + "fig4_coldstart_clean.png", dpi=300); plt.close(fig)

# calibration experiment: k = 0 (prior only) to 80 responses, bootstrap 95% CIs over items as error bars
F = json.load(open(OUT + "calibration_experiment.json"))
KS = [int(k) for k in F["k"]]
pos = np.arange(len(KS))
style = [("weak N(0,3^2)", "#8a96a3", "o", "Weak prior N(0, 3²)", -0.12),
         ("population N(mean, var)", "#b8860b", "s", "Population prior", 0.0),
         ("text: XLM-R large (pre-specified)", C, "^", "Text prior (XLM-R large)", 0.12)]
fig, ax = plt.subplots(figsize=(3.4, 2.7))
for pn, col, mk, lab, dx in style:
    r = np.array([F["rmse"][pn][str(k)] for k in KS])
    lo = np.array([F["rmse_CI"][pn][str(k)][0] for k in KS]); hi = np.array([F["rmse_CI"][pn][str(k)][1] for k in KS])
    ax.errorbar(pos + dx, r, yerr=[r - lo, hi - r], color=col, marker=mk, ms=4, lw=1.1, capsize=2.5, elinewidth=1.0, label=lab)
ax.set_xticks(pos); ax.set_xticklabels([str(k) for k in KS])
ax.set_xlabel("Responses to the new item (k; 0 = prior only)"); ax.set_ylabel("RMSE of difficulty (logit)")
ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(OUT + "fig_calibration.png", dpi=300); plt.close(fig)
print("figures written")
