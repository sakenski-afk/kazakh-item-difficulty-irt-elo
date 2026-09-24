"""Publication figures (column width) re-plotted from result tables."""
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "font.family": "DejaVu Sans"})
C = "#1f5f8b"
EN = {16: "Medicine", 10: "Sport", 6: "Literature", 15: "Brain training", 7: "Literacy", 4: "Tech. progress",
      17: "Cinema & TV", 12: "Free topic", 1: "History of KZ", 5: "Art & culture", 13: "World history",
      9: "Geography", 18: "Songs & dance"}

# Fig. 5: category linkage between Part 3 (Elo) and Part 1 (item bank)
l = pd.read_csv("results/table_elo_category_linkage.csv")
with plt.rc_context({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 10}):
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    small = l.n_resp < 1000
    ax.scatter(l.bank_b[~small], l.elo_difficulty[~small], color="#0b4f9c", s=30, zorder=3, label="≥ 1,000 responses")
    ax.scatter(l.bank_b[small], l.elo_difficulty[small], facecolor="white", edgecolor="#0b4f9c", lw=1.2, s=30,
               zorder=3, label="< 1,000 responses")
    POS = {18: (-0.45, 0.12, "left"), 13: (-0.03, 0.21, "left"), 5: (0.265, 0.06, "center"), 1: (-0.60, -0.14, "left"),
           9: (-0.52, -0.25, "left"), 12: (0.07, -0.29, "left"), 17: (0.05, -0.44, "left"),
           4: (-0.57, -0.40, "left"), 7: (-0.74, -0.49, "left"), 15: (-0.48, -0.68, "left"),
           6: (0.045, -0.72, "left"), 10: (-0.05, -1.07, "left"), 16: (-0.50, -1.38, "left")}
    for r in l.itertuples():
        tx, ty, ha = POS[r.cat_id]
        ax.annotate(EN[r.cat_id], (r.bank_b, r.elo_difficulty), xytext=(tx, ty), fontsize=8.5, va="center", ha=ha,
                    arrowprops=dict(arrowstyle="-", color="#9aa4ae", lw=0.6, shrinkA=1, shrinkB=4))
    ax.set_xlim(-0.8, 0.4); ax.set_ylim(-1.6, 0.27)
    ax.set_xlabel("Mean item difficulty in item bank (logit)"); ax.set_ylabel("Elo category difficulty")
    ax.legend(fontsize=8.5, loc="lower right", frameon=False, handletextpad=0.3, borderaxespad=0.2)
    fig.tight_layout(pad=0.3); fig.savefig("results/figures/paper_fig5_categories.png", dpi=300); plt.close(fig)

# cold start on all items with text (sensitivity figure)
tour = pd.read_csv("data/tournament_items.csv").set_index("id")
files = glob.glob("results/finetune/kaz-roberta-conversational__text__s*__tour.csv") or ["results/tfidf__text__tour.csv"]
label = "KazRoBERTa" if "finetune" in files[0] else "TF-IDF"
p = pd.concat([pd.read_csv(f).set_index("id").pred for f in files], axis=1).mean(axis=1).loc[tour.index]
r = pearsonr(p, tour.b_2PL)[0]
fig, ax = plt.subplots(figsize=(3.4, 2.9))
ax.scatter(p, tour.b_2PL, s=7, alpha=0.55, color=C, lw=0)
k, b0 = np.polyfit(p, tour.b_2PL, 1); xs = np.linspace(p.min(), p.max(), 10); ax.plot(xs, k * xs + b0, color="#9a5a00", lw=1.2)
ax.set_xlabel(f"Difficulty predicted from text ({label}, logit)"); ax.set_ylabel("IRT 2PL difficulty b")
ax.text(0.03, 0.95, f"n = {len(p)}, r = {r:.2f}", transform=ax.transAxes, va="top", fontsize=7.5)
fig.tight_layout(); fig.savefig("results/figures/paper_fig4_coldstart.png", dpi=300); plt.close(fig)

# Fig. 2: cross-lingual
it = pd.read_csv("data/items.csv.gz"); idx = it.set_index("id")
pr = it[(it.is_translation == 1) & it.main_question_id.isin(idx.index)]
bk = idx.loc[pr.main_question_id, "b_logit"].values; br = pr.b_logit.values
fig, ax = plt.subplots(figsize=(3.4, 2.9))
ax.scatter(bk, br, s=3, alpha=0.25, color=C, lw=0)
ax.plot([-4, 4], [-4, 4], color="#8a96a3", lw=0.9, ls="--")
ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
ax.set_xlabel("Difficulty of Kazakh original (logit)"); ax.set_ylabel("Difficulty of Russian translation (logit)")
ax.text(0.03, 0.95, f"n = {len(br):,} pairs, r = {pearsonr(bk, br)[0]:.2f}\nmean shift = {np.mean(br - bk):+.2f}", transform=ax.transAxes, va="top", fontsize=7.5)
fig.tight_layout(); fig.savefig("results/figures/paper_fig3_crosslingual.png", dpi=300); plt.close(fig)

# difficulty distribution (supplementary figure, not in the article)
fig, ax = plt.subplots(figsize=(3.4, 2.3))
ax.hist(it.p, bins=40, color=C, edgecolor="white", lw=0.3)
ax.set_xlabel("Proportion correct"); ax.set_ylabel("Number of items")
ax.axvline(it.p.mean(), color="#9a5a00", lw=1); ax.text(it.p.mean() + 0.02, ax.get_ylim()[1] * 0.92, f"mean {it.p.mean():.2f}", fontsize=7)
fig.tight_layout(); fig.savefig("results/figures/paper_fig2_hist.png", dpi=300); plt.close(fig)
print("ok")
