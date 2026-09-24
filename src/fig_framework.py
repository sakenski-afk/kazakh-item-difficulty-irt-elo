"""Fig. 1: overview of the three-part framework and the links between parts (numbers taken from the analyses)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix"})
fig, ax = plt.subplots(figsize=(7.2, 3.21))
ax.set_xlim(0, 100); ax.set_ylim(-6, 50); ax.axis("off")
C, FILL, T = "#1f5f99", "#e4eef7", "#111111"
XS, W = (1, 35.5, 70), 29          # left edges and width of the three columns (gaps of 5.5 units)

def box(x, y, w, h, title, lines, fill):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.4", fc=fill, ec=C, lw=1.6))
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=11.5, weight="bold", color=T)
    ax.text(x + w / 2, y + h - 6.8, "\n".join(lines), ha="center", va="top", fontsize=9.6, color=T, linespacing=1.3)

box(XS[0], 36, W, 12.5, "D1  Item bank", ["30,444 items, 177.5 M attempts", "kz / ru / en, 50 categories"], "white")
box(XS[1], 36, W, 12.5, "D2  Tournament responses", ["67,990 responses, 680 learners", "416 items (2025–2026)"], "white")
box(XS[2], 36, W, 12.5, "D3  Duels", ["100,000 games, 5,920 players", "322,046 responses (2022)"], "white")
box(XS[0], 5, W, 21.5, "Part 1  Difficulty from text",
    ["TF-IDF, embeddings + Ridge", "fine-tuned mBERT, XLM-R,", "KazRoBERTa", r"target: logit difficulty $b$"], FILL)
box(XS[1], 5, W, 21.5, "Part 2  IRT calibration",
    ["1PL / 2PL / 3PL, marginal MAP", "held-out AUC, log-loss, Brier;", "approximate AIC / BIC", r"item $b$, $a$, $c$; learner $\theta$"], FILL)
box(XS[2], 5, W, 21.5, "Part 3  Rating models",
    ["Elo, Glicko-2, TrueSkill", "vs. LR and LightGBM", r"learner $\times$ category Elo", r"category difficulty $d_c$"], FILL)
centers = [x + W / 2 for x in XS]
for x in centers:
    ax.annotate("", xy=(x, 27.6), xytext=(x, 35.4), arrowprops=dict(arrowstyle="-|>", color=C, lw=1.4))
# Part 1 -> Part 2: text predictions evaluated against IRT difficulty on new items
gap_mid = (XS[0] + W + XS[1]) / 2
ax.annotate("", xy=(XS[1] - 0.5, 8.0), xytext=(XS[0] + W + 0.5, 8.0),
            arrowprops=dict(arrowstyle="-|>", color=C, lw=1.4, mutation_scale=12))
ax.text(gap_mid, 18.0, "text predictions\non 327 new items", ha="center", va="center", fontsize=7.6,
        color=C, rotation=90, linespacing=1.15)
ax.annotate("", xy=(centers[2], 4.2), xytext=(centers[0], 4.2),
            arrowprops=dict(arrowstyle="<|-|>", color=C, lw=1.4, connectionstyle="arc3,rad=0.1"))
ax.text(50, -4.3, r"category level ($d_c$): Elo difficulty vs. item-bank difficulty", ha="center", fontsize=9.6, color=C)
fig.tight_layout(pad=0.2)
fig.savefig("results/figures/fig_framework.png", dpi=300)
plt.close(fig)
print("ok")
