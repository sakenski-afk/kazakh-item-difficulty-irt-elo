"""Revision figure: cross-fitted calibration experiment (Fig. 4 of the revised paper), from the saved JSON."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

F = json.load(open("results/revision/part2_calibration_crossfit.json"))
dm = F["designs"]["a=1 | reference: disjoint half"]
KS = [5, 10, 20, 40]; xpos = [2.5] + KS
plt.rcParams.update({"font.size": 8, "font.family": "Times New Roman", "mathtext.fontset": "stix"})
fig, ax = plt.subplots(figsize=(3.4, 2.7))
style = {"weak": ("#8a96a3", "o", "Weak prior", 0.0), "population": ("#9a5a00", "s", "Population prior", -0.04),
         "text: XLM-R large": ("#1f5f8b", "^", "Text prior (XLM-R large)", 0.04)}
for pn, (col, mk, lab, dx) in style.items():
    ks = ["0"] + [str(k) for k in KS]; xx = xpos
    r = np.array([dm["rmse"][pn][k] for k in ks]); lo = np.array([dm["rmse_CI"][pn][k][0] for k in ks]); hi = np.array([dm["rmse_CI"][pn][k][1] for k in ks])
    ax.errorbar(np.array(xx) * np.exp(dx), r, yerr=[r - lo, hi - r], marker=mk, color=col, lw=1.1, ms=4, capsize=2, label=lab)
ax.set_xscale("log"); ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticks(xpos); ax.set_xticklabels(["0"] + [str(k) for k in KS])
ax.set_xlabel("Responses to the new item (k)"); ax.set_ylabel("RMSE of difficulty (logit)")
ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig("results/revision/fig_calibration_crossfit.png", dpi=1200); plt.close(fig)
print("saved")
