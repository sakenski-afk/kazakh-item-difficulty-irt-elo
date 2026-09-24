"""Compare key numbers of a fresh run (results/) with the published ones (results/reference/tables/).
Deterministic stages must match exactly (to 1e-6); fine-tuning and embedding results may differ slightly on
different GPUs, so they are reported with a tolerance."""
import json

REF, NEW = "results/reference/tables/", "results/"

def load(name, base):
    return json.load(open(base + name))

def get(d, path):
    for k in path:
        d = d[k] if not isinstance(d, list) else d[int(k)]
    return d

CHECKS = [  # (file, path, tolerance)
    ("part1_baselines.json", ["TF-IDF + metadata Ridge", "Pearson"], 1e-6),
    ("part1_baselines.json", ["TF-IDF + metadata Ridge", "RMSE_b"], 1e-6),
    ("part1_options_ablation.json", ["stem + key + distractors + surface features", "Pearson"], 1e-6),
    ("part2_irt.json", ["heldout", "2PL", "AUC"], 1e-4),
    ("part2_irt.json", ["fit", "2PL", "AIC"], 0.5),
    ("part2_irt.json", ["eap_reliability"], 1e-3),
    ("part3_elo.json", ["3a_game_outcome", "Elo", "AUC"], 1e-6),
    ("part3_elo.json", ["3b_response", "Elo + player-category skill", "AUC"], 1e-6),
    ("part1_embeddings.json", ["BGE-M3 + Ridge (text+meta)", "Pearson"], 5e-3),
    ("summary.json", ["part1_best_vs_baseline", "delta_r"], 1e-2),
    ("summary.json", ["cross_lingual", "pearson_b"], 1e-6),
    ("summary.json", ["elo_linkage", "pearson_elo_vs_bank"], 1e-6),
]
ok = True
for f, path, tol in CHECKS:
    try:
        r, n = get(load(f, REF), path), get(load(f, NEW), path)
        good = abs(r - n) <= tol
    except (FileNotFoundError, KeyError) as e:
        r, n, good = None, f"missing ({e})", False
    ok &= good
    print(f"{'OK  ' if good else 'DIFF'} {f}:{'/'.join(path):55s} reference={r}  new={n}")
print("ALL MATCH" if ok else "SOME VALUES DIFFER (see above)")
