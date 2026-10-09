"""Run the full pipeline in dependency order.
    python run_all.py                  # everything (fine-tuning grid needs a GPU: ~3 h on an RTX 5070 Ti laptop)
    python run_all.py --skip-finetune  # all CPU stages + embeddings; analysis uses whatever fine-tune runs exist
    python run_all.py --from part3     # resume from a stage
"""
import argparse, subprocess, sys, time

STAGES = [
    ("audit", ["src/audit_export.py"]),
    ("prepare", ["src/prepare_items.py"]),
    ("part2", ["src/part2_irt.py"]),
    ("tournament", ["src/prepare_tournament_items.py"]),
    ("baselines", ["src/part1_baselines.py"]),
    ("ablation", ["src/part1_options_ablation.py"]),
    ("embeddings", ["src/part1_embeddings.py"]),
    ("finetune", ["src/run_finetune_grid.py"]),
    ("part3", ["src/part3_elo.py"]),
    ("analysis", ["src/analysis.py"]),
    ("uncertainty", ["src/uncertainty_analysis.py"]),
    ("irt_robustness", ["src/irt_robustness.py"]),
    ("language", ["src/language_check.py"]),
    ("figures", ["src/fig_paper.py"]),
    ("figures_extended", ["src/figures_extended.py"]),
    ("framework", ["src/fig_framework.py"]),
    ("rev_part1", ["src/rev_part1_text.py"]),          # revision: near duplicates, truncation, ablation CIs, screening
    ("rev_part2", ["src/rev_part2_irt.py"]),           # revision: connectivity, dimensionality, Q3 null, cross-fitted calibration
    ("rev_part2b", ["src/rev_part2b_scale.py"]),       # revision: cold-start sensitivity to the common-scale assumption
    ("rev_part3", ["src/rev_part3_ratings.py"]),       # revision: rating-model sensitivity, draws, two-way player bootstrap
    ("rev_part3_table", ["src/rev_part3_table.py"]),
    ("rev_figures", ["src/rev_figures.py"]),
    ("compare", ["src/compare_to_reference.py"]),
]

ap = argparse.ArgumentParser()
ap.add_argument("--skip-finetune", action="store_true")
ap.add_argument("--skip-embeddings", action="store_true")
ap.add_argument("--from", dest="start", choices=[s for s, _ in STAGES])
args = ap.parse_args()

names = [s for s, _ in STAGES]
first = names.index(args.start) if args.start else 0
for name, cmd in STAGES[first:]:
    if (name == "finetune" and args.skip_finetune) or (name == "embeddings" and args.skip_embeddings):
        print(f"--- skip {name}")
        continue
    t0 = time.time()
    print(f"=== {name}: {' '.join(cmd)}", flush=True)
    subprocess.run([sys.executable] + cmd, check=True)
    print(f"=== {name} done in {time.time() - t0:.0f}s", flush=True)
