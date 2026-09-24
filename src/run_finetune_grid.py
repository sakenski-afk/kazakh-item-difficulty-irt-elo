"""Part 1 fine-tuning grid: 4 encoders x 2 inputs x 3 successful seeds.
A run whose best validation Pearson r is below DIVERGED_R has collapsed to a near-constant predictor
(a known instability of large encoders); it is moved to results/finetune_diverged/ and replaced by the next seed.
Existing runs are skipped, so the grid can be resumed after an interruption."""
import argparse, glob, json, os, shutil, subprocess, sys

MODELS = {  # model id -> extra arguments
    "google-bert/bert-base-multilingual-cased": [],
    "kz-transformers/kaz-roberta-conversational": [],
    "FacebookAI/xlm-roberta-base": [],
    "FacebookAI/xlm-roberta-large": ["--lr", "1e-5", "--grad_ckpt"],
}
INPUTS = ["cat_text", "text"]
N_SEEDS, MAX_SEED, DIVERGED_R = 3, 8, 0.2
OUT, BAD = "results/finetune", "results/finetune_diverged"

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="*", default=list(MODELS))
ap.add_argument("--epochs", type=int, default=5)
args = ap.parse_args()
os.makedirs(OUT, exist_ok=True); os.makedirs(BAD, exist_ok=True)
env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYTHONIOENCODING="utf-8",
           TRANSFORMERS_VERBOSITY="error", HF_HUB_DISABLE_SYMLINKS_WARNING="1")

def diverged(meta):
    r = meta.get("val_pearson")
    return r is not None and r < DIVERGED_R

for model in args.models:
    for inp in INPUTS:
        base = f"{model.split('/')[-1]}__{inp}"
        ok = 0
        for seed in range(1, MAX_SEED + 1):
            if ok == N_SEEDS:
                break
            tag = f"{base}__s{seed}"
            path = f"{OUT}/{tag}.json"
            if not os.path.exists(path):
                if os.path.exists(f"{BAD}/{tag}.json"):
                    continue
                cmd = [sys.executable, "src/part1_finetune.py", "--model", model, "--input", inp, "--seed", str(seed),
                       "--epochs", str(args.epochs), "--out_dir", OUT] + MODELS.get(model, [])
                print(">>", " ".join(cmd), flush=True)
                subprocess.run(cmd, check=True, env=env)
            meta = json.load(open(path))
            if diverged(meta):
                print(f"!! {tag} diverged (val r = {meta['val_pearson']:.3f}); replacing with the next seed", flush=True)
                for f in glob.glob(f"{OUT}/{tag}*"):
                    shutil.move(f, os.path.join(BAD, os.path.basename(f)))
                continue
            ok += 1
        print(f"== {base}: {ok} successful seeds", flush=True)
