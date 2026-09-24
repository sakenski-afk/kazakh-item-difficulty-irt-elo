"""Build the open (level-2) data package: item-level statistics WITHOUT item texts or any learner-level records.

Privacy / IP rules applied:
  * no question or answer texts, no author (user) ids, no learner ids, no timestamps;
  * platform item ids are replaced by random ids (separate namespaces for the item bank and tournament items);
    the id mapping is written to --private_dir and must NOT be published;
  * only derived, non-identifying item features are kept (length in characters/words, flags).
Usage: python src/export_open_data.py --out ../open_data --private_dir ../private_id_mapping
"""
import argparse, glob, hashlib, json, os, secrets
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="open_data")
ap.add_argument("--private_dir", default="private_id_mapping")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True); os.makedirs(args.private_dir, exist_ok=True)
rng = np.random.default_rng(secrets.randbits(64))  # unrecorded seed: the mapping cannot be regenerated

def new_ids(old, prefix):
    perm = rng.permutation(len(old))
    return dict(zip(old, [f"{prefix}{i + 1:05d}" for i in perm]))

# ---------------- item bank (Part 1) ----------------
it = pd.read_csv("data/items.csv.gz")
item_map = new_ids(it.id.tolist(), "I")
it["item_id"] = it.id.map(item_map)
it["original_item_id"] = it.main_question_id.map(lambda m: item_map.get(int(m)) if pd.notna(m) and int(m) in item_map else None)
grp_map = new_ids(sorted(it.group.unique()), "G")
it["group_id"] = it.group.map(grp_map)
cats = pd.read_csv("data/raw/cats.csv.gz").set_index("id")
it["category_name_ru"] = it.cat_id.map(cats.name_ru)
items = it[["item_id", "group_id", "split", "cat_id", "cat_name", "category_name_ru", "lang_det", "is_translation",
            "original_item_id", "has_image", "user_submitted", "has_options", "n_chars", "n_words", "has_digit",
            "has_qmark", "n", "correct", "fail", "p", "b_logit"]].rename(columns={
    "cat_id": "category_id", "cat_name": "category_name_kz", "lang_det": "language", "n": "n_attempts",
    "correct": "n_correct", "fail": "n_incorrect", "p": "p_correct"})
items["category_id"] = items.category_id.astype("Int64")
items.sort_values("item_id").to_csv(f"{args.out}/items_difficulty.csv", index=False)

# ---------------- Part 1 test-set predictions ----------------
test = it[it.split == "test"][["id", "item_id", "b_logit"]].set_index("id")
pred = pd.DataFrame(index=test.index)
base = pd.read_csv("results/part1_baseline_preds.csv").set_index("id")
for c in base.columns:
    if c != "b_true":
        pred[f"baseline | {c}"] = base.loc[test.index, c]
for f in sorted(glob.glob("results/embeddings/*__test.csv")):
    name, variant = os.path.basename(f).split("__")[:2]
    pred[f"embeddings | {name} + Ridge | {variant}"] = pd.read_csv(f).set_index("id").pred.loc[test.index]
for d, tag in (("results/finetune", ""), ("results/finetune_diverged", " | DIVERGED (excluded)")):
    for f in sorted(glob.glob(f"{d}/*__test.csv")):
        mdl, inp, seed = os.path.basename(f).split("__")[:3]
        pred[f"finetuned | {mdl} | {inp} | {seed}{tag}"] = pd.read_csv(f).set_index("id").pred.loc[test.index]
pred.insert(0, "b_logit", test.b_logit); pred.insert(0, "item_id", test.item_id)
pred.sort_values("item_id").to_csv(f"{args.out}/part1_test_predictions.csv", index=False, float_format="%.6f")

# ---------------- tournament items: IRT + cold-start predictions (Part 2) ----------------
ip = pd.read_csv("results/part2_item_params.csv")
tour = pd.read_csv("data/tournament_items.csv").set_index("id")
cats_t = pd.read_csv("data/raw/questions.csv.gz", low_memory=False, usecols=["id", "cat_id"]).set_index("id").cat_id
tmap = new_ids(ip.question_id.tolist(), "T")
irt = ip.copy()
irt.insert(0, "tournament_item_id", irt.question_id.map(tmap))
irt["category_name_kz"] = irt.question_id.map(cats_t).map(cats.name)
irt["has_text_in_export"] = irt.question_id.isin(tour.index).astype(int)
clean_path = "results/extended/coldstart_clean_items.csv"      # from uncertainty_analysis.py (duplicate check)
if os.path.exists(clean_path):
    clean = set(pd.read_csv(clean_path).id)
    irt["duplicate_in_item_bank"] = (irt.question_id.isin(tour.index) & ~irt.question_id.isin(clean)).astype(int)
sources = {"TF-IDF + Ridge": ["results/tfidf__text__tour.csv"]}
for f in sorted(glob.glob("results/embeddings/*__text__tour.csv")):
    sources[os.path.basename(f).split("__")[0] + " + Ridge"] = [f]
for f in sorted(glob.glob("results/finetune/*__text__s*__tour.csv")):
    sources.setdefault(os.path.basename(f).split("__")[0] + " (fine-tuned, seed ensemble)", []).append(f)
for name, files in sources.items():
    p = pd.concat([pd.read_csv(f).set_index("id").pred for f in files], axis=1).mean(axis=1)
    irt[f"pred_text | {name}"] = irt.question_id.map(p)
irt = irt.drop(columns=["question_id"]).rename(columns={"n_resp": "n_responses", "p_value": "p_correct"})
irt.sort_values("tournament_item_id").to_csv(f"{args.out}/tournament_items_irt.csv", index=False, float_format="%.6f")

# ---------------- category level (Part 3 link) ----------------
cd = pd.read_csv("results/table_elo_category_linkage.csv")
cd["category_name_ru"] = cd.cat_id.map(cats.name_ru)
cd = cd.rename(columns={"cat_id": "category_id", "cat_name": "category_name_kz", "bank_b": "item_bank_mean_b",
                        "n_items": "item_bank_n_items", "n_resp": "duel_n_responses", "round_accuracy": "duel_accuracy",
                        "model_b_test": "part1_model_mean_b_test"})
cd.to_csv(f"{args.out}/category_difficulty.csv", index=False, float_format="%.6f")

# ---------------- private mapping (never publish) ----------------
pd.DataFrame({"platform_item_id": list(item_map), "item_id": list(item_map.values())}).to_csv(
    f"{args.private_dir}/item_id_mapping.csv", index=False)
pd.DataFrame({"platform_item_id": list(tmap), "tournament_item_id": list(tmap.values())}).to_csv(
    f"{args.private_dir}/tournament_item_id_mapping.csv", index=False)

# ---------------- checksums ----------------
with open(f"{args.out}/SHA256SUMS.txt", "w") as fh:
    for f in sorted(os.listdir(args.out)):
        if f.endswith(".csv"):
            fh.write(f"{hashlib.sha256(open(os.path.join(args.out, f), 'rb').read()).hexdigest()}  {f}\n")
print(f"items {len(items)}, test predictions {len(pred)} x {pred.shape[1] - 2} models, "
      f"tournament items {len(irt)}, categories {len(cd)} -> {args.out}")
