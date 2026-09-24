"""Cold-start item set for Part 1 -> Part 2 validation: tournament items that have stored text,
joined with their IRT parameters from part2_irt.py. Output: data/tournament_items.csv."""
import pandas as pd

RAW = "data/raw/"
q = pd.read_csv(RAW + "questions.csv.gz", low_memory=False)
cats = pd.read_csv(RAW + "cats.csv.gz").set_index("id")
ip = pd.read_csv("results/part2_item_params.csv")

t = q[q.id.isin(ip.question_id)][["id", "questions", "cat_id", "image"]].rename(columns={"questions": "text"})
t["cat_name"] = t.cat_id.map(cats.name)
t["has_image"] = t.image.notna().astype(int)
t = t.merge(ip, left_on="id", right_on="question_id").drop(columns=["image", "question_id"])
t.to_csv("data/tournament_items.csv", index=False)
print(f"tournament items with text: {len(t)} of {len(ip)} calibrated")
