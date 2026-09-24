"""Part 1 dataset: calibrated items (>=30 attempts) with text, metadata, target and leakage-safe splits."""
import re
import numpy as np
import pandas as pd

RAW = "data/raw/"
MIN_N = 30
KZ_CHARS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")

q = pd.read_csv(RAW + "questions.csv.gz", low_memory=False)
a = pd.read_csv(RAW + "answers.csv.gz")
ca = pd.read_csv(RAW + "correct_answers.csv.gz")
cats = pd.read_csv(RAW + "cats.csv.gz")

q["n"] = q.correct + q.fail
q = q[q.n >= MIN_N].copy()
q["text"] = q.questions.fillna("").astype(str).str.strip()
q = q[q.text.str.len() >= 3]

# target: proportion correct and its logit (IRT-like difficulty: higher = harder)
q["p"] = q.correct / q.n
p_s = (q.correct + 0.5) / (q.n + 1.0)
q["b_logit"] = -np.log(p_s / (1 - p_s))

# language by script: Kazakh-specific letters => kz, Cyrillic otherwise => ru, Latin => en
def lang_of(t):
    if any(c in KZ_CHARS for c in t):
        return "kz"
    if re.search(r"[а-яА-ЯёЁ]", t):
        return "ru"
    return "en"
q["lang_det"] = q.text.map(lang_of)
q["is_translation"] = q.main_question_id.notna().astype(int)
q["has_image"] = q.image.notna().astype(int)
q["user_submitted"] = q.user_id.notna().astype(int)
q["n_chars"] = q.text.str.len()
q["n_words"] = q.text.str.split().str.len()
q["has_digit"] = q.text.str.contains(r"\d").astype(int)
q["has_qmark"] = q.text.str.contains(r"\?").astype(int)
q["report_rate"] = q.report / q.n * 1000
q["year"] = pd.to_datetime(q.created_at).dt.year

cat_names = cats.set_index("id")
q["cat_name"] = q.cat_id.map(cat_names.name)
q["cat_name_ru"] = q.cat_id.map(cat_names.name_ru)

# answer options (available for a subset only)
key = set(ca.answer_id)
a["is_key"] = a.id.isin(key)
a["text"] = [str(t) if isinstance(t, str) else "" for t in a.text]
a = a.sort_values("id")
opts = pd.DataFrame({
    "key_text": a[a.is_key].groupby("question_id").text.agg(" | ".join),
    "distractors": a[~a.is_key].groupby("question_id").text.agg(" | ".join),
})
q = q.merge(opts, left_on="id", right_index=True, how="left")
q["has_options"] = q.key_text.notna().astype(int)

# leakage-safe split: a kz original and its ru translation, and identical stems, share one group
q["norm"] = q.text.str.lower().str.replace(r"[^\w]+", " ", regex=True).str.strip()
parent = {}
def find(x):
    while parent.get(x, x) != x:
        x = parent[x]
    return x
def union(x, y):
    rx, ry = find(x), find(y)
    if rx != ry:
        parent[ry] = rx
ids = set(q.id)
for i, m in zip(q.id, q.main_question_id):
    if pd.notna(m) and int(m) in ids:
        union(int(m), i)
for _, grp in q.groupby("norm").id:
    g = list(grp)
    for x in g[1:]:
        union(g[0], x)
q["group"] = q.id.map(find)

rng = np.random.default_rng(42)
groups = q.group.unique()
rng.shuffle(groups)
cut = np.cumsum(q.groupby("group").size().reindex(groups).values) / len(q)
split = np.where(cut <= 0.8, "train", np.where(cut <= 0.9, "val", "test"))
q["split"] = q.group.map(dict(zip(groups, split)))

cols = ["id", "text", "key_text", "distractors", "cat_id", "cat_name", "cat_name_ru", "lang_det",
        "is_translation", "main_question_id", "has_image", "user_submitted", "has_options", "n_chars",
        "n_words", "has_digit", "has_qmark", "report", "report_rate", "year", "correct", "fail", "n",
        "p", "b_logit", "group", "split"]
q[cols].to_csv("data/items.csv.gz", index=False)

print("items", len(q), "| splits", q.split.value_counts().to_dict())
print("lang", q.lang_det.value_counts().to_dict(), "| translations", q.is_translation.sum(),
      "| with options", q.has_options.sum(), "| categories", q.cat_id.nunique())
print("p mean/sd", round(q.p.mean(), 3), round(q.p.std(), 3))
# split balance check
print(q.groupby("split")[["p", "n_chars"]].mean().round(3))
