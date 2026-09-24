"""Part 1: frozen sentence embeddings (LaBSE, multilingual-e5-large, bge-m3) + Ridge, with and without metadata."""
import json, os
import torch
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from metrics import regression_report

MODELS = {"LaBSE": ("sentence-transformers/LaBSE", ""),
          "mE5-large": ("intfloat/multilingual-e5-large", "query: "),
          "BGE-M3": ("BAAI/bge-m3", "")}
df = pd.read_csv("data/items.csv.gz")
tour = pd.read_csv("data/tournament_items.csv")
df["cat_id"] = df.cat_id.fillna(-1).astype(int)
tr, va, te = ((df.split == s).values for s in ("train", "val", "test"))
y = df.b_logit.values
num = ["n_chars", "n_words", "has_digit", "has_qmark", "has_image", "is_translation", "user_submitted"]
ohe = OneHotEncoder(handle_unknown="ignore").fit(df.loc[tr, ["cat_id", "lang_det"]])
sc = StandardScaler().fit(df.loc[tr, num])
meta = hstack([ohe.transform(df[["cat_id", "lang_det"]]), csr_matrix(sc.transform(df[num]))]).tocsr()
os.makedirs("results/embeddings", exist_ok=True)
res = {}
for name, (path, prefix) in MODELS.items():
    cache = f"results/embeddings/{name}.npy"
    model = SentenceTransformer(path, device="cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(cache):
        E = np.load(cache)
    else:
        E = model.encode([prefix + t for t in df.text.astype(str)], batch_size=128, normalize_embeddings=True,
                         show_progress_bar=False)
        np.save(cache, E)
    Et = model.encode([prefix + t for t in tour.text.astype(str)], batch_size=128, normalize_embeddings=True)
    for variant, X, Xt in (("text", csr_matrix(E), csr_matrix(Et)), ("text+meta", hstack([csr_matrix(E), meta]).tocsr(), None)):
        best = None
        for a in (0.1, 0.3, 1, 3, 10):
            m = Ridge(alpha=a).fit(X[tr], y[tr])
            r = np.sqrt(np.mean((m.predict(X[va]) - y[va]) ** 2))
            if best is None or r < best[0]:
                best = (r, a, m)
        pred = best[2].predict(X[te])
        key = f"{name} + Ridge ({variant})"
        res[key] = regression_report(y[te], pred, df.p.values[te])
        print(f"{key:36s} alpha={best[1]}", {k: round(v, 4) for k, v in res[key].items()}, flush=True)
        pd.DataFrame({"id": df.id.values[te], "pred": pred}).to_csv(f"results/embeddings/{name}__{variant}__test.csv", index=False)
        if Xt is not None:  # text-only model applied to cold-start tournament items
            pd.DataFrame({"id": tour.id, "pred": best[2].predict(Xt)}).to_csv(f"results/embeddings/{name}__text__tour.csv", index=False)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
json.dump(res, open("results/part1_embeddings.json", "w"), indent=2)
