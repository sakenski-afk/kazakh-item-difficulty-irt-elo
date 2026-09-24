"""Catalogue of the database export: rows, columns, excluded (PII) columns and created_at range per table.
Reads data/raw/manifest.json and the gzip CSVs; writes results/export_catalog.csv."""
import json, os
import pandas as pd

RAW = "data/raw/"
m = json.load(open(RAW + "manifest.json", encoding="utf-8"))
rows = []
for t, info in sorted(m["tables"].items(), key=lambda kv: -kv[1]["rows"]):
    path = RAW + info["file"]
    lo = hi = None
    if info["rows"] and os.path.exists(path):
        df = pd.read_csv(path, low_memory=False, usecols=lambda c: c == "created_at")
        if "created_at" in df:
            c = pd.to_datetime(df.created_at, errors="coerce").dropna()
            if len(c):
                lo, hi = c.min().date(), c.max().date()
    rows.append({"table": t, "rows": info["rows"], "n_columns": len(info["columns"]),
                 "truncated_at_limit": info["rows"] == m.get("row_limit_per_table"),
                 "file_present": os.path.exists(path), "excluded_columns": ";".join(info["excluded_columns"]),
                 "created_from": lo, "created_to": hi, "column_names": ";".join(info["columns"])})
out = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
out.to_csv("results/export_catalog.csv", index=False)
print(f"{len(out)} tables, {out.rows.sum():,} rows, {out.n_columns.sum()} columns, "
      f"{out.truncated_at_limit.sum()} truncated at {m.get('row_limit_per_table')}, "
      f"{(out.rows == 0).sum()} empty, {(~out.file_present).sum()} files missing")
