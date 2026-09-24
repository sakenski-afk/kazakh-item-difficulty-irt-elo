"""Part 1: fine-tune a transformer encoder for item-difficulty regression (target: logit difficulty).
Usage: python src/part1_finetune.py --model FacebookAI/xlm-roberta-base --input cat_text --seed 1
Outputs (in --out_dir): <tag>.json (metrics), <tag>__test.csv (test predictions), <tag>__tour.csv
(predictions for the cold-start tournament items used in Part 2)."""
import argparse, json, math, os, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from metrics import regression_report

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--input", choices=["text", "cat_text"], default="cat_text")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--max_len", type=int, default=64)
ap.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing (used for XLM-R large on 12 GB)")
ap.add_argument("--out_dir", default="results/finetune")
args = ap.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
AMP = torch.bfloat16 if dev == "cuda" else torch.float32
tag = f"{args.model.split('/')[-1]}__{args.input}__s{args.seed}"
os.makedirs(args.out_dir, exist_ok=True)

df = pd.read_csv("data/items.csv.gz")
tour = pd.read_csv("data/tournament_items.csv")
def inp(d):
    if args.input == "text":
        return d.text.astype(str).tolist()
    return (d.cat_name.fillna("").astype(str) + " : " + d.text.astype(str)).tolist()
tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
mu, sd = tr.b_logit.mean(), tr.b_logit.std()

tok = AutoTokenizer.from_pretrained(args.model)
def enc(texts):
    return tok(texts, truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")

class Reg(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(args.model)
        h = self.enc.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(h, 1))
    def forward(self, **x):
        o = self.enc(**x).last_hidden_state
        m = x["attention_mask"].unsqueeze(-1).float()
        pooled = (o * m).sum(1) / m.sum(1)  # mean pooling
        return self.head(pooled).squeeze(-1)

model = Reg().to(dev)
if args.grad_ckpt:  # keeps large encoders inside 12 GB VRAM (avoids WDDM spill to system RAM)
    model.enc.gradient_checkpointing_enable()
opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": args.lr},
                         {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=0.01)
steps = math.ceil(len(tr) / args.bs) * args.epochs
sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
Xtr, ytr = inp(tr), torch.tensor(((tr.b_logit - mu) / sd).values, dtype=torch.float32)

@torch.no_grad()
def predict(texts):
    model.eval(); out = []
    for i in range(0, len(texts), 128):
        x = {k: v.to(dev) for k, v in enc(texts[i:i + 128]).items()}
        with torch.autocast(dev, dtype=AMP, enabled=dev == "cuda"):
            out.append(model(**x).float().cpu())
    return torch.cat(out).numpy() * sd + mu

best, best_state, t0 = 1e9, None, time.time()
for ep in range(args.epochs):
    model.train(); perm = np.random.permutation(len(tr))
    for i in range(0, len(tr), args.bs):
        idx = perm[i:i + args.bs]
        x = {k: v.to(dev) for k, v in enc([Xtr[j] for j in idx]).items()}
        with torch.autocast(dev, dtype=AMP, enabled=dev == "cuda"):
            loss = nn.functional.mse_loss(model(**x).float(), ytr[idx].to(dev))
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
    if dev == "cuda":
        torch.cuda.empty_cache()
    pv = predict(inp(va)); rmse = float(np.sqrt(np.mean((pv - va.b_logit.values) ** 2)))
    print(f"[{tag}] epoch {ep + 1} val RMSE {rmse:.4f} ({time.time() - t0:.0f}s)", flush=True)
    if rmse < best:
        best_val_r = float(np.corrcoef(pv, va.b_logit.values)[0, 1]) if np.std(pv) > 1e-9 else 0.0
        best = rmse; best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
model.load_state_dict(best_state)
pt = predict(inp(te))
rep = regression_report(te.b_logit, pt, te.p)
# cold-start tournament items (new items): predictions saved for IRT validation
ptour = predict(inp(tour))
print(f"[{tag}] TEST", {k: round(v, 4) for k, v in rep.items()}, flush=True)
json.dump({"tag": tag, "args": vars(args), "val_rmse": best, "val_pearson": best_val_r, "test": rep, "seconds": time.time() - t0},
          open(f"{args.out_dir}/{tag}.json", "w"), indent=2)
pd.DataFrame({"id": te.id, "pred": pt}).to_csv(f"{args.out_dir}/{tag}__test.csv", index=False)
pd.DataFrame({"id": tour.id, "pred": ptour}).to_csv(f"{args.out_dir}/{tag}__tour.csv", index=False)
