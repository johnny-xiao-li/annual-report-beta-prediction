from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ar_beta.utils.io import load_yaml, ensure_dir
from ar_beta.utils.seed import seed_everything
from ar_beta.data.dataset import Item1ABetaDataset, DatasetPaths
from ar_beta.models.transformer_regressor import TransformerRegressor
from ar_beta.utils.metrics import mse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(cfg.get("seed", 42))

    paths = DatasetPaths(
        item_text_file=cfg["data"]["item_text_file"],
        beta_label_file=cfg["data"]["beta_label_file"],
    )
    horizon = int(cfg["train"]["horizon_days"])

    ds = Item1ABetaDataset(paths, horizon_days=horizon)

    model_name = cfg["model"]["base_encoder"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def collate(batch):
        texts = [b["text"] for b in batch]
        y = torch.tensor([b["beta"] for b in batch], dtype=torch.float32)
        tok = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        return tok, y

    dl = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, collate_fn=collate)

    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    model = TransformerRegressor(model_name=model_name, dropout=float(cfg["model"]["dropout"])).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]))
    loss_fn = torch.nn.MSELoss()

    model.train()
    for epoch in range(int(cfg["train"]["epochs"])):
        losses = []
        for tok, y in dl:
            tok = {k: v.to(device) for k, v in tok.items()}
            y = y.to(device)
            pred = model(**tok)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"epoch={epoch} loss={sum(losses)/len(losses):.6f}")

    out_dir = ensure_dir("results/checkpoints")
    torch.save({"model_state": model.state_dict(), "model_name": model_name}, out_dir / "model.pt")
    print(f"saved -> {out_dir / 'model.pt'}")

if __name__ == "__main__":
    main()
