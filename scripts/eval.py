from __future__ import annotations
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ar_beta.utils.io import load_yaml
from ar_beta.data.dataset import Item1ABetaDataset, DatasetPaths
from ar_beta.models.transformer_regressor import TransformerRegressor
from ar_beta.utils.metrics import mse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default="results/checkpoints/model.pt")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    paths = DatasetPaths(
        item_text_file=cfg["data"]["item_text_file"],
        beta_label_file=cfg["data"]["beta_label_file"],
    )
    horizon = int(cfg["train"]["horizon_days"])
    ds = Item1ABetaDataset(paths, horizon_days=horizon)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model_name = ckpt["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def collate(batch):
        texts = [b["text"] for b in batch]
        y = torch.tensor([b["beta"] for b in batch], dtype=torch.float32)
        tok = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        return tok, y

    dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerRegressor(model_name=model_name).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ys, ps = [], []
    with torch.no_grad():
        for tok, y in dl:
            tok = {k: v.to(device) for k, v in tok.items()}
            pred = model(**tok).cpu().numpy()
            ys.extend(y.numpy().tolist())
            ps.extend(pred.tolist())

    print(f"MSE: {mse(ys, ps):.6f}")

if __name__ == "__main__":
    main()
