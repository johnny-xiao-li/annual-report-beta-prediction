from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

@dataclass
class DatasetPaths:
    item_text_file: str
    beta_label_file: str

class Item1ABetaDataset(Dataset):
    """A simple tabular dataset (text + beta label)."""

    def __init__(self, paths: DatasetPaths, horizon_days: int):
        df_text = pd.read_parquet(paths.item_text_file)
        df_beta = pd.read_parquet(paths.beta_label_file)
        df_beta = df_beta[df_beta["horizon_days"] == horizon_days].copy()

        # Join on ticker + date (adapt as needed)
        df = df_text.merge(
            df_beta,
            left_on=["ticker", "filing_date"],
            right_on=["ticker", "asof_date"],
            how="inner",
        )
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        return {
            "ticker": r["ticker"],
            "filing_date": r["filing_date"],
            "text": r["text"],
            "beta": float(r["beta"]),
        }
