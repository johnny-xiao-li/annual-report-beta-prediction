from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import yfinance as yf

@dataclass
class PortfolioConfig:
    top_k: int = 10
    bottom_k: int = 10
    holding_days: int = 180
    benchmark: str = "^GSPC"

def build_portfolio(ticker_to_beta: Dict[str, float], cfg: PortfolioConfig) -> Tuple[List[str], List[str]]:
    """Return (high_beta, low_beta) ticker lists."""
    items = sorted(ticker_to_beta.items(), key=lambda x: x[1])
    low = [t for t, _ in items[: cfg.bottom_k]]
    high = [t for t, _ in items[-cfg.top_k :]]
    return high, low

def download_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    df = yf.download(tickers=tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    else:
        df = df[["Close"]].rename(columns={"Close": tickers[0]})
    return df

def cumulative_return(price_df: pd.DataFrame) -> pd.Series:
    ret = price_df.pct_change().fillna(0.0)
    return (1.0 + ret).prod(axis=1) - 1.0
