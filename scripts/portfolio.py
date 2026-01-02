from __future__ import annotations
import argparse
import json
from datetime import timedelta

import pandas as pd
import yfinance as yf

from ar_beta.utils.io import load_yaml
from ar_beta.portfolio.simulate import PortfolioConfig, build_portfolio

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--beta_json", default="results/beta_predictions.json",
                   help="A JSON dict {ticker: beta_pred} for a given filing date.")
    ap.add_argument("--start", required=False, help="Start date YYYY-MM-DD (defaults to today-365).")
    ap.add_argument("--end", required=False, help="End date YYYY-MM-DD (defaults to today).")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    pcfg = PortfolioConfig(
        top_k=int(cfg["portfolio"]["top_k"]),
        bottom_k=int(cfg["portfolio"]["bottom_k"]),
        holding_days=int(cfg["portfolio"]["holding_days"]),
        benchmark=str(cfg["portfolio"]["benchmark"]),
    )

    with open(args.beta_json, "r", encoding="utf-8") as f:
        betas = json.load(f)

    high, low = build_portfolio(betas, pcfg)
    tickers = sorted(set(high + low + [pcfg.benchmark]))

    # Dates
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    start = pd.Timestamp(args.start) if args.start else (end - pd.Timedelta(days=365))

    prices = yf.download(tickers=tickers, start=str(start.date()), end=str(end.date()),
                         auto_adjust=True, progress=False)["Close"]
    rets = prices.pct_change().fillna(0.0)

    # Simple equal-weight baskets (you can replace with CML/MCMC optimization)
    port = rets[low].mean(axis=1) - rets[high].mean(axis=1)
    bench = rets[pcfg.benchmark]

    cum_port = (1 + port).cumprod() - 1
    cum_bench = (1 + bench).cumprod() - 1

    out = pd.DataFrame({"portfolio": cum_port, "benchmark": cum_bench})
    out.to_csv("results/portfolio_curve.csv")
    print("saved -> results/portfolio_curve.csv")

if __name__ == "__main__":
    main()
