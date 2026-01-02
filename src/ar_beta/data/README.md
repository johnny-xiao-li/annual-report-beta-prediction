# Data format (expected)

This repo does **not** redistribute EDGAR-CORPUS or market datasets.

Create these files locally:

## 1) Item 1A text
`data/processed/item1a_text.parquet`

Required columns:
- `ticker` (str) OR `cik` (str/int)
- `filing_date` (datetime)
- `text` (str) — Item 1A Risk Factors content

## 2) Beta labels
`data/processed/beta_labels.parquet`

Required columns:
- `ticker` / `cik`
- `asof_date` (datetime) — usually aligned to filing_date
- `horizon_days` (int) — e.g., 180
- `beta` (float)

Optional:
- `mkt_ticker` (benchmark index)
- `rf` (risk-free)

## Notes
The paper computes Beta using CAPM and evaluates prediction using MSE.
