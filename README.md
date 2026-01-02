# NLP-Based Analysis of Annual Reports: Beta Prediction & Portfolio Simulation

This repository is a **reproducibility-oriented codebase scaffold** for the AICS 2024 paper:

> **NLP-Based Analysis of Annual Reports: Asset Volatility Prediction and Portfolio Strategy Application**  
> Xiao Li, Yang Xu, Linyi Yang, Yue Zhang, Ruihai Dong (AICS 2024)

The paper predicts **CAPM Beta (β)** from **Form 10‑K Item 1A (Risk Factors)** text, and then uses the **predicted Beta** to construct portfolios and evaluate market performance. The paper highlights that a **Hierarchical Transformer-based** model better handles long risk disclosures by modelling token-level and sentence-level structure.  

## What’s inside (and what’s not)

- ✅ A clean repo layout, CLI entrypoints, configs, CI, and **reference implementations** for:
  - XGBoost + TF‑IDF baseline
  - Transformer regressors (BERT/RoBERTa/Longformer)
  - A practical **Hierarchical BERT** model: sentence encoder → document-level Transformer → regression head
  - Portfolio simulation pipeline (top/bottom beta selection + holding window)
- ⚠️ **No dataset is redistributed here.** You plug in EDGAR‑CORPUS (and your returns / risk‑free sources) locally.

## Repo structure

```
.
├─ src/ar_beta/                 # library code
│  ├─ data/                     # preprocessing & dataset builders
│  ├─ models/                   # baselines + hierarchical transformer
│  ├─ portfolio/                # portfolio simulation
│  └─ utils/
├─ scripts/                     # runnable scripts
├─ slides/
│  └─ oral_presentation.pdf     
└─ configs/                     # YAML configs

```

## Quickstart

### 1) Create env
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Prepare data
You should create a `data/processed/item1a_text.parquet` with at least:
- `ticker` (or CIK)
- `filing_date`
- `text` (Item 1A content)

And `data/processed/beta_labels.parquet` with:
- `ticker` (or CIK)
- `asof_date`
- `horizon_days`
- `beta`

See `src/ar_beta/data/README.md` for the expected schema and helper scripts.

### 3) Train / evaluate
```bash
python scripts/train.py --config configs/default.yaml
python scripts/eval.py  --config configs/default.yaml
```

### 4) Portfolio simulation
```bash
python scripts/portfolio.py --config configs/default.yaml
```

## Paper & Oral presentation

- 📄 Paper (AICS 2024, CEUR-WS): https://ceur-ws.org/Vol-3910/aics2024_p31.pdf
- 🎤 Oral presentation:
  - Slides (PDF): [slides/oral_presentation.pdf](slides/oral_presentation.pdf)


## Citation

If you use this repo, please cite the paper:

```bibtex
@inproceedings{li2024aics_beta,
  title={NLP-Based Analysis of Annual Reports: Asset Volatility Prediction and Portfolio Strategy Application},
  author={Li, Xiao and Xu, Yang and Yang, Linyi and Zhang, Yue and Dong, Ruihai},
  booktitle={Proceedings of the 32nd Irish Conference on Artificial Intelligence and Cognitive Science (AICS 2024)},
  year={2024}
}
```

## License

- Code: MIT (see `LICENSE`)
- Paper: see the proceedings / publisher’s license
