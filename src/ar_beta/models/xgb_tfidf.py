from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBRegressor

@dataclass
class XgbTfidfConfig:
    max_features: int = 200_000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2

def fit_xgb_tfidf(texts, y, cfg: XgbTfidfConfig):
    vec = TfidfVectorizer(
        max_features=cfg.max_features,
        ngram_range=cfg.ngram_range,
        min_df=cfg.min_df,
        stop_words="english",
    )
    X = vec.fit_transform(texts)
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X, np.asarray(y, dtype=float))
    return vec, model

def predict_xgb_tfidf(vec, model, texts):
    X = vec.transform(texts)
    return model.predict(X)
