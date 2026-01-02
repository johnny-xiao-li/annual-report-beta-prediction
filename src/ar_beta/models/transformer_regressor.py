from __future__ import annotations
import torch
from torch import nn
from transformers import AutoModel, AutoConfig

class TransformerRegressor(nn.Module):
    """A thin regression head on top of a HuggingFace encoder.

    For BERT/RoBERTa/Longformer, you may use CLS pooling or mean pooling.
    """
    def __init__(self, model_name: str, dropout: float = 0.1, pool: str = "cls"):
        super().__init__()
        self.pool = pool
        cfg = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=cfg)
        hidden = cfg.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state  # [B, T, H]
        if self.pool == "mean":
            mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
            pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = last[:, 0]  # CLS
        x = self.dropout(pooled)
        y = self.regressor(x).squeeze(-1)
        return y
