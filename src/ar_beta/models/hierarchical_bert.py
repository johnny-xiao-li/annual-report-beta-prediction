from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig

@dataclass
class HierBertConfig:
    base_encoder: str = "bert-base-uncased"
    max_sentences: int = 256
    max_tokens_per_sentence: int = 128
    doc_transformer_layers: int = 2
    doc_transformer_heads: int = 8
    dropout: float = 0.1
    sentence_pool: str = "cls"  # cls | mean

class HierarchicalBertRegressor(nn.Module):
    """Sentence encoder (BERT) + document-level Transformer + regression head.

    This follows the paper's idea of token-level encoding per sentence, then
    sentence-level encoding across the document. See Algorithm 1 in the paper.
    """

    def __init__(self, cfg: HierBertConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base_encoder)
        base_cfg = AutoConfig.from_pretrained(cfg.base_encoder)
        self.sent_encoder = AutoModel.from_pretrained(cfg.base_encoder, config=base_cfg)

        hidden = base_cfg.hidden_size
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=cfg.doc_transformer_heads,
            dim_feedforward=4 * hidden,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.doc_encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.doc_transformer_layers)
        self.dropout = nn.Dropout(cfg.dropout)
        self.regressor = nn.Linear(hidden, 1)

    @torch.no_grad()
    def encode_sentences(self, sentences: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (embeddings, mask) with shapes [S, H], [S]."""
        sentences = sentences[: self.cfg.max_sentences]
        tok = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_tokens_per_sentence,
            return_tensors="pt",
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        out = self.sent_encoder(**tok)
        last = out.last_hidden_state  # [S, T, H]
        if self.cfg.sentence_pool == "mean":
            mask = tok["attention_mask"].unsqueeze(-1).float()
            emb = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            emb = last[:, 0]  # CLS
        sent_mask = torch.ones(emb.size(0), device=device, dtype=torch.bool)
        return emb, sent_mask

    def forward(self, batch_sent_emb: torch.Tensor, batch_sent_mask: torch.Tensor) -> torch.Tensor:
        """Forward with precomputed sentence embeddings.

        batch_sent_emb: [B, S, H]
        batch_sent_mask: [B, S] True for real sentences
        """
        # Transformer expects src_key_padding_mask with True = padding.
        padding_mask = ~batch_sent_mask  # [B, S]
        h = self.doc_encoder(batch_sent_emb, src_key_padding_mask=padding_mask)  # [B, S, H]

        # Pool at document level: mean over valid sentences
        mask = batch_sent_mask.unsqueeze(-1).float()
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        y = self.regressor(self.dropout(pooled)).squeeze(-1)
        return y
