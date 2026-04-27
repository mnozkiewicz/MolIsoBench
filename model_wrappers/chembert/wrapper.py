import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List

from src.core.interfaces import EmbeddingModel


class ChemBERT(EmbeddingModel):
    MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"

    def __init__(self, pooling: str = "cls", device: str = "cpu"):
        self.pooling = pooling
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME, add_pooling_layer=False).to(self.device)
        self.model.eval()

    @property
    def embedding_dim(self) -> int:
        return self.model.config.hidden_size

    def embed(self, smiles: List[str]) -> np.ndarray:
        inputs = self.tokenizer(smiles, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state
        if self.pooling == "cls":
            embeddings = hidden[:, 0, :]
        elif self.pooling == "mean":
            mask = inputs["attention_mask"].unsqueeze(-1)
            embeddings = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        elif self.pooling == "sum":
            mask = inputs["attention_mask"].unsqueeze(-1)
            embeddings = (hidden * mask).sum(1)
        else:
            raise ValueError(f"Unknown pooling '{self.pooling}', choose cls/mean/sum")
        return embeddings.cpu().numpy()


def get_embedder(**kwargs) -> ChemBERT:
    return ChemBERT(**kwargs)
