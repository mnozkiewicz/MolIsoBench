from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from src.core.interfaces import EmbeddingModel


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFEmbedder(EmbeddingModel):
    MODEL_NAME: str = None

    def __init__(self, device: str = None, **tokenizer_kwargs):
        if device is None:
            device = get_device()
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME, trust_remote_code=True).to(self.device)
        self.model.eval()
        self._tokenizer_kwargs = tokenizer_kwargs

    @property
    def embedding_dim(self) -> int:
        return self.model.config.hidden_size

    def _pool(self, outputs, inputs) -> torch.Tensor:
        raise NotImplementedError

    def embed(self, smiles: List[str]) -> np.ndarray:
        inputs = self.tokenizer(smiles, return_tensors="pt", padding=True, truncation=True, **self._tokenizer_kwargs).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return self._pool(outputs, inputs).cpu().numpy()


class ChemBERT(HFEmbedder):
    MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"

    def __init__(self, pooling: str = "cls", device: str = None):
        super().__init__(device=device)
        self.pooling = pooling

    def _pool(self, outputs, inputs) -> torch.Tensor:
        hidden = outputs.last_hidden_state
        if self.pooling == "cls":
            return hidden[:, 0, :]
        mask = inputs["attention_mask"].unsqueeze(-1)
        if self.pooling == "mean":
            return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        if self.pooling == "sum":
            return (hidden * mask).sum(1)
        raise ValueError(f"Unknown pooling '{self.pooling}', choose cls/mean/sum")


class ChemBERTa10M(HFEmbedder):
    MODEL_NAME = "DeepChem/ChemBERTa-10M-MTR"

    def _pool(self, outputs, inputs) -> torch.Tensor:
        return outputs.last_hidden_state[:, 0, :]


class MolFormer(HFEmbedder):
    MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"

    def _pool(self, outputs, inputs) -> torch.Tensor:
        return outputs.pooler_output


class ChemFM(EmbeddingModel):
    EOS_TOKEN = "<eos>"

    def __init__(self, model_name: str, device: str = None):
        if device is None:
            device = get_device()
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()

    @property
    def embedding_dim(self) -> int:
        return self.model.config.hidden_size

    def embed(self, smiles: List[str]) -> np.ndarray:
        texts = [s + self.EOS_TOKEN for s in smiles]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, return_token_type_ids=False).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_pos = inputs["attention_mask"].sum(dim=1) - 1
        embeddings = outputs.last_hidden_state[torch.arange(len(smiles)), last_pos]
        return embeddings.cpu().numpy()


_MODELS = {
    "chemberta-77m": ChemBERT,
    "chemberta-10m": ChemBERTa10M,
    "molformer": MolFormer,
    "chemfm-1b": lambda **kw: ChemFM(model_name="ChemFM/ChemFM-1B", **kw),
    "chemfm-3b": lambda **kw: ChemFM(model_name="ChemFM/ChemFM-3B", **kw),
}


def get_embedder(model_name: str = "chemberta-77m", **kwargs) -> EmbeddingModel:
    cls = _MODELS.get(model_name.lower())
    if cls is None:
        raise ValueError(f"Unknown model '{model_name}', choose from: {list(_MODELS)}")
    return cls(**kwargs)
