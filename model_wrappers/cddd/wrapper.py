from typing import List

import numpy as np
from cddd_onnx import InferenceModel

from src.core.interfaces import EmbeddingModel


class CDDDEmbedder(EmbeddingModel):
    def __init__(self):
        self._model = InferenceModel()
        self._dim = self._model.seq_to_emb(["CCCCO"]).shape[1]

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def embed(self, smiles: List[str]) -> np.ndarray:
        return self._model.seq_to_emb(smiles, batch_size=len(smiles))


def get_embedder(**kwargs) -> CDDDEmbedder:
    return CDDDEmbedder()
