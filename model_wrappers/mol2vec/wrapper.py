from typing import List

import numpy as np
from deepchem.feat import Mol2VecFingerprint

from src.core.interfaces import EmbeddingModel


class Mol2VecEmbedder(EmbeddingModel):
    def __init__(self):
        self._feat = Mol2VecFingerprint()

    @property
    def embedding_dim(self) -> int:
        return 300

    def embed(self, smiles: List[str]) -> np.ndarray:
        embeddings = self._feat.featurize(smiles)

        dim = 300
        result = np.full((len(smiles), dim), float("nan"), dtype=np.float32)
        for i, emb in enumerate(embeddings):
            if emb.shape[0] == dim:
                result[i] = emb

        return result


def get_embedder(**kwargs) -> Mol2VecEmbedder:
    return Mol2VecEmbedder()
