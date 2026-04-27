from typing import List

import numpy as np

from src.core.interfaces import EmbeddingModel


class CLAMPEmbedder(EmbeddingModel):
    def __init__(self, device: str = "cpu"):
        from clamp.models.pretrained import PretrainedCLAMP
        self._model = PretrainedCLAMP(device=device)
        self._model.to(device)
        self._model.eval()

    @property
    def embedding_dim(self) -> int:
        return self._model.compound_features_size

    def embed(self, smiles: List[str]) -> np.ndarray:
        return self._model.encode_smiles(smiles).detach().cpu().numpy()


def get_embedder(**kwargs) -> CLAMPEmbedder:
    return CLAMPEmbedder(**kwargs)
