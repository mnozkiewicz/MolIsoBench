import sys
from pathlib import Path
from typing import List

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "molbert_repo"))

import numpy as np
from molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer

from src.core.interfaces import EmbeddingModel
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

_CHECKPOINT = _HERE / "weights" / "molbert_100epochs" / "checkpoints" / "last.ckpt"
_EMB_DIM = 768


class MolBertEmbedder(EmbeddingModel):
    def __init__(self, device: str = "cpu"):
        self._feat = MolBertFeaturizer(str(_CHECKPOINT), device=device)

    @property
    def embedding_dim(self) -> int:
        return _EMB_DIM

    def embed(self, smiles: List[str]) -> np.ndarray:
        embeddings, _ = self._feat.transform(smiles)
        return embeddings.astype(np.float32)


def get_embedder(**kwargs) -> MolBertEmbedder:
    return MolBertEmbedder(**kwargs)
