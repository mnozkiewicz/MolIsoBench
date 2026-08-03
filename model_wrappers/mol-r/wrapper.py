import sys
import warnings
from pathlib import Path
from typing import List

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "mol_r_repo" / "src"))

import numpy as np

from featurizer import MolEFeaturizer
from src.core.interfaces import EmbeddingModel

_VARIANTS = ["gcn_1024", "gat_1024", "sage_1024", "tag_1024"]


class MolREmbedder(EmbeddingModel):
    def __init__(self, variant: str = "gcn_1024", batch_size: int = 128):
        if variant not in _VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from: {_VARIANTS}")
        self._model = MolEFeaturizer(str(_HERE / "mol_r_repo" / "saved" / variant), gpu=None)
        self._dim = self._model.dim
        self._batch_size = batch_size

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def embed(self, smiles: List[str]) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            embeddings, flags = self._model.transform(smiles, batch_size=self._batch_size)
        embeddings = embeddings.astype(np.float32)
        # transform() leaves unparsed SMILES as zero rows; mark them NaN so they
        # are distinguishable from genuine embeddings (flags[i] is False on failure).
        embeddings[~flags] = np.nan
        return embeddings


def get_embedder(variant: str = "gcn_1024", **kwargs) -> MolREmbedder:
    return MolREmbedder(variant=variant, **kwargs)
