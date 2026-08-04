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


def _lenient_is_valid_single(molecule: str) -> bool:
    # MolBERT's default validity gate parses with sanitize=True, which rejects our
    # graph-derived SMILES (hypervalent carbons). Accept anything RDKit can parse
    # WITHOUT sanitization; the featurizer's standardise() then tokenizes it via
    # its own sanitize=False path. This is purely additive — molecules that passed
    # the strict gate are tokenized identically, we only admit extra ones.
    from rdkit import Chem
    return len(molecule) > 0 and Chem.MolFromSmiles(molecule, False, {}) is not None


class MolBertEmbedder(EmbeddingModel):
    def __init__(self, device: str = "cpu", strict_validation: bool = False):
        self._feat = MolBertFeaturizer(str(_CHECKPOINT), device=device)
        if not strict_validation:
            # instance-level override; assigned plain function is called as
            # is_valid_single(molecule) (no self bound), matching the signature.
            self._feat.featurizer.is_valid_single = _lenient_is_valid_single

    @property
    def embedding_dim(self) -> int:
        return _EMB_DIM

    def embed(self, smiles: List[str]) -> np.ndarray:
        embeddings, valid = self._feat.transform(smiles)
        embeddings = embeddings.astype(np.float32)
        # MolBERT validates SMILES with RDKit (sanitize=True) and zeroes invalid
        # rows; mark them NaN instead so they're distinguishable from real
        # embeddings. Hypervalent graph SMILES land here (like clamp).
        embeddings[~np.asarray(valid, dtype=bool)] = np.nan
        return embeddings


def get_embedder(**kwargs) -> MolBertEmbedder:
    return MolBertEmbedder(**kwargs)
