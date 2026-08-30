import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from rdkit import Chem, RDLogger

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))  # for the vendored chemeleon_fingerprint.py

from chemeleon_fingerprint import CheMeleonFingerprint

from src.core.interfaces import EmbeddingModel

RDLogger.DisableLog("rdApp.*")

# fixed by the pretrained checkpoint (chemeleon_mp.pt: d_h = 2048)
_EMB_DIM = 2048


def _relaxed_mol(smiles: str):
    """Parse a SMILES without valence sanitization so graph-derived molecules
    (e.g. hypervalent carbons) still featurize; do the minimum ChemProp's
    featurizer needs. Returns None for genuinely unparseable strings."""
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol)
    except Exception:
        return None
    return mol


class CheMeleonEmbedder(EmbeddingModel):
    def __init__(self, device: Optional[str] = None):
        # CheMeleonFingerprint downloads/loads ~/.chemprop/chemeleon_mp.pt and
        # builds a BondMessagePassing MPNN with mean aggregation.
        self._fp = CheMeleonFingerprint(device=device)

    @property
    def embedding_dim(self) -> int:
        return _EMB_DIM

    def embed(self, smiles: List[str]) -> np.ndarray:
        mols = [_relaxed_mol(s) for s in smiles]
        valid_idx = [i for i, mol in enumerate(mols) if mol is not None]

        out = np.full((len(smiles), _EMB_DIM), np.nan, dtype=np.float32)
        if not valid_idx:
            return out

        valid_mols = [mols[i] for i in valid_idx]
        try:
            emb = np.asarray(self._fp(valid_mols), dtype=np.float32)
            for out_i, in_i in enumerate(valid_idx):
                out[in_i] = emb[out_i]
        except Exception:
            # fall back per-molecule so one un-featurizable graph doesn't drop the batch
            for in_i in valid_idx:
                try:
                    out[in_i] = np.asarray(self._fp([mols[in_i]]), dtype=np.float32)[0]
                except Exception:
                    pass  # leaves a NaN row
        return out


def get_embedder(device: Optional[str] = None, **_) -> CheMeleonEmbedder:
    return CheMeleonEmbedder(device=device)
