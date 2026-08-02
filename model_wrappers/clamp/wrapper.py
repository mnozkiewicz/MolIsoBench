from typing import List, Optional

import numpy as np
from rdkit import Chem, RDLogger
from skfp.fingerprints import CLAMPFingerprint

from src.core.interfaces import EmbeddingModel

RDLogger.DisableLog("rdApp.*")

# CLAMP's compound encoder featurizes each molecule through RDKit fingerprints
# (ECFP + RDKit counts). RDKit's *sanitization* rejects our graph-derived SMILES
# (e.g. hypervalent carbons like CC(C)(C)(C)(C)(C)(C)C) even though they parse
# fine as graphs. We therefore build molecules with sanitization skipped and do
# only the minimum needed to fingerprint them:
#   * MolFromSmiles(sanitize=False) - parse the graph, skip the valence check
#   * UpdatePropertyCache(strict=False) - compute valences without raising
#   * FastFindRings - ring info required by Morgan/ECFP
# The resulting ECFP encodes the graph structure, which is exactly what matters
# for the isomorphism benchmark. skfp accepts RDKit Mol objects directly and does
# not re-sanitize them. Genuinely unparseable strings (None) fall back to NaN.


def _relaxed_mol(smiles: str) -> Optional[Chem.Mol]:
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol)
    except Exception:
        return None
    return mol


class CLAMPEmbedder(EmbeddingModel):
    def __init__(self, batch_size: Optional[int] = None, n_jobs: Optional[int] = None):
        self._fp = CLAMPFingerprint(batch_size=batch_size, n_jobs=n_jobs)
        self._dim = 768

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def embed(self, smiles: List[str]) -> np.ndarray:
        mols = [_relaxed_mol(smi) for smi in smiles]
        valid_mask = [mol is not None for mol in mols]
        valid_mols = [mol for mol in mols if mol is not None]

        out = np.full((len(smiles), self._dim), np.nan, dtype=np.float32)
        if valid_mols:
            emb = np.asarray(self._fp.transform(valid_mols), dtype=np.float32)
            out[np.array(valid_mask, dtype=bool)] = emb
        return out


def get_embedder(**kwargs) -> CLAMPEmbedder:
    return CLAMPEmbedder(**kwargs)
