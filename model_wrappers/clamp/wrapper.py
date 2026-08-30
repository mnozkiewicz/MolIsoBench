import types
from typing import List, Optional

import numpy as np
from rdkit import Chem, RDLogger
from skfp.fingerprints import CLAMPFingerprint, ECFPFingerprint, RDKitFingerprint
from skfp.utils import ensure_mols

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

_DEFAULT_MAX_DEPTH = 6  # RDKit path-fingerprint max_path used by stock CLAMP


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


def _make_get_input_features(max_path: int):
    # Re-implementation of CLAMPFingerprint.get_input_features with a configurable
    # RDKit path length. The RDKit topological fingerprint enumerates all paths up
    # to `max_path` bonds, which explodes on dense graphs; lowering max_path makes
    # it tractable BUT changes the input distribution the pretrained MLP expects,
    # so embeddings with max_depth != 6 are NOT the faithful CLAMP model (use the
    # same value across all datasets if you go this route).
    def get_input_features(self, X):
        X = ensure_mols(X)
        ecfp = ECFPFingerprint(
            fp_size=8192, radius=2,
            use_pharmacophoric_invariants=True, include_chirality=True, count=True,
        )
        rdkit_fp = RDKitFingerprint(
            fp_size=8192, max_path=max_path, num_bits_per_feature=1, count=True,
        )
        ecfpc = np.asarray(ecfp.transform(X))
        rdkc = np.asarray(rdkit_fp.transform(X))
        return np.log(1.0 + ecfpc + rdkc).astype(np.float32)

    return get_input_features


class CLAMPEmbedder(EmbeddingModel):
    def __init__(
        self,
        batch_size: Optional[int] = None,
        n_jobs: Optional[int] = None,
        max_depth: int = _DEFAULT_MAX_DEPTH,
    ):
        self._fp = CLAMPFingerprint(batch_size=batch_size, n_jobs=n_jobs)
        self.max_depth = max_depth
        if max_depth != _DEFAULT_MAX_DEPTH:
            # override the RDKit path length (faster on dense graphs, but no longer
            # the faithful pretrained-CLAMP featurization — see note above)
            self._fp.get_input_features = types.MethodType(
                _make_get_input_features(max_depth), self._fp
            )
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
