from typing import List

import numpy as np
import cddd_onnx.main as cddd_main
import cddd_onnx.tokenizer as cddd_tok
from cddd_onnx import InferenceModel

from src.core.interfaces import EmbeddingModel

# The stock CDDD pipeline runs `preprocess_smiles` (RDKit salt/stereo removal,
# canonicalization and an organic/property filter) in TWO places: inside
# `InferenceModel.seq_to_emb` (cddd_onnx.main) and inside
# `InputPipelineInferEncode.__init__` (cddd_onnx.tokenizer). Both modules bound
# the function with `from cddd_onnx.preprocessing import preprocess_smiles`, so
# patching `cddd_onnx.preprocessing.preprocess_smiles` has no effect -- the
# consumers already hold the original object. We instead rebind the name in each
# consumer module; the bare `preprocess_smiles` call is resolved against the
# module globals at call time, so this reliably toggles preprocessing on/off.
#
# Preprocessing is disabled by default because our graph-derived molecules (e.g.
# pure-carbon graphs with hypervalent atoms) are rejected by RDKit, which would
# otherwise drop every SMILES and raise "No valid SMILES found after preprocessing".

_ORIGINAL_PREPROCESS = cddd_main.preprocess_smiles


def _identity(sml: str) -> str:
    return sml


def _set_preprocessing(enabled: bool) -> None:
    fn = _ORIGINAL_PREPROCESS if enabled else _identity
    cddd_main.preprocess_smiles = fn
    cddd_tok.preprocess_smiles = fn


class CDDDEmbedder(EmbeddingModel):
    def __init__(self, preprocess: bool = False):
        self.preprocess = preprocess
        _set_preprocessing(preprocess)
        self._model = InferenceModel()
        self._dim = self._model.seq_to_emb(["CCCCO"]).shape[1]

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def embed(self, smiles: List[str]) -> np.ndarray:
        # re-assert in case another embedder toggled the global in between
        _set_preprocessing(self.preprocess)
        return self._model.seq_to_emb(smiles, batch_size=len(smiles))


def get_embedder(**kwargs) -> CDDDEmbedder:
    return CDDDEmbedder(**kwargs)
