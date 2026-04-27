import sys
from pathlib import Path
from typing import List

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "cddd_repo"))

import numpy as np
from cddd.inference import InferenceModel

from src.core.interfaces import EmbeddingModel

_DEFAULT_MODEL_DIR = str(_HERE / "default_model")


class CDDDEmbedder(EmbeddingModel):
    def __init__(self, model_dir: str = _DEFAULT_MODEL_DIR, use_gpu: bool = False,
                 batch_size: int = 512, cpu_threads: int = 6):
        self._model = InferenceModel(
            model_dir=model_dir,
            use_gpu=use_gpu,
            batch_size=batch_size,
            cpu_threads=cpu_threads,
        )
        self._dim = int(self._model.hparams.emb_size)

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def embed(self, smiles: List[str]) -> np.ndarray:
        return self._model.seq_to_emb(smiles)


def get_embedder(**kwargs) -> CDDDEmbedder:
    return CDDDEmbedder(**kwargs)
