import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from omegaconf import OmegaConf

if not hasattr(os, "sched_getaffinity"):
    os.sched_getaffinity = lambda pid: set()

from molbart.models.chemformer import Chemformer
from molbart.data import SynthesisDataModule
from molbart.utils.data_utils import DEFAULT_MAX_SEQ_LEN

from src.core.interfaces import EmbeddingModel

_HERE = Path(__file__).parent
_VOCAB_PATH = str(_HERE / "Chemformer" / "bart_vocab.json")
_DEFAULT_WEIGHTS = str(_HERE / "weights" / "chemformer_combined.ckpt")


class ChemformerEmbedder(EmbeddingModel):
    def __init__(self, weights_path: str = _DEFAULT_WEIGHTS, device: str = "cpu", batch_size: int = 128):
        self._batch_size = batch_size
        config = OmegaConf.create({
            "train_mode": "eval",
            "data_device": device,
            "n_gpus": 1 if device == "cuda" else 0,
            "vocabulary_path": _VOCAB_PATH,
            "model_path": weights_path,
            "model_type": "bart",
            "task": "forward_prediction",
            "n_beams": 10,
        })
        self._model = Chemformer(config)

    @property
    def embedding_dim(self) -> int:
        return self._model.model.d_model

    def embed(self, smiles: List[str]) -> np.ndarray:
        datamodule = SynthesisDataModule(
            reactants=smiles,
            products=smiles,
            tokenizer=self._model.tokenizer,
            batch_size=self._batch_size,
            max_seq_len=DEFAULT_MAX_SEQ_LEN,
            dataset_path="",
        )
        datamodule.setup()
        encoded = self._model.encode(dataloader=datamodule.full_dataloader())
        pooled = torch.stack([t.mean(dim=0) for t in encoded])
        return pooled.detach().cpu().numpy()


def get_embedder(**kwargs) -> ChemformerEmbedder:
    return ChemformerEmbedder(**kwargs)
