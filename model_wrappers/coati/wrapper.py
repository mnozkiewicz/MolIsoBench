import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "COATI"))
os.environ.setdefault("S3_CACHE_DIR", str(Path(__file__).parent))

from coati.models.simple_coati2.io import load_coati2
from coati.generative.coati_purifications import embed_smiles_batch

from src.core.interfaces import EmbeddingModel

MODEL_URL = "s3://terray-public/models/coati2_chiral_03-08-24.pkl"


class COATI(EmbeddingModel):
    def __init__(self, device: str = "cpu", doc_url: str = MODEL_URL):
        self.encoder, self.tokenizer = load_coati2(
            doc_url=doc_url,
            device=device,
            freeze=True,
            force_cpu=(device == "cpu"),
        )
        self.encoder.eval()

    @property
    def embedding_dim(self) -> int:
        return self.encoder.embed_dim

    def embed(self, smiles: List[str]) -> np.ndarray:
        with torch.no_grad():
            vecs = embed_smiles_batch(smiles, self.encoder, self.tokenizer)
        return vecs.cpu().numpy()


def get_embedder(**kwargs) -> COATI:
    return COATI(**kwargs)
