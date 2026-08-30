import contextlib
import io
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
        # COATI's tokenizer has a hard n_seq cap (256); SMILES that tokenize
        # longer (e.g. brec's large graphs) raise "Oversized String" and would
        # kill the whole batch. Pre-filter those (and any that fail to tokenize)
        # -> NaN rows, and embed the rest in one batch. Truncating is not an
        # option (it would encode a different molecule), so oversized = NaN.
        dim = self.embedding_dim
        out = np.full((len(smiles), dim), np.nan, dtype=np.float32)

        ok_idx, ok_smiles = [], []
        # the tokenizer prints the (huge) offending SMILES on oversize; mute it
        with contextlib.redirect_stdout(io.StringIO()):
            for i, smi in enumerate(smiles):
                try:
                    self.tokenizer.tokenize_text("[SMILES]" + smi + "[STOP]", pad=True)
                    ok_idx.append(i)
                    ok_smiles.append(smi)
                except Exception:
                    pass  # oversized / untokenizable -> stays NaN

        if ok_smiles:
            with torch.no_grad():
                vecs = embed_smiles_batch(ok_smiles, self.encoder, self.tokenizer)
            vecs = vecs.cpu().numpy().astype(np.float32)
            for j, i in enumerate(ok_idx):
                out[i] = vecs[j]
        return out


def get_embedder(**kwargs) -> COATI:
    return COATI(**kwargs)
