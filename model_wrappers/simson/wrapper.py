from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from x_transformers import TransformerWrapper, Encoder

from src.core.interfaces import EmbeddingModel

_HERE = Path(__file__).parent
_CHECKPOINT = _HERE / "weights" / "pretrained_best_model.pth"
_TOKENIZER = _HERE / "pubchem_part_tokenizer.json"

_DIC_SIZE = 300
_MAX_LEN = 512
_D_MODEL = 768
_NLAYERS = 2
_NHEAD = 8


def _global_ap(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(x.view(x.size(0), x.size(1), -1), dim=1)


class _SimSonEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TransformerWrapper(
            num_tokens=_DIC_SIZE,
            max_seq_len=_MAX_LEN,
            emb_dropout=0.1,
            attn_layers=Encoder(
                dim=_D_MODEL,
                depth=_NLAYERS,
                heads=_NHEAD,
            ),
        )
        self.linear = nn.Linear(_DIC_SIZE, _MAX_LEN)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        mask = src.gt(0).bool()
        x = self.encoder(src, mask=mask)
        x = _global_ap(x)
        return self.linear(x)


class SimSonEmbedder(EmbeddingModel):
    def __init__(self, device: str = None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = torch.device(device)
        self._model = _SimSonEncoder()
        self._model.load_state_dict(
            torch.load(str(_CHECKPOINT), map_location=self._device)
        )
        self._model.to(self._device)
        self._model.eval()
        self._tokenizer = Tokenizer.from_file(str(_TOKENIZER))

    @property
    def embedding_dim(self) -> int:
        return _MAX_LEN

    def embed(self, smiles: List[str]) -> np.ndarray:
        all_ids = [self._tokenizer.encode(s).ids for s in smiles]
        seq_len = min(max(len(ids) for ids in all_ids), _MAX_LEN)
        padded = torch.zeros(len(smiles), seq_len, dtype=torch.long, device=self._device)
        for i, ids in enumerate(all_ids):
            trunc = ids[:seq_len]
            padded[i, : len(trunc)] = torch.tensor(trunc, device=self._device)
        with torch.no_grad():
            out = self._model(padded)
        return out.cpu().numpy().astype(np.float32)


def get_embedder(device: str = None, **_) -> EmbeddingModel:
    return SimSonEmbedder(device=device)
