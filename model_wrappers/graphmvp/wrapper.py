import sys
from pathlib import Path
from typing import List

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "graphmvp_repo" / "src_classification"))

import numpy as np
import torch
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Batch
from rdkit import Chem

from models import GNN
from datasets.molecule_datasets import mol_to_graph_data_obj_simple

from src.core.interfaces import EmbeddingModel

_EMB_DIM = 300
_WEIGHTS_DIR = _HERE / "weights"

_POOL_FNS = {
    "mean": global_mean_pool,
    "max":  global_max_pool,
    "sum":  global_add_pool,
}


class GraphMVPEmbedder(EmbeddingModel):
    def __init__(self, weights_file: str, pooling: str = "mean", device: str = "cpu"):
        self.device = torch.device(device)
        pool_fn = _POOL_FNS.get(pooling)
        if pool_fn is None:
            raise ValueError(f"Unknown pooling '{pooling}'. Choose from: {list(_POOL_FNS)}")
        self._pool = pool_fn

        self._gnn = GNN(num_layer=5, emb_dim=_EMB_DIM, JK="last", drop_ratio=0.0, gnn_type="gin")
        ckpt = torch.load(_WEIGHTS_DIR / weights_file, map_location=self.device)
        self._gnn.load_state_dict(ckpt)
        self._gnn.to(self.device).eval()

    @property
    def embedding_dim(self) -> int:
        return _EMB_DIM

    def _embed_one(self, smiles: str) -> torch.Tensor:
        try:
            mol = Chem.MolFromSmiles(smiles)
            pyg = mol_to_graph_data_obj_simple(mol)
            return pyg
        except Exception:
            return None

    def embed(self, smiles: List[str]) -> np.ndarray:
        graphs = []
        nan_mask = []
        for s in smiles:
            g = self._embed_one(s)
            if g is None:
                nan_mask.append(True)
                graphs.append(None)
            else:
                nan_mask.append(False)
                graphs.append(g)

        valid_graphs = [g for g in graphs if g is not None]
        result = torch.full((len(smiles), _EMB_DIM), float("nan"))

        if valid_graphs:
            batch = Batch.from_data_list(valid_graphs).to(self.device)
            with torch.no_grad():
                node_repr = self._gnn(batch)
            pooled = self._pool(node_repr, batch.batch).cpu()
            valid_idx = [i for i, m in enumerate(nan_mask) if not m]
            for out_i, in_i in enumerate(valid_idx):
                result[in_i] = pooled[out_i]

        return result.numpy()


def get_embedder(weights_file: str = "GraphMVP/pretraining_model.pth", **kwargs) -> GraphMVPEmbedder:
    return GraphMVPEmbedder(weights_file=weights_file, **kwargs)
