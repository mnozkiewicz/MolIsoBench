import os
import sys
from pathlib import Path
from typing import List

_HERE = Path(__file__).parent
_MOL_DIR = _HERE / "GraphFP" / "mol"
sys.path.insert(0, str(_MOL_DIR))

import numpy as np
import torch
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Batch
from rdkit import Chem

_orig_dir = os.getcwd()

os.chdir(_MOL_DIR)
import downstream_old as base
from loader import mol_to_graph_data_obj_simple
os.chdir(_orig_dir)

from src.core.interfaces import EmbeddingModel

_PRETRAIN_DIR = _HERE / "GraphFP" / "mol" / "pretrain"
_EMB_DIM = 300

_MODEL_FILES = {
    "C":  "GIN_C.pth",
    "CP": "GIN_CP_03.pth",
}


class GraphFPEmbedder(EmbeddingModel):
    def __init__(self, mode: str = "CP", device: str = "cpu"):
        self.device = torch.device(device)
        filename = _MODEL_FILES.get(mode.upper())
        if filename is None:
            raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(_MODEL_FILES)}")

        model = base.GNN_graphpred(1, emb_dim=_EMB_DIM, num_gnn_layers=5, dropout=0, graph_pooling="mean")
        ckpt = torch.load(_PRETRAIN_DIR / filename, map_location=self.device)
        model.gnn.load_state_dict(ckpt["gnn"] if "gnn" in ckpt else ckpt["mol_gnn"])
        self._gnn = model.gnn.to(self.device)
        self._gnn.eval()

    @property
    def embedding_dim(self) -> int:
        return _EMB_DIM

    def _embed_one(self, smiles: str) -> torch.Tensor:
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            pyg = mol_to_graph_data_obj_simple(mol)
            batch = Batch.from_data_list([pyg]).to(self.device)
            with torch.no_grad():
                node_repr = self._gnn(batch.x, batch.edge_index, batch.edge_attr)
            return global_mean_pool(node_repr, batch.batch).squeeze(0)
        except Exception as e:
            print(f"Error processing '{smiles}': {e}")
            return torch.full((_EMB_DIM,), float("nan"))

    def embed(self, smiles: List[str]) -> np.ndarray:
        return torch.stack([self._embed_one(s) for s in smiles]).cpu().numpy()


def get_embedder(mode: str = "CP", **kwargs) -> GraphFPEmbedder:
    return GraphFPEmbedder(mode=mode, **kwargs)
