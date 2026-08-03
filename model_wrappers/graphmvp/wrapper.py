import sys
from pathlib import Path
from typing import List

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "graphmvp_repo" / "src_classification"))

import numpy as np
import torch
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Batch
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

from models import GNN
from datasets.molecule_datasets import mol_to_graph_data_obj_simple

from src.core.interfaces import EmbeddingModel

_EMB_DIM = 300
_WEIGHTS_DIR = _HERE / "weights"

_BASE = "GraphMVP_simple/output"
_MODEL_FILES = {
    # main GraphMVP variants
    "graphmvp":        f"{_BASE}/3D_hybrid_02_masking/GEOM_3D_nmol50000_nconf5_nupper1000/CL_1_VAE_1/6_51_10_0.1/0.15_EBM_dot_prod_0.2_normalize_l2_detach_target_2_100_0/pretraining_model.pth",
    "graphmvp-cp":     f"{_BASE}/3D_hybrid_03_masking/GEOM_3D_nmol50000_nconf5_nupper1000/CL_1_VAE_1_CP_0.1/6_51_10_0.1/0.15_EBM_dot_prod_0.2_normalize_l2_detach_target_2_100_0/pretraining_model.pth",
    "graphmvp-am":     f"{_BASE}/3D_hybrid_03_masking/GEOM_3D_nmol50000_nconf5_nupper1000/CL_1_VAE_1_AM_1/6_51_10_0.1/0.15_EBM_dot_prod_0.05_normalize_l2_detach_target_2_100_0/pretraining_model.pth",
    # baselines
    "graphcl":         "baseline_weights/GraphCL.pth",
    "am":              "baseline_weights/AM.pth",
    "cp":              "baseline_weights/CP.pth",
    "motif":           "baseline_weights/Motif.pth",
    "contextual":      "baseline_weights/Contextual.pth",
    "ep":              "baseline_weights/EP.pth",
    "ig":              "baseline_weights/IG.pth",
    "gpt-tnn":         "baseline_weights/GPT_TNN.pth",
}

_POOL_FNS = {
    "mean": global_mean_pool,
    "max":  global_max_pool,
    "sum":  global_add_pool,
}


class GraphMVPEmbedder(EmbeddingModel):
    def __init__(self, model: str = "graphmvp", pooling: str = "mean", device: str = "cpu"):
        self.device = torch.device(device)
        pool_fn = _POOL_FNS.get(pooling)
        if pool_fn is None:
            raise ValueError(f"Unknown pooling '{pooling}'. Choose from: {list(_POOL_FNS)}")
        self._pool = pool_fn

        weights_file = _MODEL_FILES.get(model)
        if weights_file is None:
            raise ValueError(f"Unknown model '{model}'. Choose from: {list(_MODEL_FILES)}")

        self._gnn = GNN(num_layer=5, emb_dim=_EMB_DIM, JK="last", drop_ratio=0.0, gnn_type="gin")
        ckpt = torch.load(_WEIGHTS_DIR / weights_file, map_location=self.device)
        self._gnn.load_state_dict(ckpt)
        self._gnn.to(self.device).eval()

    @property
    def embedding_dim(self) -> int:
        return _EMB_DIM

    def _embed_one(self, smiles: str) -> torch.Tensor:
        try:
            # Prefer a sanitized parse (faithful for valid molecules); fall back
            # to sanitize=False so graph-derived SMILES with hypervalent atoms
            # still featurize (the featurizer only reads atom/bond type + chirality
            # + bond dir, none of which need sanitization).
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                mol = Chem.MolFromSmiles(smiles, sanitize=False)
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


def get_embedder(model: str = "graphmvp", **kwargs) -> GraphMVPEmbedder:
    return GraphMVPEmbedder(model=model, **kwargs)
