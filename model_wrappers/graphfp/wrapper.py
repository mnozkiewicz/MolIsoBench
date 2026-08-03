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

# GraphFP's `mol` code is cwd-sensitive: utils/chem_utils.py runs
# `open("vocab.txt")` at import time, and several modules use relative imports.
# Import them (including the fragment Tokenizer) with cwd set to the mol dir.
_orig_dir = os.getcwd()
os.chdir(_MOL_DIR)
import downstream_old as base
import downstream_frag_old as base_frag  # frag GNN uses Embedding(908), not 800
from loader import mol_to_graph_data_obj_simple
from mol_bpe import Tokenizer
os.chdir(_orig_dir)

from src.core.interfaces import EmbeddingModel

_PRETRAIN_DIR = _MOL_DIR / "pretrain"
_VOCAB = _MOL_DIR / "vocab.txt"
_EMB_DIM = 300
_MOL_LAYERS = 5
_FRAG_LAYERS = 2

# Single-atom fragment SMILES that FragmentDataset strips brackets from before
# the vocab lookup (mirrors `organic_major_ish` in prepare_data_old.py).
_ORGANIC_MAJOR_ISH = {"[C]", "[O]", "[N]", "[F]", "[Cl]", "[Br]", "[I]", "[S]", "[P]", "[B]", "[H]"}

# mode -> pretrained checkpoint. C/CP use only the atom (molecular) GNN -> 300-d.
# CPF additionally runs a GNN over the fragment graph and concatenates -> 600-d.
_MODEL_FILES = {
    "C":   "GIN_C.pth",
    "CP":  "GIN_CP_03.pth",
    "CPF": "GIN_CPF_01.pth",
}


class GraphFPEmbedder(EmbeddingModel):
    def __init__(self, mode: str = "CP", device: str = "cpu"):
        self.mode = mode.upper()
        self.device = torch.device(device)
        filename = _MODEL_FILES.get(self.mode)
        if filename is None:
            raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(_MODEL_FILES)}")

        ckpt = torch.load(_PRETRAIN_DIR / filename, map_location=self.device)

        # atom (molecular) GNN — present in every checkpoint
        mol_model = base.GNN_graphpred(
            1, emb_dim=_EMB_DIM, num_gnn_layers=_MOL_LAYERS, dropout=0, graph_pooling="mean"
        )
        mol_sd = ckpt["mol_gnn"] if "mol_gnn" in ckpt else ckpt["gnn"]
        mol_model.gnn.load_state_dict(mol_sd)
        self._mol_gnn = mol_model.gnn.to(self.device)
        self._mol_gnn.eval()

        # fragment GNN — only for CPF
        self._frag_gnn = None
        self._tokenizer = None
        self._vocab_dict = None
        if self.mode == "CPF":
            # NB: the fragment GNN lives in downstream_frag_old, whose atom=False
            # branch uses Embedding(908) (the pretrained frag-vocab size); the
            # copy in downstream_old hardcodes 800 and won't load this checkpoint.
            frag_model = base_frag.GNN_graphpred(
                1, emb_dim=_EMB_DIM, num_gnn_layers=_FRAG_LAYERS, dropout=0,
                graph_pooling="mean", atom=False,
            )
            frag_model.gnn.load_state_dict(ckpt["frag_gnn"])
            self._frag_gnn = frag_model.gnn.to(self.device)
            self._frag_gnn.eval()

            self._tokenizer = Tokenizer(str(_VOCAB))
            # fragment SMILES -> contiguous vocab id, matching training (Embedding(800))
            self._vocab_dict = {
                smi: i for i, smi in enumerate(self._tokenizer.vocab_dict.keys())
            }

    @property
    def embedding_dim(self) -> int:
        return _EMB_DIM * 2 if self.mode == "CPF" else _EMB_DIM

    def _mol_embed(self, smiles: str) -> torch.Tensor:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        pyg = mol_to_graph_data_obj_simple(mol)
        batch = Batch.from_data_list([pyg]).to(self.device)
        with torch.no_grad():
            node_repr = self._mol_gnn(batch.x, batch.edge_index, batch.edge_attr)
        return global_mean_pool(node_repr, batch.batch).squeeze(0)

    def _frag_embed(self, smiles: str) -> torch.Tensor:
        # Build the fragment graph exactly as FragmentDataset.process does:
        # fragment the molecule, map each fragment node to its vocab id, and
        # build a bidirectional (unfeatured) edge list between fragments.
        tree = self._tokenizer(smiles)

        frag = [[0] for _ in range(len(tree.nodes))]
        for node_i in tree.nodes:
            node = tree.get_node(node_i)
            for _ in node.atom_mapping.keys():
                node_smile = node.smiles
                if node_smile in _ORGANIC_MAJOR_ISH:
                    node_smile = node_smile[1:-1]
                frag[node_i][0] = self._vocab_dict[node_smile]

        frag_edge_index = [[], []]
        for src, dst in tree.edges:
            frag_edge_index[0].extend([src, dst])
            frag_edge_index[1].extend([dst, src])

        frag_x = torch.LongTensor(frag).to(self.device)                 # [num_frags, 1]
        edge_index = torch.LongTensor(frag_edge_index).to(self.device)  # [2, 2*num_edges]
        batch_vec = torch.zeros(frag_x.size(0), dtype=torch.long, device=self.device)

        with torch.no_grad():
            # fragment GNN has no edge features (called with edge_attr=None)
            node_repr = self._frag_gnn(frag_x.squeeze(-1), edge_index, None)
        return global_mean_pool(node_repr, batch_vec).squeeze(0)

    def _embed_one(self, smiles: str) -> torch.Tensor:
        try:
            mol_emb = self._mol_embed(smiles)
            if self.mode == "CPF":
                frag_emb = self._frag_embed(smiles)
                return torch.cat([mol_emb, frag_emb], dim=-1)
            return mol_emb
        except Exception as e:
            print(f"Error processing '{smiles}': {e}")
            return torch.full((self.embedding_dim,), float("nan"), device=self.device)

    def embed(self, smiles: List[str]) -> np.ndarray:
        return torch.stack([self._embed_one(s) for s in smiles]).cpu().numpy()


def get_embedder(mode: str = "CP", **kwargs) -> GraphFPEmbedder:
    return GraphFPEmbedder(mode=mode, **kwargs)
