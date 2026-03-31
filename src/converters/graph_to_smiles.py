from ..core.interfaces import Converter
import networkx as nx
from rdkit import Chem


def _nx_to_rdkit(graph: nx.Graph) -> Chem.Mol:
    mol = Chem.RWMol()
    node_to_idx = {}
        
    # 1. Add atom nodes
    for node in graph.nodes():
        idx = mol.AddAtom(Chem.Atom('C'))
        node_to_idx[node] = idx
            
    # 2. Add Single bonds
    for u, v in graph.edges():
        mol.AddBond(node_to_idx[u], node_to_idx[v], Chem.BondType.SINGLE)
            
    final_mol = mol.GetMol()
    return final_mol


class GraphToCarbonSMILESConverter(Converter):
    def convert(self, graph: nx.Graph) -> str:
        """
        Converts an nx.Graph into a SMILES string of pure Carbon atoms.
        """            
        # Generate SMILES
        graph_mol = _nx_to_rdkit(graph)
        smiles = Chem.MolToSmiles(graph_mol)
        return smiles