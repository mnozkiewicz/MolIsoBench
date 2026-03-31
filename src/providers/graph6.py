from ..core.interfaces import GraphProvider
from typing import Iterator, Dict, Any, Tuple
import networkx as nx

class Graph6Provider(GraphProvider):
    def __init__(self, filepath: str):
        self.filepath = filepath

    def __iter__(self) -> Iterator[Tuple[Dict[str, Any], nx.Graph]]:
        """
        Reads a graph6 file line by line.
        Yields a dict with the line number as 'id' and the nx.Graph as 'graph'.
        """
        with open(self.filepath, 'r') as f:
            for line_idx, line in enumerate(f):
                g6_string = line.strip()
                if not g6_string:
                    continue

                graph = nx.from_graph6_bytes(g6_string.encode('ascii'))                
                yield {"id": line_idx}, graph