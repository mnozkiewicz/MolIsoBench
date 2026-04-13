import networkx as nx
from abc import ABC, abstractmethod
from typing import Any, Iterator, Dict, Tuple
import numpy as np


class GraphProvider(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[Dict[str, Any], nx.Graph]]:
        """
        Must yield a tuple of (graph_metadata, networkx_graph).
        This guarantees a standard Intermediate Representation (IR).
        """
        pass


class Converter(ABC):
    @abstractmethod
    def convert(self, graph: nx.Graph) -> Any:
        """
        Takes the standard nx.Graph and converts it into some other format
        """
        pass


class EmbeddingModel(ABC):

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        pass

    @abstractmethod
    def embed(self, formatted_data: Any) -> np.ndarray:
        """Takes the output of the Converter and returns a vector."""
        pass