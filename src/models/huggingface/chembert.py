import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from ...core.interfaces import EmbeddingModel
from typing import List

class ChemBert(EmbeddingModel):

    MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"

    def __init__(self, pooling: str = "cls", device: str = "cpu"):

        self.pooling = pooling
        self.device = torch.device(device)
        
        print(f"Loading {self.MODEL_NAME} onto {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME, add_pooling_layer=False).to(self.device)
        self.model.eval()

    @property
    def embedding_dim(self) -> int:
        return self.model.config.hidden_size

    def embed(self, smiles: List[str]) -> np.ndarray:

        inputs = self.tokenizer(
            smiles, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get the embeddings for all tokens (Shape: [batch_size, seq_len, hidden_size])
        last_hidden_state = outputs.last_hidden_state
        
        if self.pooling == "cls":
            graph_embedding = last_hidden_state[:, 0, :]
        elif self.pooling == "mean":
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            masked_embeddings = last_hidden_state * attention_mask
            sum_embeddings = torch.sum(masked_embeddings, dim=1)
            seq_lengths = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            graph_embedding = sum_embeddings / seq_lengths
        elif self.pooling == "sum":
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            masked_embeddings = last_hidden_state * attention_mask
            graph_embedding = torch.sum(masked_embeddings, dim=1)
        else:
            raise ValueError("Pooling must be 'cls', 'mean' ot 'sum'")
        
        return graph_embedding.cpu().numpy()