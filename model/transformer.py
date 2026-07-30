import numpy as np

from core.tensor import Tensor
from core.nn import  Module, RMSLayerNorm, LinearLayer, EmbeddingLayer  
from model.blocks import TransformerBlock

class BareMetalTransformer(Module):
    
    def __init__(self, 
            vocab_size: int, 
            d_model: int,
            hidden_dim: int, 
            num_heads: int, 
            n_layers: int
    ): 

        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.t_embedding = EmbeddingLayer(vocab_size, d_model)
        
        self.blocks = [
            TransformerBlock(d_model, num_heads, hidden_dim)
            for _ in range(n_layers)
        ]

        self.final_norm = RMSLayerNorm(d_model)

        self.lm_head = LinearLayer(d_model, vocab_size, bias=False)

    def forward(self, input_ids: np.ndarray) -> Tensor:

        x = self.t_embedding(input_ids)

        for block in self.blocks:
            x = block(x)
        
        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits


