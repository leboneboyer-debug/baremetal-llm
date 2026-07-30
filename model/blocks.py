import numpy as np

from core.nn import Module
from core.nn import RMSLayerNorm, LinearLayer, SiLU
from core.attention import CausalSelfAttention

class SwiGLU(Module):
    """
    Swish-Gated Linear Unit (SwiGLU) for Feee-Forward Network

    """

    def __init__(self, 
                d_model: int, 
                hidden_dim: int):
        super().__init__()

        self.gate_proj = LinearLayer(d_model, hidden_dim, bias=True)
        self.value_proj = LinearLayer(d_model, hidden_dim, bias=True)
        self.out_proj = LinearLayer(hidden_dim, d_model, bias=False)
        self.silu = SiLU()

    def forward(self, x):

        gate = self.silu(self.gate_proj(x))
        up = self.value_proj(x)
        gated_hidden = gate * up
        return self.out_proj(gated_hidden)

class TransformerBlock(Module):

    def __init__(self, 
                d_model: int, 
                num_heads: int, 
                hidden_dim: int
    ):

        super().__init__()
        self.rms_norm_1 = RMSLayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model=d_model, num_heads=num_heads)
        
        self.rms_norm_2 = RMSLayerNorm(d_model)
        self.ffn = SwiGLU(d_model, hidden_dim)

    def forward(self, x):

        norm_x1 = self.rms_norm_1(x)
        attn = self.attention(norm_x1)
        x = x + attn

        norm_x2 = self.rms_norm_2(x)
        ffn_out = self.ffn(norm_x2)
        x = x + ffn_out

        return x





