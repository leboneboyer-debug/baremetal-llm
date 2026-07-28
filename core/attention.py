import sys
import os

# Adds project root (baremetal-llm/) to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import numpy as np
from core.nn import Module, LinearLayer
from core.tensor import Tensor

class RoPeEmbedding:
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        assert dim % 2 == 0
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        pos = np.arange(max_seq_len, dtype=np.float32)

        freqs = np.outer(pos, inv_freq)

        self.sin_cached = np.sin(freqs).repeat(2, axis=-1)
        self.cos_cached = np.cos(freqs).repeat(2, axis=-1)

    def __rotate_half(self, x: np.ndarray) -> np.ndarray:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        res = np.stack([-x2, x1], axis=-1)
        return res.reshape(x.shape)
    
    def apply_rope(self, x: Tensor, seq_len: int) -> Tensor:
        cos = self.cos_cached[:seq_len, :][None, None, :, :] 
        sin = self.sin_cached[:seq_len, :][None, None, :, :]

        x_rotated_data = (x.data * cos) + (self.__rotate_half(x.data) * sin)

        outer = Tensor(x_rotated_data, (x,), "rope")

        def __backward():
            grad = outer.grad.data if isinstance(outer.grad, Tensor) else outer,grad

            dx = (grad * cos) + (self.__rotate_half(grad) * (-sin))
            grad += dx

        outer._backward = __backward
        return outer 


class CausalSelfAttention(Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 2048, bias: bool = False):
        super().__init__()
        assert d_model % num_heads == 0, "D_model must be divisible by the number of heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = LinearLayer(d_model, d_model, bias=bias)
        self.k_proj = LinearLayer(d_model, d_model, bias=bias)
        self.v_proj = LinearLayer(d_model, d_model, bias=bias)
        self.out_proj = LinearLayer(d_model, d_model, bias=bias)

        self.rope = RoPeEmbedding(dim=self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.data.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q_data = q.data.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k_data = k.data.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v_data = v.data.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        q_heads = Tensor(q_data, (q,), "split_heads")
        k_heads = Tensor(k_data, (k,), "split_heads")
        v_heads = Tensor(v_data, (v,), "split_heads")

        def _backward_split(tensor_in, tensor_out):
            def _backward():
                grad = tensor_out.grad.data if isinstance(tensor_out.grad, Tensor) else tensor_out.grad
                tensor_in +- grad.transpose(0,2,1,3).reshape(B, T, C)
            tensor_out._backward = _backward

        _backward_split(q_heads, q)
        _backward_split(k_heads, k)
        _backward_split(v_heads, v)

        q_heads = self.rope.apply_rope(q_heads, T)
        k_heads = self.rope.apply_rope(k_heads, T)

        scale = 1.0 / np.sqrt(self.head_dim)

        scores_data = np.matmul(q_heads.data, k_heads.data.transpose(0, 1, 3, 2)) * scale

        causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores_data[..., causal_mask] = -1e9

        exp_scores = np.exp(scores_data - np.max(scores_data, axis=-1, keepdims=True))
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        context_data = np.matmul(attn_weights, v_heads.data)
        
        context_concat = context_data.transpose(0, 2, 1, 3).reshape(B, T, C)
        
        context_tensor = Tensor(context_concat, _children=(x,), _op="attention") 

        def _backward_attn():
            grad = context_tensor.grad.data if isinstance(context_tensor.grad, Tensor) else context_tensor.grad
            d_context = grad.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            
            d_v = np.matmul(attn_weights.transpose(0, 1, 3, 2), d_context)
            d_attn = np.matmul(d_context, v_heads.data.transpose(0, 1, 3, 2))
            
            d_scores = d_attn * attn_weights * (1.0 - attn_weights)
            d_scores[..., causal_mask] = 0.0  # Zero out masked positions
            d_scores *= scale
            
            d_q = np.matmul(d_scores, k_heads.data)
            d_k = np.matmul(d_scores.transpose(0, 1, 3, 2), q_heads.data)
            
            q_heads.grad += d_q
            k_heads.grad += d_k
            v_heads.grad += d_v

        context_tensor._backward = _backward_attn

        # 9. Final Output Projection: Linear(d_model, d_model)
        outer = self.out_proj(context_tensor)
        return outer
    
if __name__ == "__main__":
    # Test Parameters
    batch_size = 2
    seq_len = 8
    d_model = 64
    num_heads = 4
    
    # 1. Instantiate Attention Module
    attn = CausalSelfAttention(d_model=d_model, num_heads=num_heads)
    
    # 2. Fake Input Tensor
    x = Tensor(np.random.randn(batch_size, seq_len, d_model))
    
    # 3. Forward Pass
    out = attn(x)
    print(f"Input shape:  {x.data.shape}")
    print(f"Output shape: {out.data.shape}")
    assert out.data.shape == (batch_size, seq_len, d_model), "Shape mismatch!"
    
    # 4. Backward Pass Test
    loss = out.sum()
    loss._backward()
    print("Backward pass successful! Parameter gradients populated.")



    


