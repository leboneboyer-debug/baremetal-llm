# core/nn.py
# Neural Network (lol)
import numpy as np
from core.tensor import Tensor
from core.ops import gelu


# Base Module Class:
class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must call forward()")
    
    def parameters(self):
        params = []
        for attr in self.__dict__.values():
            if isinstance(attr, Tensor) and attr.requires_grad:
                params.append(attr)
            elif isinstance(attr, Module):
                params.extend(attr.parameters())
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params
    
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()
    
# Embedding Layer
class EmbeddingLayer(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim

        weight_data = np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.02
        self.weight = Tensor(weight_data)
    
    def forward(self, index):

        indices = np.array(index, dtype=np.int64)

        out_data = self.weight.data[index]
        outer = Tensor(data=out_data, _children=(self.weight,), _op="embedding")

        def _backward():
            grad = outer.grad.data if isinstance(outer.grad, Tensor) else outer.grad

            np.add.at(self.weight.grad, indices, grad)
        
        outer._backward = _backward
        return outer

class LinearLayer(Module):
    def __init__(self, in_f, out_f, bias=True):
        super().__init__()
        self.in_f = in_f
        self.out_f = out_f

        # Xavier Initialization
        bound = np.sqrt(6 / (in_f + out_f))
        weight_data = np.random.uniform(-bound, bound, (in_f, out_f))

        # Initalized Tensor consists of our weights
        self.weight = Tensor(weight_data)

        # If bias exists, intialize a tensor of our biases
        if bias:
            bias_data = np.zeros((out_f,), dtype=np.float32)
            self.bias = Tensor(bias_data)
        else:
            self.bias = None

    def forward(self, x): 
        # Affine Transformation: y = W * x + b
        outer = x @ self.weight

        if self.bias is not None:
            outer += self.bias

        return outer 
    
class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape, )
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps

        self.gamma = Tensor(np.ones(self.normalized_shape, dtype=np.float32))
        self.beta = Tensor(np.zeros(self.normalized_shape, dtype=np.float32))

        def forward(self, x):

            mean = x.sum(axis=-1, keepdims=True) * (1.0 / self.normalized_shape[-1])
            x_centered = x - mean

            variance = (x_centered ** 2).sum(axis=1, keepdims=True) * (1.0 / self.normalized_shape[-1])
            inv_std = (variance + self.eps) ** -0.5

            x_norm = x_centered * inv_std
            out = self.gamma * x_norm + self.beta

            return out

class RMSLayerNorm(Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.gamma = Tensor(np.ones(dim, ))

    def forward(self, x):

        ms = (x * x).sum(axis=-1, keepdims=True) * (1.0 / self.dim)
        rms = (ms + self.eps) ** -0.5

        x_norm = x * rms * self.gamma
        return x_norm

class GELU(Module):
    def forward(self, x):
        return gelu(x)

class SiLU(Module):
    
    def forward(self, x):
        sigmoid_x = 1.0 / (1.0 + (-x).exp())
        return x * sigmoid_x

        


        