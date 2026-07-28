import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        """
        Initializes a Tensor object
        """
        if isinstance(data, Tensor):
            data = data.data

        self.data = np.array(data, dtype=np.float32)

        self.grad = np.zeros_like(self.data, dtype=np.float32)

        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op})"
    
    # Helper function "unbroadcast"
    @staticmethod
    def unbroadcast(grad, shape):
        """Sum-out broadcasted dimensions of grad to match shape."""
        grad = grad.data if isinstance(grad, Tensor) else grad
        # Reduce extra leading dimensions
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)

        # Reduce dimensions that were broadcast (size 1 in target shape)
        for i, dim in enumerate(shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)

        return grad

    ###############################
    # Primary Arithmetic Operations
    ###############################
    
    def __add__(self, other):

        other = other if isinstance(other, Tensor) else Tensor(other)

        # Forward Pass using tensor addition
        outer = Tensor(self.data + other.data, (self, other), "+")

        def _backward():

            og = outer.grad.data if isinstance(outer.grad, Tensor) else outer.grad

            grad_self = Tensor.unbroadcast(og, self.data.shape)
            grad_other = Tensor.unbroadcast(og, other.data.shape)
            
            self.grad = self.grad + grad_self
            other.grad = other.grad + grad_other
            
        outer._backward = _backward
        return outer
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        # Forward Pass using Tensor multplication
        outer = Tensor(self.data * other.data, (self, other), "*")

        def _backward():

            self.grad += Tensor.unbroadcast(other.data * outer.grad, self.data.shape)
            other.grad += Tensor.unbroadcast(self.data * outer.grad, other.data.shape)

        outer._backward = _backward
        return outer
    
    def __pow__(self, power):

        outer = Tensor(self.data ** power, (self, ), "**")

        def _backward():
            self.grad += Tensor.unbroadcast((power * (self.data ** (power - 1)) * outer.grad), self.data.shape)

        outer._backward = _backward
        return outer
    
    def sqrt(self):
        # Forward pass: x ** 0.5
        outer = Tensor(np.sqrt(self.data), (self,), "sqrt")

        def _backward():
            # d/dx(sqrt(x)) = 1 / (2 * sqrt(x))
            og = outer.grad.data if isinstance(outer.grad, Tensor) else outer.grad
            self.grad += Tensor.unbroadcast(og / (2.0 * outer.data), self.data.shape)

        outer._backward = _backward
        return outer
    
    def __truediv__(self, other):
    # self / other  ->  self * (other**-1)
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self * (other ** -1.0)

    def __rtruediv__(self, other):
        # other / self  ->  other * (self**-1)
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other * (self ** -1.0)
    

    def exp(x):
        outer = Tensor(np.exp(x.data), _children=(x,), _op="exp")

        def _backward():
            grad = outer.grad.data if isinstance(outer.grad, Tensor) else outer.grad
            x.grad += grad * outer.data

        outer._backward = _backward
        return outer
    
    
    def __sub__(self, other): return self + (-other)
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other

    # =============================
    # Matrix and Tensor Operations
    # =============================
    
    def relu(self):
        outer = Tensor(np.maximum(0, self.data), (self, ), "reLU")

        def _backward():
            self.grad += (self.data > 0) * outer.grad

        outer._backward = _backward
        return outer
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        outer = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():

            grad_self = outer.grad @ other.data.T
            grad_other = self.data.T @ outer.grad

            self.grad = self.grad + grad_self
            other.grad = other.grad + grad_other

        outer._backward = _backward
        return outer
    
    def sum(self, axis=None, keepdims=False):

        # 1. Forward Pass
        out_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        outer = Tensor(out_data, (self,), "sum")

        # 2. Backward Pass
        def _backward():
            grad = outer.grad.data if isinstance(outer.grad, Tensor) else outer.grad
        
            # If axis was specified and keepdims was False, we must re-insert 
            # singleton dimensions (size 1) so grad can broadcast back to self.data.shape
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else axis
                shape = list(self.data.shape)
                for ax in axes:
                    ax = ax % len(shape)  # Handle negative axis indices (e.g., -1)
                    shape[ax] = 1
                grad = grad.reshape(shape)

        # Broadcast gradient across the summed dimensions and accumulate
            self.grad += np.broadcast_to(grad, self.data.shape)

        outer._backward = _backward
        return outer
    

    # ===================
    # Backpropogation
    # ===================

    def backward_prop(self):

        order = []
        visited = set()

        def build_order(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_order(child)
                order.append(v)

        build_order(self)
        self.grad = np.ones_like(self.data, dtype=np.float32)

        for node in reversed(order):
            node._backward()
            # print("Node Operation:",node._op)


    # ======================
    # Reshaping and Transposing
    # ====================== 

    def reshape(self, *shape):
        shape = shape[0] if isinstance(shape[0], (tuple, list)) else shape
        outer = Tensor(self.data.reshape(shape), (self, ), "reshape" )

        def _backward():
            self.grad = outer.grad.reshape(self.data.shape)

        outer._backward = _backward
        return outer

    def transpose(self, *axes):
        outer = Tensor(self.data.transpose(*axes), (self,), "transpose") 

        def _backward():
            reversed_axes = self.data.transpose(*axes)
            self.grad = outer.grad.transpose(*reversed_axes)

        outer._backward = _backward
        return outer
        


            



