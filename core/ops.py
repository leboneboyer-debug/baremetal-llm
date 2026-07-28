import numpy as np
from core.tensor import Tensor

def softmax(x, axis=-1):
    # Ensure the input is a Tensor
    x = x if isinstance(x, Tensor) else Tensor(x)

    # Foward Pass (Using Log-Sum_Exp for numerical stability)
    max_x = np.max(x.data, axis=axis, keepdims=0)
    exp_x = np.exp(x - max_x) 

    # Compute Softmax Probabilities
    probs = exp_x / np.sum(exp_x, axis=axis, keepdims=0)

    # Create a outside tensor of softmax probabilties
    outer = Tensor(probs, (x, ), "softmax")

    def _backward():


        sum_G_S = np.sum(outer.grad * outer.data, axis=axis, keepdims=True)
        x.grad = (outer.data * (outer.grad - sum_G_S))

    outer._backward = _backward
    return outer

def cross_entropy_loss(logits, targets):

    logits = logits if isinstance(logits, Tensor) else Tensor(logits)
    targets = np.array(targets)

    # Reshaping the logits to 2-dimesions
    orig_shape = logits.data.shape
    logits_2d = logits.data.reshape(-1, orig_shape[-1])
    targets_flat = targets.reshape(-1)
    N = logits_2d.shape[0]

    # Foward Pass using Log-Sum-Max
    max_logits = np.max(logits_2d, axis=-1, keepdims=True)
    exp_logits = np.exp(logits_2d - max_logits)
    sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)

    # Softmax 
    probs_2d = exp_logits / sum_exp

    # Loss
    log_probs = (logits_2d - max_logits) - np.log(sum_exp)
    correct_log_probs = log_probs[np.arange(N), targets_flat]
    loss_val = -np.mean(correct_log_probs)


    outer = Tensor(loss_val, (logits, ), "cross-entropy")

    def _backward():

        grad_2d = probs_2d.copy()
        grad_2d[np.arange(N), targets_flat] -= 1.0
        grad_2d = (grad_2d / N) * outer.grad

        logits.grad += grad_2d.reshape(orig_shape)

    outer._backward = _backward
    return outer

def gelu(x):

    x = x if isinstance(x, Tensor) else Tensor(x)

    c = np.sqrt(2 / np.pi) 
    u_x = c * (x.data + 0.044175 * (x.data ** 3))
    a_x = np.tanh(u_x)

    outer_data = 0.5 * x.data * (1 + a_x)

    outer = Tensor(outer_data, (x, ), "GeLU")

    def _backward():

        du_dx = c * (1 + 3 * 0.044175 * (x.data **2))
        dtanh_dx = (1.0 - (a_x ** 2)) * du_dx

        local_grad = 0.5 * (1.0 + u_x) + 0.5 * x.data * dtanh_dx
        x.grad = local_grad * outer.grad
    
    outer._backward = _backward
    return outer

def layer_loss(x, gamma, beta, eps=1e-5):

    x = x if isinstance(x, Tensor) else Tensor(x)
    gamma = gamma if isinstance(gamma, Tensor) else Tensor(gamma)
    beta = beta if isinstance(beta, Tensor) else Tensor(beta)

    # 1. Forward Pass
    d = x.data.shape[-1]
    mean = np.mean(x.data, axis=-1, keepdims=True)
    var = np.var(x.data, axis=-1, keepdims=True)
    
    std_inv = 1.0 / np.sqrt(var + eps)
    x_hat = (x.data - mean) * std_inv
    outer_data = gamma.data * x_hat + beta.data

    outer = Tensor(outer_data, (x, gamma, beta), "layer_loss")

    def _backward():

    
        sum_axes = tuple(range(outer.grad.ndim - 1))

        beta.grad += np.sum(outer.grad, axis=sum_axes)
        gamma.grad += np.sum(outer.grad * x_hat, axis=sum_axes)

        dx_hat = outer.grad * gamma.data
        
        sum_dx_hat = np.sum(dx_hat, axis=-1, keepdims=True)
        sum_dx_hat_x_hat = np.sum(dx_hat * x_hat, axis=-1, keepdims=True)
        
        dx = (1.0 / d) * std_inv * (d * dx_hat - sum_dx_hat - x_hat * sum_dx_hat_x_hat)
        x.grad += dx
    
    outer._backward = _backward
    return outer







     













    






