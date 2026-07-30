import numpy as np

class Optimizer:
    # Base class for all optimizers
    def __init__(self, params):
        self.params = list(params)

    def reset_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data, dtype=np.float32)

    def step(self):
        raise NotImplementedError

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            grad = p.grad

            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad ** 2)

            m_hat = self.m[i] / (1.0 - (self.beta1 ** self.t))
            v_hat = self.v[i] / (1.0 - (self.beta2 ** self.t))

            adaptive_update = m_hat / (np.sqrt(v_hat) + self.eps)

            p.data = p.data - self.lr * (adaptive_update + self.weight_decay * p.data)
            

