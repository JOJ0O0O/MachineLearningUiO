import numpy as np

class MomentumOptimizer:
    """Turns a list of gradients into a list of updates; knows nothing about the network."""

    def __init__(self, gamma, beta=0.9):
        self.gamma, self.beta, self.velocity = gamma, beta, None

    def step(self, layer_grads):
        if self.velocity is None:
            self.velocity = [(np.zeros_like(dW), np.zeros_like(db)) for dW, db in layer_grads]
        self.velocity = [(self.beta * vW + self.gamma * dW, self.beta * vb + self.gamma * db)
                         for (vW, vb), (dW, db) in zip(self.velocity, layer_grads)]
        return self.velocity

class AdamOptimizer:
    """Turns a list of gradients into a list of updates, no knowledge of the network"""
 
    def __init__(self, gamma, beta1=0.9, beta2=0.999):
        self.gamma, self.beta1, self.beta2, self.r, self.m, self.t = gamma, beta1, beta2, None, None, 1
 
    def step(self, layer_grads):
        """
        Calculates the update for a single Adam step given the gradients of the weights and biases of every layer
        """
        # AI usage: Claude (Anthropic) suggested replacing `== None` with `is None`
        # in the two initialisation checks. The Adam algorithm itself: no AI usage.
        if self.r is None:
            self.r = [(np.zeros_like(dW), np.zeros_like(db)) for dW, db in layer_grads]
        if self.m is None:
            self.m = [(np.zeros_like(dW), np.zeros_like(db)) for dW, db in layer_grads]
 
        self.m = [(self.beta1 * mW + (1.0 - self.beta1) * dW, self.beta1 * mb + (1.0 - self.beta1) * db) for (mW, mb), (dW, db) in zip(self.m, layer_grads)]       # Eq. (4.51)
        self.r = [(self.beta2 * rW + (1.0 - self.beta2) * dW * dW, self.beta2 * rb + (1.0 - self.beta2) * db * db) for (rW, rb), (dW, db) in zip(self.r, layer_grads)]       # Eq. (4.52)
        m_hat = [(mW / (1.0 - self.beta1**self.t), mb / (1.0 - self.beta1**self.t)) for (mW, mb) in self.m]
        r_hat = [(rW / (1.0 - self.beta2**self.t), rb / (1.0 - self.beta2**self.t)) for (rW, rb) in self.r]
 
        update = [((self.gamma * m_hatW) / (np.sqrt(r_hatW + 1e-10)),
                   self.gamma * m_hatb / (np.sqrt(r_hatb + 1e-10)))
                  for (m_hatW, m_hatb), (r_hatW, r_hatb) in zip(m_hat, r_hat)]
        self.t += 1
 
        return update