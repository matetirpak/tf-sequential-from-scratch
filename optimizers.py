'''All optimizers are defined here.'''

from abc import ABC, abstractmethod
import numpy as np



class Optimizer(ABC):
    '''Abstract Optimizer layout'''
    def __init__(self, eta):
        self.eta = eta
        self.math = None

    @abstractmethod
    def update(self, w_params, b_params, w_grads, b_grads):
        '''Applies gradients to respective parameters'''
        pass
        
    def cpu_or_gpu(self, gpu=False):
        '''Loads respective math library'''
        if not gpu:
            import numpy as np
            self.math = np
        else:
            import cupy as cp
            self.math = cp


class DefaultOptimizer(Optimizer):
    '''No optimization, gradients are applied directly'''
    def __init__(self, eta):
        super().__init__(eta) 
        

    def update(self, w_params, b_params, w_grads, b_grads):
        n = len(w_params)
        for i in range(n):
            w_params[i] -= self.eta * w_grads[i]
            b_params[i] -= self.eta * b_grads[i]
        return w_params, b_params


class Adam(Optimizer):
    '''Adam optimizer'''
    def __init__(self, eta=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(eta) 
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        
        self.m_w = None
        self.v_w = None
        self.m_b = None
        self.v_b = None

    
    def set_eta(self, eta):
        self.eta = eta

    
    def update(self, w_params, b_params, w_grads, b_grads):
        if not self.math:
            raise Exception("Math library couldn't be determined.")
            
        if self.m_w is None:
            self.m_w = [self.math.zeros_like(w) for w in w_params]
            self.v_w = [self.math.zeros_like(w) for w in w_params]
            self.m_b = [self.math.zeros_like(b) for b in b_params]
            self.v_b = [self.math.zeros_like(b) for b in b_params]

        self.t += 1

        # Update weights
        for i, (w, grad_w) in enumerate(zip(w_params, w_grads)):
            self.m_w[i] = (self.beta1 * self.m_w[i] 
                          + (1 - self.beta1) * grad_w)
            self.v_w[i] = (self.beta2 * self.v_w[i] 
                          + (1 - self.beta2) * (grad_w ** 2))
            
            m_hat_w = self.m_w[i] / (1 - self.beta1 ** self.t)
            v_hat_w = self.v_w[i] / (1 - self.beta2 ** self.t)
            
            w_params[i] -= (self.eta * m_hat_w
                           / (self.math.sqrt(v_hat_w) + self.epsilon))

        # Update biases
        for i, (b, grad_b) in enumerate(zip(b_params, b_grads)):
            self.m_b[i] = (self.beta1 * self.m_b[i] 
                          + (1 - self.beta1) * grad_b)
            self.v_b[i] = (self.beta2 * self.v_b[i] 
                          + (1 - self.beta2) * (grad_b ** 2))
            
            m_hat_b = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_hat_b = self.v_b[i] / (1 - self.beta2 ** self.t)
            
            b_params[i] -= (self.eta * m_hat_b 
                           / (self.math.sqrt(v_hat_b) + self.epsilon))

        return w_params, b_params
