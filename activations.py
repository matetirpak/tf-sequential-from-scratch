'''This module contains the activation functions for neural network layers.'''

from abc import ABC, abstractmethod
import numpy as np


class AcFunc(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def activate(self):
        '''Activation for the forward pass'''
        pass

    @abstractmethod
    def derivative(self):
        '''Derivative for gradient calculation / backpropagation'''
        pass


class Linear(AcFunc):
    def __init__(self):
        super().__init__()

    def activate(self, Z):
        return Z

    def derivative(self, Z):
        return 1


class ReLU(AcFunc):
    def __init__(self):
        super().__init__()

    def activate(self, Z):
        return np.maximum(0, Z)

    def derivative(self, Z):
        return (Z > 0).astype(float)


class LeakyReLU(AcFunc):
    def __init__(self, alpha=None):
        super().__init__()
        if not alpha:
            raise Exception("LeakyReLU is missing argument 'alpha'.")
        self.alpha = alpha

    def activate(self, Z):
        return np.where(Z > 0, Z, Z * self.alpha)

    def derivative(self, Z):
        return np.where(Z > 0, 1, self.alpha)


class Sigmoid(AcFunc):
    def __init__(self):
        super().__init__()

    def activate(self, Z):
        Z = np.clip(Z, -10, 10)
        pos = 1 / (1 + np.exp(-Z))
        neg = np.exp(Z) / (1 + np.exp(Z))
        Z = np.where(Z > 0, pos, neg)
        return Z
    
    def derivative(self, Z):
        S = self.activate(Z)
        return S * (1 - S)


class Softmax(AcFunc):
    def __init__(self):
        super().__init__()
        
    def activate(self, Z):
        # Numerical stability: subtract max value for each row
        e_Z = np.exp(Z - np.max(Z, axis=-1, keepdims=True))
        
        Z = e_Z / np.sum(e_Z, axis=-1, keepdims=True)
        return Z
    
    def derivative(self, Z):
        S = self.activate(Z)
        return S * (1 - S)
