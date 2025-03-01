'''All neural network layers are defined here.'''

from abc import ABC, abstractmethod
import numpy as np

from activations import *
from losses import *
from optimizers import *
from weightinit import *


class Layer(ABC):
    '''Abstract layout of a neural network layer.'''
    def __init__(self):
        self.math = None
        self.weights = None
        self.biases = None

    def cpu_or_gpu(self, gpu=False):
        '''Loads respective math library'''
        if not gpu:
            import numpy as np
            self.math = np
        else:
            import cupy as cp
            self.math = cp
    
    @abstractmethod
    def init_flow(self):
        ''' Initializes fitting weights and biases to allow
            forward propagation between layers.
        '''
        pass
    
    @abstractmethod
    def forward(self):
        ''' Computes the forward propagation in the current layer
            and returns the layer output.
        '''
        pass

    @abstractmethod
    def backprop(self):
        ''' Computes the backpropagation in the current layer
            and returns weight and gradient information.
        '''
        pass


class Preprocessing(Layer):
    ''' Abstract layout of a neural network layer used 
        for data processing.
    '''
    def __init__(self):
        super().__init__()
        pass


class Input(Layer):
    '''The model will discard this layer after extracting dimensions.'''
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def init_flow(self, prev_data):
        return self.dim
        
    def forward(self):
        pass

    def backprop(self):
        pass


class StandardScaler(Preprocessing):
    def __init__(self):
        super().__init__()
        self.dim = None
    
    def init_flow(self, prev_data):
        self.dim = prev_data
        if len(self.dim) != 1:
            raise Exception(f"""Failed to synchronize layers.
                StandardScaler only works on flattened input.""")
        return self.dim
        
    def forward(self, X):
        if (X[1].shape != self.dim):
            raise Exception(f"""Expected input shape{self.dim}, 
                but got {X.shape[1]}.""")
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        A = (X - mean) / (std + 1e-8)
        return A
    
    def backprop(self, dA, eta):
        return dA


class Flatten(Preprocessing):
    def __init__(self):
        super().__init__()
        self.dim = None
    
    def init_flow(self, prev_data):
        self.dim = prev_data
        next_dim = (np.prod(self.dim),)
        return next_dim
        
    def forward(self, X):
        if (X[1].shape != self.dim):
            raise Exception(f"""Expected input shape {self.dim},
                but got {X.shape[1]}.""")
        A = X.reshape(X.shape[0], -1)
        return A
    
    def backprop(self, dA, eta):
        return dA


class Dense(Layer):
    '''Fully connected layer'''
    def __init__(self, n_neurons, ac_func, l2=0):
        super().__init__()
        if not isinstance(ac_func, AcFunc):
            raise TypeError(f"""{type(ac_func).__name__} is not 
                a valid activation function.""")
        self.dim = (n_neurons,)
        self.ac_func = ac_func
        self.l2 = l2

    
    def init_flow(self, prev_data):
        # Amount of neurons
        n = self.dim[0]
        # Amount of inputs from previous layer
        m = prev_data[0]
        if isinstance(self.ac_func, Sigmoid) \
        or isinstance(self.ac_func, Softmax):
            self.weights = xavier_weight_init(m,n)
        elif isinstance(self.ac_func, ReLU) \
        or isinstance(self.ac_func, LeakyReLU) \
        or isinstance(self.ac_func, Linear):
            self.weights = he_weight_init(m,n)
        else:
            raise Exception(f"""Activation function 
                {type(self.ac_func).__name__} is not defined
                for layer {self.__class__.__name__}.""")
        self.biases = self.math.zeros(n)
        
        return self.dim

    
    def forward(self, X):
        if self.weights is None or self.biases is None:
            raise Exception(f"Weights haven't been initialized.")
        self.A_prev = X
        self.Z = X @ self.weights.T + self.biases
        self.A = self.ac_func.activate(self.Z)
        
        return self.A

    
    def backprop(self, dA, eta):
        dZ = dA * self.ac_func.derivative(self.Z)
        m = self.A_prev.shape[1]
        dW = self.math.dot(dZ.T, self.A_prev) / m
        dB = self.math.sum(dZ, axis=0) / m
        dA_prev = self.math.dot(dZ, self.weights)
        
        dW += (self.l2 / m) * self.weights

        return dA_prev, self.weights, self.biases, dW, dB

    def replace_weights(self, W, B):
        ''' Allows to set new weights according to
            the optimizer's calculations.
        '''
        self.weights = W
        self.biases = B


# Yet to be implemented
class Conv2D(Layer):
    def __init__(self):
        super().__init__()

    def init_flow(self):
        pass
        
    def forward(self, X):
        return X

    def backprop(self, dA, eta):
        return dA
        