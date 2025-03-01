'''All losses are defined here.'''

from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    """ Abstract layout of the loss functions.
        Args:
            y_pred (numpy array): Predicted probabilities, 
                                  shape: (n_samples, n_classes)
            y_true (numpy array): True labels, 
                                  shape: (n_samples, n_classes)
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def compute(self, y_pred, y_true):
        '''Loss computation'''
        pass

    @abstractmethod
    def derivative(self, y_pred, y_true):
        '''Derivative for gradient calculation / backpropagation'''
        pass


class BinaryCrossEntropy(Loss):
    ''' Used for multilabel classification in combination with
        sigmoid activation.
    '''
    def __init__(self):
        super().__init__() 
    
    def compute(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
        return loss.mean()

    def derivative(self, y_pred, y_true):
        return y_pred - y_true


class CategoricalCrossEntropy(Loss):
    ''' Used for multiclass classification in combination with
        softmax activation.
    '''
    def __init__(self):
        super().__init__() 
        
    def compute(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(y_true * np.log(y_pred), axis=-1)
        return loss.mean()

    def derivative(self, y_pred, y_true):
        return y_pred - y_true


class MeanAbsoluteError(Loss):
    '''Loss = sum(abs(pred - true))'''
    def __init__(self):
        super().__init__() 
    
    def compute(self, y_pred, y_true):
        return np.sum(np.abs(y_pred - y_true))

    def derivative(self, y_pred, y_true):
        return 2 * (y_pred - y_true)


class MeanSquaredError(Loss):
    '''Loss = sum((pred - true)^2)'''
    def __init__(self):
        super().__init__() 
    
    def compute(self, y_pred, y_true):
        return np.sum(np.square(y_pred - y_true))

    def derivative(self, y_pred, y_true):
        return 2 * (y_pred - y_true)
