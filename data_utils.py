'''Data processing utils.'''

import numpy as np


def one_hot(y, num_labels: int):
    ''' Applies one hot encoding.
        Input: y, shape: (n)
        Output: ret, shape: (n,num_labels)
    '''
    n = len(y)
    ret = np.zeros((n,num_labels))
    for i in range(n):
        ret[i][y[i]] = 1
    return ret


def decode_one_hot(Y):
    ''' Reverts one hot encoding.
        Input: Y, shape: (n,num_labels)
        Output: ret, shape: (n)
    '''
    n = len(Y)
    ret = np.zeros(n)
    for i in range(n):
        ret[i] = np.argmax(Y[i])
    return ret