'''Weight initializations for neural network layers.'''

import numpy as np


def he_weight_init(n_input,m_output):
    std_dev = np.sqrt(2 / n_input)
    weights = np.random.normal(0, std_dev, size=(m_output, n_input))
    
    return weights


def xavier_weight_init(n_input,m_output):
    lower_bound = -1 / np.sqrt(n_input)
    upper_bound = 1 / np.sqrt(n_input)
    
    weights = np.random.uniform(lower_bound, upper_bound, 
                                size=(m_output, n_input))
    
    return weights