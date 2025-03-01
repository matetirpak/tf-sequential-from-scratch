'''This module contains the Sequential Neural Network.'''

import numpy as np
import time
import h5py

from activations import *
from layers import *
from losses import *
from optimizers import *
from data_utils import *
from weightinit import *


class SeqNN:
    '''Sequential Neural Network'''
    
    def __init__(self, layers, loss_func, eta, optimizer=None):
        self.check_for_gpu()
        self.set_layers(layers)
        self.set_loss_func(loss_func)
        self.eta = eta
        self.set_optimizer(optimizer, eta)
        self.layer_sync()
    
    
    def check_for_gpu(self):
        '''Activates gpu acceleration if possible.'''
        self.gpu = False
        import numpy as np
        self.math = np
        
        self.gpu_status = 1
        self.gpu_exception = None
        
        try:
            import cupy as cp
        except ImportError:
            self.gpu_status = 2
            return
    
        try:
            gpu_count = cp.cuda.runtime.getDeviceCount()
            if gpu_count == 0:
                self.gpu_status = 3
                return
        except cp.cuda.runtime.CUDARuntimeError as e:
            self.gpu_status = 4
            self.gpu_exception = e
            return
        
        self.math = cp
        self.gpu = True
        self.gpu_status = 0

    def verbose_gpu(self):
        '''Prints the gpu status.'''
        if self.gpu:
            print("GPU acceleration is active.")
        else:
            print("GPU acceleration is not active.")

        if self.gpu_status == 1:
            print("Failed to enable GPU acceleration for unknown reason.")
        elif self.gpu_status == 2:
            print("CuPy is not installed. Install it using `pip install cupy`.")
        elif self.gpu_status == 3:
            print("CuPy is installed, but no compatible GPU is available.")
        elif self.gpu_status == 4:
            print("""CuPy is installed, but no compatible GPU is available or 
                  CUDA runtime is not properly configured.""")

        if self.gpu_exception:
            print("Error details:", self.gpu_exception)

        
    def set_layers(self, layers):
        '''Called upon initialization. Layers are saved if not flawed.'''
        if not isinstance(layers[0], Input):
            raise TypeError(f"""Expected the first layer to be of type Input,
                but got {type(layers[0]).__name__}.""")
            
        if len(layers) < 2:
            raise Exception(f"Model can't consist of a single input layer")
            
        for i, layer in enumerate(layers):
            if not isinstance(layer, Layer):
                raise TypeError(f"""Layer validation failed at index {i}.
                    Expected an instance of Layer, but
                    got {type(layer).__name__}.""")
        
        self.layers = layers

    
    def set_loss_func(self, loss_func):
        '''Called upon initialization. Loss function gets saved if not flawed.'''
        if not isinstance(loss_func, Loss):
                raise TypeError(f"""Expected loss to be of type Loss, 
                    but got {type(loss).__name__}.""")
        
        self.loss_func = loss_func

    
    def set_optimizer(self, optimizer, eta):
        ''' Called upon initialization. Optimizer gets saved if not flawed.
            If none was specified, the default optimizer is being used.
        '''
        if not optimizer:
            self.optimizer = DefaultOptimizer(eta)
            return
            
        if not isinstance(optimizer, Optimizer):
                raise TypeError(f"""Expected loss to be of type Optimizer, 
                    but got {type(optimizer).__name__}.""")
        if isinstance(optimizer, Adam):
            optimizer.set_eta(eta)
        self.optimizer = optimizer
        self.optimizer.cpu_or_gpu(gpu=self.gpu)

    
    def layer_sync(self):
        ''' Initializes flow between layers and separates a preprocessing-head
            if possible.
        '''
        # Weight matrices: NxM, N: current num neurons, M: previous num neurons
        prev_sync = None
        for layer in self.layers:
            layer.cpu_or_gpu(gpu=self.gpu)
            prev_sync = layer.init_flow(prev_sync)
            
        # Delete the unnecessary input layer
        self.layers = self.layers[1:]
        
        # Preprocessing head
        self.preprocessing = None
        for i, layer in enumerate(self.layers):
            if not isinstance(layer, Preprocessing):
                break
        if i > 0:
            self.preprocessing = self.layers[:i]
            self.layers = self.layers[i:]

    
    def loss(self, y_true, y_pred):
        '''Computes the loss with the saved loss function.'''
        loss = self.loss_func.calculate_loss(y_pred, y_true)
        return loss
    
    
    def l2_loss_penalty(self, batch_size):
        '''Collects L2 losses across all layers.'''
        loss = 0
        for i, layer in enumerate(self.layers):
            if not isinstance(layer, Dense):
                continue
            if layer.l2 == 0:
                continue
            loss += (2*layer.l2 / batch_size) * self.math.sum(
                        self.math.square(layer.weights))
        return loss

    
    def forward_prop(self, X, preprocessing=False):
        ''' Performs forward propagation across all layers.
            When using the model for predictions, preprocessing
            should be set to True.
        '''
        A = X
        it_obj = self.layers
        if preprocessing:
            it_obj = self.preprocessing
        for i, layer in enumerate(it_obj):
            A = layer.forward(A)
        return A
    
    def back_prop(self, pred, y):
        ''' Performs backpropagation across all layers and returns
            all weights and their gradients.
        '''
        w_params = list()
        b_params = list()
        w_grads = list()
        b_grads = list()
        
        dA = self.loss_func.derivative(pred, y)
        n_layers = len(self.layers)
        for i in range(n_layers-1, -1, -1):
            dA, W, B, dW, dB = self.layers[i].backprop(dA, self.eta)
            w_params.insert(0, W)
            b_params.insert(0, B)
            w_grads.insert(0, dW)
            b_grads.insert(0, dB)

        return w_params, b_params, w_grads, b_grads

    def update_weights(self, w_params, b_params):
        '''Applies weight updates to every layer.'''
        for i, layer in enumerate(self.layers):
            layer.replace_weights(w_params[i], b_params[i])
    
    def create_batches(self, X, y, batch_size):
        '''Creates batches of specified size from given data.'''
        n_batches = len(X) // batch_size + (len(X) % batch_size > 0)
        X_batches = [X[i * batch_size: (i + 1) * batch_size] 
                     for i in range(n_batches)]
        y_batches = [y[i * batch_size: (i + 1) * batch_size] 
                     for i in range(n_batches)]
        return X_batches, y_batches

    
    def fit(self, X, y, batch_size=32, epochs=10, valid=None, metrics=[]):
        history = {"loss": []}
        
        X_orig = X.copy()

        # If possible, preprocess the training data
        if self.preprocessing:
            X = self.forward_prop(X, preprocessing=True)
        
        # Extract the validation set
        if valid:
            X_valid, y_valid = valid
                
        # Create Batches
        X_batches, y_batches = self.create_batches(X, y, batch_size)
        n_batches = len(X_batches)    
        
        eval_accuracy = False
        if "accuracy" in metrics:
            eval_accuracy = True
            history["acc"] = []
            if valid:
                history["val_acc"] = []
            
        for e in range(epochs):
            start = time.time()
            epoch_loss = 0
            
            for b in range(n_batches):
                X_batch = X_batches[b]
                y_batch = y_batches[b]
                
                pred = self.forward_prop(X_batch)

                w_params, b_params, w_grads, b_grads = self.back_prop(
                    pred, y_batch)
                w_params, b_params = self.optimizer.update(
                    w_params, b_params, w_grads, b_grads)
                self.update_weights(w_params, b_params)
                
                
                loss = self.loss_func.compute(pred, y_batch)
                l2_penalty = self.l2_loss_penalty(len(X_batch))
                loss += l2_penalty
                epoch_loss += loss
            epoch_loss /= n_batches
            history["loss"].append(epoch_loss)
            
            if eval_accuracy:
                y_pred = self.predict(X_orig)
                train_acc = self.accuracy(y_pred, y)
                history["acc"].append(train_acc)
            
            if valid:
                if eval_accuracy:
                    y_pred_valid = self.predict(X_valid)
                    valid_acc = self.accuracy(y_pred_valid, y_valid)
                    history["val_acc"].append(valid_acc)
            
            end = time.time()
            print(f"Epoch {e+1}/{epochs} - loss: {epoch_loss:.4f}", end="")

            if eval_accuracy:
                print(f" - accuracy: {train_acc:.4f}", end="")
            
            if valid:
                if eval_accuracy:
                    print(f" - validation accuracy: {valid_acc:.4f}", end="")
            
            elapsed_time = end - start
            print(f" - time elapsed: {elapsed_time:.4f} seconds")
        return history


    def predict(self, X):
        '''Returns predictions on given data.'''
        if self.preprocessing:
            X = self.forward_prop(X, preprocessing=True)
        pred = self.forward_prop(X)
        return pred

    def accuracy(self, y_pred, y_true):
        '''Computes the accuracy over one-hot encoded predictions.'''
        n = len(y_true)
        correct = 0
        for i in range(n):
            if self.math.argmax(y_pred[i]) == self.math.argmax(y_true[i]):
                correct += 1
        return correct / n

    # Following functions are used for saving the model.
    def get_all_weights(self):
        '''Extracts all weights of the model without processing.'''
        W_list = list()
        B_list = list()
        for layer in self.layers:
            if isinstance(layer, Preprocessing):
                continue
            W = layer.weights
            B = layer.biases
            W_list.append(W)
            B_list.append(B)
        return W_list, B_list

    def make_weights_homogenious(self, W_list, B_list):
        '''Processes extracted weights to save them in a single numpy array.'''
        # W_list and B_list are of equal length
        # Determine maximum dimensions
        wx, wy, wz = len(W_list), 0, 0
        bx, by = len(B_list), 0
        for i_layer in range(wx):
            W = W_list[i_layer]
            B = B_list[i_layer]
            if not isinstance(W, np.ndarray):
                raise Exception(f"""Expected weights to be of type 
                    numpy.ndarray, but got {type(W)}.""")
            if not isinstance(B, np.ndarray):
                raise Exception(f"""Expected biases to be of type
                    numpy.ndarray, but got {type(B)}.""")
            wy = max(wy, W.shape[0])
            wz = max(wz, W.shape[1])
            by = max(by, B.shape[0])

        # Collect with padding
        weights = np.zeros(shape=(wx,wy,wz))
        biases = np.zeros(shape=(bx, by))
        for i_layer in range(wx):
            W = W_list[i_layer]
            B = B_list[i_layer]
            n, m = W.shape
            weights[i_layer, :n, :m] = W
            n = B.shape[0]
            biases[i_layer, :n] = B
            
        return weights, biases

    
    def save_weights(self, file='weights.hdf5'):
        '''Saves weights in a specified file.'''
        weights, biases = self.get_all_weights()
        weights, biases = self.make_weights_homogenious(weights, biases)
        with h5py.File(file, 'w') as f:
            df_weights = f.create_dataset('weights', data=weights, 
                                          compression='gzip')
            df_biases = f.create_dataset('biases', data=biases, 
                                          compression='gzip')

    
    def load_weights(self, file='weights.hdf5'):
        '''Loads weights from a specified file.'''
        with h5py.File(file, 'r') as f:
            weights = np.array(f['weights'])
            biases = np.array(f['biases'])
        self.apply_loaded_weights(weights, biases)


    def apply_loaded_weights(self, weights, biases):
        '''Helper function for 'load_weights'.'''
        i = 0
        for layer in self.layers:
            if isinstance(layer, Preprocessing):
                continue
            n, m = layer.weights.shape
            W = weights[i, :n, :m]
            n = layer.biases.shape[0]
            B = biases[i, :n]
            layer.replace_weights(W, B)
            i += 1
            