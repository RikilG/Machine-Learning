#!/bin/env python

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


class InputLayer:

    def __init__(self, units):
        self.size = units
        self.outputs = np.zeros(units)
        self.next_layer = None
    
    def _setNextLayer(self, next_layer):
        self.next_layer = next_layer
    
    def forward_prop(self, inputs):
        self.outputs = inputs


class HiddenLayer:

    def __init__(self, units, prev_layer, weights_init, activation, dropout=0):
        self.weights = weights_init(size=(prev_layer.size, units))
        self.outputs = np.zeros(units)
        self.bias    = weights_init(size=units)
        self.activation = activation
        self.size    = units
        self.prev_layer = prev_layer
        self.next_layer = None
    
    def _setNextLayer(self, next_layer):
        self.next_layer = next_layer
    
    def forward_prop(self):
        self.raw_outputs = (self.prev_layer.outputs @ self.weights) + self.bias # y = w*x + b
        self.outputs = self.activation(self.raw_outputs)
    
    def backward_prop(self, alpha):
        # Yo, using Dynamic Programming (DP) to get stored error from next layer
        self.error = self.activation(self.raw_outputs, derivative=True) * (self.next_layer.weights @ self.next_layer.error)
        grad_E = self.prev_layer.outputs[:,None] * self.error
        self.weights = self.weights - alpha*grad_E
        # same for biases: a unit which always throws 1 as output but has edge weight bi
        self.bias = self.bias - alpha*(self.error)


class OutputLayer:

    def __init__(self, units, prev_layer, weights_init, activation):
        self.weights = weights_init(size=(prev_layer.size, units))
        self.size = units
        self.bias = weights_init(size=units)
        self.prev_layer = prev_layer
        self.activation = activation

    def forward_prop(self):
        self.raw_outputs = (self.prev_layer.outputs @ self.weights) + self.bias # y = w*x + b
        self.outputs = self.activation(self.raw_outputs)
    
    def backward_prop(self, targets, alpha):
        self.error = self.activation(self.raw_outputs, derivative=True)*(self.outputs - targets)
        grad_E = self.prev_layer.outputs[:, None] * self.error
        self.weights = self.weights - alpha*grad_E
        # same for biases
        self.bias = self.bias - alpha*(self.error)


class NeuralNet:

    def __init__(self, learning_rate=0.1, weights_init="normal", random_seed=None):
        self.layers = []
        self.input_layer_attached = False
        self.output_layer_attached = False
        self.alpha = learning_rate
        if weights_init == "normal" or weights_init == "gaussian":
            self.weights_init = np.random.normal
        elif weights_init == "uniform":
            self.weights_init = np.random.uniform
        elif weights_init == "random":
            self.weights_init == np.random.random
        else: assert False, "unknown weight initilization type"
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def sigmoid(self, z, derivative=False):
        if derivative:
            temp = self.sigmoid(z)
            return temp*(1-temp)
        return 1 / (1 + np.exp(-z))
    
    def tanh(self, z, derivative=False):
        if derivative:
            return 1 - np.square(np.tanh(z))
        return np.tanh(z)
    
    def relu(self, z, derivative=False):
        if derivative:
            return 1*(z>0) # 1 is multiplied to convert bool to int
        return z*(z>0) # same as np.maximum(0, z) but slightly faster
    
    def identity(self, z, derivative=False):
        if derivative:
            return 1
        return z

    def _get_activation_func(self, activation):
        if activation == "sigmoid":
            return self.sigmoid
        elif activation == "identity":
            return self.identity
        elif activation == "tanh":
            return self.tanh
        elif activation == "relu":
            return self.relu
        else:
            assert False, "No such activation function exists"
    
    def attach_input(self, units):
        assert not self.input_layer_attached, "Input layer already attached"
        self.layers.append(InputLayer(units=units))
        self.layers[0]._activation = "none"
        self.input_layer_attached = True
    
    def attach_hidden(self, units=5, layers=1, activation="sigmoid"):
        assert self.input_layer_attached, "No input layer in neural net"
        _activation = activation
        activation = self._get_activation_func(activation)
        for i in range(len(self.layers)-1, len(self.layers)+layers-1):
            self.layers.append(HiddenLayer(units=units, prev_layer=self.layers[i], weights_init=self.weights_init, activation=activation))
            self.layers[i+1]._activation = _activation # just the name of activation
        for i in range(len(self.layers)-1):
            self.layers[i]._setNextLayer(self.layers[i+1])
        
    def attach_output(self, units, activation="identity"):
        assert self.input_layer_attached, "No input layer in neural net"
        assert not self.output_layer_attached, "Output layer already attached"
        _activation = activation
        activation = self._get_activation_func(activation)
        self.layers.append(OutputLayer(units=units, prev_layer=self.layers[-1], weights_init=self.weights_init, activation=activation))
        self.layers[-2]._setNextLayer(self.layers[-1])
        self.layers[-1]._activation = _activation
        self.output_layer_attached = True
    
    def feed_forward(self, input):
        assert self.input_layer_attached, "No input layer in neural net"
        assert self.output_layer_attached, "No output layer in neural net"
        self.layers[0].forward_prop(input)
        for i in range(1, len(self.layers)):
            self.layers[i].forward_prop()
        return self.layers[-1].outputs
    
    def back_prop(self, targets):
        self.layers[-1].backward_prop(targets, self.alpha)
        for i in range(len(self.layers)-2, 0, -1):
            self.layers[i].backward_prop(self.alpha)
    
    def train(self, x, y, epochs=5, plot_every=None):
        tx = np.array(x)
        ty = np.array(y)
        if type(y[0]) in [np.int, np.int64, np.float, np.float64]:
            ty = ty[:, None]
        assert len(tx) == len(ty), f"input output size mismatch {len(tx)} != {len(ty)}"
        assert len(tx[0]) == self.layers[0].size, "Input size and input layer size mismatch"
        assert len(ty[0]) == self.layers[-1].size, "Output size and output layer size mismatch"
        self.plt_iters = []
        self.plt_errors = []
        print(f"Training epochs: {epochs}")
        for i in tqdm(range(epochs)):
            for r in range(len(tx)):
                self.feed_forward(tx[r])
                self.back_prop(ty[r])
            if plot_every is not None and i % plot_every == 0:
                self.plt_iters.append(i)
                self.plt_errors.append(self.test(x, y, error_measure=True))
        if plot_every is not None:
            plt.plot(self.plt_iters[int(0.2*epochs/plot_every):], self.plt_errors[int(0.2*epochs/plot_every):])
            plt.show()
    
    def classify(self, z, threshold=0.5):
        if z[0]>=threshold: return [1]
        else: return [0]
    
    def test(self, x, y, threshold=0.5, error_measure=False):
        x = np.array(x)
        y = np.array(y)
        if type(y[0]) in [np.int, np.int64, np.float, np.float64]:
            y = y[:, None]
        predicted = np.zeros(y.shape)
        for i in range(len(x)):
            predicted[i] = self.classify(self.feed_forward(x[i]), threshold=threshold)
        misclassfied = (predicted != y).sum()
        mse =  np.square(predicted - y).sum()/len(y)
        acc = (predicted == y).sum()/len(y)
        if not error_measure:
            TP = np.logical_and(predicted == y, y == 1).sum()
            TN = np.logical_and(predicted == y, y == 0).sum()
            FP = np.logical_and(predicted != y, y == 0).sum()
            FN = np.logical_and(predicted != y, y == 1).sum()
            precision = TP/(TP+FP)
            recall = TP/(TP+FN)
            f_score = (2*precision*recall)/(precision+recall)
            print(f"Stats: MSE: {round(mse,4)}, Mis-classified: {misclassfied}")
            print(f"Precision: {round(precision, 4)}, Recall: {round(recall, 4)}")
            print(f"Accuracy: {round(acc,4)}, F-Score: {round(f_score, 4)}")
        return mse

    def print_info(self):
        print(f"Neural net params:")
        print(f"  Alpha: {self.alpha}")
        for i in range(len(self.layers)):
            print(f"  Layer: {i}, Units: {self.layers[i].size}, Activation: {self.layers[i]._activation}")
        
        