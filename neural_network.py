#!/bin/env python

import numpy as np
import pandas as pd


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

    def __init__(self, units, prev_layer, activation, dropout=0):
        self.weights = np.random.rand(prev_layer.size, units)
        self.outputs = np.zeros(units)
        self.bias    = np.random.rand(units)
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

    def __init__(self, units, prev_layer, activation=None):
        self.weights = np.random.rand(prev_layer.size, units)
        self.size = units
        self.bias = np.random.rand(units)
        self.prev_layer = prev_layer
        self.activation = activation

    def forward_prop(self):
        self.raw_outputs = (self.prev_layer.outputs @ self.weights) + self.bias # y = w*x + b
        if self.activation is not None:
            self.outputs = self.activation(self.raw_outputs)
        else:
            self.outputs = self.raw_outputs
    
    def backward_prop(self, targets, alpha):
        self.error = self.outputs - targets
        # can add output sigmoid derivative to error if present. although not in bishop
        grad_E = self.prev_layer.outputs[:, None] * self.error
        self.weights = self.weights - alpha*grad_E
        # same for biases
        self.bias = self.bias - alpha*(self.error)


class NeuralNet:

    def __init__(self, learning_rate=0.1):
        self.layers = []
        self.input_layer_attached = False
        self.output_layer_attached = False
        self.alpha = learning_rate
    
    def sigmoid(self, z, derivative=False):
        if derivative:
            temp = self.sigmoid(z)
            return temp*(1-temp)
        return 1 / (1 + np.exp(-z))
    
    def identity(self, z, derivative=False):
        if derivative:
            return 1
        return z

    def _get_activation_func(self, activation):
        if activation == "sigmoid":
            return self.sigmoid
        elif activation == "identity":
            return self.identity
        else:
            assert False, "No such activation function exists"
    
    def attach_input(self, units):
        assert not self.input_layer_attached, "Input layer already attached"
        self.layers.append(InputLayer(units=units))
        self.input_layer_attached = True
    
    def attach_hidden(self, units=5, layers=1, activation="sigmoid"):
        assert self.input_layer_attached, "No input layer in neural net"
        activation = self._get_activation_func(activation)
        for i in range(len(self.layers)-1, len(self.layers)+layers-1):
            self.layers.append(HiddenLayer(units=units, prev_layer=self.layers[i], activation=activation))
        for i in range(len(self.layers)-1):
            self.layers[i]._setNextLayer(self.layers[i+1])
        
    def attach_output(self, units, activation="identity"):
        assert self.input_layer_attached, "No input layer in neural net"
        assert not self.output_layer_attached, "Output layer already attached"
        activation = self._get_activation_func(activation)
        self.layers.append(OutputLayer(units=units, prev_layer=self.layers[-1], activation=activation))
        self.layers[-2]._setNextLayer(self.layers[-1])
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
    
    def train(self, x, y, iters=5):
        tx = np.array(x)
        ty = np.array(y)
        if type(y[0]) in [np.int, np.int64]:
            ty = ty[:, None]
        assert len(tx) == len(ty), f"input output size mismatch {len(tx)} != {len(ty)}"
        assert len(tx[0]) == self.layers[0].size, "Input size and input layer size mismatch"
        assert len(ty[0]) == self.layers[-1].size, "Output size and output layer size mismatch"
        for i in range(iters):
            for r in range(len(tx)):
                self.feed_forward(tx[r])
                self.back_prop(ty[r])


def train_test_split(dataset, split=0.8):
    break_at = int(split*dataset.shape[0])
    y = dataset['AboveMedianPrice']
    x = dataset.drop('AboveMedianPrice', axis=1)
    x_train = x.iloc[:break_at, :]
    x_test = x.iloc[break_at:, :]
    y_train = y[:break_at]
    y_test = y[break_at:]
    return x_train, x_test, y_train, y_test


def main():
    np.random.seed(42)

    dataset = pd.read_csv('datasets/housepricedata.csv')
    x_train, x_test, y_train, y_test = train_test_split(dataset)

    nn = NeuralNet(learning_rate=0.001)
    nn.attach_input(units=10)
    nn.attach_hidden(units=3, layers=1, activation="sigmoid")
    nn.attach_output(units=1, activation="identity")
    nn.train(x_train, y_train, 20)
    print(nn.feed_forward(np.array(x_train.iloc[2])), y_train[2])


if __name__ == "__main__":
    main()