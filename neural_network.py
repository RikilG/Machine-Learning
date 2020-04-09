#!/bin/env python

import numpy as np
import pandas as pd


class InputLayer:

    def __init__(self, units):
        self.size = units
        self.outputs = np.zeros(units)
    
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
    
    def forward_prop(self):
        self.outputs = (self.prev_layer.outputs @ self.weights) + self.bias # y = w*x + b
        self.outputs = self.activation(self.outputs)


class OutputLayer:

    def __init__(self, units, prev_layer, activation=None):
        self.weights = np.random.rand(prev_layer.size, units)
        self.size = units
        self.bias = np.random.rand(units)
        self.prev_layer = prev_layer

    def forward_prop(self):
        self.outputs = (self.prev_layer.outputs @ self.weights) + self.bias # y = w*x + b


class NeuralNet:

    def __init__(self):
        self.layers = []
        self.input_layer_attached = False
        self.output_layer_attached = False
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def identity(self, z):
        return z
    
    def attach_input(self, units):
        assert not self.input_layer_attached, "Input layer already attached"
        self.layers.append(InputLayer(units=units))
        self.input_layer_attached = True
    
    def attach_hidden(self, units=5, count=1, activation="sigmoid"):
        assert self.input_layer_attached, "No input layer in neural net"
        if activation == "sigmoid":
            activation = self.sigmoid
        elif activation == "identity":
            activation = self.identity
        else:
            assert False, "No such activation function exists"
        for i in range(len(self.layers)-1, len(self.layers)+count-1):
            self.layers.append(HiddenLayer(units=units, prev_layer=self.layers[i], activation=activation))
    
    def attach_output(self, units, activation="identity"):
        assert self.input_layer_attached, "No input layer in neural net"
        assert not self.output_layer_attached, "Output layer already attached"
        self.layers.append(OutputLayer(units=units, prev_layer=self.layers[-1]))
        self.output_layer_attached = True
    
    def feed_forward(self, input):
        assert self.input_layer_attached, "No input layer in neural net"
        assert self.output_layer_attached, "No output layer in neural net"
        self.layers[0].forward_prop(input)
        for i in range(1, len(self.layers)):
            self.layers[i].forward_prop()
        return self.layers[-1].outputs


def main():
    dataset = pd.read_csv('datasets/housepricedata.csv')
    nn = NeuralNet()
    nn.attach_input(units=2)
    nn.attach_hidden(units=3, count=1, activation="identity")
    nn.attach_output(units=1)
    print(nn.feed_forward(np.array([1,2])))

if __name__ == "__main__":
    main()