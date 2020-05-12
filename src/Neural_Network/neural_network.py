#!/bin/env python

import numpy as np
import pandas as pd
from nn_core import NeuralNet


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
    dataset = pd.read_csv('./../../datasets/housepricedata.csv')
    # Standadization
    # dataset = (dataset-np.mean(dataset))/np.std(dataset)
    # min-max scaling / Normalization
    dataset = (dataset - np.min(dataset))/(np.max(dataset) - np.min(dataset))
    x_train, x_test, y_train, y_test = train_test_split(dataset)

    # allowed weights init: random, normal/gaussian, uniform
    nn = NeuralNet(learning_rate=0.1, weights_init="normal", random_seed=42)
    nn.attach_input(units=10)
    nn.attach_hidden(units=12, layers=1, activation="sigmoid")
    nn.attach_hidden(units=2, layers=1, activation="sigmoid")
    nn.attach_output(units=1, activation="sigmoid")
    nn.print_info()
    nn.train(x_train, y_train, epochs=1800, plot_every=50)
    nn.test(x_test, y_test, threshold=0.5)
    # for i in range(1,len(nn.layers)):
    #     print(nn.layers[i].weights)


# def xor_gate():
#     dataset = pd.read_csv('datasets/xorgate.csv')
#     print(dataset)
#     y = dataset[dataset.columns[-1]]
#     x = dataset.drop(dataset.columns[-1], axis=1)
#     nn = NeuralNet()
#     nn.attach_input(units=2)
#     nn.attach_hidden(units=2)
#     nn.attach_output(units=1)
#     nn.train(x, y, iters=900)
#     for index, row in x.iterrows():
#         temp = np.array(row)
#         print(temp, nn.feed_forward(temp))


if __name__ == "__main__":
    main()
    # xor_gate()

"""
The ones who accomplish something are the fools who keep pressing forward,
The ones who accomplish nothing are the wise who cease advancing.

10 12 12 1 / 5000 / 0.1 - 29/1168 m
10 12 1 / 5000 / 0.1 - 30 m - sigmoid(all)
10 12 1 / 5000 / 0.1 - 33 m - tanh(all)
10 12 1 / 5000 / 0.1 - 32 m - relu(all)
10 12 2 1 / 1800 / 0.1 - 29 m - sigmoid(all)
10 12 2 1 / 4500 / 0.03 - 28 m - sigmoid(all)
"""
