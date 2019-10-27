import random

import numpy as np
import math


def sigmoid(value):
    return 1 / (1 + math.exp(-value))


def d_sigmoid(value, sigmoided=False):
    sig = value if sigmoided else sigmoid(value)
    return sig * (1 - sig)


class Multilayer:
    learning_rate = 0.1

    def __init__(self, inputs_size, hidden_size, output_size):
        self.inputs = Layer(inputs_size, bias=random.random())
        self.hidden = Layer(hidden_size, bias=random.random())
        self.output = Layer(output_size)

        self.layers = [self.inputs, self.hidden, self.output]

        self.weights = []

        for i in range(len(self.layers) - 1):
            matrix = np.random.rand(self.layers[i].size, self.layers[i + 1].size)
            self.weights.append(matrix)

    def feedforward(self, inputs):
        for i in range(len(self.layers) - 1):
            layer = self.layers[i]
            inputs = [sigmoid(x + layer.bias) for x in self.feed_layer(i, inputs)]
        return inputs

    def feed_layer(self, layer_index, inputs):
        return np.dot(inputs, self.weights[layer_index])

    def backpropagate(self, err):
        for i in range(len(self.weights) - 1, 0, -1):
            weight = self.weights[i]
            err = np.dot(weight, err) * self.learning_rate
            self.weights[i] = np.subtract(weight, err)
        return err


class Layer:
    def __init__(self, size, bias=0):
        self.size = size
        self.bias = bias

    def __repr__(self):
        return f"<Layer size={self.size} bias={self.bias}>"


class Trainer:
    def __init__(self, mlp, dataset):
        self.mlp = mlp
        self.dataset = dataset

    def train_times(self, times, output_as_array=True):
        for i in range(times):
            for (inputs, outputs) in self.dataset:
                self.mlp.train(inputs, outputs, output_as_array)

