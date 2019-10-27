import random

import numpy as np


class Multilayer:
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
            inputs = [x + layer.bias for x in self.feed_layer(i, inputs)]
        return inputs

    def feed_layer(self, layer_index, inputs):
        return np.dot(inputs, self.weights[layer_index])


class Layer:
    def __init__(self, size, bias=0):
        self.size = size
        self.bias = bias

    def __repr__(self):
        return f"<Layer size={self.size} bias={self.bias}>"
