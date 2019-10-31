import random

import numpy as np
import math


def sigmoid(value):
    return 1 / (1 + np.exp(-value))


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
        self.link_layers()

    def link_layers(self):
        layers_index_count = len(self.layers) - 1
        for i in range(layers_index_count + 1):
            if i != layers_index_count:
                matrix = np.random.rand(self.layers[i].size, self.layers[i + 1].size)
                self.layers[i].forward_weights = matrix
            if i != 0:
                self.layers[i].backward_weights = self.layers[i-1].forward_weights

    def predicted(self, inputs):
        return self.feedforward(inputs)[-1]

    def feedforward(self, inputs):
        layers = [inputs]
        for layer in self.layers:
            if layer.forward_weights is None:
                continue
            inputs = layer.forward(inputs)
            layers.append(inputs)
        return layers

    def train(self, inputs, outputs):
        self.backpropagate(inputs, outputs)


class Layer:
    forward_weights = None
    backward_weights = None

    def __init__(self, size, bias=0):
        self.size = size
        self.bias = bias

    def forward(self, inputs):
        return np.dot(inputs, self.forward_weights) + self.bias
        
    def backward(self, inputs):
        return np.dot(inputs, self.backward_weights.T)

    def __repr__(self):
        return f"<Layer size={self.size} bias={self.bias}>"


class Trainer:
    def __init__(self, mlp, dataset):
        self.mlp = mlp
        self.dataset = dataset

    def train_times(self, times):
        for i in range(times):
            for (inputs, outputs) in self.dataset:
                self.mlp.train(inputs, outputs)

    def analyze(self):
        print("[Analyzing]")
        errors = []
        for inpt, output in self.dataset:
            predicted = self.mlp.feedforward(inpt)
            print(f"expected: {output} got:{predicted}")
            errors.append(0 if output == predicted else 1)

        print(f"mlp weights: {self.mlp.weights}")
        print(f"Error: {np.mean(errors)}")
