from functools import reduce

import numpy as np

from neuralnetwork.trainer import Trainable
from neuralnetwork import activation_funcs

import random


def square(val):
    return val ** 2


v_square = np.vectorize(square)


class Multilayer(Trainable):
    learning_rate = 0.01

    def __init__(self, inputs_size, hidden_size, output_size):
        self.inputs = Layer(inputs_size, bias=0.5)
        self.hidden = Layer(hidden_size, bias=0.5)
        self.output = Layer(output_size)

        self.layers = [self.inputs, self.hidden, self.output]

        self.create_weights()
        self.link_layers()

    def create_weights(self):
        for i in range(len(self.layers) - 1):
            matrix = np.random.rand(self.layers[i].size, self.layers[i + 1].size)
            self.layers[i].forward_weights = matrix

    def link_layers(self):
        for i in range(len(self.layers) - 1):
            self.layers[i+1].backward_weights = self.layers[i].forward_weights
            self.layers[i+1].previous_layer = self.layers[i]

    def predict(self, inputs):
        return self.feedforward(inputs)[-1]

    def feedforward(self, inputs):
        layers = [inputs]
        for layer in self.layers:
            if layer.forward_weights is None:
                continue
            inputs = layer.forward(inputs)
            layers.append(inputs)
        return layers

    def train(self, inputs, target_outputs):
        backward_input = target_outputs
        elegible_layers = [
            l for l in self.layers[::-1] if l.backward_weights is not None
        ]

        predicted_outputs = self.feedforward(inputs)

        for layer in elegible_layers:
            output = predicted_outputs.pop()
            print(f"output: {output}")
            errors = v_square(backward_input - output)
            print(f"errors: {errors}")

            import pdb;pdb.set_trace()

            for weights, error in zip(layer.backward_weights.T, errors.T):

                weight_sum = reduce(lambda x, y: x + y, weights)
                pondered_error = error / weight_sum * self.learning_rate
                for w in layer.backward_weights:
                    layer.backward_weights += (
                    w * pondered_error * predicted_outputs[-1]
                )
                layer.bias += layer.bias * pondered_error

            backward_input = layer.backward(error)

    def explain(self):
        print(f"MLP layers")
        for l in self.layers:
            print(l)
            print(l.forward_weights)
            print("---")

class Layer:
    forward_weights = None
    backward_weights = None
    activate_func = v_gate

    def __init__(self, size, bias=0):
        self.size = size
        self.bias = bias

    def forward(self, inputs):
        return self.activate_func(np.dot(inputs, self.forward_weights) + self.bias)

    def backward(self, inputs):
        return np.dot(inputs, self.backward_weights.T)

    def __repr__(self):
        return f"<Layer size={self.size} bias={self.bias}>"
