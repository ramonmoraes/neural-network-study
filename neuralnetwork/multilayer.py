from functools import reduce

import numpy as np

import random
import math


def gate(val):
    return 1 if val > 0.5 else 0


v_gate = np.vectorize(gate)


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

    def train(self, inputs, target_outputs):
        elegible_layers = [
            l for l in self.layers[::-1] if l.backward_weights is not None
        ]

        predicted_outputs = self.feedforward(inputs)
        error = target_outputs - predicted_outputs[-1]

        for layer in elegible_layers:
            output = predicted_outputs.pop()

            import pdb; pdb.set_trace()
            for weights in layer.backward_weights.T:
                
                weight_sum = reduce(lambda x, y: x + y, weights)
                pondered_error = error / weight_sum * self.learning_rate
                
                layer.backward_weights += layer.backward_weights * pondered_error * output
                layer.bias += layer.bias * pondered_error
                

            error = layer.backward(error)


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
            predicted = self.mlp.predicted(inpt)
            print(f"expected: {output} got:{predicted}")
            errors.append(0 if output == predicted else 1)

        print(f'errors {np.mean(errors)}')
        print(f"mlp layers")
        for x in self.mlp.layers:
            print(x)
            print(x.forward_weights)
            print("---")
