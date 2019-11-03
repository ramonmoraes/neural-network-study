import random

from neuralnetwork.trainer import Trainable
from functools import reduce


class Perceptron(Trainable):
    learning_rate = 0.1

    def __init__(self, inputs):
        self.weights = [random.random() for i in inputs]
        self.bias = 1

    def predict(self, inputs):
        pondered_inputs = [inpt * weight for inpt, weight in zip(inputs, self.weights)]
        predicted = reduce(lambda a, b: a + b, pondered_inputs) + self.bias
        return self.activate_function(predicted)

    def activate_function(self, value):
        return 1 if value > 0.5 else 0  # step

    def train(self, inputs, output):
        predicted = self.predict(inputs)
        pondered_error = self.learning_rate * (output - predicted)
        self.bias += pondered_error
        for (index, inpt) in enumerate(inputs):
            self.weights[index] = self.weights[index] + pondered_error * inpt

    def explain(self):
        print(f"Perceptron weights: {self.weights}")
        print(f"Perceptron bias: {self.bias}")
