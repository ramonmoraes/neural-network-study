import random
from numpy import mean
from functools import reduce


class Perceptron:
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


class Trainer:
    def __init__(self, dataset, perceptron):
        self.dataset = dataset
        self.perceptron = perceptron

    def train_times(self, times=100):
        for i in range(times):
            for inpt, output in self.dataset:
                self.perceptron.train(inpt, output)

    def analyze(self):
        print("[Analyzing]")
        errors = []
        for inpt, output in self.dataset:
            predicted = self.perceptron.predict(inpt)
            print(f"expected: {output} got:{predicted}")
            errors.append(0 if output == predicted else 1)

        print(f"Perceptron weights: {self.perceptron.weights}")
        print(f"Perceptron bias: {self.perceptron.bias}")
        print(f"Error: {mean(errors)}")
