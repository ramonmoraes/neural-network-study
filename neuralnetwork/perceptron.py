import random
from numpy import mean
from functools import reduce


class Perceptron:
    learning_rate = 0.1

    def __init__(self, inputs):
        self.weights = [random.random() for i in inputs]
        self.bias = random.random()

    def predict(self, inputs):
        pondered_inputs = [inpt * weight for inpt, weight in zip(inputs, self.weights)]
        predicted = reduce(lambda a, b: a + b, pondered_inputs) + self.bias
        return self.activate_function(predicted)

    def activate_function(self, value):
        return 1 if value > 0.5 else 0  # step

    def train(self, inpt, output):
        predicted = self.predict(inpt)
        error = output - predicted
        for w in [*self.weights, self.bias]:
            w += self.learning_rate * error


class Trainer:
    def __init__(self, dataset, perceptron):
        self.dataset = dataset
        self.perceptron = perceptron

    def train_times(self, times=100):
        for i in range(times):
            for inpt, output in self.dataset:
                self.perceptron.train(inpt, output)

    def analyze(self):
        for inpt, output in self.dataset:
            print(f"expected {output}")
            print(f"got {self.perceptron.predict(inpt)}")
