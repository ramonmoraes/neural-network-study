import random
from functools import reduce

class Perceptron:
    learning_rate = 0.1
    
    def __init__(self, inputs):
        self.weights = [random.random() for i in inputs]
        self.bias = random.random()

    def predict(self, inputs):
        pondered_inputs = [inpt * weight for inpt, weight in zip(inputs, self.weights)]
        predicted = reduce(lambda a,b: a+b, pondered_inputs) + self.bias
        return self.activate_function(predicted)
    
    def activate_function(self, value):
        return 1 if value > 0.5 else 0 #step