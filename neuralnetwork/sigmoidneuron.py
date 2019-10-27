from neuralnetwork.perceptron import Perceptron


class SigmoidNeuron(Perceptron):
    def activate_function(self, value):
        return 1 / (1 + math.exp(-value))
