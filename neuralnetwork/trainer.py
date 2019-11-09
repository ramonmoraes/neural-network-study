import numpy as np


class Trainer:
    def __init__(self, dataset, trainable):
        self.dataset = dataset
        self.trainable = trainable

    def train_times(self, times=1):
        for i in range(times):
            for inpt, output in self.dataset:
                self.trainable.train(inpt, output)

    def analyze(self):
        print("[Analyzing]")
        errors = []
        for inpt, output in self.dataset:
            predicted = self.trainable.predict(inpt)
            print(f"expected: {output} got:{predicted}")
            errors.append(0 if output == predicted else 1)
        print(f"Error: {np.mean(errors)}")

        self.trainable.explain()
        return errors


class Trainable:
    def predict(self, inputs):
        raise NotImplementedError()

    def train(self, inputs, outputs):
        raise NotImplementedError()

    def explain(self):
        raise NotImplementedError()

