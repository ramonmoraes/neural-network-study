
from collections import namedtuple

class Dataset:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __iter__(self):
        return zip(self.inputs, self.outputs)

    def __getitem__(self, key):
        return (
            self.inputs[key],
            self.outputs[key],
        )

OR_DATASET = Dataset([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], [0, 1, 1, 1])

AND_DATASET = Dataset([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], [0, 0, 0, 1])