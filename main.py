from neuralnetwork.dataset import OR_DATASET, AND_DATASET
from neuralnetwork.perceptron import Perceptron, Trainer

perceptron = Perceptron(AND_DATASET.inputs[0])

trainer = Trainer(AND_DATASET, perceptron)

trainer.analyze()
trainer.train_times(100)
trainer.analyze()
