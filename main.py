from neuralnetwork.dataset import OR_DATASET, AND_DATASET, XOR_DATASET
from neuralnetwork.perceptron import Perceptron
from neuralnetwork.multilayer import Multilayer
from neuralnetwork.trainer import Trainer

perceptron = Perceptron(AND_DATASET.inputs[0])
mlp = Multilayer(2,2,1)

trainer = Trainer(OR_DATASET, mlp)

# trainer.analyze()
# trainer.train_times(100)
# trainer.analyze()

# mlp.layers[1].backward_weights
# mlp.layers[2].backward_weights == mlp.layers[1].forward_weights

# output = mlp.predicted([[0.5, 1]])
# back = mlp.layers[2].backward([[1,1]])

# trainer = Trainer(mlp, XOR_DATASET)


# trainer.train_times(1)

# trainer.analyze()
trainer.train_times(100)
trainer.analyze()
