from neuralnetwork.dataset import XOR_DATASET
from neuralnetwork.multilayer import Multilayer
from neuralnetwork.trainer import Trainer

mlp = Multilayer(2,2,1)
trainer = Trainer(XOR_DATASET, mlp)

trainer.train_times(50)
trainer.analyze()