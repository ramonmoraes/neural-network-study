from neuralnetwork.perceptron import Perceptron
from neuralnetwork.trainer import Trainer
from neuralnetwork.dataset import AND_DATASET, OR_DATASET

def test_perceptron_predict():
    for dataset in [AND_DATASET, OR_DATASET]:
        perceptron = Perceptron(2)
        trainer = Trainer(dataset,perceptron)
        errors_list = trainer.analyze()
        # Where is the time the prediction have failed
        assert 1 in errors_list

        trainer.train_times(50)
        errors_list = trainer.analyze()
        assert 1 not in errors_list
