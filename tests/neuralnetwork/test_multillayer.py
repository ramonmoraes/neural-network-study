from neuralnetwork.multilayer import Multilayer
import numpy as np


def test_mlp_link():
    mlp = Multilayer(2, 2, 2)
    assert mlp.layers[0].forward_weights is mlp.layers[1].backward_weights
    assert mlp.layers[1].forward_weights is mlp.layers[2].backward_weights

    assert mlp.layers[2].previous_layer is mlp.layers[1]
    assert mlp.layers[1].previous_layer is mlp.layers[0]


def test_predict():
    mlp = Multilayer(2, 2, 1)
    mlp.layers[0].bias = 1
    mlp.layers[1].bias = 1

    mlp.layers[0].forward_weights = np.array([[0.5, 0.4], [0.2, 0.04]])
    mlp.layers[1].forward_weights = np.array([[0.2], [0.8]])
    mlp.link_layers()

    prediction = mlp.predict([1,1]) 
    assert len(prediction) == 1
    assert round(prediction[0],2) == 0.96