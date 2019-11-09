from neuralnetwork.multilayer import Multilayer


def test_mlp_link():
    mlp = Multilayer(2, 2, 2)
    assert mlp.layers[0].forward_weights is mlp.layers[1].backward_weights
    assert mlp.layers[1].forward_weights is mlp.layers[2].backward_weights

    assert mlp.layers[2].previous_layer is mlp.layers[1]
    assert mlp.layers[1].previous_layer is mlp.layers[0]

