class Multilayer:
    def __init__(self, inputs_size, hidden_size, output_size):
        self.inputs = inputs_size * [0]
        self.hidden = hidden_size * [0]
        self.output = output_size * [0]

