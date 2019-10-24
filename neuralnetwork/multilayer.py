class Multilayer:
"""
    I1 - H1
            - o
    I2 - H2
---------------------
    | I1*W1 I2*W1 |  -\  | H1 |
    | I1*W2 I2*W2 |  -/  | H2 |
"""
    def __init__(self, inputs, hidden, output):
        self.inputs = inputs
        self.hidden = hidden
        self.output = output

