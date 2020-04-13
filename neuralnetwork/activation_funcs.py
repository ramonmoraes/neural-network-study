import numpy as np
import math


def gate(val):
    return 1 if val > 0.5 else 0


v_gate = np.vectorize(gate)


def sigmoid(value):
    return 1 / (1 + np.exp(-value))


v_sigmoid = np.vectorize(sigmoid)


def d_sigmoid(value, sigmoided=False):
    sig = value if sigmoided else sigmoid(value)
    return sig * (1 - sig)


v_d_sigmoid = np.vectorize(d_sigmoid)

def tanh(val):
    p_exp = np.exp(val)
    n_exp = np.exp(-val)
    
    return (p_exp - n_exp) / (p_exp + n_exp)

v_tanh = np.vectorize(tanh)

def d_tanh(val):
    tan = val
    # tan = tanh(val)
    return 1 - tan ** 2

v_d_tanh = np.vectorize(d_tanh)
