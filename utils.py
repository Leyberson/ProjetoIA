import numpy as np
CONST = 418.9829

def schwefel_function(x):
    cum_fun = len(x)*CONST
    for x_i in x:
        cum_fun -= (x_i * np.sin(np.sqrt(np.abs(x_i))))
    return cum_fun

def initialize_X(max=500, min=-500, size=5):
    return [np.random.rand()*(max-min) + min for _ in range(size)]
