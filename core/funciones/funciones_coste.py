import numpy as np
# from .funciones_activacion import tanh
cross_entropy = {
    "funcion": lambda y, a: -1 / (1 if isinstance(y, int) else y.size) * np.sum(
        y * np.log(a) + (np.ones(len(y)) - y) * np.log(np.ones(len(a)) - a)),
    "derivada": lambda y, a: -y / a + (1 - y) / (1 - a)}

# mse = {"funcion": lambda y, a: 1 / (2 * (1 if isinstance(y, int) else y.size)) * np.sum((y - a) ** 2),
#        "derivada": lambda y, a: 1 / (1 if isinstance(y, int) else y.size) * np.sum(tanh["derivada"]())}
