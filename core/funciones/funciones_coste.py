import numpy as np

np.seterr(divide='raise', invalid='raise')


# from .funciones_activacion import tanh
# cross_entropy = {
#     "funcion": lambda y, a: -1 / (1 if isinstance(y, int) else y.size) * np.sum(
#         y * np.log(a) + (np.ones(len(y)) - y) * np.log(np.ones(len(a)) - a)),
#     "derivada": lambda y, a: -y / a + (1 - y) / (1 - a)}
# mse = {"funcion": lambda y, a: 1 / (2 * (1 if isinstance(y, int) else y.size)) * np.sum((y - a) ** 2),
#        "derivada": lambda y, a: 1 / (1 if isinstance(y, int) else y.size) * np.sum(tanh["derivada"]())}

def funcion(y, a):
    try:
        cost = -1 / (1 if isinstance(y, int) else y.size) * np.sum(
            y * np.log(a) + (np.ones(y.shape) - y) * np.log(np.ones(a.shape) - a))
    except FloatingPointError:
        a = np.maximum(np.minimum(a, 1 - 1e-3), 1e-3)
        cost = -1 / (1 if isinstance(y, int) else y.size) * np.sum(
            y * np.log(a) + (np.ones(y.shape) - y) * np.log(np.ones(a.shape) - a))
    return cost


def derivada(y, a):
    try:
        cost = -y / a + (1 - y) / (1 - a)
    except FloatingPointError:
        a = np.maximum(np.minimum(a, 1 - 1e-3), 1e-3)
        cost = -y / a + (1 - y) / (1 - a)
    return cost


cross_entropy = {
    "funcion": funcion,
    "derivada": derivada}
