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
    # try:
    cost = -1 / (1 if isinstance(y, int) else y.size) * np.sum(np.nan_to_num(
        y * np.log(a) + (np.ones(len(y)) - y) * np.log(np.ones(len(a)) - a)))
    # except(FloatingPointError):
    # try:
    #     a = max(min(a, 1-1e-3), 1e-3)
    #     cost = -y / a + (1 - y) / (1 - a)
    # except(TypeError):
    #     a = max(min(a[0], 1 - 1e-3), 1e-3)
    #     cost = -y / a + (1 - y) / (1 - a)
    return cost


def derivada(y, a, model):
    # try:
    cost = np.nan_to_num(-y / a + (1 - y) / (1 - a))
    # except(FloatingPointError):
    #     a = max(min(a, 1-1e-3), 1e-3)
    #     cost = -y / a + (1 - y) / (1 - a)
    return cost


cross_entropy = {
    "funcion": funcion,
    "derivada": derivada}
