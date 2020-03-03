import numpy as np

sigma = lambda x: 1 / (1 + np.exp(-x))
identidad = {"funcion": lambda x: x, "derivada": 1}
sigmoide = {"funcion": lambda x: sigma(x),
            "derivada": lambda x: sigma(x) * (1 - sigma(x))}
relu = {"funcion": lambda x: np.maximum(0, x), "derivada": lambda x: np.where(x <= 0, 0, 1)}
def softmax(x): return x / np.sum(x)
# t = lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
# tanh = {"funcion": lambda x: t(x),
#         "derivada": lambda x: 1 - t(x) ** 2}
