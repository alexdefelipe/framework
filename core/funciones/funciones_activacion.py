import numpy as np

sigma = lambda x: 1 / (1 + np.exp(-x))
identidad = {"funcion": lambda x: x, "derivada": 1}
sigmoide = {"funcion": lambda x: sigma(x),
            "derivada": lambda x: sigma(x) * (1 - sigma(x))}
