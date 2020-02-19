from core.capas import Entrada, Densa
from core import Modelo

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs


modelo = Modelo()

modelo.add(Entrada(2))
modelo.add(Densa(8))
modelo.add(Densa(1))

X, Y = make_blobs(n_samples=50, centers=2, n_features=2)
Y = Y[:, np.newaxis]
plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c="skyblue")
plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c="salmon")
plt.axis("equal")
plt.show()
# X = np.ones((1, 2))
# Y = np.ones((1, 1))

modelo.train(X, Y, 2000)
plt.plot(modelo.coste)
plt.show()
