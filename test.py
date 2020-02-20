from core.capas import Entrada, Densa
from core import Modelo

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.model_selection import train_test_split

modelo = Modelo()

modelo.add(Entrada(2))
modelo.add(Densa(1))

X, Y = make_blobs(n_samples=300, centers=2, n_features=2, random_state=106)
# X, Y = make_circles(n_samples=500, factor=0.1, noise=0.05)
Y = Y[:, np.newaxis]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
# X = np.ones((1, 2))
# Y = np.ones((1, 1))

modelo.train(X_train, y_train, epochs=500, lr=0.01)
preds = modelo.predict(X_test)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax2 = modelo.diagnose_boundaries([min(X_train[:, 0]),max(X_train[:, 0])], [min(X_train[:, 1]), max(X_train[:, 1])], res=50, axis = ax2)
fig.suptitle('Horizontally stacked subplots')
ax1.plot(modelo.coste)
ax1.set_title("Coste")
ax2.scatter(X_test[:, 0], X_test[:, 1], c=preds)
ax2.set_title("Predicciones")
plt.show()


