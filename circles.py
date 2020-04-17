import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from core import Modelo
from core.capas import Entrada, Densa


def nuestro_modelo(X_train, X_test, y_train, y_test, topologia):
    modelo = Modelo()

    modelo.add(Entrada(X_train.shape[1]))
    for n in topologia:
        modelo.add(Densa(n))

    modelo.add(Densa(3, funcion_activacion="softmax"))

    modelo.train(X_train, y_train, epochs=100, batch_size=100, lr=0.01, diagnose=False)
    y_pred, scores = modelo.predict(X_test, return_scores=True)

    # y_pred_bool = np.round(scores).astype(np.int)
    print("Resultados de nuestro modelo:\n")
    # print(classification_report(np.argmax(y_test, axis=0), y_pred))

    preds = modelo.predict(X_test)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax2 = modelo.plot_probability_map([min(X_train[:, 0]), max(X_train[:, 0])],
                                      [min(X_train[:, 1]), max(X_train[:, 1])],
                                      res=50, axis=ax2)
    fig.suptitle('Horizontally stacked subplots')
    ax1.plot(modelo.cost)
    ax1.set_title("Coste")
    ax2.scatter(X_test[:, 0], X_test[:, 1], c=preds)
    ax2.set_title("Predicciones")
    plt.show()


if __name__ == '__main__':
    X, y = make_blobs(n_samples=300, centers=3, n_features=2, random_state=105)
    # y = np.expand_dims(y, axis=1)
    # X, y = make_circles(n_samples=1000, factor=0.1, noise=0.05, random_state=106)
    # X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    # X, y = load_digits(return_X_y=True)
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    onehotencoder = OneHotEncoder(categories="auto")
    y = onehotencoder.fit_transform(np.expand_dims(y, axis=1)).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=5000, test_size=10000, random_state=42)
    nuestro_modelo(X_train, X_test, y_train, y_test, [3])
