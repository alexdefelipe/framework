import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from core import Modelo
from core.capas import Entrada, Densa


def nuestro_modelo(X_train, X_test, y_train, y_test):
    modelo = Modelo()

    modelo.add(Entrada(2))
    modelo.add(Densa(3))
    modelo.add(Densa(1))

    modelo.train(X_train, y_train, epochs=100, lr=0.05, diagnose=True)
    # y_pred, scores = modelo.predict(X_test, True)

    # y_pred_bool = np.round(scores).astype(np.int)
    # print("Resultados de nuestro modelo:\n")
    # print(classification_report(y_test, np.squeeze(y_pred_bool)))

    preds = modelo.predict(X_test)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax2 = modelo.diagnose_boundaries([min(X_train[:, 0]), max(X_train[:, 0])], [min(X_train[:, 1]), max(X_train[:, 1])],
                                     res=50, axis=ax2)
    fig.suptitle('Horizontally stacked subplots')
    ax1.plot(modelo.coste)
    ax1.set_title("Coste")
    ax2.scatter(X_test[:, 0], X_test[:, 1], c=preds)
    ax2.set_title("Predicciones")
    plt.show()


def modelo_keras(X_train, X_test, y_train, y_test):
    modelo = Sequential()
    modelo.add(Dense(2, input_dim=2, activation='sigmoid'))
    modelo.add(Dense(2, activation='sigmoid'))
    modelo.add(Dense(1, activation='sigmoid'))
    modelo.compile(loss='binary_crossentropy', optimizer='SGD')
    modelo.fit(X_train, y_train, epochs=200, verbose=0)
    y_pred = modelo.predict_proba(X_test)
    y_pred_bool = np.round(y_pred)

    print("Resultados del modelo de keras:\n")
    print(classification_report(y_test, y_pred_bool))


if __name__ == '__main__':
    # X, Y = make_blobs(n_samples=300, centers=2, n_features=2, random_state=105)
    X, Y = make_circles(n_samples=1000, factor=0.1, noise=0.05, random_state=106)
    X_trai, X_tes, y_trai, y_tes = train_test_split(X, Y, test_size=0.33, random_state=42)
    # modelo_keras(X_trai, X_tes, y_trai, y_tes)
    nuestro_modelo(X_trai, X_tes, y_trai, y_tes)
