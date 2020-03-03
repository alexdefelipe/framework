import numpy as np
from keras import Sequential, optimizers
from keras.layers import Dense
from sklearn.datasets import make_circles
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from core import Modelo
from core.capas import Entrada, Densa


def nuestro_modelo(topologia, X_train, X_test, y_train, y_test, lr, epochs, batch_size):
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    n_classes = np.unique(y_train).size

    modelo = Modelo()
    modelo.add(Entrada(n_features))
    for n_units in topologia:
        modelo.add(Densa(n_units, funcion_activacion='sigmoide'))

    activation = 'sigmoide' if n_classes is 2 else 'softmax'
    n_units = 1 if n_classes is 2 else n_classes
    modelo.add(Densa(n_units, funcion_activacion=activation))

    modelo.train(X_train, y_train, epochs=epochs, batch_size=batch_size, lr=lr)
    y_pred, scores = modelo.predict(X_test, return_scores=True)

    y_pred_bool = np.round(scores).astype(np.int)
    print("Resultados de nuestro modelo:\n")
    print(classification_report(y_test, np.squeeze(y_pred_bool)))


def modelo_keras(topologia, X_train, X_test, y_train, y_test, lr, epochs, batch_size):
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    n_classes = np.unique(y_train).size

    model = Sequential()
    for n_units in topologia:
        model.add(Dense(n_units, activation='sigmoid', input_dim=n_features))

    activation = 'sigmoid' if n_classes is 2 else 'softmax'
    n_units = 1 if n_classes is 2 else n_classes
    model.add(Dense(n_units, activation=activation))

    sgd = optimizers.SGD(lr=lr, decay=0, momentum=0, nesterov=False)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    y_pred = model.predict_classes(X_test, verbose=0)

    print("Resultados del modelo de keras:\n")
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    # X, Y = make_blobs(n_samples=300, centers=2, n_features=2, random_state=105)
    X, Y = make_circles(n_samples=1000, factor=0.1, noise=0.05, random_state=106)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    topologia = [3]

    nuestro_modelo(topologia, X_train, X_test, y_train, y_test, 0.05, 100, 20)
    modelo_keras(topologia, X_train, X_test, y_train, y_test, 0.05, 100, 20)
