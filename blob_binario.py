from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from core import Modelo
from core.callbacks import MetricsCallback
from core.capas import Entrada, Densa
from core.metrics import Accuracy, Recall


def nuestro_modelo(X_train, X_test, y_train, y_test, topologia):
    modelo = Modelo()

    modelo.add(Entrada(X_train.shape[1]))
    for n in topologia:
        modelo.add(Densa(n))

    modelo.add(Densa(1, funcion_activacion="sigmoide"))

    modelo.train(X_train, y_train, epochs=100, batch_size=100, lr=0.01, diagnose=False, callbacks=set_callbacks())
    y_pred, scores = modelo.predict(X_test, return_scores=True)

    preds = modelo.predict(X_test)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax2 = modelo.plot_probability_map([min(X_train[:, 0]), max(X_train[:, 0])],
                                      [min(X_train[:, 1]), max(X_train[:, 1])],
                                      res=50, axis=ax2, plot=True)
    fig.suptitle('Horizontally stacked subplots')
    ax1.plot(modelo.cost)
    ax1.set_title("Coste")
    ax2.scatter(X_test[:, 0], X_test[:, 1], c=preds)
    ax2.set_title("Predicciones")
    plt.show()


def set_callbacks():
    metrics_callback = MetricsCallback(selected_metrics=[Accuracy.name, Recall.name], on_epoch_functions=[lambda x: print(x)])
    return [metrics_callback]


if __name__ == '__main__':
    X, y = make_blobs(n_samples=300, centers=2, n_features=4, random_state=105)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    nuestro_modelo(X_train, X_test, y_train, y_test, [3])
