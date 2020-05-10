from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

from core import Modelo
from core.capas import Entrada, Densa


def nuestro_modelo(X_train, X_test, y_train, y_test, topologia):
    modelo = Modelo()

    modelo.add(Entrada(X_train.shape[1]))
    for n in topologia:
        modelo.add(Densa(n))

    modelo.add(Densa(1, funcion_activacion="sigmoide"))

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
    X, y = make_circles(n_samples=200, random_state=105)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    nuestro_modelo(X_train, X_test, y_train, y_test, [7, 5, 3])
