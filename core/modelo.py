import numpy as np

from core.capas import Entrada
from core.funciones import funciones_coste


class Modelo:
    def __init__(self, funcion_coste=funciones_coste.cross_entropy):
        self.funcion_coste = funcion_coste
        self.capas = []
        self.W = []
        self.b = []
        self.deltas = []
        self.coste = []

    def diagnose_boundaries(self, x_lims, y_lims, res, axis=None, plot=False):
        import matplotlib.pyplot as plt

        if axis is None:
            fig, axis = plt.figure(figsize=(30, 20))
        x_values = np.linspace(x_lims[0], x_lims[1], res)
        y_values = np.linspace(y_lims[0], y_lims[1], res)

        grid = np.zeros((res, res))
        inputs = []
        for x in x_values:
            for y in y_values:
                inputs.append([x, y])
        preds, scores = self.predict(inputs, return_scores=True)
        scores = np.reshape(scores, (res, res)).T

        axis.pcolormesh(x_values, y_values, scores, cmap="coolwarm")
        axis.set_xlim(*x_lims)
        axis.set_ylim(*y_lims)

        if plot:
            plt.show()
        else:
            return axis

    def add(self, capa):
        n_capas = len(self.capas)

        if n_capas == 0:
            if not isinstance(capa, Entrada):
                raise Exception("La primera capa debe ser de tipo Entrada")

            self.capas.append(capa)
        else:
            n_anterior = self.capas[-1].n
            n = capa.n
            self.capas.append(capa)
            # Inicialización He
            self.W.append(np.random.rand(n_anterior, n) * (np.sqrt(2 / n_anterior)))
            self.b.append(np.random.rand(1, n) * np.sqrt(2 / n_anterior))

    def comprobar_entradas(self, inputs, targets):
        if type(inputs) is not np.ndarray or type(targets) is not np.ndarray:
            raise Exception("Tanto las entradas como las etiquetas deben de estar contenidas en un array de numpy")

        i = inputs.shape[0]
        j = targets.shape[0]
        if i is not j:
            raise Exception(
                "La dimensión 0, correspondiente al número de muestras, de los inputsy de las etiquetas debe de coincidir ")

        return True

    def train(self, inputs, targets, epochs=1, lr=0.001, batch_size=100):
        self.comprobar_entradas(inputs, targets)
        for epo in range(epochs):
            last_activations = []
            for x, y in zip(inputs, targets):
                # Feedforward
                for i, capa in enumerate(self.capas[1:]):
                    if i == 0:
                        activaciones = capa.__propagar__(x, self.W[i], self.b[i])
                    else:
                        activaciones = capa.__propagar__(activaciones, self.W[i], self.b[i])
                # Backpropagation
                # Calcular deltas

                delta = self.funcion_coste["derivada"](y, capa.a, self) * capa.funcion_activacion["derivada"](capa.z)
                self.deltas.insert(0, delta)
                for i in reversed(range(1, len(self.capas) - 1)):
                    capa = self.capas[i]
                    delta = self.W[i] @ self.deltas[0] * capa.funcion_activacion["derivada"](capa.z).T
                    self.deltas.insert(0, delta)
                # Gradient descent
                for i in reversed(range(len(self.W))):
                    self.W[i] = self.W[i] - lr * np.sum(self.deltas[i] @ self.capas[i + 1].a)
                    self.b[i] = self.b[i] - lr * np.sum(self.deltas[i])
                self.deltas = []
                last_activations.append(self.capas[-1].a)
            last_activations = [act[0] for act in last_activations]
            # print(last_activations)
            # print(self.deltas)
            self.coste.append(self.funcion_coste["funcion"](targets, last_activations))

    def predict(self, inputs, return_scores=False):
        predictions = []
        scores = []
        for x in inputs:
            activaciones = x
            # Feedforward
            for i, capa in enumerate(self.capas[1:]):
                if i == 0:
                    activaciones = capa.__propagar__(x, self.W[i], self.b[i])
                else:
                    activaciones = capa.__propagar__(activaciones, self.W[i], self.b[i])
            a = self.capas[-1].a[0]
            if not np.isnan(a).any():
                scores.append(a)
                predictions.append(int(np.round(a)))
        if return_scores:
            return predictions, scores
        else:
            return predictions
