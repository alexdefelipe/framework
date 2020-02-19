from core.capas import Entrada
from core.funciones import funciones_coste

import numpy as np


class Modelo:
    def __init__(self, funcion_coste=funciones_coste.cross_entropy):
        self.funcion_coste = funcion_coste
        self.capas = []
        self.W = []
        self.b = []
        self.deltas = []
        self.lr = 0.1
        self.coste = []

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

    def train(self, inputs, targets, epochs=1):
        for epo in range(epochs):
            last_activations = []
            for x, y in zip(inputs, targets):
                activaciones = x
                # Feedforward
                for i, capa in enumerate(self.capas):
                    if i == 0:
                        activaciones = capa.__propagar__(activaciones)
                    else:
                        activaciones = capa.__propagar__(activaciones, self.W[i - 1], self.b[i - 1])
                # Backpropagation
                # Calcular deltas
                for i, capa in enumerate(reversed(self.capas[1:])):
                    if i == 0:
                        delta = self.funcion_coste["derivada"](y,
                                                               capa.a)  # * capa.funcion_activacion["derivada"](capa.a)
                        self.deltas.append(delta)
                    else:
                        delta = np.dot(self.W[-i], self.deltas[i - 1]) * capa.funcion_activacion["derivada"](capa.a)
                        self.deltas.append(delta)
                # Gradient descent
                for i in range(1, len(self.capas)):
                    self.W[-i] = self.W[-i] - self.lr * np.dot(self.deltas[i - 1], self.capas[-i].a.T)
                self.deltas = []
                last_activations.append(self.capas[-1].a)
            last_activations = [act[0] for act in last_activations]
            # print(last_activations)
            # print(self.deltas)
            self.coste.append(self.funcion_coste["funcion"](targets, last_activations))

    def predict(self, inputs):
        predictions = []
        for x in inputs:
            activaciones = x
            # Feedforward
            for i, capa in enumerate(self.capas):
                if i == 0:
                    activaciones = capa.__propagar__(activaciones)
                else:
                    activaciones = capa.__propagar__(activaciones, self.W[i - 1], self.b[i - 1])
            predictions.append(int(np.round(self.capas[-1].a[0])))
        return predictions
