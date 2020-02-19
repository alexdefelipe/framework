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
        self.lr = 0.001
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
                # Feedforward
                for i, capa in enumerate(self.capas[1:]):
                    if i == 0:
                        activaciones = capa.__propagar__(x,  self.W[i], self.b[i])
                    else:
                        activaciones = capa.__propagar__(activaciones, self.W[i], self.b[i])
                # Backpropagation
                # Calcular deltas

                delta = self.funcion_coste["derivada"](y, capa.a) * capa.funcion_activacion["derivada"](capa.z)
                self.deltas.insert(0, delta)
                for i in reversed(range(1, len(self.capas) - 1)):
                    capa = self.capas[i]
                    delta = self.W[i] @ self.deltas[0] * capa.funcion_activacion["derivada"](capa.z).T
                    self.deltas.insert(0, delta)
                # Gradient descent
                for i in reversed(range(len(self.W))):
                    self.W[i] = self.W[i] - self.lr * np.sum(self.deltas[i] @ self.capas[i+1].a)
                    self.b[i] = self.b[i] - self.lr * np.sum(self.deltas[i])
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
            for i, capa in enumerate(self.capas[1:]):
                if i == 0:
                    activaciones = capa.__propagar__(x, self.W[i], self.b[i])
                else:
                    activaciones = capa.__propagar__(activaciones, self.W[i], self.b[i])
            a = self.capas[-1].a[0]
            if not np.isnan(a).any():
                predictions.append(int(np.round(a)))
        return predictions
