import copy
import os
import tempfile

import imageio
import matplotlib.pyplot as plt
import numpy as np

from core.capas import Entrada
from core.funciones import funciones_coste
from core.optimizers import optimizers


class Modelo:
    def __init__(self, funcion_coste=funciones_coste.cross_entropy, optimizer="SGD"):
        self.funcion_coste = funcion_coste
        self.capas = []
        self.W = []
        self.b = []
        self.deltas = []
        self.coste = []
        self.bound_tracking = []
        self.weights = []
        self.optimizer = optimizers[optimizer]

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
        if i != j:
            raise Exception(
                "La dimensión 0, correspondiente al número de muestras, de los inputs y de las etiquetas debe de coincidir ")

    def generate_random_batch(self, inputs, targets, batch_size):
        batch_idx = np.random.choice(targets.size, batch_size, replace=False)
        batch_inputs = inputs[batch_idx, :]
        batch_targets = targets[batch_idx]

        return batch_inputs, batch_targets

    def feed_forward(self, x, capas, W, b):
        activaciones = np.reshape(x, (1, x.size))
        for i, capa in enumerate(capas):
            if i == 0:
                activaciones = capa.__propagar__(activaciones)
            else:
                activaciones = capa.__propagar__(activaciones, W[i - 1], b[i - 1])

        prediccion = int(np.round(activaciones))
        score = activaciones

        return prediccion, score

    def train(self, inputs, targets, epochs=1, lr=0.001, batch_size=None, diagnose=False):
        self.comprobar_entradas(inputs, targets)
        self.cost = np.empty((epochs,))

        if batch_size is None:
            batch_size = targets.size

        for epo in range(epochs):
            batch_inputs, batch_targets = self.generate_random_batch(inputs, targets, batch_size)
            last_activations = np.empty(batch_targets.shape)
            for n_sample, (x, y) in enumerate(zip(batch_inputs, batch_targets)):
                # Feedforward
                self.feed_forward(x, self.capas, self.W, self.b)

                # Backpropagation
                delta = self.funcion_coste["derivada"](y, self.capas[-1].a) * self.capas[-1].funcion_activacion[
                    "derivada"](self.capas[-1].z)
                self.deltas.insert(0, delta)
                for i in reversed(range(1, len(self.capas) - 1)):
                    capa = self.capas[i]
                    delta = self.deltas[0] @ self.W[i].T * capa.funcion_activacion["derivada"](capa.z)
                    self.deltas.insert(0, delta)

                # Weight optimization
                self.optimize_parameters(lr)

                self.deltas = []
                last_activations[n_sample] = self.capas[-1].a
            self.cost[epo] = self.funcion_coste["funcion"](batch_targets, last_activations)
            self.weights.append(copy.deepcopy(self.W))
        if diagnose:
            self.decision_boundary_tracking(inputs, targets)

    def predict(self, inputs, weights=None, return_scores=False):
        predictions = []
        scores = []

        if weights is None:
            weights = self.W

        for x in inputs:
            prediction, score = self.feed_forward(x, self.capas, weights, self.b)
            predictions.append(prediction)
            if return_scores:
                scores.append(score)
        return (predictions, scores) if return_scores else predictions

    def plot_probability_map(self, x_lims, y_lims, res, data=None, weights=None, axis=None, plot=False):
        axis = axis or plt.gca()
        x_values = np.linspace(x_lims[0], x_lims[1], res)
        y_values = np.linspace(y_lims[0], y_lims[1], res)

        inputs = np.array([(x, y) for x in x_values for y in y_values])
        preds, scores = self.predict(inputs, weights=weights, return_scores=True)
        scores = np.reshape(scores, (res, res)).T

        axis.pcolormesh(x_values, y_values, scores, cmap="coolwarm")
        axis.set_xlim(*x_lims)
        axis.set_ylim(*y_lims)

        if data is not None:
            axis.scatter(data[0][:, 0], data[0][:, 1], c=data[1])

        if plot:
            plt.show()
        else:
            return axis

    def decision_boundary_tracking(self, inputs, targets):
        n_epochs = self.cost.size
        fig, ax = plt.subplots(2, 1, figsize=(5, 6), gridspec_kw={'height_ratios': [1, 3]})
        tmpdir = tempfile.gettempdir()
        savepath = '/'.join([tmpdir, 'diagnostic_images'])
        filename = '/'.join([savepath, 'image.jpg'])
        os.makedirs(savepath, exist_ok=True)

        for epoch in range(1, n_epochs + 1):
            weights = self.weights[epoch - 1]
            ax[0].set_title("Cost over epochs")
            ax[0].set_xlabel("Epochs")
            ax[0].set_ylabel("Cost")
            ax[0].plot(self.cost[:epoch])
            ax[0].set_xlim(0, epoch)
            ax[1] = self.plot_probability_map([min(inputs[:, 0]), max(inputs[:, 0])],
                                              [min(inputs[:, 1]), max(inputs[:, 1])], axis=ax[1], res=50,
                                              data=(inputs, targets), weights=weights)

            ax[1].set_title("Decision map evolution")
            fig.tight_layout()
            plt.savefig(filename)
            self.bound_tracking.append(imageio.imread(filename))
            ax[0].cla()
            ax[1].cla()

        duration = [max(3 / n_epochs, 0.05) for i in range(n_epochs)]
        duration[-1] = 1
        imageio.mimsave('/'.join([savepath, 'anim.gif']), self.bound_tracking, duration=duration)
        print("Se ha guardado un GIF en " + savepath)

    def optimize_parameters(self, lr):
        self.optimizer(self, lr)
