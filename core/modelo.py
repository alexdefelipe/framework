import os
import tempfile

import imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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
        self.bound_tracking = []

    def diagnose_boundaries(self, x_lims, y_lims, res, axis=None, plot=False, data=None):
        import matplotlib.pyplot as plt

        axis = axis or plt.gca()
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

        if data is not None:
            axis.scatter(data[0][:, 0], data[0][:, 1], c=data[1])

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
        if i != j:
            raise Exception(
                "La dimensión 0, correspondiente al número de muestras, de los inputs y de las etiquetas debe de coincidir ")

    def train(self, inputs, targets, epochs=1, lr=0.001, batch_size=100, diagnose=False):
        self.comprobar_entradas(inputs, targets)
        self.coste = np.empty((epochs,))

        if diagnose:
            fig, ax = plt.subplots(2, 1, figsize=(5, 6), gridspec_kw={'height_ratios': [1, 3]})
            # ax = fig.add_subplot(111)
            tmpdir = tempfile.gettempdir()
            savepath = '/'.join([tmpdir, 'diagnostic_images'])
            os.makedirs(savepath, exist_ok=True)
            n = 0

        for epo in range(epochs):
            last_activations = np.empty(targets.shape)
            for n_sample, (x, y) in enumerate(zip(inputs, targets)):
                # Feedforward
                activaciones = np.reshape(x, (1, x.size))
                for i, capa in enumerate(self.capas):
                    if i == 0:
                        activaciones = capa.__propagar__(activaciones)
                    else:
                        activaciones = capa.__propagar__(activaciones, self.W[i - 1], self.b[i - 1])
                # Backpropagation
                # Calcular deltas

                delta = self.funcion_coste["derivada"](y, capa.a) * capa.funcion_activacion["derivada"](capa.z)
                self.deltas.insert(0, delta)
                for i in reversed(range(1, len(self.capas) - 1)):
                    capa = self.capas[i]
                    delta = self.deltas[0] @ self.W[i].T * capa.funcion_activacion["derivada"](capa.z)
                    self.deltas.insert(0, delta)
                # Gradient descent
                for i in reversed(range(len(self.W))):
                    self.W[i] = self.W[i] - lr * self.capas[i].a.T @ self.deltas[i]
                    self.b[i] = self.b[i] - lr * self.deltas[i]
                self.deltas = []
                last_activations[n_sample] = self.capas[-1].a
            self.coste[epo] = self.funcion_coste["funcion"](targets, last_activations)
            if diagnose:
                ax[0].plot(self.coste[:epo])
                ax[0].set_title("Cost over epochs")
                ax[0].set_xlim(0, epochs)
                ax[0].set_xlabel("Epochs")
                ax[0].set_ylabel("Cost")
                ax[1] = self.diagnose_boundaries([min(inputs[:, 0]), max(inputs[:, 0])],
                                                 [min(inputs[:, 1]), max(inputs[:, 1])], axis=ax[1], res=50,
                                                 data=(inputs, targets))
                ax[1].set_title("Decision map evolution")
                # fig.suptitle("Training diagnostics")
                fig.tight_layout()
                # fig.subplots_adjust(top=0.92)
                filename = '/'.join([savepath, 'image.jpg'])
                plt.savefig(filename)
                self.bound_tracking.append(imageio.imread(filename))
                # plt.cla()
                ax[0].cla()
                ax[1].cla()
        if diagnose:
            duration = [max(3 / epochs, 0.05) for i in range(epochs)]
            duration[-1] = 1
            imageio.mimsave('/'.join([savepath, 'anim.gif']), self.bound_tracking, duration=duration)
            im = Image.open('/'.join([savepath, 'anim.gif']))

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
