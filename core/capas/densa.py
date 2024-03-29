from ..funciones import funciones_activacion


class Densa():
    def __init__(self, n, funcion_activacion='sigmoide'):
        self.n = n
        self.funcion_activacion = getattr(funciones_activacion, funcion_activacion)

    def __propagar__(self, x, W, b):
        self.z = x @ W + b
        self.a = self.funcion_activacion["funcion"](self.z)
        return self.a
