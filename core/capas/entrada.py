from ..funciones import funciones_activacion


class Entrada():
    def __init__(self, n):
        self.n = n
        self.funcion_activacion = funciones_activacion.identidad

    def __propagar__(self, x):
        self.z = x
        self.a = self.funcion_activacion["funcion"](x)
        return self.a
