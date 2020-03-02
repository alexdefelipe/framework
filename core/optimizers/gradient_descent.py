def gradient_descent(modelo, lr):
    for i in reversed(range(len(modelo.W))):
        modelo.W[i] = modelo.W[i] - lr * modelo.capas[i].a.T @ modelo.deltas[i]
        modelo.b[i] = modelo.b[i] - lr * modelo.deltas[i]
