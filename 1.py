import numpy as np

def f(x):

    return x[0]**2 + 3 * (x[1]**2)


def gradient(f, x):

    h = 1e-5
    grad = np.zeros_like(x, dtype=float)

    for i in range(2):

        x_forward = x.copy()
        x_backward = x.copy()

        x_forward[i] += h
        x_backward[i] -= h

        grad[i] = (f(x_forward) - f(x_backward)) / (2 * h)

    return grad

x = np.array([2.0, 1.0])
grad = gradient(f, x)

print("Исходная функция: y = (x0)^2 + 3 * (x1)^2")
print("x:", [float(xi) for xi in x])
print("Значение функции:", f(x))
print("градиент:", [float(gi) for gi in grad])