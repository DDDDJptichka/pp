import numpy as np
import matplotlib.pyplot as plt


def numerical_gradient(f, x, eps=1e-6):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] += eps
        x2[i] -= eps
        grad[i] = (f(x1) - f(x2)) / (2 * eps)
    return grad


def test_sum():
    print("sum gradient test")
    x = np.array([1.0, 2.0, 3.0])

    f = lambda v: np.sum(v)
    num = numerical_gradient(f, x)
    ana = np.ones_like(x)

    print("analytical:", [float(i) for i in ana])
    print("numerical :", [float(i) for i in num])
    print()


def test_dot():
    print("dot product gradient test")
    w = np.array([0.5, -0.2])
    x = np.array([2.0, 3.0])

    f = lambda w_: np.dot(w_, x)
    num = numerical_gradient(f, w)
    ana = x

    print("analytical:", [float(i) for i in ana])
    print("numerical :", [float(i) for i in num])
    print()


def relu(z):
    return np.maximum(0, z)


def test_relu():
    print("relu derivative test")
    pts = np.linspace(-2, 2, 9)

    for z in pts:
        num = (relu(z + 1e-6) - relu(z - 1e-6)) / (2e-6)
        print(f"{z:.2f} -> {num:.2f}")
    print()


def precision_plot():
    def f(x):
        return np.sin(x) + x**2

    def df(x):
        return np.cos(x) + 2*x

    x0 = 1.0
    eps = np.logspace(-16, -1, 40)
    err = []

    for e in eps:
        num = (f(x0 + e) - f(x0 - e)) / (2 * e)
        err.append(abs(num - df(x0)))

    plt.figure()
    plt.loglog(eps, err)
    plt.xlabel("epsilon")
    plt.ylabel("error")
    plt.title("numerical gradient precision")
    plt.grid()
    plt.show()


def train_neuron():
    np.random.seed(0)

    X = np.random.randn(100, 2)
    true_w = np.array([2.0, 3.0])
    y = X @ true_w

    w = np.random.randn(2) * 0.1
    lr = 0.01

    losses = []

    for _ in range(200):
        pred = X @ w
        loss = np.mean((pred - y) ** 2)
        losses.append(loss)

        grad = (2 / len(X)) * X.T @ (pred - y)
        w -= lr * grad

    print("final weights:", w)

    plt.figure()
    plt.plot(losses)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("training")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    test_sum()
    test_dot()
    test_relu()
    precision_plot()
    train_neuron()