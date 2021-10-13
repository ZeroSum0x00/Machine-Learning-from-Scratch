import numpy as np


def central_diff_weights(Np, ndiv=1):

    if Np < ndiv + 1:
        raise ValueError("Number of points must be at least the derivative order + 1.")
    if Np % 2 == 0:
        raise ValueError("The number of points must be odd.")

    ho = Np >> 1
    x = np.arange(-ho, ho + 1.0)
    x = x[:, np.newaxis]
    X = x**0.0
    for k in range(1, Np):
        X = np.hstack([X, x**k])
    w = np.prod(np.arange(1, ndiv + 1), axis=0) * np.linalg.inv(X)[ndiv]
    return w


def derivative(func, x0, dx=1.0, n=1, args=(), order=3):

    if order < n + 1:
        raise ValueError("'order' (the number of points used to compute the derivative), "
                         "must be at least the derivative order 'n' + 1.")
    if order % 2 == 0:
        raise ValueError("'order' (the number of points used to compute the derivative) "
                         "must be odd.")
    # pre-computed for n=1 and 2 and low-order for speed.
    if n == 1:
        if order == 3:
            weights = np.array([-1, 0, 1]) / 2.0
        elif order == 5:
            weights = np.array([1, -8, 0, 8, -1]) / 12.0
        elif order == 7:
            weights = np.array([-1, 9, -45, 0, 45, -9, 1]) / 60.0
        elif order == 9:
            weights = np.array([3, -32, 168, -672, 0, 672, -168, 32, -3]) / 840.0
        else:
            weights = central_diff_weights(order, 1)
    elif n == 2:
        if order == 3:
            weights = np.array([1, -2.0, 1])
        elif order == 5:
            weights = np.array([-1, 16, -30, 16, -1]) / 12.0
        elif order == 7:
            weights = np.array([2, -27, 270, -490, 270, -27, 2]) / 180.0
        elif order == 9:
            weights = np.array([-9, 128, -1008, 8064, -14350, 8064, -1008, 128, -9]) / 5040.0
        else:
            weights = central_diff_weights(order, 2)
    else:
        weights = central_diff_weights(order, n)
    val = 0.0
    ho = order >> 1
    for k in range(order):
        val += weights[k] * func(x0 + (k - ho) * dx, *args)
    return val / np.prod((dx, ) * n, axis=0)