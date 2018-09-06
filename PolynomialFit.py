"""
Implement simple polynomial interpolation to help draw smooth curves
on the merge trees
"""
import numpy as np
import matplotlib.pyplot as plt

def polyFit(X, xs, doPlot = False):
    """
    Given a Nx2 array X of 2D coordinates, fit an N^th order polynomial
    and evaluate it at the coordinates in xs.
    This function assumes that all of the points have a unique X position
    """
    x = X[:, 0]
    y = X[:, 1]
    N = X.shape[0]
    A = np.zeros((N, N))
    for i in range(N):
        A[:, i] = x**i
    AInv = np.linalg.inv(A)
    b = AInv.dot(y[:, None])

    M = xs.size
    Y = np.zeros((M, 2))
    Y[:, 0] = xs
    for i in range(N):
        Y[:, 1] += b[i]*(xs**i)
    if doPlot:
        plt.plot(Y[:, 0], Y[:, 1], 'b')
        plt.hold(True)
        plt.scatter(X[:, 0], X[:, 1], 20, 'r')
        plt.show()
    return Y

if __name__ == '__main__':
    [x1, y1, x3, y3] = [100, 100, 101, 104]
    x2 = 0.5*(x1 + x3)
    y2 = 0.25*y1 + 0.75*y3
    xs = np.linspace(x1, x3, 50)
    X = np.array([[x1, y1], [x2, y2], [x3, y3]])
    Y = polyFit(X, xs, doPlot = False)
    plt.plot(Y[:, 0], Y[:, 1], 'k')
    plt.hold(True)
    plt.scatter(X[:, 0], X[:, 1], 20)
    plt.axis('equal')
    plt.show()
