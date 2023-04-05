import numpy as np


def polynomial_kernel(X, Y, c, p):
    """
        Computes the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """

    matrix = np.matmul(X, np.transpose(Y)) + c
    kernel_matrix = np.power(matrix, p)
    return kernel_matrix



def rbf_kernel(X, Y, gamma):
    """
        Computes the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """

    X_norm = np.sum(X ** 2, axis=-1)
    Y_norm = np.sum(Y ** 2, axis=-1)
    kernel_matrix = np.exp(-gamma * (X_norm[:, None] + Y_norm[None, :] - 2 * np.dot(X, np.transpose(Y))))

    return kernel_matrix
