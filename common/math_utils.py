"""
math_utils.py
=============
Pure mathematical utilities shared across all calculation scripts.

Functions
---------
Im, Re                  vectorised imaginary / real part extraction
is_invertible           check if a square matrix is invertible
print_matrix            LaTeX-formatted matrix display (Jupyter)
normalize               normalise an array by its max absolute value
gaussian_function       normalised 1-D Gaussian lineshape
lorentzian_function     normalised 1-D Lorentzian lineshape
"""

import numpy as np
from IPython.display import display, Math


# ---------------------------------------------------------------------------
# Complex-number helpers
# ---------------------------------------------------------------------------

def Imaginary(z):
    return z.imag

Im = np.vectorize(Imaginary)


def Real(z):
    return z.real

Re = np.vectorize(Real)


# ---------------------------------------------------------------------------
# Linear-algebra helpers
# ---------------------------------------------------------------------------

def is_invertible(matrix):
    """Return True if *matrix* is square and full-rank."""
    return (matrix.shape[0] == matrix.shape[1] and
            np.linalg.matrix_rank(matrix) == matrix.shape[0])


def print_matrix(array):
    """Display *array* as a LaTeX bmatrix in a Jupyter notebook."""
    matrix = ''
    for row in array:
        try:
            for number in row:
                matrix += f'{number}&'
        except TypeError:
            matrix += f'{row}&'
        matrix = matrix[:-1] + r'\\'
    display(Math(r'\begin{bmatrix}' + matrix + r'\end{bmatrix}'))


# ---------------------------------------------------------------------------
# Array normalisation
# ---------------------------------------------------------------------------

def normalize(x):
    """Return *x* divided by its maximum absolute value."""
    return x / np.max(np.abs(x))


# ---------------------------------------------------------------------------
# Spectral lineshapes
# ---------------------------------------------------------------------------

def gaussian_function(x, x0=0, width=1e-3):
    """
    Normalised 1-D Gaussian.

    Parameters
    ----------
    x     : array_like  evaluation points
    x0    : float       centre (default 0)
    width : float       standard deviation σ (default 1e-3)

    Returns
    -------
    ndarray  value of  (1 / (σ√(2π))) exp(−½ ((x−x0)/σ)²)
    """
    return 1 / (width * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - x0) / width) ** 2)


def lorentzian_function(x, x0=0, width=1e-3):
    """
    Normalised 1-D Lorentzian.

    Parameters
    ----------
    x     : array_like  evaluation points
    x0    : float       centre (default 0)
    width : float       half-width at half-maximum η (default 1e-3)

    Returns
    -------
    ndarray  value of  (η/π) / ((x−x0)² + η²)
    """
    return (width / np.pi) / ((x - x0) ** 2 + width ** 2)
