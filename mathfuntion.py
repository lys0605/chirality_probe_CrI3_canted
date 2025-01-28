import numpy as np
from IPython.display import display, Math

def Imaginary(z):
    return z.imag
Im = np.vectorize(Imaginary)

def Real(z):
    return z.real
Re = np.vectorize(Real)

def is_invertible(matrix):
    return matrix.shape[0] == matrix.shape[1] and np.linalg.matrix_rank(matrix) == matrix.shape[0]

def print_matrix(array):
    matrix = ''
    for row in array:
        try:
            for number in row:
                matrix += f'{number}&'
        except TypeError:
            matrix += f'{row}&'
        matrix = matrix[:-1] + r'\\'
    display(Math(r'\begin{bmatrix}'+matrix+r'\end{bmatrix}'))