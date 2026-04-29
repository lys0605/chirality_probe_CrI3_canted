"""
CrI3_model.py
=============
Core CrI3 FM magnon Hamiltonian on the honeycomb lattice.

All energies are in meV; lattice constant a = 1.
Default parameters from Phys. Rev. X 8, 041028 (2018).

Constants
---------
NN    nearest-neighbour bond vectors,          shape (3, 2)
NNN   next-nearest-neighbour bond vectors,     shape (3, 2)
NNNN  next-next-nearest-neighbour bond vectors, shape (3, 2)

Functions
---------
FM_eigs_exact   Bogoliubov eigenpairs at a single k-point
"""

import numpy as np

# ---------------------------------------------------------------------------
# Honeycomb lattice bond vectors  (a = 1)
# ---------------------------------------------------------------------------
NN   = np.sqrt(3) * np.array([[ 0,    1/np.sqrt(3)],
                               [-1/2, -0.5/np.sqrt(3)],
                               [ 1/2, -0.5/np.sqrt(3)]])

NNN  = -np.sqrt(3) * np.array([[ 1/2,  np.sqrt(3)/2],
                                [-1,    0            ],
                                [ 1/2, -np.sqrt(3)/2]])

NNNN = np.array([[ 0,           -2],
                 [ np.sqrt(3),   1],
                 [-np.sqrt(3),   1]])


def FM_eigs_exact(k, J1=2.01, J2=0.16, J3=-0.08, D=0.31, Az=0.49, S=3/2):
    """
    Bogoliubov eigenpairs of the CrI3 FM magnon Hamiltonian at a single k.

    Parameters
    ----------
    k  : array_like, shape (2,)   reciprocal-space point
    J1 : float  nearest-neighbour Heisenberg coupling (meV)
    J2 : float  next-nearest-neighbour coupling (meV)
    J3 : float  next-next-nearest-neighbour coupling (meV)
    D  : float  Dzyaloshinskii-Moriya coupling (meV)
    Az : float  single-ion anisotropy (meV)
    S  : float  spin quantum number

    Returns
    -------
    E  : ndarray, shape (2,)    eigenvalues [epsilon_+, epsilon_-]
    Uk : ndarray, shape (2, 2)  eigenvectors stored as rows
    """
    M_ex     = 3*J1*S + 6*J2*S + 3*J3*S
    gamma_1  = J1  * S * np.sum(np.exp(1j * k @ NN.T))
    gamma_2  = 2*J2 * S * np.sum(np.cos(k @ NNN.T))
    gamma_3  = J3  * S * np.sum(np.exp(1j * k @ NNNN.T))
    lambda_k = 2*D  * S * np.sum(np.sin(k @ NNN.T))

    d0  = M_ex + 2*Az*S - gamma_2
    dx  = -np.real(gamma_1 + gamma_3)
    dy  =  np.imag(gamma_1 + gamma_3)
    dz  = lambda_k

    d_abs   = np.emath.sqrt(dx**2 + dy**2 + dz**2)
    dxy_abs = np.emath.sqrt(dx**2 + dy**2)

    ep = d0 + d_abs
    em = d0 - d_abs

    cos_h    = np.emath.sqrt((d_abs + dz) / (2 * d_abs))
    sin_h    = np.emath.sqrt((d_abs - dz) / (2 * d_abs))
    varphi_k = 1j * np.log((dx + 1j*dy) / dxy_abs)
    phase    = np.exp(1j * varphi_k / 2)
    phase_c  = np.conj(phase)

    u1 = np.array([ cos_h * phase,  sin_h * phase_c])
    u2 = np.array([-sin_h * phase,  cos_h * phase_c])

    return np.array([ep, em]), np.array([u1, u2])
