"""
bose_statistics.py
==================
Temperature-dependent statistical functions for magnon calculations.

All energies are assumed to be in **meV** and temperatures in **Kelvin**,
consistent with the rest of the codebase (J ~ 1 meV energy scale).
k_B = 8.617333262145e-2 meV / K is used throughout.

Functions
---------
bose_einstein       Bose–Einstein occupation number n(E, T)
boltzmann_factor    Boltzmann weight exp(−E / k_B T)
occupation_function 1 / (1 − boltzmann_factor)  (bosonic)
"""

import numpy as np
from .model_parameters import k_B_meV

# Expose the constant so callers can verify the unit convention
k_B = k_B_meV   # meV / K


def bose_einstein(E, T, mu=0):
    """
    Bose–Einstein distribution.

    Parameters
    ----------
    E  : array_like  energy in meV
    T  : float       temperature in K  (must be > 0)
    mu : float       chemical potential in meV (default 0)

    Returns
    -------
    ndarray   1 / (exp((E − μ) / k_B T) − 1)
    """
    return 1 / (np.exp((E - mu) / (k_B * T)) - 1)


def boltzmann_factor(E, T, mu=0):
    """
    Boltzmann weight exp(−(E − μ) / k_B T).

    Returns 0 everywhere when T == 0 (ground state limit).

    Parameters
    ----------
    E  : array_like  energy in meV
    T  : float       temperature in K
    mu : float       chemical potential in meV (default 0)
    """
    if T == 0:
        return 0 * np.asarray(E)
    return np.exp(-(E - mu) / (k_B * T))


def occupation_function(E, T, mu=0):
    """
    Bosonic occupation  1 / (1 − boltzmann_factor(E, T, mu)).

    Parameters
    ----------
    E  : array_like  energy in meV
    T  : float       temperature in K
    mu : float       chemical potential in meV (default 0)
    """
    return 1 / (1 - boltzmann_factor(E, T=T, mu=mu))
