"""
parameters.py
=============
Centralised material parameters and physical constants for all calculations.

Import with:
    from parameters import CrI3, CANTED_AFM, k_B_meV, k_B_eV
"""

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
k_B_meV = 8.617333262145e-2   # Boltzmann constant in meV / K
k_B_eV  = 8.617333262145e-5   # Boltzmann constant in eV  / K

# ---------------------------------------------------------------------------
# CrI3 — ferromagnetic honeycomb insulator (energies in meV)
# Reference: J. Phys. Chem. Lett. 12, 6 (2021)
# ---------------------------------------------------------------------------
CrI3 = dict(
    J1 =  2.01,   # meV  nearest-neighbour Heisenberg coupling
    J2 =  0.16,   # meV  next-nearest-neighbour Heisenberg coupling
    J3 = -0.08,   # meV  next-next-nearest-neighbour Heisenberg coupling
    D  =  0.31,   # meV  Dzyaloshinskii–Moriya interaction (DMI)
    Az =  0.49,   # meV  single-ion anisotropy
    S  =  3/2,    #      spin quantum number
)

# ---------------------------------------------------------------------------
# Canted antiferromagnet on honeycomb lattice (energies normalised to J)
# All exchange and DMI values are given as ratios relative to J.
# ---------------------------------------------------------------------------
CANTED_AFM = dict(
    J  = 1.0,    # meV  Heisenberg exchange (sets the energy scale)
    D  = 0.1,    # meV  DMI  (D/J = 0.1)
    S  = 5/2,    #      spin quantum number
)
