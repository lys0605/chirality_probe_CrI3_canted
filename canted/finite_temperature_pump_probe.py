# %%
# NOTE: This script is incomplete (work in progress).
# Goal: finite-temperature RCD spectra chi(omega, T) for the canted AFM,
# analogous to CrI3/CrI3_pump_probe.py for the ferromagnet.
# Several sections are not yet functional and are commented out with TODO/FIXME notes.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from common.plot_utils import (plot, letter_annotation, panel,
                               plot_frequency_resolved_RCD,
                               plot_frequency_temperature_resolved_RCD)
from common.math_utils import gaussian_function, lorentzian_function, normalize
from common.bose_statistics import bose_einstein, boltzmann_factor, occupation_function
from common.honeycomb_lattice import *
from canted.canted_raman_cross_section import *
import scienceplots


def partition_function(energy_array, T=0, mu=0):
    """
    Grand-canonical partition function for bosonic magnons on the BZ.

    Z = exp( ∫ d²k/(2π)² ln(1 / (1 − exp(−E_k / k_B T))) )

    Parameters
    ----------
    energy_array : ndarray  magnon energies on the BZ k-mesh
    T            : float    temperature in K
    mu           : float    chemical potential (default 0)
    """
    return np.exp(
        bz_integration_honeycomb(
            np.log(1 / (1 - boltzmann_factor(energy_array, T=T, mu=mu)))
        ) / (2 * np.pi) ** 2
    )


# %% Canted AFM — Raman arrays for each canting value B/Bs
J = 1
D = 0.1
S = 5/2
s_values = [0, 0.25, 0.5, 0.75, 1]

raman_array = [get_raman_cross_section_exact(J=J, D=D, S=S, B0=s) for s in s_values]

len_s = len(s_values)
cross_section_array = np.array([raman_array[i][0] for i in range(len_s)])  # (len_s, 6, N, N)
berry_array         = np.array([raman_array[i][1] for i in range(len_s)])  # (len_s, 2, N, N)
berry_rcd_array     = np.array([raman_array[i][2] for i in range(len_s)])
energy_array        = np.array([raman_array[i][3] for i in range(len_s)])  # (len_s, 2, N, N)
# energy_array[j][0] = upper band for canting j
# energy_array[j][1] = lower band for canting j

# %%
w      = np.linspace(0, 25, 200)
len_w  = len(w)
width  = (w[1] - w[0]) * 2

# %% Temperature grid
temperatures = [0, 2, 4, 6, 24, 58]
len_T        = len(temperatures)
temp         = temperatures[3]   # single temperature used in the fixed-T chi loops below

# TODO: partition_array should be computed per (canting j, temperature T) pair for the
# full canted finite-T calculation.  Current placeholder computes it for a single
# canting (j=0) across all temperatures, mirroring the CrI3 pattern.
# Generalise to shape (len_s, len_T) when implementing the full sweep.
partition_array = np.array([
    partition_function(energy_array[0][0], T=T) * partition_function(energy_array[0][1], T=T)
    for T in temperatures
])

# %% chi at fixed temperature `temp`, varying canting j
# Convention for cross_section_array[j] channel indices (from canted_raman_cross_section):
#   [0] two-magnon upper,  [1] two-magnon lower
#   [2] FM upper,          [3] FM lower
#   [4] AFM upper,         [5] AFM lower
chi_two_magnons_lower_T = np.zeros((len_s, len_w))
chi_two_magnons_upper_T = np.zeros((len_s, len_w))
chi_AFM_lower_T         = np.zeros((len_s, len_w))
chi_AFM_upper_T         = np.zeros((len_s, len_w))
chi_FM_lower_T          = np.zeros((len_s, len_w))
chi_FM_upper_T          = np.zeros((len_s, len_w))

for j in range(len_s):
    for i in range(len_w):
        weight = boltzmann_factor(energy_array[j][1], T=temp)
        chi_two_magnons_lower_T[j][i] = bz_integration_honeycomb(weight * cross_section_array[j][1] * gaussian_function(w[i], x0=2 * energy_array[j][1],                                   width=width))
        chi_two_magnons_upper_T[j][i] = bz_integration_honeycomb(weight * cross_section_array[j][0] * gaussian_function(w[i], x0=2 * energy_array[j][0],                                   width=width))
        chi_AFM_lower_T[j][i]         = bz_integration_honeycomb(weight * cross_section_array[j][5] * gaussian_function(w[i], x0=energy_array[j][1] + energy_array[j][0],                 width=width))
        chi_AFM_upper_T[j][i]         = bz_integration_honeycomb(weight * cross_section_array[j][4] * gaussian_function(w[i], x0=energy_array[j][1] + energy_array[j][0],                 width=width))
        chi_FM_lower_T[j][i]          = bz_integration_honeycomb(weight * cross_section_array[j][3] * gaussian_function(w[i], x0=np.abs(energy_array[j][1] - energy_array[j][0]),         width=width))
        chi_FM_upper_T[j][i]          = bz_integration_honeycomb(weight * cross_section_array[j][2] * gaussian_function(w[i], x0=np.abs(energy_array[j][1] - energy_array[j][0]),         width=width))

# %% Bare DOS
dos_two_magnons_lower = np.zeros((len_s, len_w))
dos_two_magnons_upper = np.zeros((len_s, len_w))
dos_AFM               = np.zeros((len_s, len_w))
dos_FM                = np.zeros((len_s, len_w))

for j in range(len_s):
    for i in range(len_w):
        dos_two_magnons_lower[j][i] = bz_integration_honeycomb(gaussian_function(w[i], x0=2 * energy_array[j][1],                               width=width))
        dos_two_magnons_upper[j][i] = bz_integration_honeycomb(gaussian_function(w[i], x0=2 * energy_array[j][0],                               width=width))
        dos_AFM[j][i]               = bz_integration_honeycomb(gaussian_function(w[i], x0=energy_array[j][1] + energy_array[j][0],              width=width))
        dos_FM[j][i]                = bz_integration_honeycomb(gaussian_function(w[i], x0=np.abs(energy_array[j][1] - energy_array[j][0]),      width=width))

# %% Berry-curvature-weighted DOS
dos_weighted_two_magnons_lower = np.zeros((len_s, len_w))
dos_weighted_two_magnons_upper = np.zeros((len_s, len_w))
dos_weighted_AFM_upper         = np.zeros((len_s, len_w))
dos_weighted_AFM_lower         = np.zeros((len_s, len_w))
dos_weighted_FM                = np.zeros((len_s, len_w))

for j in range(len_s):
    for i in range(len_w):
        dos_weighted_two_magnons_lower[j][i] = bz_integration_honeycomb(berry_array[j][1] * gaussian_function(w[i], x0=2 * energy_array[j][1],                               width=width))
        dos_weighted_two_magnons_upper[j][i] = bz_integration_honeycomb(berry_array[j][0] * gaussian_function(w[i], x0=2 * energy_array[j][0],                               width=width))
        dos_weighted_AFM_upper[j][i]         = bz_integration_honeycomb(berry_array[j][0] * gaussian_function(w[i], x0=energy_array[j][1] + energy_array[j][0],              width=width))
        dos_weighted_AFM_lower[j][i]         = bz_integration_honeycomb(berry_array[j][1] * gaussian_function(w[i], x0=energy_array[j][1] + energy_array[j][0],              width=width))
        dos_weighted_FM[j][i]                = bz_integration_honeycomb(berry_array[j][1] * gaussian_function(w[i], x0=np.abs(energy_array[j][1] - energy_array[j][0]),      width=width))

# %%
rcd_label = [r'$\chi_{\alpha^{\prime}\bar{\alpha}^\prime}$', r'$\chi_{\beta^{\prime}\bar{\beta}^\prime}$',
             r'$\chi_{\alpha^{\prime}\beta}$', r'$\chi_{\bar{\beta}^{\prime}\bar{\alpha}}$',
             r'$\chi_{\alpha^{\prime}\bar{\beta}^\prime}$', r'$\chi_{\beta^{\prime}\alpha^\prime}$']

# TODO: compute chi_vacuum, chi_one_magnon, and their dos_weighted counterparts,
# then divide by partition_array to get the proper finite-T normalised spectra.
# chi_vacuum       = chi_two_magnons_lower_T + chi_two_magnons_upper_T + chi_AFM_lower_T + chi_AFM_upper_T
# chi_one_magnon   = 2*chi_two_magnons_lower_T + chi_two_magnons_upper_T + 2*chi_AFM_lower_T + chi_AFM_upper_T + chi_FM_upper_T
# dos_weighted_vacuum     = dos_weighted_two_magnons_lower + dos_weighted_two_magnons_upper + dos_weighted_AFM_lower + dos_weighted_AFM_upper
# dos_weighted_one_magnon = 2*dos_weighted_two_magnons_lower + dos_weighted_two_magnons_upper + 2*dos_weighted_AFM_lower + dos_weighted_AFM_upper + dos_weighted_FM

colors_vacuum      = ['#00215d', '#0071bc', '#8fd0ff']
colors_one_magnon  = ['#b6000f', '#ff814b', '#ffc883']
colors_temperature = ['#1a1a1a', '#192225', '#143349', '#245074', '#2f6fa4', '#3989d6']

# %% Plot chi(omega, T) — FM channel at fixed canting, one curve per canting j
# TODO: replace the canting sweep with a proper temperature sweep (loop over temperatures)
# to produce chi(omega, T) at a fixed canting, mirroring CrI3/CrI3_pump_probe.py.
with plt.style.context(['science', 'ieee']):
    fig, ax = plt.subplots(figsize=(4, 3))
    plot_frequency_temperature_resolved_RCD(ax, w, chi_FM_lower_T, temperatures, plot_length=len_T, color=colors_temperature, label=r'')
    ax.set_xlabel(r'$\hslash\omega/J$', fontsize=13)
    ax.set_ylabel(r'$\chi(\omega, T)$', fontsize=13)
    ax.legend(loc="lower right", bbox_to_anchor=(1.7, -0.2), fontsize=13)
    plt.show()
    # fig.savefig('figures/pump_probe/canted_TRCD_FM_vary_T.png', dpi=300, bbox_inches='tight')

# %% Plot chi^(0) and chi^(1) for three canting values
# FIXME: chi_vacuum and chi_one_magnon must be computed above before enabling this block
# with plt.style.context(['science', 'ieee']):
#     fig = plt.figure(figsize=(4, 6))
#     gs = fig.add_gridspec(2, hspace=0)
#     axes = gs.subplots(sharex=True)
#     for ax in axes:
#         ax.set_box_aspect(0.75)
#     plot_frequency_resolved_RCD(axes[0], w, chi_vacuum[1:4],     s_values[0:3], plot_length=3, color=colors_vacuum,     label=r'')
#     plot_frequency_resolved_RCD(axes[1], w, chi_one_magnon[1:4], s_values[0:3], plot_length=3, color=colors_one_magnon, label=r'')
#     axes[1].set_xlabel(r'$\hslash\omega/J$', fontsize=15)
#     axes[0].set_ylabel(r'$\chi^{(0)}(\omega)$', fontsize=15)
#     axes[1].set_ylabel(r'$\chi^{(1)}(\omega)$', fontsize=15)
#     axes[0].legend(loc="lower left", fontsize=13)
#     axes[1].legend(loc="lower left", fontsize=13)
#     plt.show()
#     # fig.savefig('figures/pump_probe/canted_TRCD_vacuum_one_magnon.png', dpi=300, bbox_inches='tight')

# %% Normalised chi^(0), chi^(1), weighted DOS at two canting values
# FIXME: requires chi_vacuum, chi_one_magnon, dos_weighted_one_magnon defined above
# with plt.style.context(['science', 'ieee']):
#     fig = plt.figure(figsize=(4, 6))
#     gs = fig.add_gridspec(2, hspace=0)
#     axes = gs.subplots(sharex=True)
#     for ax in axes:
#         ax.set_box_aspect(0.75)
#     plot_frequency_resolved_RCD(axes[0], w, normalize(chi_vacuum[1]),           s_values[1], plot_length=1, color=colors_vacuum[-1],     label=r'$\chi^{(0)}(\omega)$')
#     plot_frequency_resolved_RCD(axes[0], w, normalize(chi_one_magnon[1]),       s_values[1], plot_length=1, color=colors_one_magnon[-1], label=r'$\chi^{(1)}(\omega)$')
#     plot_frequency_resolved_RCD(axes[0], w, normalize(dos_weighted_one_magnon[1]), s_values[1], ls='--', plot_length=1, color='grey',    label=r'$\Lambda$')
#     plot_frequency_resolved_RCD(axes[1], w, normalize(chi_vacuum[3]),           s_values[3], plot_length=1, color=colors_vacuum[-1],     label=r'$\chi^{(0)}(\omega)$')
#     plot_frequency_resolved_RCD(axes[1], w, normalize(chi_one_magnon[3]),       s_values[3], plot_length=1, color=colors_one_magnon[-1], label=r'$\chi^{(1)}(\omega)$')
#     plot_frequency_resolved_RCD(axes[1], w, normalize(dos_weighted_one_magnon[3]), s_values[3], ls='--', plot_length=1, color='black',   label=r'$\Lambda^\prime$')
#     axes[1].set_xlabel(r'$\hslash\omega/J$', fontsize=15)
#     axes[0].set_ylabel(r'', fontsize=15)
#     axes[1].set_ylabel(r'', fontsize=15)
#     axes[0].legend(loc="lower left", fontsize=13)
#     axes[1].legend(loc="lower left", fontsize=13)
#     plt.show()
#     # fig.savefig('figures/pump_probe/canted_TRCD_vacuum_2dos.png', dpi=300, bbox_inches='tight')
# %%
