# %%
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from common.plot_utils import plot, panel
from common.math_utils import Im
from CrI3.CrI3_model import NN, NNN, NNNN, FM_eigs_exact
from common.honeycomb_lattice import bzmesh, honeycomb_bz, bz_integration_honeycomb
from common.model_parameters import CrI3
import scienceplots


def FM_derivatives(k, J1=2.01, J2=0.16, J3=-0.08, D=0.31, Az=0.49, S=3/2):
    """
    First-order k-derivatives of the CrI3 magnon Hamiltonian at a single k.

    Parameters
    ----------
    k  : array_like, shape (2,)
    J1, J2, J3, D, Az, S : float  (same as FM_eigs_exact)

    Returns
    -------
    Hx : ndarray, shape (2, 2)  dH/dkx
    Hy : ndarray, shape (2, 2)  dH/dky
    """
    gamma_1x  =  1j * np.dot(NN.T[0],   np.exp(1j * k @ NN.T))   * J1 * S
    gamma_1y  =  1j * np.dot(NN.T[1],   np.exp(1j * k @ NN.T))   * J1 * S
    gamma_3x  =  1j * np.dot(NNNN.T[0], np.exp(1j * k @ NNNN.T)) * J3 * S
    gamma_3y  =  1j * np.dot(NNNN.T[1], np.exp(1j * k @ NNNN.T)) * J3 * S
    lambda_kx =     np.dot(NNN.T[0], np.cos(k @ NNN.T)) * 2 * D * S
    lambda_ky =     np.dot(NNN.T[1], np.cos(k @ NNN.T)) * 2 * D * S
    gamma_2x  = -np.dot(NNN.T[0], np.sin(k @ NNN.T)) * 2 * J2 * S
    gamma_2y  = -np.dot(NNN.T[1], np.sin(k @ NNN.T)) * 2 * J2 * S

    Hx = np.array([[-gamma_2x + lambda_kx, -gamma_1x - gamma_3x],
                   [-gamma_1x.conj() - gamma_3x.conj(), -gamma_2x - lambda_kx]])
    Hy = np.array([[-gamma_2y + lambda_ky, -gamma_1y - gamma_3y],
                   [-gamma_1y.conj() - gamma_3y.conj(), -gamma_2y - lambda_ky]])
    return Hx, Hy


def FM_berry_curvature(k, J1=2.01, J2=0.16, J3=-0.08, D=0.31, Az=0.49, S=3/2, m=0):
    """
    Berry curvature of band m at a single k-point.

    Parameters
    ----------
    k : array_like, shape (2,)
    m : int  band index  (0 = upper band epsilon_+, 1 = lower band epsilon_-)

    Returns
    -------
    berry_curvature : float
    """
    E,  Uk  = FM_eigs_exact(k, J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S)
    Hx, Hy  = FM_derivatives(k, J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S)
    Uik     = Uk.conj()          # rows = conjugate eigenvectors

    index    = np.array([0, 1])
    excluded = np.delete(index, m)

    berry_curvature = 0
    for n in excluded:
        berry_curvature += -Im(
            ((Uik[m] @ Hx @ Uk[n]) * (Uik[n] @ Hy @ Uk[m])
           - (Uik[m] @ Hy @ Uk[n]) * (Uik[n] @ Hx @ Uk[m]))
            / (E[n] - E[m])**2
        )
    return berry_curvature


def FM_quantum_metric(k, J1=2.01, J2=0.16, J3=-0.08, D=0.31, Az=0.49, S=3/2, m=0):
    """
    Quantum metric of band m at a single k-point.

    Parameters
    ----------
    k : array_like, shape (2,)
    m : int  band index  (0 = upper, 1 = lower)

    Returns
    -------
    quantum_metric : float
    """
    from math_utils import Re
    E,  Uk  = FM_eigs_exact(k, J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S)
    Hx, Hy  = FM_derivatives(k, J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S)
    Uik     = Uk.conj()

    index    = np.array([0, 1])
    excluded = np.delete(index, m)

    quantum_metric = 0
    for n in excluded:
        quantum_metric += Re(
            ((Uik[m] @ Hx @ Uk[n]) * (Uik[n] @ Hy @ Uk[m]))
            / (E[n] - E[m])**2
        )
    return quantum_metric


def get_FM_berry_curvature(J1=2.01, J2=0.16, J3=-0.08, D=0.31, Az=0.49, S=3/2, n=200):
    """
    Berry curvature over the full BZ on an (2n+1)×(2n+1) square mesh.

    Returns
    -------
    ndarray, shape (2, 2n+1, 2n+1)  [omega_upper, omega_lower]
    """
    kx, ky = bzmesh(n=n, m=2)
    omega_upper = np.zeros(kx.shape)
    omega_lower = np.zeros(kx.shape)
    kx0 = kx[0]
    ky0 = ky.T[0]
    for i in range(ky0.size):
        for j in range(kx0.size):
            k = np.array([kx0[j], ky0[i]])
            omega_upper[i, j] = FM_berry_curvature(k, J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S, m=0)
            omega_lower[i, j] = FM_berry_curvature(k, J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S, m=1)
        if i % 50 == 0:
            print(f'{i}: done  J1={J1}, J2={J2}, J3={J3}, D={D}, Az={Az}, S={S}')
    return np.array([omega_upper, omega_lower])


def get_FM_quantum_metric(J1=2.01, J2=0.16, J3=-0.08, D=0.31, Az=0.49, S=3/2, n=200):
    """
    Quantum metric over the full BZ on an (2n+1)×(2n+1) square mesh.

    Returns
    -------
    ndarray, shape (2, 2n+1, 2n+1)  [g_upper, g_lower]
    """
    kx, ky = bzmesh(n=n, m=2)
    g_upper = np.zeros(kx.shape)
    g_lower = np.zeros(kx.shape)
    kx0 = kx[0]
    ky0 = ky.T[0]
    for i in range(ky0.size):
        for j in range(kx0.size):
            k = np.array([kx0[j], ky0[i]])
            g_upper[i, j] = FM_quantum_metric(k, J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S, m=0)
            g_lower[i, j] = FM_quantum_metric(k, J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S, m=1)
        if i % 50 == 0:
            print(f'{i}: done  J1={J1}, J2={J2}, J3={J3}, D={D}, Az={Az}, S={S}')
    return np.array([g_upper, g_lower])


# %%  Computation and plotting  (only runs when executed directly)
if __name__ == '__main__':

    # %%  Compute Berry curvature over the BZ
    CrI3_berry_curvatures = get_FM_berry_curvature(**CrI3, n=200)

    honeycomb_bz_x, honeycomb_bz_y = honeycomb_bz()
    kx, ky = bzmesh(n=200, m=2)

    # %%  Plot lower band Berry curvature
    color_bar_title = [r"$\Omega_{+}$", r"$\Omega_{-}$"]
    pads = [7, 12]

    with plt.style.context(['science', 'ieee']):
        fig, ax = panel(figsize=(4, 3), nrows=1, ncols=1,
                        width_ratios=[1], height_ratios=[1],
                        hspace=0.1, wspace=0.4)
        fig.subplots_adjust(top=0.95, bottom=0.15, right=0.99)

        pc = ax.pcolormesh(kx, ky, CrI3_berry_curvatures[1], cmap='jet')
        plot(honeycomb_bz_x, honeycomb_bz_y, ax=ax, linestyle='-', linewidth=1, color='k')

        clb = fig.colorbar(pc, ax=ax, shrink=0.9)
        clb.ax.set_title(color_bar_title[1], loc='left', fontsize=24, pad=pads[1])
        clb.ax.tick_params(labelsize=24)

        ax.set_axis_on()
        ax.grid(False)
        ax.set_xticks([-0.5*2*np.pi, 0, 0.5*2*np.pi])
        ax.set_xticklabels(['-1', '0', '1'], fontsize=24)
        ax.set_yticks([-0.5*2*np.pi, 0, 0.5*2*np.pi])
        ax.set_yticklabels(['-1', '0', '1'], fontsize=24)
        ax.set_xlabel(r'$k_x(\pi/a)$', fontsize=28)
        ax.set_ylabel(r'$k_y(\pi/a)$', fontsize=28)
        plt.show()

    fig.savefig('figures/CrI3_berry_curvatures/CrI3_berry_curvature_lower.png',
                dpi=300, bbox_inches='tight')

    # %%  Plot upper band Berry curvature  (required by panel_plot.py)
    with plt.style.context(['science', 'ieee']):
        fig, ax = panel(figsize=(4, 3), nrows=1, ncols=1,
                        width_ratios=[1], height_ratios=[1],
                        hspace=0.1, wspace=0.4)
        fig.subplots_adjust(top=0.95, bottom=0.15, right=0.99)

        pc = ax.pcolormesh(kx, ky, CrI3_berry_curvatures[0], cmap='jet')
        plot(honeycomb_bz_x, honeycomb_bz_y, ax=ax, linestyle='-', linewidth=1, color='k')

        clb = fig.colorbar(pc, ax=ax, shrink=0.9)
        clb.ax.set_title(color_bar_title[0], loc='left', fontsize=24, pad=pads[0])
        clb.ax.tick_params(labelsize=24)

        ax.set_axis_on()
        ax.grid(False)
        ax.set_xticks([-0.5*2*np.pi, 0, 0.5*2*np.pi])
        ax.set_xticklabels(['-1', '0', '1'], fontsize=24)
        ax.set_yticks([-0.5*2*np.pi, 0, 0.5*2*np.pi])
        ax.set_yticklabels(['-1', '0', '1'], fontsize=24)
        ax.set_xlabel(r'$k_x(\pi/a)$', fontsize=28)
        ax.set_ylabel(r'$k_y(\pi/a)$', fontsize=28)
        plt.show()

    fig.savefig('figures/CrI3_berry_curvatures/CrI3_berry_curvature_upper.png',
                dpi=300, bbox_inches='tight')
