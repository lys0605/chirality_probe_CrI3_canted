# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from plot_utils import plot
from CrI3_model import NN, NNN, NNNN
from honeycomb_lattice import (get_kvectors, get_path, get_total_path,
                                group_kvectors, GAMMA, K, K_PRIME, M_POINT)
from parameters import CrI3
import scienceplots

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['axes.labelsize']  = 24

# High-symmetry point aliases matching standard paper notation
K1 = K
K2 = K_PRIME
M  = M_POINT


def magnon_energy(k, J1=2.01, J2=0.16, J3=-0.08, D=0.31, Az=0.49, S=3/2):
    """
    CrI3 FM magnon dispersion over an array of k-points.

    Parameters
    ----------
    k  : ndarray, shape (N, 2)  array of reciprocal-space points
    J1 : float  nearest-neighbour Heisenberg coupling (meV)
    J2 : float  next-nearest-neighbour coupling (meV)
    J3 : float  next-next-nearest-neighbour coupling (meV)
    D  : float  Dzyaloshinskii-Moriya coupling (meV)
    Az : float  single-ion anisotropy (meV)
    S  : float  spin quantum number

    Returns
    -------
    energy : ndarray, shape (2, N)   band energies [epsilon_+, epsilon_-]
    """
    M_ex     = 3*J1*S + 6*J2*S + 3*J3*S
    gamma_1  = J1  * S * np.sum(np.exp(1j * k @ NN.T),   axis=1)
    gamma_2  = 2*J2 * S * np.sum(np.cos(k @ NNN.T),      axis=1)
    gamma_3  = J3  * S * np.sum(np.exp(1j * k @ NNNN.T), axis=1)
    lambda_k = 2*D  * S * np.sum(np.sin(k @ NNN.T),      axis=1)

    d0    = M_ex + 2*Az*S - gamma_2
    dx    = -np.real(gamma_1 + gamma_3)
    dy    =  np.imag(gamma_1 + gamma_3)
    dz    = lambda_k
    d_abs = np.emath.sqrt(dx**2 + dy**2 + dz**2)

    ep = d0 + d_abs
    em = d0 - d_abs
    return np.array([ep, em])


# %%  Build k-path:  K' → Γ → K → M → K' → Γ
k0 = get_kvectors(-1*K1, GAMMA)
k1 = get_kvectors(GAMMA, K1)
k2 = get_kvectors(K1, M, num=51)
k3 = get_kvectors(M, K2, num=51)
k4 = get_kvectors(K2, GAMMA)
k_vectors = group_kvectors(k0, k1, k2, k3, k4)

# %%  Compute band structure
magnon_bands = magnon_energy(k_vectors, **CrI3)

K2_Gamma   = get_path(k0)
Gamma_K1   = get_path(k1)[1:]
K1_M       = get_path(k2)[1:]
M_K2       = get_path(k3)[1:]
K2_Gamma_2 = get_path(k4)[1:]

k_label = [r"K$^\prime$", r"$\Gamma$", r"K", "M", r"K$^\prime$", r"$\Gamma$"]
path, k_index = get_total_path(K2_Gamma, Gamma_K1, K1_M, M_K2, K2_Gamma_2)

blue_colors = ['#8fd0ff', '#589fef', '#0071bc', '#00468b', '#00215d']
red_colors  = ['#ffc883', '#ff9f4d', '#ff6f00', '#c94c00', '#7f2e00']

# %%  Plot
with plt.style.context('science'):
    fig, ax = plt.subplots(figsize=(6, 4))
    plot(np.arange(len(path)), magnon_bands[1], ax=ax,
         color=blue_colors[0], linestyle='-', linewidth=2.5,
         label=r"$\epsilon_{\bf k -}$")
    plot(np.arange(len(path)), magnon_bands[0], ax=ax,
         color=red_colors[0],  linestyle='-', linewidth=2.5,
         label=r"$\epsilon_{\bf k +}$")

    for i in range(len(k_index) - 2):
        ax.axvline(k_index[i + 1], color='black', ls='-', linewidth=1.0)

    ax.set_xticks(k_index, k_label, fontsize=24)
    ax.yaxis.grid()
    ax.set_xlim(0, len(path))
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels(['' if t == 0 else str(int(t)) for t in yticks], fontsize=28)
    ax.set_ylim(0, 20)
    ax.set_in_layout(True)
    ax.legend(loc='lower center', bbox_to_anchor=(0.62, 0.01), fontsize=28, frameon=True)
    ax.set_ylabel(r"$\epsilon$ (meV)", fontsize=32)
    plt.show()

fig.savefig('figures/CrI3_bands/CrI3_band_structure.png', dpi=300, bbox_inches='tight')

# %%
print(np.min(magnon_bands[1]))
