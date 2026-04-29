# %%
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import plot, panel
from CrI3_model import NN, NNN, NNNN, FM_eigs_exact
from honeycomb_lattice import bzmesh, honeycomb_bz, bz_integration_honeycomb
from parameters import CrI3
import scienceplots


def raman_cross_section_ham(k, qq=0, J1=2.01, J2=0.16, J3=-0.08, D=0.31, Az=0.49, S=3/2):
    """
    Raman Hamiltonian matrix elements at a single k-point.

    Computes H_R^{ss'} = 0.5*(L_xx + ss'*L_yy + i(s'-s)*L_xy) and rotates
    it to the magnon eigenbasis.

    Parameters
    ----------
    k  : array_like, shape (2,)
    qq : int  polarisation channel:
              0 -> RL (L-in, R-out)
              1 -> LL
              2 -> RR
              3 -> LR (R-in, L-out)
    J1, J2, J3, D, Az, S : float  (same as FM_eigs_exact)

    Returns
    -------
    Hr : ndarray, shape (2, 2)  Raman Hamiltonian in the eigenbasis
    """
    _, Uk_rows = FM_eigs_exact(k, J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S)
    Uk  = Uk_rows.T       # columns = eigenvectors
    Uik = Uk.conj().T     # rows    = bra vectors

    # Second k-derivatives of the Hamiltonian matrix elements
    gamma_1xx = -np.dot(NN.T[0]   * NN.T[0],   np.exp(1j * k @ NN.T))   * J1 * S
    gamma_1xy = -np.dot(NN.T[0]   * NN.T[1],   np.exp(1j * k @ NN.T))   * J1 * S
    gamma_1yy = -np.dot(NN.T[1]   * NN.T[1],   np.exp(1j * k @ NN.T))   * J1 * S
    gamma_3xx = -np.dot(NNNN.T[0] * NNNN.T[0], np.exp(1j * k @ NNNN.T)) * J3 * S
    gamma_3xy = -np.dot(NNNN.T[0] * NNNN.T[1], np.exp(1j * k @ NNNN.T)) * J3 * S
    gamma_3yy = -np.dot(NNNN.T[1] * NNNN.T[1], np.exp(1j * k @ NNNN.T)) * J3 * S
    lambda_kxx = -np.dot(NNN.T[0] * NNN.T[0],  np.sin(k @ NNN.T)) * 2 * D  * S
    lambda_kxy = -np.dot(NNN.T[0] * NNN.T[1],  np.sin(k @ NNN.T)) * 2 * D  * S
    lambda_kyy = -np.dot(NNN.T[1] * NNN.T[1],  np.sin(k @ NNN.T)) * 2 * D  * S
    gamma_2xx  = -np.dot(NNN.T[0] * NNN.T[0],  np.cos(k @ NNN.T)) * 2 * J2 * S
    gamma_2xy  = -np.dot(NNN.T[0] * NNN.T[1],  np.cos(k @ NNN.T)) * 2 * J2 * S
    gamma_2yy  = -np.dot(NNN.T[1] * NNN.T[1],  np.cos(k @ NNN.T)) * 2 * J2 * S

    L_xx = np.array([[-gamma_2xx + lambda_kxx, -gamma_1xx - gamma_3xx],
                     [-gamma_1xx.conj() - gamma_3xx.conj(), -gamma_2xx - lambda_kxx]])
    L_xy = np.array([[-gamma_2xy + lambda_kxy, -gamma_1xy - gamma_3xy],
                     [-gamma_1xy.conj() - gamma_3xy.conj(), -gamma_2xy - lambda_kxy]])
    L_yy = np.array([[-gamma_2yy + lambda_kyy, -gamma_1yy - gamma_3yy],
                     [-gamma_1yy.conj() - gamma_3yy.conj(), -gamma_2yy - lambda_kyy]])

    # Raman Hamiltonians for each circular-polarisation channel
    # Convention: L = +1, R = -1;  H_R^{ss'} = 0.5*(L_xx + ss'*L_yy + i(s'-s)*L_xy)
    L, R = 1, -1
    ram = {
        0: 0.5 * (L_xx + R*L*L_yy + 1j*(L - R)*L_xy),   # RL
        1: 0.5 * (L_xx + L*L*L_yy + 1j*(L - L)*L_xy),   # LL
        2: 0.5 * (L_xx + R*R*L_yy + 1j*(R - R)*L_xy),   # RR
        3: 0.5 * (L_xx + L*R*L_yy + 1j*(R - L)*L_xy),   # LR
    }
    return Uik @ ram[qq] @ Uk


def get_raman_cross_section(J1=2.01, J2=0.16, J3=-0.08, D=0.31, Az=0.49, S=3/2):
    """
    Raman cross-section |t_{+-}^{ss'}|^2 over the full BZ for all four
    circular-polarisation channels.

    Returns
    -------
    ndarray, shape (4, 2n+1, 2n+1)
        [p_RL, p_LL, p_RR, p_LR]  interband (+ -> -) matrix elements squared
    """
    kx, ky = bzmesh(m=2)
    p_RL = np.zeros(kx.shape)
    p_LL = np.zeros(kx.shape)
    p_RR = np.zeros(kx.shape)
    p_LR = np.zeros(kx.shape)
    kx0 = kx[0]
    ky0 = ky.T[0]
    for i in range(ky0.size):
        for j in range(kx0.size):
            k = np.array([kx0[j], ky0[i]])
            Hr_RL = raman_cross_section_ham(k, qq=0, J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S)
            Hr_LL = raman_cross_section_ham(k, qq=1, J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S)
            Hr_RR = raman_cross_section_ham(k, qq=2, J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S)
            Hr_LR = raman_cross_section_ham(k, qq=3, J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S)
            p_RL[i, j] = np.abs(Hr_RL[0, 1])**2
            p_LL[i, j] = np.abs(Hr_LL[0, 1])**2
            p_RR[i, j] = np.abs(Hr_RR[0, 1])**2
            p_LR[i, j] = np.abs(Hr_LR[0, 1])**2
        if i % 50 == 0:
            print(f'{i}: done  J1={J1}, J2={J2}, J3={J3}, D={D}, Az={Az}, S={S}')
    return np.array([p_RL, p_LL, p_RR, p_LR])


def get_RCD(J1=2.01, J2=0.16, J3=-0.08, D=0.31, Az=0.49, S=3/2):
    """
    Raman circular dichroism (RCD) over the full BZ.

    RCD_k = 2 Im[(L_tilde_xx - L_tilde_yy)_{01} * (L_tilde_xy)_{01}^*]

    Returns
    -------
    RCD : ndarray, shape (2n+1, 2n+1)
    """
    kx, ky = bzmesh(m=2)
    RCD = np.zeros(kx.shape)
    kx0 = kx[0]
    ky0 = ky.T[0]
    for i in range(ky0.size):
        for j in range(kx0.size):
            k = np.array([kx0[j], ky0[i]])

            _, Uk_rows = FM_eigs_exact(k, J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S)
            Uk  = Uk_rows.T
            Uik = Uk.conj().T

            gamma_1xx = -np.dot(NN.T[0]   * NN.T[0],   np.exp(1j * k @ NN.T))   * J1 * S
            gamma_1xy = -np.dot(NN.T[0]   * NN.T[1],   np.exp(1j * k @ NN.T))   * J1 * S
            gamma_1yy = -np.dot(NN.T[1]   * NN.T[1],   np.exp(1j * k @ NN.T))   * J1 * S
            gamma_3xx = -np.dot(NNNN.T[0] * NNNN.T[0], np.exp(1j * k @ NNNN.T)) * J3 * S
            gamma_3xy = -np.dot(NNNN.T[0] * NNNN.T[1], np.exp(1j * k @ NNNN.T)) * J3 * S
            gamma_3yy = -np.dot(NNNN.T[1] * NNNN.T[1], np.exp(1j * k @ NNNN.T)) * J3 * S
            lambda_kxx = -np.dot(NNN.T[0] * NNN.T[0],  np.sin(k @ NNN.T)) * 2 * D  * S
            lambda_kxy = -np.dot(NNN.T[0] * NNN.T[1],  np.sin(k @ NNN.T)) * 2 * D  * S
            lambda_kyy = -np.dot(NNN.T[1] * NNN.T[1],  np.sin(k @ NNN.T)) * 2 * D  * S
            gamma_2xx  = -np.dot(NNN.T[0] * NNN.T[0],  np.cos(k @ NNN.T)) * 2 * J2 * S
            gamma_2xy  = -np.dot(NNN.T[0] * NNN.T[1],  np.cos(k @ NNN.T)) * 2 * J2 * S
            gamma_2yy  = -np.dot(NNN.T[1] * NNN.T[1],  np.cos(k @ NNN.T)) * 2 * J2 * S

            L_xx = np.array([[-gamma_2xx + lambda_kxx, -gamma_1xx - gamma_3xx],
                              [-gamma_1xx.conj() - gamma_3xx.conj(), -gamma_2xx - lambda_kxx]])
            L_xy = np.array([[-gamma_2xy + lambda_kxy, -gamma_1xy - gamma_3xy],
                              [-gamma_1xy.conj() - gamma_3xy.conj(), -gamma_2xy - lambda_kxy]])
            L_yy = np.array([[-gamma_2yy + lambda_kyy, -gamma_1yy - gamma_3yy],
                              [-gamma_1yy.conj() - gamma_3yy.conj(), -gamma_2yy - lambda_kyy]])

            L_tilde_xx = Uik @ L_xx @ Uk
            L_tilde_xy = Uik @ L_xy @ Uk
            L_tilde_yy = Uik @ L_yy @ Uk

            RCD[i, j] = 2 * np.imag(
                (L_tilde_xx[0, 1] - L_tilde_yy[0, 1]) * L_tilde_xy[0, 1].conj()
            )
        if i % 50 == 0:
            print(f'{i}: done  J1={J1}, J2={J2}, J3={J3}, D={D}, Az={Az}, S={S}')
    return RCD


def get_energy(J1=2.01, J2=0.16, J3=-0.08, D=0.31, Az=0.49, S=3/2):
    """
    Magnon band energies over the full BZ.

    Returns
    -------
    ndarray, shape (2, 2n+1, 2n+1)   [epsilon_+, epsilon_-]
    """
    kx, ky = bzmesh(m=2)
    ep = np.zeros(kx.shape)
    em = np.zeros(kx.shape)
    kx0 = kx[0]
    ky0 = ky.T[0]
    for i in range(ky0.size):
        for j in range(kx0.size):
            k = np.array([kx0[j], ky0[i]])
            E, _ = FM_eigs_exact(k, J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S)
            ep[i, j] = E[0]
            em[i, j] = E[1]
        if i % 50 == 0:
            print(f'{i}: done  J1={J1}, J2={J2}, J3={J3}, D={D}, Az={Az}, S={S}')
    return np.array([ep, em])


# %%  Computation and plotting  (only runs when executed directly)
if __name__ == '__main__':

    # %%  Compute RCD and energy gap over the BZ
    RCD = get_RCD(**CrI3)
    gap = get_energy(**CrI3)

    honeycomb_bz_x, honeycomb_bz_y = honeycomb_bz()
    kx, ky = bzmesh(m=2)

    # %%  Plot RCD over the BZ
    with plt.style.context(['science', 'ieee']):
        fig, axes = panel(figsize=(4, 3), nrows=1, ncols=1,
                          width_ratios=[1], height_ratios=[1])
        fig.subplots_adjust(top=0.95, bottom=0.15, right=0.99)

        pc = axes.pcolormesh(kx, ky, RCD, cmap='jet')
        plot(honeycomb_bz_x, honeycomb_bz_y, ax=axes,
             linestyle='-', linewidth=1, color='k')

        clb = fig.colorbar(pc, ax=axes, shrink=0.9)
        clb.ax.set_title(r"$\chi$", loc='left', fontsize=16, pad=5)
        clb.ax.tick_params(labelsize=16)

        axes.set_axis_on()
        axes.grid(False)
        axes.set_xticks([-0.5*2*np.pi, 0, 0.5*2*np.pi])
        axes.set_xticklabels(['-1', '0', '1'], fontsize=16)
        axes.set_yticks([-0.5*2*np.pi, 0, 0.5*2*np.pi])
        axes.set_yticklabels(['-1', '0', '1'], fontsize=16)
        axes.set_xlabel(r'$k_x(\pi/a)$', fontsize=18)
        axes.set_ylabel(r'$k_y(\pi/a)$', fontsize=18)
        plt.show()

    # %%
    print('Max energy gap:', gap.max())
    print('BZ-integrated RCD / (2pi):', bz_integration_honeycomb(RCD, n=200) / (2*np.pi))
