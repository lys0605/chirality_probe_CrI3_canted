# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from plot_utils import plot, letter_annotation, panel
from mathfuntion import Im, Re, is_invertible
from honeycomb_lattice import *
import scienceplots
import matplotlib.path as mpath
import matplotlib.patches as mpatches
Path = mpath.Path
# %%


def FM_eigs_exact(k,J1=2.01, J2=0.16, J3=-0.08, D=0.31, Az=0.49 ,S=3/2):
    '''
    obtain a the eigenvectors of tauz@matrix H_k for a given k
    energy in meV 
    Parameters:
        k (np.ndarray): The k-vectors.
        J1 (float): Nearest neighbour Heisenberg coupling.
        J2 (float): Next nearest neighbour Heisenberg coupling.
        J3 (float): Next next nearest neighbour Heisenberg coupling.
        Az (float): Anisotropy.
        D (float): DMI coupling.
        S (float): Spin number.
    Returns:
        energy (np.ndarray): The energy bands.
    '''
    # parameters
    M = 3*J1*S+6*J2*S+3*J3*S
    
    # lattice structure
#     a = 6.324 # 6.324 Å
    n_n = np.sqrt(3)*np.array([[0,1/np.sqrt(3)],[-1/2,-0.5/np.sqrt(3)],[1/2,-0.5/np.sqrt(3)]])
    next_n_n = -np.sqrt(3)*np.array([[1/2,np.sqrt(3)/2],[-1,0],[1/2,-np.sqrt(3)/2]]) # next nearest neighbour vectors
    next_next_n_n =  np.array([[0, -2], [np.sqrt(3),1], [-np.sqrt(3),1]]) # next next nearest neighbour vectors

    # vectorizing function
    dot = np.vectorize(np.dot,signature='(n),(m)->()')
    gamma_1 = J1*S*np.sum(np.exp(1j*k@n_n.T))
    gamma_2 = 2*J2*S*np.sum(np.cos(k@next_n_n.T))
    gamma_3 = J3*S*np.sum(np.exp(1j*k@next_next_n_n.T))
    lambda_k = 2*D*S*np.sum(np.sin(k@next_n_n.T))
    
    # bloch vector
    d0 = M+2*Az*S-gamma_2
    dx = -np.real(gamma_1+gamma_3)
    dy = np.imag(gamma_1+gamma_3)
    dz = lambda_k
    d_abs = np.emath.sqrt(dx**2+dy**2+dz**2)
    dxy_abs = np.emath.sqrt(dx**2+dy**2)
 
    # energy band -,+
    em = d0-d_abs
    ep = d0+d_abs
    
    # subspace parameters  
    cos = np.emath.sqrt((d_abs+dz)/(2*d_abs))
    sin = np.emath.sqrt((d_abs-dz)/(2*d_abs))
    
    # angles
    varphi_k = 1j*np.log((dx+1j*dy)/dxy_abs)
    psi_k = np.arcsin(sin)
    
    phase = np.exp(1j*varphi_k/2)
    phase_conj = np.conj(phase)
    
    
    # eigenvectors
    u1 = np.array([cos*phase, sin*phase_conj])
    u2 = np.array([-sin*phase, cos*phase_conj])
    
    Uk = np.array([u1,u2])
    E = np.array([ep,em])
    
#     norm = np.abs(u1[0])**2+np.abs(u1[1])**2-np.abs(u1[2])**2-np.abs(u1[3])**2
    return E,Uk


def FM_derivatives(k,J1=2.01, J2=0.16, J3=-0.08, D=0, Az=0.49 ,S=3/2):
    '''
    obtain a the derivatives of tauz@matrix H_k for a given k
    energy in meV 
    Parameters:
        k (np.ndarray): The k-vectors.
        J1 (float): Nearest neighbour Heisenberg coupling.
        J2 (float): Next nearest neighbour Heisenberg coupling.
        J3 (float): Next next nearest neighbour Heisenberg coupling.
        Az (float): Anisotropy.
        D (float): DMI coupling.
        S (float): Spin number.
    Returns:
        Hx (np.ndarray): The derivative of matrix H_k with respect to x.
        Hy (np.ndarray): The derivative of matrix H_k with respect to y.
    '''
     # parameters
    M = 3*J1*S+6*J2*S+3*J3*S
    
    # lattice structure
#     a = 6.324 # 6.324 Å
    n_n = np.sqrt(3)*np.array([[0,1/np.sqrt(3)],[-1/2,-0.5/np.sqrt(3)],[1/2,-0.5/np.sqrt(3)]])
    next_n_n = -np.sqrt(3)*np.array([[1/2,np.sqrt(3)/2],[-1,0],[1/2,-np.sqrt(3)/2]]) # next nearest neighbour vectors
    next_next_n_n =  np.array([[0, -2], [np.sqrt(3),1], [-np.sqrt(3),1]]) # next next nearest neighbour vectors

    # vectorizing function
    gamma_1x = 1j*np.dot(n_n.T[0],np.exp(1j*k@n_n.T))*J1*S
    gamma_1y = 1j*np.dot(n_n.T[1],np.exp(1j*k@n_n.T))*J1*S
    gamma_3x = 1j*np.dot(next_next_n_n.T[0],np.exp(1j*k@next_next_n_n.T))*J3*S
    gamma_3y = 1j*np.dot(next_next_n_n.T[1],np.exp(1j*k@next_next_n_n.T))*J3*S
    lambda_kx = np.dot(next_n_n.T[0],np.cos(k@next_n_n.T))*2*D*S
    lambda_ky = np.dot(next_n_n.T[1],np.cos(k@next_n_n.T))*2*D*S
    gamma_2x = -np.dot(next_n_n.T[0],np.sin(k@next_n_n.T))*2*J2*S
    gamma_2y = -np.dot(next_n_n.T[1],np.sin(k@next_n_n.T))*2*J2*S
    
    Hx = np.array([[-gamma_2x+lambda_kx,-gamma_1x-gamma_3x],[-gamma_1x.conj()-gamma_3x.conj(),-gamma_2x-lambda_kx]])
    Hy = np.array([[-gamma_2y+lambda_ky,-gamma_1y-gamma_3y],[-gamma_1y.conj()-gamma_3y.conj(),-gamma_2y-lambda_ky]])
    
    return Hx, Hy 


def FM_berry_curvature(k,J1=2.01, J2=0.16, J3=-0.08, D=0.31, Az=0.49 ,S=3/2, m=0):
    '''
    obtain a the derivatives of tauz@matrix H_k for a given k
    energy in meV 
    Parameters:
    ----------
        k: reciprocal vectors
        J: exchange interaction coefficient
        D: DMI coefficient
        S: spin number
        s: saturation field ratio (sin\theta)
        m: 0 for e_+, 1 for e_-

    Returns:
    -------
        berry_curvature (float): The Berry curvature.
    '''  
    E, Uk = FM_eigs_exact(k=k,J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S)
    Hx, Hy = FM_derivatives(k=k,J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S)

    Uik = Uk.conj().T # inverse.T
    Uik = Uik.T # we want to take the column vector
#     Uik = LA.inv(Uk).T
    
    # calculate berry
    index = np.array([0,1]) # total bands
    excluded = np.delete(index,m)
    
    berry_curvature = 0
    
    for n in excluded:
        berry_curvature += -Im(((Uik[m]@Hx@Uk[n])*(Uik[n]@Hy@Uk[m])-(Uik[m]@Hy@Uk[n])*(Uik[n]@Hx@Uk[m]))/(E[n]-E[m])**2)
    
    return berry_curvature

def FM_quantum_metric(k,J1=2.01, J2=0.16, J3=-0.08, D=0.31, Az=0.49 ,S=3/2, m=0):
    '''
    obtain a the derivatives of tauz@matrix H_k for a given k
    energy in meV 
    Parameters:
    ----------
        k: reciprocal vectors
        J: exchange interaction coefficient
        D: DMI coefficient
        S: spin number
        s: saturation field ratio (sin\theta)
        m: 0 for e_+, 1 for e_-

    Returns:
    -------
        berry_curvature (float): The Berry curvature.
    '''  
    E, Uk = FM_eigs_exact(k=k,J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S)
    Hx, Hy = FM_derivatives(k=k,J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S)

    Uik = Uk.conj().T # inverse.T
    Uik = Uik.T # we want to take the column vector
#     Uik = LA.inv(Uk).T
    
    # calculate berry
    index = np.array([0,1]) # total bands
    excluded = np.delete(index,m)
    
    quantum_metric = 0
    
    for n in excluded:
        quantum_metric += Re(((Uik[m]@Hx@Uk[n])*(Uik[n]@Hy@Uk[m]))/(E[n]-E[m])**2)
    
    return quantum_metric

def get_FM_berry_curvature(J1=2.01, J2=0.16, J3=-0.08, D=0.31, Az=0.49 ,S=3/2, n=200):
    # k-mesh
    kx,ky = bzmesh(n=n, m=2)
    omega_upper = np.zeros(kx.shape) #square mesh
    omega_lower = np.zeros(kx.shape)
    kx = kx[0]
    ky = ky.T[0]
    for i in range(ky.size):
        for j in range(kx.size):
            k = np.vstack((kx[j],ky[i])).T
            k = k.reshape(k.shape[1]) 
            omega_upper[i,j] = FM_berry_curvature(k,J1=J1, J2=J2, J3=J3, D=D, Az=Az ,S=S, m=0)
            omega_lower[i,j] = FM_berry_curvature(k,J1=J1, J2=J2, J3=J3, D=D, Az=Az ,S=S, m=1)
        if i%50 == 0:
            print(f'{i}: done with parameters J1={J1}, J2={J2} , J3={J3}, D={D}, Az={Az}, S={S}')
    return np.array([omega_upper, omega_lower])

def get_FM_quantum_metric(J1=2.01, J2=0.16, J3=-0.08, D=0.31, Az=0.49 ,S=3/2, n=200):
    # k-mesh
    kx,ky = bzmesh(n=n, m=2)
    g_upper = np.zeros(kx.shape) #square mesh
    g_lower = np.zeros(kx.shape)
    kx = kx[0]
    ky = ky.T[0]
    for i in range(ky.size):
        for j in range(kx.size):
            k = np.vstack((kx[j],ky[i])).T
            k = k.reshape(k.shape[1]) 
            g_upper[i,j] = FM_quantum_metric(k,J1=J1, J2=J2, J3=J3, D=D, Az=Az ,S=S, m=0)
            g_lower[i,j] = FM_quantum_metric(k,J1=J1, J2=J2, J3=J3, D=D, Az=Az ,S=S, m=1)
        if i%50 == 0:
            print(f'{i}: done with parameters J1={J1}, J2={J2} , J3={J3}, D={D}, Az={Az}, S={S}')
    return np.array([g_upper, g_lower])


#%% 
J1 = 2.01 # n.n Heisenberg coupling meV
J2 = 0.16 # n.n.n Heisenberg coupling meV
J3 = -0.08 # n.n.n.n Heisenberg coupling meV
D = 0.31 # DMI meV
Az = 0.49 # anisotropy
S = 3/2 # spin number

CrI3_berry_curvatures = get_FM_berry_curvature(J1=J1, J2=J2, J3=J3, D=J1/10, Az=Az, S=S, n=200) 
#CrI3_quantum_metrics = get_FM_quantum_metric(J1=J1, J2=J2, J3=J3, D=0, Az=Az, S=S, n=200)
# %%
honeycomb_bz_x, honeycomb_bz_y = honeycomb_bz()

kx,ky = bzmesh(n=200, m=2)

# %% ploting 
color_bar_title = [r"$\Omega_{+}$",r"$\Omega_{-}$"]
color_bar_title_RCD = [r"$\Omega_{+}^{\text{RCD}}$",r"$\Omega_{-}^{\text{RCD}}$"]
color_bar_title_2M = [r"$\Omega_{+}^{\text{2M}}$",r"$\Omega_{-}^{\text{2M}}$"]
color_bar_title_AFM = [r"$\Omega_{+}^{\text{AFM}}$",r"$\Omega_{-}^{\text{AFM}}$"]
color_bar_title_FM = [r"$\Omega_{+}^{\text{FM}}$",r"$\Omega_{-}^{\text{FM}}$"]
color_bar_title_noFM = [r"$\Omega_{+}^{\text{noFM}}$",r"$\Omega_{-}^{\text{noFM}}$"]
color_bar_title_rest = [r"$\Omega_{+}^{\text{Rest}}$",r"$\Omega_{-}^{\text{Rest}}$"]
color_bar_title_compare = [r"$\Omega_{+}^{\text{2M}^{(0)}}$",r"$\Omega_{+}^{\text{2M}^{(2)}}$"]
color_bar_title_rho = [r"$\rho^{\text{RL}}$",r"$\tilde{\rho}$"]
color_bar_title_hyperbolic = [r"$\sinh 2\chi_{+}$",r"$\sinh 2\chi_{-}$"]
color_bar_title_M = [r"$AM_{+}$",r"$AM_{-}$"]
color_bar_title_J = [r"$J_{+}$",r"$J_{-}$"]
# color_bar_title_exact = [r"$$", r"$-\rho_k^{\text{RL}}\sin 2\psi_k$"]

pads = [7, 12]
with plt.style.context(['science','ieee']):
    fig, ax = panel(figsize=(4,3), nrows=1, ncols=1, width_ratios=[1], height_ratios=[1], hspace=0.1, wspace=0.4)

    fig.subplots_adjust(top=0.95, bottom=0.15, right=0.99)

    pc = ax.pcolormesh(kx, ky, CrI3_berry_curvatures[1], cmap="jet",) 
    
    plot(honeycomb_bz_x, honeycomb_bz_y, ax=ax, linestyle='-', linewidth=1, color='k')

    clb = fig.colorbar(pc, ax=ax, shrink=0.9)
    clb.ax.set_title(color_bar_title[1], loc='left', fontsize=24, pad=pads[1])
    clb.ax.tick_params(labelsize=24)
    ax.set_axis_on() # make sure the axis is on
    ax.grid(False) # make sure the grid is off

    ax.set_xticks([-0.5 * 2 * np.pi, 0, 0.5 * 2 * np.pi])
    ax.set_xticklabels(['-1', '0', '1'], fontsize=24)
    ax.set_yticks([-0.5 * 2 * np.pi, 0, 0.5 * 2 * np.pi])
    ax.set_yticklabels(['-1', '0', '1'], fontsize=24)

    ax.set_xlabel(r'$k_x(\pi/a)$', fontsize=28)
    ax.set_ylabel(r'$k_y(\pi/a)$', fontsize=28)
    plt.show()
# %%
fig.savefig('figures/CrI3_berry_curvatures/CrI3_berry_curvature_lower.png', dpi=300 ,bbox_inches='tight')

# %%
