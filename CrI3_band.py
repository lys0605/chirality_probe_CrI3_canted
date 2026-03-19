#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from plot_utils import plot, letter_annotation, panel
from mathfuntion import Im, Re, is_invertible
import scienceplots

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['axes.labelsize'] = 24

dot = np.vectorize(np.dot,signature='(n),(m)->()')

Gamma = np.array([0, 0]) # Gamma
K1 = 2*np.pi*np.array([2/3,0])/np.sqrt(3) # K
M = 2*np.pi*np.array([1/2,1/(2*np.sqrt(3))])/np.sqrt(3) # M
K2 = 2*np.pi*np.array([1/3,1/np.sqrt(3)])/np.sqrt(3) # K'

def get_kvectors(pt1,pt2,num=101):
    """
    2D version
    """
    # get k_vectors
    kx = np.linspace(pt1[0],pt2[0],num=num)
    ky = np.linspace(pt1[1],pt2[1],num=num)
    k = np.vstack((kx,ky)).T
    return k

def get_path(k):
    dot = np.vectorize(np.dot,signature='(n),(m)->()')
    return np.sqrt(dot(k,k))

def get_total_path(*arg):
    '''
    connect all the paths
    '''
    length = len(arg)
    lengths = np.zeros(length)
    k_index = np.zeros(length+1)
    path = np.concatenate(arg)
    for j in range(length):
        lengths[j] = len(arg[j])
    k_index[-1] = lengths.sum()
    for i in range(length-1):
        if i != 0:
            k_index[i+1] = k_index[i]+lengths[i]   
        else:
            k_index[i+1] = k_index[i]+lengths[i]-1
    return path, k_index

def group_kvectors(*arg):
    k_vectors = arg[0]
    for i in range(len(arg)-1):
        k_vectors = np.concatenate((k_vectors,arg[i+1][1:]))
    return k_vectors

def get_band(energy_function,k_vectors,**kwarg):
    """
    get band
    """
    J1 = kwarg.get('J1')
    J2 = kwarg.get('J2')
    J3 = kwarg.get('J3')
    D = kwarg.get('D')
    Az = kwarg.get('Az')
    S = kwarg.get('S')
    energy = energy_function(k_vectors,J1=J1, J2=J2, J3=J3, D=D, Az=Az,S=S)
    return energy

def magnon_energy(k,J1=2.01, J2=0.16, J3=-0.08, D=0.31, Az=0.49 ,S=3/2):
    '''
    band structure of the CrI3
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
    next_n_n = -1*np.sqrt(3)*np.array([[1/2,np.sqrt(3)/2],[-1,0],[1/2,-np.sqrt(3)/2]]) # next nearest neighbour vectors
    next_next_n_n =  np.array([[0, -2], [np.sqrt(3),1], [-np.sqrt(3),1]]) # next next nearest neighbour vectors

    # vectorizing function
    dot = np.vectorize(np.dot, signature='(n),(m)->()')
    gamma_1 = J1*S*np.sum(np.exp(1j*k@n_n.T), axis=1)
    gamma_2 = 2*J2*S*np.sum(np.cos(k@next_n_n.T), axis=1)
    gamma_3 = J3*S*np.sum(np.exp(1j*k@next_next_n_n.T), axis=1)
    lambda_k = 2*D*S*np.sum(np.sin(k@next_n_n.T), axis=1)

    # bloch vector
    d0 = M+2*Az*S-gamma_2
    dx = -np.real(gamma_1+gamma_3)
    dy = np.imag(gamma_1+gamma_3)
    dz = lambda_k
    d_abs = np.emath.sqrt(dx**2+dy**2+dz**2)
 
    # energy band -,+
    em = d0-d_abs
    ep = d0+d_abs

    # energy band +,-
    energy = np.array([ep,em])
    return energy

#%%
J1 = 2.01 # n.n Heisenberg coupling meV
J2 = 0.16 # n.n.n Heisenberg coupling meV
J3 = -0.08 # n.n.n.n Heisenberg coupling meV
D = 0.31 # DMI meV
Az = 0.49 # anisotropy
S = 3/2 # spin number

k0 = get_kvectors(-1*K1, Gamma)
k1 = get_kvectors(Gamma, K1)
k2 = get_kvectors(K1, M, num=51)
k3 = get_kvectors(M, K2, num=51)
k4 = get_kvectors(K2, Gamma)
k_vectors = group_kvectors(k0, k1, k2, k3, k4)

#%%
magnon_bands = get_band(magnon_energy, k_vectors,J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S)

K2_Gamma = get_path(k0)
Gamma_K1 = get_path(k1)[1:]
K1_M = get_path(k2)[1:]
M_K2 = get_path(k3)[1:]
K2_Gamma_2 = get_path(k4)[1:]

k_label = [r"K$^\prime$",r"$\Gamma$",r"K$^\prime$","M",r"K$^\prime$",r"$\Gamma$"]

path, k_index = get_total_path(K2_Gamma,Gamma_K1,K1_M,M_K2,K2_Gamma_2)

blue_colors = ['#8fd0ff', '#589fef', '#0071bc', '#00468b', '#00215d'] # blue colors for the bands from light to dark
red_colors = ['#ffc883', '#ff9f4d', '#ff6f00', '#c94c00', '#7f2e00'] # red colors for the bands from light to dark

#%%
with plt.style.context('science'):
    fig, ax = plt.subplots(figsize=(6,4))
    plot(np.arange(len(path)), magnon_bands[1], ax=ax, color=blue_colors[0], linestyle='-', linewidth=2.5, label=r"$\epsilon_{\bf k -}$")
    plot(np.arange(len(path)), magnon_bands[0], ax=ax, color=red_colors[0], linestyle='-', linewidth=2.5, label=r"$\epsilon_{\bf k +}$")

    for i in range(len(k_index)-2):  
        ax.axvline(k_index[i+1], color="black", ls = '-' ,linewidth=1.0)

    ax.set_xticks(k_index,k_label)
    ax.yaxis.grid()
    ax.set_xlim(0,len(path)) 
    ax.set_ylim(0,22) 
    ax.set_in_layout(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.62,0.01), fontsize=18, frameon=True)
    ax.set_ylabel(r"$\epsilon$ (meV)")
    # $\mathrm{(meV)}$
    fig.tight_layout()
    plt.show()
fig.savefig('figures/CrI3_bands/CrI3_band_structure.png', dpi=300 ,bbox_inches='tight')
#

# %%
print(np.min(magnon_bands[1]))
# %%

# %%
