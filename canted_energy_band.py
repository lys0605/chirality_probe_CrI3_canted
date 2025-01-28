import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from plot_utils import plot, letter_annotation, panel
from mathfuntion import Im, Re, is_invertible, print_matrix
import scienceplots

mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.labelsize'] = 22

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
    J = kwarg.get('J')
    D = kwarg.get('D')
    S = kwarg.get('S')
    s = kwarg.get('s')
    energy = energy_function(k_vectors,J=J,D=D,S=S,s=s)
    return energy

def canted_energy(k,J=1.54,D=0.1,S=5/2,s=0.6):
    '''
    band structure of the canted antiferromagnet
    energy in meV 
    k: reciprocal vectors
    J: exchange interaction coefficient
    D: DMI coefficient
    S: spin number
    s: saturation field ratio (sin\theta)
    '''
    Bs = 6*J*S
    B = s*Bs
    
    # parameters
    M = 6*J*S
    v = s**2
    
    # lattice structure
#     a = 6.324 # 6.324 Å
    n_n = np.sqrt(3)*np.array([[0,1/np.sqrt(3)],[-1/2,-0.5/np.sqrt(3)],[1/2,-0.5/np.sqrt(3)]])
    next_n_n = -1*np.sqrt(3)*np.array([[1/2,np.sqrt(3)/2],[-1,0],[1/2,-np.sqrt(3)/2]]) # next nearest neighbour vectors
    
    # vectorizing function
    dot = np.vectorize(np.dot, signature='(n),(m)->()')
    gamma = np.sum(np.exp(-1j*k@n_n.T), axis=1)
    gamma_sin = np.sum(np.sin(k@next_n_n.T), axis=1)
    
    # complex parameters
    phi_k = 2*J*S*gamma
    lam_k = 4*D*S*s*gamma_sin
    phi_k_sq = np.abs(phi_k)**2
    delta_k = np.sqrt(lam_k**2+v**2*phi_k_sq)
    varphi_k = 1j*np.log(phi_k/np.abs(phi_k))
    
    # energy band -,+
    energy = [np.emath.sqrt((M-delta_k)**2-(1-v)**2*phi_k_sq),
              np.emath.sqrt((M+delta_k)**2-(1-v)**2*phi_k_sq)]

    return energy

J = 1 # meV
D = 0.1 # D/J = 0.1
S = 5/2 # spin number
s = 0.25 # saturation field ratio (sin\theta) = B/Bs

k0 = get_kvectors(-1*K1, Gamma)
k1 = get_kvectors(Gamma, K1)
k2 = get_kvectors(K1, M, num=51)
k3 = get_kvectors(M, K2, num=51)
k4 = get_kvectors(K2, Gamma)
k_vectors = group_kvectors(k0, k1, k2, k3, k4)

s_values = [0.25, 0.5, 0.75]
D_values = [0, 0.6, 1.2, 1.8]
magnon_bands = [get_band(canted_energy,k_vectors,J=J,D=D,S=S,s=s) for s in s_values for D in D_values]

K2_Gamma = get_path(k0)
Gamma_K1 = get_path(k1)[1:]
K1_M = get_path(k2)[1:]
M_K2 = get_path(k3)[1:]
K2_Gamma_2 = get_path(k4)[1:]

k_label = ["K\'",r"$\Gamma$","K","M","K\'",r"$\Gamma$"]

path, k_index = get_total_path(K2_Gamma,Gamma_K1,K1_M,M_K2,K2_Gamma_2)

blue_colors = ['#8fd0ff', '#589fef', '#0071bc', '#00468b', '#00215d'] # blue colors for the bands from light to dark
red_colors = ['#ffc883', '#ff9f4d', '#ff6f00', '#c94c00', '#7f2e00'] # red colors for the bands from light to dark

with plt.style.context(['science']):
    fig, ax = plt.subplots(figsize=(8,4))
    for i in range(len(D_values)):
        plot(np.arange(len(path)), magnon_bands[i][0], ax=ax, color=blue_colors[i], linestyle='-', linewidth=1, label=r"$B=0.25B_s$"+rf"$D={D_values[i]}$")
        plot(np.arange(len(path)), magnon_bands[i][1], ax=ax, color=blue_colors[i], linestyle='-', linewidth=1)

    for i in range(len(k_index)-2):  
        ax.axvline(k_index[i+1], color="black", ls = '-' ,linewidth=1.0)

    ax.set_xticks(k_index,k_label)
    ax.yaxis.grid()
    ax.set_xlim(0,len(path))
    #ax.set_ylim(0,25) 
    ax.set_in_layout(True)
    ax.legend(loc='lower center', bbox_to_anchor=(1.4,0.01), fontsize=8, frameon=True)
    ax.set_ylabel(r"$\epsilon$ $\mathrm{(meV)}$")
    fig.tight_layout()
    plt.show()
fig.savefig('canted_energy_bands_B=0.25.png', dpi=600 ,bbox_inches='tight')