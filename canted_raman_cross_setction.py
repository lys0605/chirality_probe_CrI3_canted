# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from plot_utils import plot, letter_annotation, panel
from mathfuntion import Im, Re, is_invertible
from honeycomb_lattice import *
import scienceplots
# %%

def canted_eigs_2(k,J=1,D=0.1,S=5/2,s=0.6):
    '''
    obtain a the eigenvectors of tauz@matrix H_k for a given k
    energy in meV 
    k: reciprocal vectors
    J: exchange interaction coefficient
    D: DMI coefficient
    S: spin number
    s: saturation field ratio (sin\theta)
    '''
    # parameters
    M = 3*J*S
    v = s**2
    
    # lattice parameters
    a = 1
    n_n = a*np.sqrt(3)*np.array([[0,1/np.sqrt(3)],[1/2,-0.5/np.sqrt(3)],[-1/2,-0.5/np.sqrt(3)]]) #a1,a2,a3
    next_n_n = -a*np.sqrt(3)*np.array([[-1,0],[1/2,np.sqrt(3)/2],[1/2,-np.sqrt(3)/2]]) # b1,b2,b3
    
    # vectorizing function
    dot = np.vectorize(np.dot, signature='(n),(m)->()')
    gamma = np.sum(np.exp(-1j*k@n_n.T))
    gamma_sin = np.sum(np.sin(k@next_n_n.T))
    
    # complex parameters
    phi_k = J*S*gamma
    phi_bar_k = np.conjugate(phi_k) 
    aphi_k = np.abs(phi_k)**2
    lam_k = 2*D*S*s*gamma_sin
    delta_k = np.sqrt(lam_k**2+v**2*aphi_k)
    
    # energy band -,+
    em = np.emath.sqrt(M**2+lam_k**2-(1-2*v)*aphi_k-2*M*delta_k)
    ep = np.emath.sqrt(M**2+lam_k**2-(1-2*v)*aphi_k+2*M*delta_k)
    
    # (para)normalisation constant
    n1 = 1/np.emath.sqrt((M-delta_k+em)**2/((v-1)**2*aphi_k)+(v**2*aphi_k+(delta_k+lam_k)*(-M+lam_k-em))**2/(v*(v-1)*aphi_k)**2-(delta_k+lam_k)**2/(v**2*aphi_k)-1)
    n2 = 1/np.emath.sqrt((M+delta_k+ep)**2/((v-1)**2*aphi_k)+(-v**2*aphi_k+(delta_k-lam_k)*(-M+lam_k-ep))**2/(v*(v-1)*aphi_k)**2-(-delta_k+lam_k)**2/(v**2*aphi_k)-1)
    n3 = 1/np.emath.sqrt((-M+delta_k+em)**2/((v-1)**2*aphi_k)+(v**2*aphi_k+(delta_k+lam_k)*(-M+lam_k+em))**2/(v*(v-1)*aphi_k)**2-(delta_k+lam_k)**2/(v**2*aphi_k)-1)
    n4 = 1/np.emath.sqrt((M+delta_k-ep)**2/((v-1)**2*aphi_k)+(-v**2*aphi_k+(delta_k-lam_k)*(-M+lam_k+ep))**2/(v*(v-1)*aphi_k)**2-(-delta_k+lam_k)**2/(v**2*aphi_k)-1)
    
    u1 = n1*np.array([(M-delta_k+em)/((v-1)*phi_k),-(v**2*aphi_k+(delta_k+lam_k)*(-M+lam_k-em))/(v*(v-1)*aphi_k),(delta_k+lam_k)/(v*phi_k),1])
    u2 = n2*np.array([(M+delta_k+ep)/((v-1)*phi_k),(-v**2*aphi_k+(delta_k-lam_k)*(-M+lam_k-ep))/(v*(v-1)*aphi_k),(-delta_k+lam_k)/(v*phi_k),1])
    u3 = n3*np.array([(M-delta_k-em)/((v-1)*phi_k),-(v**2*aphi_k+(delta_k+lam_k)*(-M+lam_k+em))/(v*(v-1)*aphi_k),(delta_k+lam_k)/(v*phi_k),1])
    u4 = n4*np.array([(M+delta_k-ep)/((v-1)*phi_k),(-v**2*aphi_k+(delta_k-lam_k)*(-M+lam_k+ep))/(v*(v-1)*aphi_k),(-delta_k+lam_k)/(v*phi_k),1])
    
    Uk = np.array([u2,u1,u4,u3])
    E = np.array([ep,em,-ep,-em])
#     norm = np.abs(u1[0])**2+np.abs(u1[1])**2-np.abs(u1[2])**2-np.abs(u1[3])**2
    return E,Uk


def raman_cross_section_ham(k,qq=0,J=1,D=0.1,S=5/2,B0=0.5):
    '''
    for each k, we compute the elements of raman hamiltonian
    R: right circular polarized 
    L: left circular polarized 
    qq: 0: LR, 1: LL, 2:RR, 3:RL
    e_s: s-circular polarization vectors
    '''
    # field
    Bs = 3*J*S # not 6JS here
    B = (B0-0.001)*Bs
    s = B/Bs
    
    # parameters
    M = 3*J*S
    v = s**2
    
    # lattice parameters
    a = 1
    n_n = a*np.sqrt(3)*np.array([[0,1/np.sqrt(3)],[1/2,-0.5/np.sqrt(3)],[-1/2,-0.5/np.sqrt(3)]]) #a1,a2,a3
    next_n_n = -a*np.sqrt(3)*np.array([[-1,0],[1/2,np.sqrt(3)/2],[1/2,-np.sqrt(3)/2]]) # b1,b2,b3
    
    # vectorizing function
    dot = np.vectorize(np.dot,signature='(n),(m)->()')
    gamma = np.sum(np.exp(-1j*k@n_n.T))
    gamma_sin = np.sum(np.sin(k@next_n_n.T))
    
    # complex parameters
    phi_k = J*S*gamma
    phi_bar_k = np.conjugate(phi_k) 
    aphi_k = np.abs(phi_k)**2
    lam_k = 2*D*S*s*gamma_sin
    delta_k = np.sqrt(lam_k**2+v**2*aphi_k)
    
    #eigenvectors and eigenenergy of canted AFM
    E, Uk = canted_eigs_2(k, J=J,D=D,S=S,s=s)
    Uk = Uk.T
    
    tauz = np.diag([1,1,-1,-1])
    Uik = Uk.conj().T # in berry curvature we use inverse, here we use hermitian conjugate
    
    # chirality and polarisation
    zeta = 1 # +1: right; -1:left
    # left circularly polarized: (-i,1)/sqrt(2); right circularly polarized: (i,1)/sqrt(2)
    # s: out; s':in
    # e_s^* = e_out , e_s'=e_in
    e_in_1 = (1/np.sqrt(2))*np.array([-1j*zeta,1]) 
    e_in_2 = (1/np.sqrt(2))*np.array([1j*zeta,1])
    e_out_1 = (1/np.sqrt(2))*np.array([-1j*zeta,1])
    e_out_2 = (1/np.sqrt(2))*np.array([1j*zeta,1])
    
    # if extra phase factors are present
    phi = 0*np.pi # rotated angle around normal
    theta = 0*np.pi # angle tilted away from x axis (no longer orthogonal)
    e_in_1 = (1/np.sqrt(2))*np.array([-np.sin(phi)-1j*zeta*np.cos(phi)*np.cos(theta),np.cos(phi)-1j*zeta*np.sin(phi)*np.cos(theta)])
    e_in_2 = (1/np.sqrt(2))*np.array([-np.sin(phi)+1j*zeta*np.cos(phi)*np.cos(theta),np.cos(phi)+1j*zeta*np.sin(phi)*np.cos(theta)])
    e_out_1 = (1/np.sqrt(2))*np.array([-np.sin(phi)-1j*zeta*np.cos(phi)*np.cos(theta),np.cos(phi)-1j*zeta*np.sin(phi)*np.cos(theta)])
    e_out_2 = (1/np.sqrt(2))*np.array([-np.sin(phi)+1j*zeta*np.cos(phi)*np.cos(theta),np.cos(phi)+1j*zeta*np.sin(phi)*np.cos(theta)])
    
    # if zeta = 1
    # g-factor
    # LR (e_out = L, e_in = R -> ss' = RL)
    g_11_1 = e_in_1@n_n[0]*e_out_1@n_n[0]
    g_11_2 = e_in_1@n_n[1]*e_out_1@n_n[1]
    g_11_3 = e_in_1@n_n[2]*e_out_1@n_n[2]
    gk_11 = J*S*(g_11_1*np.exp(-1j*k@n_n[0])+g_11_2*np.exp(-1j*k@n_n[1])+g_11_3*np.exp(-1j*k@n_n[2]))
    gkb_11 = J*S*(g_11_1*np.exp(1j*k@n_n[0])+g_11_2*np.exp(1j*k@n_n[1])+g_11_3*np.exp(1j*k@n_n[2]))
    g0_11 = J*S*(g_11_1+g_11_2+g_11_3)
    
    # LL
    g_12_1 = e_in_1@n_n[0]*e_out_2@n_n[0]
    g_12_2 = e_in_1@n_n[1]*e_out_2@n_n[1]
    g_12_3 = e_in_1@n_n[2]*e_out_2@n_n[2]
    gk_12 = J*S*(g_12_1*np.exp(-1j*k@n_n[0])+g_12_2*np.exp(-1j*k@n_n[1])+g_12_3*np.exp(-1j*k@n_n[2]))
    gkb_12 = J*S*(g_12_1*np.exp(1j*k@n_n[0])+g_12_2*np.exp(1j*k@n_n[1])+g_12_3*np.exp(1j*k@n_n[2]))
    g0_12 = J*S*(g_12_1+g_12_2+g_12_3)
    
    # RR
    g_21_1 = e_in_2@n_n[0]*e_out_1@n_n[0]
    g_21_2 = e_in_2@n_n[1]*e_out_1@n_n[1]
    g_21_3 = e_in_2@n_n[2]*e_out_1@n_n[2]
    gk_21 = J*S*(g_21_1*np.exp(-1j*k@n_n[0])+g_21_2*np.exp(-1j*k@n_n[1])+g_21_3*np.exp(-1j*k@n_n[2]))
    gkb_21 = J*S*(g_21_1*np.exp(1j*k@n_n[0])+g_21_2*np.exp(1j*k@n_n[1])+g_21_3*np.exp(1j*k@n_n[2]))
    g0_21 = J*S*(g_21_1+g_21_2+g_21_3)
    
    # RL (actually LR, e_out = R, e_in = L -> ss' = RL)
    g_22_1 = e_in_2@n_n[0]*e_out_2@n_n[0]
    g_22_2 = e_in_2@n_n[1]*e_out_2@n_n[1]
    g_22_3 = e_in_2@n_n[2]*e_out_2@n_n[2]
    gk_22 = J*S*(g_22_1*np.exp(-1j*k@n_n[0])+g_22_2*np.exp(-1j*k@n_n[1])+g_22_3*np.exp(-1j*k@n_n[2]))
    gkb_22 = J*S*(g_22_1*np.exp(1j*k@n_n[0])+g_22_2*np.exp(1j*k@n_n[1])+g_22_3*np.exp(1j*k@n_n[2]))
    g0_22 = J*S*(g_22_1+g_22_2+g_22_3)
    
    # f-factor
    # LR
    f_11_1 = e_in_1@next_n_n[0]*e_out_1@next_n_n[0]
    f_11_2 = e_in_1@next_n_n[1]*e_out_1@next_n_n[1]
    f_11_3 = e_in_1@next_n_n[2]*e_out_1@next_n_n[2]
    #fk_11 = 1j*2*D*S*(f_11_1*np.exp(-1j*k@next_n_n[0])+f_11_2*np.exp(-1j*k@next_n_n[1])+f_11_3*np.exp(-1j*k@next_n_n[2]))
    fk_11 = 2*D*S*s*(f_11_1*np.sin(k@next_n_n[0])+f_11_2*np.sin(k@next_n_n[1])+f_11_3*np.sin(k@next_n_n[2]))
    
    # LL
    f_12_1 = e_in_1@next_n_n[0]*e_out_2@next_n_n[0]
    f_12_2 = e_in_1@next_n_n[1]*e_out_2@next_n_n[1]
    f_12_3 = e_in_1@next_n_n[2]*e_out_2@next_n_n[2]
    #fk_12 = 1j*2*D*S*(f_12_1*np.exp(-1j*k@next_n_n[0])+f_12_2*np.exp(-1j*k@next_n_n[1])+f_12_3*np.exp(-1j*k@next_n_n[2]))
    fk_12 = 2*D*S*s*(f_12_1*np.sin(k@next_n_n[0])+f_12_2*np.sin(k@next_n_n[1])+f_12_3*np.sin(k@next_n_n[2]))
    
    # RR
    f_21_1 = e_in_2@next_n_n[0]*e_out_1@next_n_n[0]
    f_21_2 = e_in_2@next_n_n[1]*e_out_1@next_n_n[1]
    f_21_3 = e_in_2@next_n_n[2]*e_out_1@next_n_n[2]
    #fk_21 = 1j*2*D*S*(f_21_1*np.exp(-1j*k@next_n_n[0])+f_21_2*np.exp(-1j*k@next_n_n[1])+f_21_3*np.exp(-1j*k@next_n_n[2]))
    fk_21 = 2*D*S*s*(f_21_1*np.sin(k@next_n_n[0])+f_21_2*np.sin(k@next_n_n[1])+f_21_3*np.sin(k@next_n_n[2]))
    
    # RL
    f_22_1 = e_in_2@next_n_n[0]*e_out_2@next_n_n[0]
    f_22_2 = e_in_2@next_n_n[1]*e_out_2@next_n_n[1]
    f_22_3 = e_in_2@next_n_n[2]*e_out_2@next_n_n[2]
    #fk_22 = 1j*2*D*S*(f_22_1*np.exp(-1j*k@next_n_n[0])+f_22_2*np.exp(-1j*k@next_n_n[1])+f_22_3*np.exp(-1j*k@next_n_n[2]))
    fk_22 = 2*D*S*s*(f_22_1*np.sin(k@next_n_n[0])+f_22_2*np.sin(k@next_n_n[1])+f_22_3*np.sin(k@next_n_n[2]))

    # set to zero to check it influence in frequency-resolved RCD
    fk_11 = fk_11*1
    fk_12 = fk_12*1
    fk_21 = fk_21*1
    fk_22 = fk_22*1
    
    # raman hamiltonian with different combination
    # LR (actually RL)
    ramham11 = np.array([[g0_11+fk_11,-v*gkb_11,0,(1-v)*gkb_11],[-v*gk_11,g0_11-fk_11,(1-v)*gk_11,0],
                        [0,(1-v)*gkb_11,g0_11-fk_11,-v*gkb_11],[(1-v)*gk_11,0,-v*gk_11,g0_11+fk_11]])
    
    # LL (actually RR)
    ramham12 = np.array([[g0_12+fk_12,-v*gkb_11,0,(1-v)*gkb_12],[-v*gk_12,g0_12-fk_12,(1-v)*gk_12,0],
                        [0,(1-v)*gkb_12,g0_12-fk_12,-v*gkb_12],[(1-v)*gk_12,0,-v*gk_12,g0_11+fk_12]])
    
    # RR (actually LL)
    ramham21 = np.array([[g0_21+fk_21,-v*gkb_21,0,(1-v)*gkb_21],[-v*gk_21,g0_21-fk_21,(1-v)*gk_21,0],
                        [0,(1-v)*gkb_21,g0_21-fk_21,-v*gkb_21],[(1-v)*gk_21,0,-v*gk_21,g0_21+fk_21]])
    
    # RL (actually LR)
    ramham22 = np.array([[g0_22+fk_22,-v*gkb_22,0,(1-v)*gkb_22],[-v*gk_22,g0_22-fk_22,(1-v)*gk_22,0],
                        [0,(1-v)*gkb_22,g0_22-fk_22,-v*gkb_22],[(1-v)*gk_22,0,-v*gk_22,g0_22+fk_22]])
    
    # change to correct basis to take the matrix elements out directly
    if qq == 0:
        Hr1 = Uik@ramham11@Uk # LR (actually RL)
        return Hr1
    elif qq == 1:
        Hr2 = Uik@ramham12@Uk # LL (actually RR)
        return Hr2
    elif qq == 2:
        Hr3 = Uik@ramham21@Uk # RR (actually LL)
        return Hr3
    elif qq == 3:
        Hr4 = Uik@ramham22@Uk # RL (acutally LR)
        return Hr4

def get_raman_cross_section(qq=0, J=1, D=0.1, S=5/2, B0=0.5):
    """
    Compute the Raman cross section for a given set of parameters.

    Parameters:
        qq (int): 0: RL; 1: RR; 2: LL, 3: LR.
        J (float): Heisenberg exchange interaction in mev
        D (float): DM interaction in meV
        S (float): Spin number
        B0 (float): Saturation field ratio (sin\theta)

    Returns:
        p_13 (np.array): Two magnon pair creation (same modes but opposite momentum) for upper band
        p_24 (np.array): Two magnon pair creation (same modes but opposite momentum) for lower band

        p_12 (np.array): FM process (interband) for upper band
        p_34 (np.array): FM process (interband) for lower band

        p_14 (np.array): AFM process (pair creation with different modes) for upper band
        p_23 (np.array): AFM process (pair creation with different modes) for lower band
    """
     # k-mesh
    kx,ky = bzmesh(m=2)
     
    # two-magnon pair creation (same modes but opposite momentum)
    p_13 = np.zeros(kx.shape)
    p_24 = np.zeros(kx.shape)
    
    # FM process (interband)
    p_12 = np.zeros(kx.shape)
    p_34 = np.zeros(kx.shape)
    
    # AFM process (pair creation with different modes)
    p_14 = np.zeros(kx.shape)
    p_23 = np.zeros(kx.shape)
    
    kx = kx[0]
    ky = ky.T[0]
    for i in range(ky.size):
        for j in range(kx.size):
            k = np.vstack((kx[j],ky[i])).T
            k = k.reshape(k.shape[1])
            Hr_qq = raman_cross_section_ham(k,qq=qq,J=J,D=D,S=S,B0=B0)
            
            # two magnon
            p_13[i,j] = np.abs(Hr_qq[0,2])**2
            p_24[i,j] = np.abs(Hr_qq[1,3])**2
            
            # FM
            p_12[i,j] = np.abs(Hr_qq[2,3])**2
            p_34[i,j] = np.abs(Hr_qq[0,1])**2
            
            # AFM
            p_14[i,j] = np.abs(Hr_qq[0,3])**2
            p_23[i,j] = np.abs(Hr_qq[1,2])**2
            
        if i%50 == 0:
            print(f'{i}: done')
    return p_13,p_24,p_12,p_34,p_14,p_23


J = 1 # meV
D = 0.1 # D/J = 0.1
S = 5/2 # spin number
s = 0.6 # saturation field ratio (sin\theta) = B/Bs

# %% RL
s_values = [0.25, 0.5, 0.75, 1]
D_values = [0.0125, 0.025, 0.05, 0.075, 0.1]
raman_cross_sections_RL = [np.array(get_raman_cross_section(qq=0,J=J,D=D,S=S,B0=s)) for s in s_values for D in D_values]
raman_cross_sections_LR = [np.array(get_raman_cross_section(qq=3,J=J,D=D,S=S,B0=s)) for s in s_values for D in D_values]

honeycomb_bz_x, honeycomb_bz_y = honeycomb_bz()

kx,ky = bzmesh(m=2)

# %%
color_bar_title_RL_upper = [r"$|t_{{\alpha}^{\prime}\bar{\alpha}^{\prime}}^{RL}|^2$",
                            r"$|t_{\bar{\beta}^{\prime}\bar{\alpha}}^{RL}|^2$",
                            r"$|t_{{\alpha}^{\prime}\bar{\beta}^{\prime}}^{RL}|^2$",] # 2M, FM, AFM

color_bar_title_RL_lower = [r"$|t_{{\beta}^{\prime}\bar{\beta}^{\prime}}^{RL}|^2$",
                            r"$|t_{\bar{\beta}^{\prime}\bar{\alpha}}^{RL}|^2$",
                            r"$|t_{{\beta}^{\prime}\bar{\alpha}^{\prime}^{RL}|^2$",] 

pads = [7, 2, 0]

with plt.style.context(['science','ieee']):
    fig, axes = panel(figsize=(12,3), nrows=1, ncols=3, width_ratios=[1, 1, 1], height_ratios=[1], hspace=0.1, wspace=0.25)

    fig.subplots_adjust(top=0.95, bottom=0.15, right=0.99)

    for i in range(3):
        pc = axes[i].pcolormesh(kx, ky, raman_cross_sections_RL[5][2*i]-raman_cross_sections_LR[5][2*i], cmap="jet")
        
        plot(honeycomb_bz_x, honeycomb_bz_y, ax=axes[i], linestyle='-', linewidth=1, color='k')

        clb = fig.colorbar(pc, ax=axes[i], shrink=0.9)
        clb.ax.set_title(color_bar_title_RL_upper[i], loc='left', fontsize=16, pad=pads[i])
        clb.ax.tick_params(labelsize=16)

        axes[i].set_axis_on() # make sure the axis is on
        axes[i].grid(False) # make sure the grid is off

        axes[i].set_xticks([-0.5 * 2 * np.pi, 0, 0.5 * 2 * np.pi])
        axes[i].set_xticklabels(['-1', '0', '1'], fontsize=16)
        axes[i].set_yticks([-0.5 * 2 * np.pi, 0, 0.5 * 2 * np.pi])
        axes[i].set_yticklabels(['-1', '0', '1'], fontsize=16)

        axes[i].set_xlabel(r'$k_x(\pi/a)$', fontsize=18)
        axes[i].set_ylabel(r'$k_y(\pi/a)$', fontsize=18)
    plt.show()
# %%

