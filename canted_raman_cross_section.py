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
    e_in_1 = (1/np.sqrt(2))*np.array([-1j*zeta,1]) # L 
    e_in_2 = (1/np.sqrt(2))*np.array([1j*zeta,1]) # R
    e_out_1 = (1/np.sqrt(2))*np.array([-1j*zeta,1]) # R
    e_out_2 = (1/np.sqrt(2))*np.array([1j*zeta,1]) # L
    
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
    ramham12 = np.array([[g0_12+fk_12,-v*gkb_12,0,(1-v)*gkb_12],[-v*gk_12,g0_12-fk_12,(1-v)*gk_12,0],
                        [0,(1-v)*gkb_12,g0_12-fk_12,-v*gkb_12],[(1-v)*gk_12,0,-v*gk_12,g0_12+fk_12]])
    
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


# %%
# %%
def get_raman_cross_section_exact(J=1,D=0.1,S=5/2,B0=0.5, f=1):
    '''
    Compute the Raman cross section for a given set of parameters.

    Parameters:
        J (float): Heisenberg exchange interaction in meV
        D (float): DM interaction in meV
        S (float): Spin number
        B0 (float): Saturation field ratio (sin\theta)
        f (int): turn on fk (1) or off (0)

    Returns:
        p_13 (np.array): Two magnon pair creation (same modes but opposite momentum) for upper band
        p_24 (np.array): Two magnon pair creation (same modes but opposite momentum) for lower band
        p_12 (np.array): FM process (interband) for upper band
        p_34 (np.array): FM process (interband) for lower band
        p_14 (np.array): AFM process (pair creation with different modes) for upper band
        p_23 (np.array): AFM process (pair creation with different modes) for lower band
    '''
    # field
    Bs = 3*J*S # not 6JS here
    B = (B0-0.001)*Bs
    s = B/Bs
    anisotropy_z = 0.22
    
    # parameters
    M = Bs + 2*anisotropy_z*S
    v = s**2
    
    # lattice parameters
    a = 1
    n_n = a*np.sqrt(3)*np.array([[0,1/np.sqrt(3)],[1/2,-0.5/np.sqrt(3)],[-1/2,-0.5/np.sqrt(3)]]) #a1,a2,a3
    next_n_n = -a*np.sqrt(3)*np.array([[-1,0],[1/2,np.sqrt(3)/2],[1/2,-np.sqrt(3)/2]]) # b1,b2,b3
    
    no_n_n = n_n.shape[0]
    no_next_n_n = next_n_n.shape[0]
    
    x_n_n = np.array([1,0])@n_n.T
    y_n_n = np.array([0,1])@n_n.T
    xy_n_n = np.outer(x_n_n,y_n_n)
    diff_n_n = np.array((n_n[0]-n_n,n_n[1]-n_n,n_n[2]-n_n)) # a_i-a_j
    x_next_n_n = np.array([1,0])@next_n_n.T
    y_next_n_n = np.array([0,1])@next_n_n.T
    
    n_n_xy_diff = x_n_n**2-y_n_n**2
    n_n_xy_prod = x_n_n*y_n_n
    next_n_n_xy_diff = x_next_n_n**2-y_next_n_n**2
    next_n_n_xy_prod = x_next_n_n*y_next_n_n
    
    next_xy_nn = np.outer(x_next_n_n,y_n_n)-np.outer(y_next_n_n,x_n_n)
    
    # vectorizing function
    dot = np.vectorize(np.dot,signature='(n),(m)->()')
    
    # chirality and polarisation
    zeta = 1 # +1: right; -1:left
    # right circularly polarized: (-i,1)/sqrt(2); left circularly polarized: (i,1)/sqrt(2)
    # e_in_1 -> (e_in_R)^*; e_in_2 -> (e_in_L)^*;
    # e_out_1 -> (e_out_L); e_out_2 -> (e_out_R)
    e_in_1 = (1/np.sqrt(2))*np.array([-1j*zeta,1])
    e_in_2 = (1/np.sqrt(2))*np.array([1j*zeta,1])
    e_out_1 = (1/np.sqrt(2))*np.array([-1j*zeta,1])
    e_out_2 = (1/np.sqrt(2))*np.array([1j*zeta,1])
    
    # calculate
    kx,ky = bzmesh(m=2)
     
    # two-magnon pair creation (same modes but opposite momentum), FM, AFM
    p_13 = np.zeros(kx.shape)
    p_24 = np.zeros(kx.shape)
    p_12 = np.zeros(kx.shape)
    p_14 = np.zeros(kx.shape)
    energy_lower = np.zeros(kx.shape)
    energy_upper = np.zeros(kx.shape)
    berry_p = np.zeros(kx.shape)
    berry_m = np.zeros(kx.shape)
    berry_rcd_p = np.zeros(kx.shape)
    berry_rcd_m = np.zeros(kx.shape)
    
    kx = kx[0]
    ky = ky.T[0]
    for i in range(ky.size):
        for j in range(kx.size):
            k = np.vstack((kx[j],ky[i])).T
            k = k.reshape(k.shape[1])
            
            gamma = np.sum(np.exp(-1j*k@n_n.T))
            gamma_sin = np.sum(np.sin(k@next_n_n.T))
            
            # complex parameters
            phi_k = J*S*gamma
            phi_bar_k = np.conjugate(phi_k) 
            aphi_k = np.abs(phi_k)**2
            lambda_k = 2*D*S*s*gamma_sin
            delta_k = np.sqrt(lambda_k**2+v**2*aphi_k)
            # energy band -,+
            Em = M-delta_k
            Ep = M+delta_k
            rho_k = (1-v)*np.abs(phi_k)
            em = np.emath.sqrt(Em**2-rho_k**2)
            ep = np.emath.sqrt(Ep**2-rho_k**2)
            energy_lower[i,j] = em
            energy_upper[i,j] = ep

            # subspace parameters  
            cosh1 = np.emath.sqrt((Ep+ep)/(2*ep))
            sinh1 = np.emath.sqrt((Ep-ep)/(2*ep))
            cosh2 = np.emath.sqrt((Em+em)/(2*em))
            sinh2 = -np.emath.sqrt((Em-em)/(2*em))
            sinh1Double = 2*sinh1*cosh1
            cosh1Double = sinh1**2+cosh1**2
            sinh2Double = 2*sinh2*cosh2
            cosh2Double = sinh2**2+cosh2**2
            sinh_12_p = sinh1*cosh2+sinh2*cosh1
            sinh_12_m = sinh1*cosh2-sinh2*cosh1
            cosh_12_p = cosh1*cosh2+sinh1*sinh2
            cosh_12_m = cosh1*cosh2-sinh2*sinh1

            cos = np.emath.sqrt((delta_k+lambda_k)/(2*delta_k))
            sin = np.emath.sqrt((delta_k-lambda_k)/(2*delta_k))
            sinDouble = 2*sin*cos
            cosDouble = cos**2-sin**2
            
            phase = 1j*np.log(phi_k/np.abs(phi_k))
            
            # LR
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

            # RL
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

            # set fk to zero (i dont know why)
            fk_11 = fk_11*f
            fk_12 = fk_12*f
            fk_21 = fk_21*f
            fk_22 = fk_22*f

            t_RL_1 = fk_11*sinh_12_m*sinDouble
            t_RL_2 = 0.5*(1-v)*(gk_11*np.exp(1j*phase)-gkb_11*np.exp(-1j*phase.conj()))*cosh_12_p*sinDouble
            t_RL_3 = (-v*gk_11)*(cosh1*sinh2*sin**2+sinh1*cosh2*cos**2)*np.exp(1j*phase)+(-v*gkb_11)*(-cosh1*sinh2*cos**2-sinh1*cosh2*sin**2)*np.exp(-1j*phase)
            t_RL = t_RL_3+t_RL_1+t_RL_2
            
            t_LR_1 = fk_22*sinh_12_m*sinDouble
            t_LR_2 = 0.5*(1-v)*(gk_22*np.exp(1j*phase)-gkb_22*np.exp(-1j*phase.conj()))*cosh_12_p*sinDouble
            t_LR_3 = (-v*gk_22)*(cosh1*sinh2*sin**2+sinh1*cosh2*cos**2)*np.exp(1j*phase)+(-v*gkb_22)*(-cosh1*sinh2*cos**2-sinh1*cosh2*sin**2)*np.exp(-1j*phase)
            t_LR = t_LR_3+t_LR_1+t_LR_2
            nu = 0
            rho = 0
            rho_tilde = 0
            for l in range(no_n_n):
                nu += (J*S)**2*np.dot(xy_n_n[l],np.sin(k@diff_n_n[l].T))
            for n in range(no_next_n_n):
                for m in range(no_n_n):
                    rho += np.sin(k@next_n_n[n].T)*(next_n_n_xy_prod[n]*n_n_xy_diff[m]-n_n_xy_prod[m]*next_n_n_xy_diff[n])*np.sin(phase-k@n_n[m].T)
                    rho_tilde += next_xy_nn[n,m]*np.cos(k@next_n_n[n])*np.cos(phase-k@n_n[m].T)
            rho = 4*J*D*S**2*s*rho
            rho_tilde = 2*J*D*S**2*s*rho_tilde
            
            xi_p_k = (1-v)*sinh1Double*cosDouble**2
            xi_m_k = (1-v)*sinh2Double*cosDouble**2 # for two-magnon pair
            zeta_k = cosh_12_m*sinDouble*((1-v)*sinh_12_p*sinDouble-v*cosh_12_p) # for FM
            sigma_k = sinh_12_m*sinDouble*((1-v)*cosh_12_p*sinDouble-v*sinh_12_p) # for AFM
            
            # (alpha mode -> +; beta mode -> -)
            
            # two magnon
            p_13_1 = -(1-v)**2*cosh1Double
            p_13_2 = v*(1-v)*sinh1Double*sinDouble
            p_13[i,j] = a**2*nu*cosDouble*(p_13_1+p_13_2)+rho*xi_p_k
            p_24_1 = -(1-v)**2*cosh2Double
            p_24_2 = v*(1-v)*sinh2Double*sinDouble
            p_24[i,j] = -a**2*nu*cosDouble*(p_24_1+p_24_2)-rho*xi_m_k
            
            # FM
            p_12_1 = v**2*cosh_12_p*cosh_12_m
            p_12_2 = -v*(1-v)*cosh_12_m*sinh_12_p*sinDouble
            p_12[i,j] =  -a**2*nu*cosDouble*(p_12_1+p_12_2)-rho*zeta_k
            
            # AFM
            p_14_1 = v**2*sinh_12_p*sinh_12_m
            p_14_2 = -v*(1-v)*sinh_12_m*cosh_12_p*sinDouble
            p_14[i,j] = a**2*nu*cosDouble*(p_14_1+p_14_2)+rho*sigma_k
            
            # berry curvature
#             berry_p_1 = p_13[i,j]/(4*ep)**2+p_14[i,j]/(ep+em)**2-p_12[i,j]/(ep-em)**2
#             berry_p_2 = (xi_p_k/(4*ep)**2+sigma_k/(ep+em)**2+zeta_k/(ep-em)**2)*(rho-rho_tilde)
#             berry_p[i,j] = (2/a**2)*(berry_p_1-berry_p_2)
            berry_p_1 = nu*cosDouble*(p_12_1+p_12_2)-(rho_tilde)*zeta_k # FM
            berry_p_2 = nu*cosDouble*(p_14_1+p_14_2)-(rho_tilde)*sigma_k # AFM
            berry_p_3 = nu*cosDouble*(p_13_1+p_13_2)-(rho_tilde)*xi_p_k # two magnon
            berry_p[i,j] = -2*(berry_p_3/(ep+ep)**2+berry_p_2/(ep+em)**2+berry_p_1/(em-ep)**2)
            berry_rcd_p[i,j] = -2*(p_13[i,j]/(ep+ep)**2+p_14[i,j]/(ep+em)**2-p_12[i,j]/(em-ep)**2)+2*(xi_p_k/(ep+ep)**2+sigma_k/(ep+em)**2+zeta_k/(em-ep)**2)*(rho+rho_tilde)
            
            berry_m_1 = -nu*cosDouble*(p_12_1+p_12_2)+(rho_tilde)*zeta_k # FM
            berry_m_2 = nu*cosDouble*(p_14_1+p_14_2)-(rho_tilde)*sigma_k # AFM
            berry_m_3 = -nu*cosDouble*(p_24_1+p_24_2)+(rho_tilde)*xi_m_k # two magnon
            berry_m[i,j] = -2*(berry_m_3/(em+em)**2+berry_m_2/(ep+em)**2+berry_m_1/(ep-em)**2)
            berry_rcd_m[i,j] = -2*(p_24[i,j]/(em+em)**2+p_14[i,j]/(ep+em)**2+p_12[i,j]/(ep-em)**2)-2*(xi_m_k/(em+em)**2-sigma_k/(ep+em)**2+zeta_k/(ep-em)**2)*(rho+rho_tilde)
            
        if i%50 == 0:
                print(f'{i}: done with parameterts J={J}, D={D}, S={S}, B0={B0}')
    p_array = np.array([p_13,p_24, p_12,-p_12,p_14,p_14])
    berry_array = np.array([berry_p,berry_m])
    berry_rcd_array = np.array([berry_rcd_p,berry_rcd_m])
    energy_array = np.array([energy_upper,energy_lower])
    #return p_array, berry_array, berry_rcd_array, energy_array
    return p_array
# %%

J = 1 # meV
D = 0.1 # D/J = 0.1
S = 5 # spin number
s = 0.74 # saturation field ratio (sin\theta) = B/Bs

# %% RL
s_values = [0.75,]
D_values = [0.1]
raman_cross_sections_RL = [np.array(get_raman_cross_section(qq=0,J=J,D=D,S=S,B0=s)) for s in s_values for D in D_values]
raman_cross_sections_RR = [np.array(get_raman_cross_section(qq=1,J=J,D=D,S=S,B0=s)) for s in s_values for D in D_values]
raman_cross_sections_LL = [np.array(get_raman_cross_section(qq=2,J=J,D=D,S=S,B0=s)) for s in s_values for D in D_values]
raman_cross_sections_LR = [np.array(get_raman_cross_section(qq=3,J=J,D=D,S=S,B0=s)) for s in s_values for D in D_values]

honeycomb_bz_x, honeycomb_bz_y = honeycomb_bz()

kx,ky = bzmesh(m=2)


# %%
color_bar_title_RL_upper = [r"$|t_{{\alpha}^{\prime}\bar{\alpha}^{\prime}}^{RL}|^2$",
                            r"$|t_{\alpha^{\prime}\beta}^{RL}|^2$",
                            r"$|t_{{\alpha}^{\prime}\bar{\beta}^{\prime}}^{RL}|^2$",] # 2M, FM, AFM

color_bar_title_RL_lower = [r"$|t_{{\beta}^{\prime}\bar{\beta}^{\prime}}^{RL}|^2$",
                            r"$|t_{\beta}^{\prime}\alpha}^{RL}|^2$",
                            r"$|t_{{\beta}^{\prime}\bar{\alpha}^{\prime}^{RL}|^2$",] 

title = [r"2M", r"AFM", r"FM"]

pads = [7, 2, 0]

# %%
intensity = np.array(raman_cross_sections_RL) + np.array(raman_cross_sections_RR) + np.array(raman_cross_sections_LL) + np.array(raman_cross_sections_LR)
RCD = np.array(raman_cross_sections_RL) + np.array(raman_cross_sections_RR) - np.array(raman_cross_sections_LL) - np.array(raman_cross_sections_LR)
with plt.style.context(['science','ieee']):
    fig, axes = panel(figsize=(12,3), nrows=1, ncols=3, width_ratios=[1, 1, 1], height_ratios=[1], hspace=0.1, wspace=0.25)

    fig.subplots_adjust(top=0.95, bottom=0.15, right=0.99)

    for i in range(3):
        pc = axes[i].pcolormesh(kx, ky, RCD[0][2*i+1] , cmap="jet")
        
        plot(honeycomb_bz_x, honeycomb_bz_y, ax=axes[i], linestyle='-', linewidth=1, color='k')

        clb = fig.colorbar(pc, ax=axes[i], shrink=0.9)
        clb.ax.set_title(title[i], loc='left', fontsize=16, pad=pads[i])
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
# %% single figure
with plt.style.context(['science','ieee']):
    fig, axes = panel(figsize=(4,3), nrows=1, ncols=1, width_ratios=[1], height_ratios=[1], hspace=0.1, wspace=0.25)

    fig.subplots_adjust(top=0.95, bottom=0.15, right=0.99)

    for i in range(1):
        pc = axes.pcolormesh(kx, ky, RCD[0][2*1], cmap="jet")
        
        plot(honeycomb_bz_x, honeycomb_bz_y, ax=axes, linestyle='-', linewidth=1, color='k')

        clb = fig.colorbar(pc, ax=axes, shrink=0.9)
        clb.ax.set_title(color_bar_title_RL_upper[1], loc='left', fontsize=16, pad=pads[2])
        clb.ax.tick_params(labelsize=16)

        axes.set_axis_on() # make sure the axis is on
        axes.grid(False) # make sure the grid is off

        axes.set_xticks([-0.5 * 2 * np.pi, 0, 0.5 * 2 * np.pi])
        axes.set_xticklabels(['-1', '0', '1'], fontsize=16)
        axes.set_yticks([-0.5 * 2 * np.pi, 0, 0.5 * 2 * np.pi])
        axes.set_yticklabels(['-1', '0', '1'], fontsize=16)

        axes.set_xlabel(r'$k_x(\pi/a)$', fontsize=18)
        axes.set_ylabel(r'$k_y(\pi/a)$', fontsize=18)
        axes.set_title(title[2], fontsize=18)
    plt.show()
   #fig.savefig('figures/raman_scattering/raman_cross_section_RL_FM_upper.png', dpi=300)
# %%
