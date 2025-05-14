import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from plot_utils import plot, letter_annotation, panel
from mathfuntion import Im, Re, is_invertible
from honeycomb_lattice import *
import scienceplots
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


def raman_cross_section_ham(k,qq=0, J1=2.01, J2=0.16, J3=-0.08, D=0.31, Az=0.49 ,S=3/2):
    '''
    for each k, we compute the elements of raman hamiltonian by proposed shortcut
    R: right circular polarized 
    L: left circular polarized 
    e_s: s-circular polarization vectors
    Parameters:
        k (np.ndarray): The k-vectors.
        qq (int): The configuration of the Raman cross-section.
        J1 (float): Nearest neighbour Heisenberg coupling.
        J2 (float): Next nearest neighbour Heisenberg coupling.
        J3 (float): Next next nearest neighbour Heisenberg coupling.
        Az (float): Anisotropy.
        D (float): DMI coupling.
        S (float): Spin number.
    Returns:
        Hr1 (np.ndarray): The Raman Hamiltonian for the first configuration.
        Hr2 (np.ndarray): The Raman Hamiltonian for the second configuration.
        Hr3 (np.ndarray): The Raman Hamiltonian for the third configuration.
        Hr4 (np.ndarray): The Raman Hamiltonian for the fourth configuration.
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
    
    Uk = np.array([u1,u2]).T
    E = np.array([ep,em])
    Uik = Uk.conj().T 
    
    # chirality and polarisation
    zeta = 1 # +1: right; -1:left
    # left circularly polarized: (-i,1)/sqrt(2); right circularly polarized: (i,1)/sqrt(2)
    # s: out; s':in
    # e_s^* = e_out , e_s'=e_in
    e_in_1 = (1/np.sqrt(2))*np.array([-1j*zeta,1]) # L
    e_in_2 = (1/np.sqrt(2))*np.array([1j*zeta,1]) # R = (L)^*
    e_out_1 = (1/np.sqrt(2))*np.array([-1j*zeta,1]) #  L = (R)^* 
    e_out_2 = (1/np.sqrt(2))*np.array([1j*zeta,1]) # R
    
    # if extra phase factors are present
    phi = 0*np.pi # rotated angle around normal
    theta = 0*np.pi # angle tilted away from x axis (no longer orthogonal)
    e_in_1 = (1/np.sqrt(2))*np.array([-np.sin(phi)-1j*zeta*np.cos(phi)*np.cos(theta),np.cos(phi)-1j*zeta*np.sin(phi)*np.cos(theta)])
    e_in_2 = (1/np.sqrt(2))*np.array([-np.sin(phi)+1j*zeta*np.cos(phi)*np.cos(theta),np.cos(phi)+1j*zeta*np.sin(phi)*np.cos(theta)])
    e_out_1 = (1/np.sqrt(2))*np.array([-np.sin(phi)-1j*zeta*np.cos(phi)*np.cos(theta),np.cos(phi)-1j*zeta*np.sin(phi)*np.cos(theta)])
    e_out_2 = (1/np.sqrt(2))*np.array([-np.sin(phi)+1j*zeta*np.cos(phi)*np.cos(theta),np.cos(phi)+1j*zeta*np.sin(phi)*np.cos(theta)])
    
    #shortcut, H_R^{ss'}=0.5*(L_xx+ss'+L_yy+i(s'-s)L_xy)
    # second derivatives of parameters
    gamma_1xx = -1*np.dot(n_n.T[0]*n_n.T[0],np.exp(1j*k@n_n.T))*J1*S
    gamma_1xy = -1*np.dot(n_n.T[0]*n_n.T[1],np.exp(1j*k@n_n.T))*J1*S
    gamma_1yy = -1*np.dot(n_n.T[1]*n_n.T[1],np.exp(1j*k@n_n.T))*J1*S
    gamma_3xx = -1*np.dot(next_next_n_n.T[0]*next_next_n_n.T[0],np.exp(1j*k@next_next_n_n.T))*J3*S
    gamma_3xy = -1*np.dot(next_next_n_n.T[0]*next_next_n_n.T[1],np.exp(1j*k@next_next_n_n.T))*J3*S
    gamma_3yy = -1*np.dot(next_next_n_n.T[1]*next_next_n_n.T[1],np.exp(1j*k@next_next_n_n.T))*J3*S
    lambda_kxx = -np.dot(next_n_n.T[0]*next_n_n.T[0],np.sin(k@next_n_n.T))*2*D*S
    lambda_kxy = -np.dot(next_n_n.T[0]*next_n_n.T[1],np.sin(k@next_n_n.T))*2*D*S
    lambda_kyy = -np.dot(next_n_n.T[1]*next_n_n.T[1],np.sin(k@next_n_n.T))*2*D*S
    gamma_2xx = -np.dot(next_n_n.T[0]*next_n_n.T[0],np.cos(k@next_n_n.T))*2*J2*S
    gamma_2xy = -np.dot(next_n_n.T[0]*next_n_n.T[1],np.cos(k@next_n_n.T))*2*J2*S
    gamma_2yy = -np.dot(next_n_n.T[1]*next_n_n.T[1],np.cos(k@next_n_n.T))*2*J2*S

    L_xx = np.array([[-gamma_2xx+lambda_kxx, -gamma_1xx-gamma_3xx],[-gamma_1xx.conj()-gamma_3xx.conj(), -gamma_2xx-lambda_kxx]])
    L_xy = np.array([[-gamma_2xy+lambda_kxy, -gamma_1xy-gamma_3xy],[-gamma_1xy.conj()-gamma_3xy.conj(), -gamma_2xy-lambda_kxy]])
    L_yy = np.array([[-gamma_2yy+lambda_kyy, -gamma_1yy-gamma_3yy],[-gamma_1yy.conj()-gamma_3yy.conj(), -gamma_2yy-lambda_kyy]])
    
    # raman hamiltonian with different combination
    L = 1
    R = -1
    # RL (s'=L=1, s=R=-1)
    ramham11 = 0.5*(L_xx+R*L*L_yy+1j*(L-R)*L_xy)
    
    # LL (s'=L=1, s=L=1)
    ramham12 = 0.5*(L_xx+L*L*L_yy+1j*(L-L)*L_xy)
    
    # RR 
    ramham21 = 0.5*(L_xx+R*R*L_yy+1j*(R-R)*L_xy)
    
    # LR
    ramham22 = 0.5*(L_xx+L*R*L_yy+1j*(R-L)*L_xy)
    
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

def get_raman_cross_section(J1=2.01, J2=0.16, J3=-0.08, D=0.31, Az=0.49 ,S=3/2):
    """
    Compute the Raman cross section for a given set of parameters.

    Parameters:
        qq (int): 0: RL; 1: RR; 2: LL, 3: LR.
        J1 (float): Nearest neighbour Heisenberg coupling.
        J2 (float): Next nearest neighbour Heisenberg coupling.
        J3 (float): Next next nearest neighbour Heisenberg coupling.
        D (float): DMI coupling.
        Az (float): Anisotropy.
        S (float): Spin number.
    Returns:
        p_12_RL (np.ndarray): The Raman cross section for L in R out.
        p_12_LL (np.ndarray): The Raman cross section for L in L out.
        p_12_RR (np.ndarray): The Raman cross section for R in R out.
        p_12_LR (np.ndarray): The Raman cross section for R in L out.
    """
     # k-mesh
    kx,ky = bzmesh(m=2)
     
    # FM process (interband)
    p_12_RL = np.zeros(kx.shape)
    p_12_LL = np.zeros(kx.shape)
    p_12_RR = np.zeros(kx.shape)
    p_12_LR = np.zeros(kx.shape)
    
    kx = kx[0]
    ky = ky.T[0]
    for i in range(ky.size):
        for j in range(kx.size):
            k = np.vstack((kx[j],ky[i])).T
            k = k.reshape(k.shape[1])

            Hr_RL = raman_cross_section_ham(k,qq=0,J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S)
            Hr_LL = raman_cross_section_ham(k,qq=1,J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S)
            Hr_RR = raman_cross_section_ham(k,qq=2,J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S)
            Hr_LR = raman_cross_section_ham(k,qq=3,J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S)
        
            
            # FM
            p_12_RL[i,j] = np.abs(Hr_RL[0,1])**2
            p_12_LL[i,j] = np.abs(Hr_LL[0,1])**2
            p_12_RR[i,j] = np.abs(Hr_RR[0,1])**2
            p_12_LR[i,j] = np.abs(Hr_LR[0,1])**2
            
        if i%50 == 0:
            print(f'{i}: done with parameters J1={J1}, J2={J2} , J3={J3}, D={D}, Az={Az}, S={S}')
    return np.array([p_12_RL, p_12_LL, p_12_RR, p_12_LR])

def get_RCD(J1=2.01, J2=0.16, J3=-0.08, D=0.31, Az=0.49 ,S=3/2):
    """
    Compute the RCD for a given set of parameters in CrI3.

    Parameters:
        qq (int): 0: RL; 1: RR; 2: LL, 3: LR.
        J1 (float): Nearest neighbour Heisenberg coupling.
        J2 (float): Next nearest neighbour Heisenberg coupling.
        J3 (float): Next next nearest neighbour Heisenberg coupling.
        D (float): DMI coupling.
        Az (float): Anisotropy.
        S (float): Spin number.
    Returns:
        p_12_RL (np.ndarray): The Raman cross section for L in R out.
        p_12_LL (np.ndarray): The Raman cross section for L in L out.
        p_12_RR (np.ndarray): The Raman cross section for R in R out.
        p_12_LR (np.ndarray): The Raman cross section for R in L out.
    """
    # parameters
    M = 3*J1*S+6*J2*S+3*J3*S
    
    # lattice structure
    # a = 6.324 # 6.324 Å
    n_n = np.sqrt(3)*np.array([[0,1/np.sqrt(3)],[-1/2,-0.5/np.sqrt(3)],[1/2,-0.5/np.sqrt(3)]])
    next_n_n = -np.sqrt(3)*np.array([[1/2,np.sqrt(3)/2],[-1,0],[1/2,-np.sqrt(3)/2]]) # next nearest neighbour vectors
    next_next_n_n =  np.array([[0, -2], [np.sqrt(3),1], [-np.sqrt(3),1]]) # next next nearest neighbour vectors

     # k-mesh
    kx,ky = bzmesh(m=2)
     
    # RCD in CrI3
    RCD = np.zeros(kx.shape)

    kx = kx[0]
    ky = ky.T[0]
    for i in range(ky.size):
        for j in range(kx.size):
            k = np.vstack((kx[j],ky[i])).T
            k = k.reshape(k.shape[1])
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
            
            Uk = np.array([u1,u2]).T
            E = np.array([ep,em])
            Uik = Uk.conj().T

            #shortcut, H_R^{ss'}=0.5*(L_xx+ss'+L_yy+i(s'-s)L_xy)
            # second derivatives of parameters
            gamma_1xx = -1*np.dot(n_n.T[0]*n_n.T[0],np.exp(1j*k@n_n.T))*J1*S
            gamma_1xy = -1*np.dot(n_n.T[0]*n_n.T[1],np.exp(1j*k@n_n.T))*J1*S
            gamma_1yy = -1*np.dot(n_n.T[1]*n_n.T[1],np.exp(1j*k@n_n.T))*J1*S
            gamma_3xx = -1*np.dot(next_next_n_n.T[0]*next_next_n_n.T[0],np.exp(1j*k@next_next_n_n.T))*J3*S
            gamma_3xy = -1*np.dot(next_next_n_n.T[0]*next_next_n_n.T[1],np.exp(1j*k@next_next_n_n.T))*J3*S
            gamma_3yy = -1*np.dot(next_next_n_n.T[1]*next_next_n_n.T[1],np.exp(1j*k@next_next_n_n.T))*J3*S
            lambda_kxx = -np.dot(next_n_n.T[0]*next_n_n.T[0],np.sin(k@next_n_n.T))*2*D*S
            lambda_kxy = -np.dot(next_n_n.T[0]*next_n_n.T[1],np.sin(k@next_n_n.T))*2*D*S
            lambda_kyy = -np.dot(next_n_n.T[1]*next_n_n.T[1],np.sin(k@next_n_n.T))*2*D*S
            gamma_2xx = -np.dot(next_n_n.T[0]*next_n_n.T[0],np.cos(k@next_n_n.T))*2*J2*S
            gamma_2xy = -np.dot(next_n_n.T[0]*next_n_n.T[1],np.cos(k@next_n_n.T))*2*J2*S
            gamma_2yy = -np.dot(next_n_n.T[1]*next_n_n.T[1],np.cos(k@next_n_n.T))*2*J2*S

            L_xx = np.array([[-gamma_2xx+lambda_kxx, -gamma_1xx-gamma_3xx],[-gamma_1xx.conj()-gamma_3xx.conj(), -gamma_2xx-lambda_kxx]])
            L_xy = np.array([[-gamma_2xy+lambda_kxy, -gamma_1xy-gamma_3xy],[-gamma_1xy.conj()-gamma_3xy.conj(), -gamma_2xy-lambda_kxy]])
            L_yy = np.array([[-gamma_2yy+lambda_kyy, -gamma_1yy-gamma_3yy],[-gamma_1yy.conj()-gamma_3yy.conj(), -gamma_2yy-lambda_kyy]])
            
            L_tilde_xx = Uik@L_xx@Uk
            L_tilde_xy = Uik@L_xy@Uk
            L_tilde_yy = Uik@L_yy@Uk

            RCD[i,j] = 2*np.imag((L_tilde_xx[0,1]-L_tilde_yy[0,1])*L_tilde_xy[0,1].conj())
            
        if i%50 == 0:
            print(f'{i}: done with parameters J1={J1}, J2={J2} , J3={J3}, D={D}, Az={Az}, S={S}')
    return RCD

def get_energy(J1=2.01, J2=0.16, J3=-0.08, D=0.31, Az=0.49 ,S=3/2):
    """
    Compute the Raman cross section for a given set of parameters.

    Parameters:
        qq (int): 0: RL; 1: RR; 2: LL, 3: LR.
        J1 (float): Nearest neighbour Heisenberg coupling.
        J2 (float): Next nearest neighbour Heisenberg coupling.
        J3 (float): Next next nearest neighbour Heisenberg coupling.
        D (float): DMI coupling.
        Az (float): Anisotropy.
        S (float): Spin number.
    Returns:
        energy (np.ndarray): energy over BZ
    """
     # k-mesh
    kx,ky = bzmesh(m=2)
    em = np.zeros(kx.shape)
    ep = np.zeros(kx.shape)

    # parameters
    M = 3*J1*S+6*J2*S+3*J3*S
    
    # lattice structure
#     a = 6.324 # 6.324 Å
    n_n = np.sqrt(3)*np.array([[0,1/np.sqrt(3)],[-1/2,-0.5/np.sqrt(3)],[1/2,-0.5/np.sqrt(3)]])
    next_n_n = -np.sqrt(3)*np.array([[1/2,np.sqrt(3)/2],[-1,0],[1/2,-np.sqrt(3)/2]]) # next nearest neighbour vectors
    next_next_n_n =  np.array([[0, -2], [np.sqrt(3),1], [-np.sqrt(3),1]]) # next next nearest neighbour vectors

    # vectorizing function
    dot = np.vectorize(np.dot,signature='(n),(m)->()')
    kx = kx[0]
    ky = ky.T[0]
    for i in range(ky.size):
        for j in range(kx.size):
            k = np.vstack((kx[j],ky[i])).T
            k = k.reshape(k.shape[1])
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
            em[i,j] = d0-d_abs
            ep[i,j] = d0+d_abs 
        if i%50 == 0:
            print(f'{i}: done with parameters J1={J1}, J2={J2} , J3={J3}, D={D}, Az={Az}, S={S}')
    return np.array([ep,em])  

# %%

J1 = 2.01 # n.n Heisenberg coupling meV
J2 = 0.16 # n.n.n Heisenberg coupling meV
J3 = -0.08 # n.n.n.n Heisenberg coupling meV
D = 0.31 # DMI meV
Az = 0.49 # anisotropy
S = 3/2 # spin number

# raman_cross_sections = get_raman_cross_section(J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S)
# RCD = raman_cross_sections[0]+raman_cross_sections[2] - raman_cross_sections[1]-raman_cross_sections[3]
RCD = get_RCD(J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S)

honeycomb_bz_x, honeycomb_bz_y = honeycomb_bz()

kx,ky = bzmesh(m=2)


# %%
gap = get_energy(J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S)

# %%
color_bar_title_cross_section = [r"$|t_{+-}^{RL}|^2$",
                                r"$|t_{+-}^{LL}|^2$",
                                r"$|t_{+-}^{RR}|^2$",
                                r"$|t_{+-}^{LR}|^2$",] 

pads = [7, 2, 0, 0]

# with plt.style.context(['science','ieee']):
#     fig, axes = panel(figsize=(16,3), nrows=1, ncols=4, width_ratios=[1, 1, 1, 1], height_ratios=[1], hspace=0.1, wspace=0.25)

#     fig.subplots_adjust(top=0.95, bottom=0.15, right=0.99)

#     for i in range(4):
#         pc = axes[i].pcolormesh(kx, ky, raman_cross_sections[i], cmap="jet")
        
#         plot(honeycomb_bz_x, honeycomb_bz_y, ax=axes[i], linestyle='-', linewidth=1, color='k')

#         clb = fig.colorbar(pc, ax=axes[i], shrink=0.9)
#         clb.ax.set_title(color_bar_title_cross_section[i], loc='left', fontsize=16, pad=pads[i])
#         clb.ax.tick_params(labelsize=16)

#         axes[i].set_axis_on() # make sure the axis is on
#         axes[i].grid(False) # make sure the grid is off

#         axes[i].set_xticks([-0.5 * 2 * np.pi, 0, 0.5 * 2 * np.pi])
#         axes[i].set_xticklabels(['-1', '0', '1'], fontsize=16)
#         axes[i].set_yticks([-0.5 * 2 * np.pi, 0, 0.5 * 2 * np.pi])
#         axes[i].set_yticklabels(['-1', '0', '1'], fontsize=16)

#         axes[i].set_xlabel(r'$k_x(\pi/a)$', fontsize=18)
#         axes[i].set_ylabel(r'$k_y(\pi/a)$', fontsize=18)
#     plt.show()
# %%
with plt.style.context(['science','ieee']):
    fig, axes = panel(figsize=(4,3), 
                     nrows=1, ncols=1, 
                     width_ratios=[1], height_ratios=[1])

    fig.subplots_adjust(top=0.95, bottom=0.15, right=0.99)

    pc = axes.pcolormesh(kx, ky, RCD, cmap="jet")
            
    plot(honeycomb_bz_x, honeycomb_bz_y, ax=axes, linestyle='-', linewidth=1, color='k')

    clb = fig.colorbar(pc, ax=axes, shrink=0.9)
    clb.ax.set_title(r"$\chi$", loc='left', fontsize=16, pad=5)
    clb.ax.tick_params(labelsize=16)

    axes.set_axis_on() # make sure the axis is on
    axes.grid(False) # make sure the grid is off

    axes.set_xticks([-0.5 * 2 * np.pi, 0, 0.5 * 2 * np.pi])
    axes.set_xticklabels(['-1', '0', '1'], fontsize=16)
    axes.set_yticks([-0.5 * 2 * np.pi, 0, 0.5 * 2 * np.pi])
    axes.set_yticklabels(['-1', '0', '1'], fontsize=16)

    axes.set_xlabel(r'$k_x(\pi/a)$', fontsize=18)
    axes.set_ylabel(r'$k_y(\pi/a)$', fontsize=18)
    #axes.set_title('RCD of '+r"$CrI_3$", fontsize=18)
    plt.show()
    #fig.savefig('figures/panel_plots/figure9_RCD_lower.png', dpi=300, bbox_inches='tight')
# %%
print(gap.max())
# %%
