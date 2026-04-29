# %%
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from common.plot_utils import plot, letter_annotation, panel
from common.math_utils import Im, Re, is_invertible
from common.honeycomb_lattice import *
import scienceplots
# %%

def get_RCD_exact(J=1,D=0.1,S=5/2,B0=0.5,m=0):
    """
    Calculate the RCD for a canted antiferromagnet on a honeycomb lattice.

    Parameters:
        J (float): Heisenberg exchange interaction in mev
        D (float): DM interaction in meV
        S (float): Spin number
        B0 (float): Saturation field ratio (sin\theta)

    Returns:
        RCD_array (np.ndarray): The RCD for different processes (2M, AFM, FM)
    """
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
            
        if i%50 == 0:
                print(f'Iteration {i}: B={B0:.2f}, D={D:.3f}: done')
    # RCD_array = np.array([p_13,p_24,p_12,-p_12,p_14,p_14])
    RCD_array = np.array([[p_13, p_14, -p_12],[p_24, p_14, p_12,]])
    return RCD_array

def get_RCD_LMC(J=1,D=0.1,S=5/2,B0=0.5,m=0):
    """
    Calculate the RCD for a canted antiferromagnet on a honeycomb lattice by LMCs

    Parameters:
        J (float): Heisenberg exchange interaction in mev
        D (float): DM interaction in meV
        S (float): Spin number
        B0 (float): Saturation field ratio (sin\theta)

    Returns:
        RCD_array (np.ndarray): The RCD for different processes (2M, AFM, FM)
    """
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

            # derivatives
            gamma_xx = -np.dot(n_n.T[0]*n_n.T[0], np.exp(-1j*k@n_n.T))
            gamma_xy = -np.dot(n_n.T[0]*n_n.T[1], np.exp(-1j*k@n_n.T))
            gamma_yy = -np.dot(n_n.T[1]*n_n.T[1], np.exp(-1j*k@n_n.T))
            gamma_sin_xx = -np.dot(next_n_n.T[0]*next_n_n.T[0], np.sin(k@next_n_n.T))
            gamma_sin_xy = -np.dot(next_n_n.T[0]*next_n_n.T[1], np.sin(k@next_n_n.T))
            gamma_sin_yy = -np.dot(next_n_n.T[1]*next_n_n.T[1], np.sin(k@next_n_n.T))
            
            # complex parameters
            phi_k = J*S*gamma
            phi_bar_k = np.conjugate(phi_k) 
            aphi_k = np.abs(phi_k)**2
            lambda_k = 2*D*S*s*gamma_sin
            delta_k = np.sqrt(lambda_k**2+v**2*aphi_k)

            phi_k_xx = J*S*gamma_xx
            phi_k_xy = J*S*gamma_xy
            phi_k_yy = J*S*gamma_yy
            phi_bar_k_xx = np.conjugate(phi_k_xx)
            phi_bar_k_xy = np.conjugate(phi_k_xy)
            phi_bar_k_yy = np.conjugate(phi_k_yy)
            lambda_k_xx = 2*D*S*s*gamma_sin_xx
            lambda_k_xy = 2*D*S*s*gamma_sin_xy
            lambda_k_yy = 2*D*S*s*gamma_sin_yy

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

            # angles
            varphi_k = 1j*np.log(phi_k/np.abs(phi_k))
            phase = np.exp(1j*varphi_k/2)
            phase_conj = np.conj(phase)
            
            # eigenvectors
            tau_x = np.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]])
            u1 = np.array([-cosh1*cos*phase, cosh1*sin*phase_conj, -sinh1*sin*phase, sinh1*cos*phase_conj])
            u2 = np.array([cosh2*sin*phase, cosh2*cos*phase_conj, sinh2*cos*phase, sinh2*sin*phase_conj])
            u3 = np.array([-sinh1*cos*phase, sinh1*sin*phase_conj, -cosh1*sin*phase, cosh1*cos*phase_conj])
            u4 = np.array([sinh2*sin*phase, sinh2*cos*phase_conj, cosh2*cos*phase, cosh2*sin*phase_conj])
            
            Uk = np.array([u1,u2,u3,u4]).T
            Uconj = Uk.conj().T
            E = np.array([ep,em,-ep,-em])

            # LMCs
            Lxx = np.array([[lambda_k_xx, -v*phi_bar_k_xx, 0, (1-v)*phi_bar_k_xx],[-v*phi_k_xx, -lambda_k_xx, (1-v)*phi_k_xx, 0],
                           [0, (1-v)*phi_bar_k_xx, -lambda_k_xx, -v*phi_bar_k_xx], [(1-v)*phi_k_xx, 0, v*phi_k_xx, lambda_k_xx]])
            Lxy = np.array([[lambda_k_xy, -v*phi_bar_k_xy, 0, (1-v)*phi_bar_k_xy],[-v*phi_k_xy, -lambda_k_xy, (1-v)*phi_k_xy, 0],
                           [0, (1-v)*phi_bar_k_xy, -lambda_k_xy, -v*phi_bar_k_xy], [(1-v)*phi_k_xy, 0, v*phi_k_xy, lambda_k_xy]])
            Lyy = np.array([[lambda_k_yy, -v*phi_bar_k_yy, 0, (1-v)*phi_bar_k_yy],[-v*phi_k_yy, -lambda_k_yy, (1-v)*phi_k_yy, 0],
                           [0, (1-v)*phi_bar_k_yy, -lambda_k_yy, -v*phi_bar_k_yy], [(1-v)*phi_k_yy, 0, v*phi_k_yy, lambda_k_yy]])
            
            Ltilde_xx = Uconj@Lxx@Uk
            Ltilde_xy = Uconj@Lxy@Uk
            Ltilde_yy = Uconj@Lyy@Uk

            # ramham ss', s= out, s'=in
            L = 1
            R = -1
            ramham_RL = 0.5*(Lxx+R*L*Lyy+1j*(L-R)*Lxy)
            ramham_LL = 0.5*(Lxx+L*L*Lyy+1j*(L-L)*Lxy)
            ramham_RR = 0.5*(Lxx+R*R*Lyy+1j*(R-R)*Lxy)
            ramham_LR = 0.5*(Lxx+L*R*Lyy+1j*(R-L)*Lxy)
            # (alpha mode -> +; beta mode -> -)
            # scattering
            tRL = Uconj@ramham_RL@Uk
            tLL = Uconj@ramham_LL@Uk
            tRR = Uconj@ramham_RR@Uk
            tLR = Uconj@ramham_LR@Uk

            
            # two magnon
            #p_13[i,j] = np.abs(tRL[0,2])**2-np.abs(tLR[0,2])**2
            #p_24[i,j] = np.abs(tRL[1,3])**2-np.abs(tLR[1,3])**2
            p_13[i,j] = 2*np.imag((Ltilde_xx[0,2]-Ltilde_yy[0,2])*Ltilde_xy[0,2].conj())
            p_24[i,j] = 2*np.imag((Ltilde_xx[1,3]-Ltilde_yy[1,3])*Ltilde_xy[1,3].conj())
            
            # FM
            #p_12[i,j] =  np.abs(Uconj.T[0]@ramham_RL@u2)**2-np.abs(Uconj.T[0]@ramham_LR@u2)**2
            p_12[i,j] = 2*np.imag((Ltilde_xx[0,1]-Ltilde_yy[0,1])*Ltilde_xy[0,1].conj())
            
            # AFM
            #p_14[i,j] = np.abs(tRL[0,3])**2-np.abs(tLR[0,3])**2
            p_14[i,j] = 2*np.imag((Ltilde_xx[0,3]-Ltilde_yy[0,3])*Ltilde_xy[0,3].conj())
            
        if i%50 == 0:
                print(f'Iteration {i}: B={B0:.2f}, D={D:.3f}: done')
    # RCD_array = np.array([p_13,p_24,p_12,-p_12,p_14,p_14])
    RCD_array = np.array([[p_13, p_14, -p_12],[p_24, p_14, p_12,]])
    return RCD_array

J = 1 # meV
D = 0.1# D/J = 0.1
S = 5 # spin number; we divided by overall two factor in each term of Hamiltonian, so S=5 to compensate
s = 0.75 # saturation field ratio (sin\theta) = B/Bs

# %%
s_values = [0.75]
D_values = [0.1]

#RCD = [get_RCD_exact(J=J, D=D, S=S, B0=s) for s in s_values for D in D_values]
RCD = get_RCD_exact(J=J, D=D, S=S, B0=s)
# %%
honeycomb_bz_x, honeycomb_bz_y = honeycomb_bz()

kx,ky = bzmesh(m=2)

# %% ploting 
color_bar_title_upper = [r"$\chi_{{\alpha}^{\prime}\bar{\alpha}^{\prime}}$", 
                         r"$\chi_{{\alpha}^{\prime}\bar{\beta}^{\prime}}$",
                         r"$\chi_{\alpha^{\prime}\beta}$",] # 2M, AFM, FM

color_bar_title_lower = [r"$\chi_{{\beta}^{\prime}\bar{\beta}^{\prime}}$",
                         r"$\chi_{{\alpha}^{\prime}{\beta}}$",
                         r"$\chi_{{\beta}^{\prime}\bar{\alpha}^{\prime}}$",]
color_bar_title = [color_bar_title_upper, color_bar_title_lower]
title = ["2M", "AFM", "FM"]
# subplot_labels = [r'$\mathrm{(a)}$', r'$\mathrm{(b)}$', r'$\mathrm{(c)}$']

# pads = [[7, 0, 0], [12, 12, 12]]
# with plt.style.context(['science','ieee']):
#     fig, axes = panel(figsize=(12,3), 
#                      nrows=1, ncols=3, 
#                      width_ratios=[1, 1, 1], height_ratios=[1], 
#                      wspace=0.25)

#     fig.subplots_adjust(top=0.95, bottom=0.15, right=0.99)

#     for i in range(1):
#         for j in range(3):

#             pc = axes[j].pcolormesh(kx, ky, RCD[2*5+4][i+1][j], cmap="jet")
            
#             plot(honeycomb_bz_x, honeycomb_bz_y, ax=axes[j], linestyle='-', linewidth=1, color='k')

#             clb = fig.colorbar(pc, ax=axes[j], shrink=0.9)
#             clb.ax.set_title(color_bar_title[i+1][j], loc='left', fontsize=16, pad=pads[i+1][j])
#             clb.ax.tick_params(labelsize=16)

#             axes[j].set_axis_on() # make sure the axis is on
#             axes[j].grid(False) # make sure the grid is off

#             axes[j].set_xticks([-0.5 * 2 * np.pi, 0, 0.5 * 2 * np.pi])
#             axes[j].set_xticklabels(['-1', '0', '1'], fontsize=16)
#             axes[j].set_yticks([-0.5 * 2 * np.pi, 0, 0.5 * 2 * np.pi])
#             axes[j].set_yticklabels(['-1', '0', '1'], fontsize=16)

#             axes[j].set_xlabel(r'$k_x(\pi/a)$', fontsize=18)
#             axes[j].set_ylabel(r'$k_y(\pi/a)$', fontsize=18)
#             axes[j].set_title(title[j], fontsize=18)
#             letter_annotation(axes[j], -0.2,1, subplot_labels[j],size=18)
#     plt.show()
#     fig.savefig('figures/panel_plots/figure9_RCD_lower.png', dpi=300, bbox_inches='tight')
# %%
pads = [7, 7, 7]
with plt.style.context(['science','ieee']):
    fig, axes = panel(figsize=(12,3), nrows=1, ncols=3, width_ratios=[1,1,1], height_ratios=[1], hspace=0.1, wspace=0.4)

    fig.subplots_adjust(top=0.95, bottom=0.15, right=0.99)

    for i in range(3):
        pc = axes[i].pcolormesh(kx, ky, RCD[0][i], cmap="jet",) 
        
        plot(honeycomb_bz_x, honeycomb_bz_y, ax=axes[i], linestyle='-', linewidth=1, color='k')

        clb = fig.colorbar(pc, ax=axes[i], shrink=0.9)
        clb.ax.set_title(color_bar_title_upper[i], loc='left', fontsize=16, pad=pads[i])
        clb.ax.tick_params(labelsize=16)

        axes[i].set_axis_on() # make sure the axis is on
        axes[i].grid(False) # make sure the grid is off
        axes[i].set_xticks([-0.5 * 2 * np.pi, 0, 0.5 * 2 * np.pi])
        axes[i].set_xticklabels(['-1', '0', '1'], fontsize=16)
        axes[i].set_yticks([-0.5 * 2 * np.pi, 0, 0.5 * 2 * np.pi])
        axes[i].set_yticklabels(['-1', '0', '1'], fontsize=16)

        axes[i].set_xlabel(r'$k_x(\pi/a)$', fontsize=18)
        axes[i].set_ylabel(r'$k_y(\pi/a)$', fontsize=18)
        axes[i].set_title(title[i], fontsize=18)
    # letter_annotation(axes[0],-0.25,1,r'$\mathrm{(a)}$',size=18)
    # letter_annotation(axes[1],-0.25,1,r'$\mathrm{(b)}$',size=18)
    # letter_annotation(axes[2],-0.25,1,r'$\mathrm{(c)}$',size=18)
    plt.show()
    # fig.savefig('figures/panel_plots/figure4_RCD_upper.png', dpi=300, bbox_inches='tight')
# %%
print(RCD[0].shape)
# %%
a =1 
n_n = a*np.sqrt(3)*np.array([[0,1/np.sqrt(3)],[1/2,-0.5/np.sqrt(3)],[-1/2,-0.5/np.sqrt(3)]]) #a1,a2,a3
next_n_n = -a*np.sqrt(3)*np.array([[-1,0],[1/2,np.sqrt(3)/2],[1/2,-np.sqrt(3)/2]]) # b1,b2,b3
# %%
print(n_n.T[0]*n_n.T[0])
# %%
