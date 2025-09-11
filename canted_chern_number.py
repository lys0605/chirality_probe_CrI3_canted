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

def get_berry_curvature(J=1, D=0.1, S=5/2, B0=0.5):
    """
    Calculate the berry curvature for a canted antiferromagnet on a honeycomb lattice.

    Parameters:
        J (float): Heisenberg exchange interaction in mev
        D (float): DM interaction in meV
        S (float): Spin number
        B0 (float): Saturation field ratio (sin\theta)

    Returns:
        Cherns (np.ndarray): The Chern number for the two bands.
    """
    # field
    Bs = 3*J*S # not 6JS here
    B = (B0)*Bs
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
    
    # meshgrid
    kx,ky = bzmesh(m=2)
     
    # two-magnon pair creation (same modes but opposite momentum), FM, AFM
    p_13 = np.zeros(kx.shape)
    p_24 = np.zeros(kx.shape)
    p_12 = np.zeros(kx.shape)
    p_14 = np.zeros(kx.shape)
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
            
            # berry curvature
            berry_rcd_p[i,j] = -2*(0*p_13[i,j]/(ep+ep)**2+0*p_14[i,j]/(ep+em)**2-1*p_12[i,j]/(em-ep)**2)+0*2*(xi_p_k/(ep+ep)**2+sigma_k/(ep+em)**2+zeta_k/(em-ep)**2)*(rho+rho_tilde)
            berry_rcd_m[i,j] = -2*(0*p_24[i,j]/(em+em)**2+0*p_14[i,j]/(ep+em)**2+1*p_12[i,j]/(ep-em)**2)-0*2*(xi_m_k/(em+em)**2-sigma_k/(ep+em)**2+zeta_k/(ep-em)**2)*(rho+rho_tilde)
            
        if i%50 == 0:
                print(f'Iteration {i}: B={B0:.2f}, D={D:.3f}: done')
    
    return berry_rcd_p, berry_rcd_m

def get_chern_number(J=1, D=0.1, S=5/2, B0=0.5, n=200, m=1):
    """
    Calculate the canted Chern number for a canted antiferromagnet on a honeycomb lattice.
    
    Parameters:
        J (float): Heisenberg exchange interaction in mev
        D (float): DM interaction in meV
        S (float): Spin number
        B0 (float): Saturation field ratio (sin\theta)

    Returns:
        Cherns (np.ndarray): The Chern number for the two bands.
    """
    berry_p, berry_m = get_berry_curvature(J=J, D=D, S=S, B0=B0)
    chern_p = bz_integration_honeycomb(berry_p, n=n, m=m)/(2*np.pi)
    chern_m = bz_integration_honeycomb(berry_m, n=n, m=m)/(2*np.pi)
    Cherns = np.array([chern_p, chern_m])
    return Cherns

J = 1 # meV
D = 0.1 # D/J = 0.1
S = 5/2 # spin number
s = 0.6 # saturation field ratio (sin\theta) = B/Bs

# %%
# s_values = np.array([0.25, 0.5, 0.75, 1])
s_values = np.linspace(-1.0, 1.0, num=11)
D_values = np.linspace(-0.1, 0.1, num=11)

C_p = np.zeros((s_values.size, D_values.size))
C_m = np.zeros((s_values.size, D_values.size))

for i in range(s_values.size):
    for j in range(D_values.size):
        C_p[i,j], C_m[i,j] = get_chern_number(J=J, D=D_values[j], S=S, B0=s_values[i])
       
Cherns = np.array([C_p, C_m])
# %%

# %% plot

mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
mpl.rcParams['axes.labelsize'] = 18

color_bar_title = [r'$C_\alpha$', r'$C_{\beta}$']
color_bar_title_RCD = [r'$C_\alpha^{\text{RCD}}$', r'$C_{\beta}^{\text{RCD}}$']
color_bar_title_FM = [r"$C_{\alpha}^{\text{FM}}$",r"$C_{\beta}^{\text{FM}}$"]
color_bar_title_noFM = [r"$C_{\alpha}^{\text{noFM}}$",r"$C_{\beta}^{\text{noFM}}$"]

pads = [7, 10]

px,py = np.meshgrid(D_values, s_values)

# with plt.style.context(['science', 'ieee']):
    
#     fig, axes = panel(figsize=(8,3), nrows=1, ncols=2, width_ratios=[1,1], height_ratios=[1], hspace=0.1, wspace=0.5)

#     fig.subplots_adjust(top=0.95, bottom=0.15, right=0.99)

#     for i in range(2):
#         pc = axes[i].pcolormesh(px, py, Cherns[i], cmap="jet")

#         clb = fig.colorbar(pc, ax=axes[i], shrink=0.9)
#         clb.ax.set_title(color_bar_title[i], loc='left', fontsize=18, pad=pads[i])
#         clb.ax.tick_params(labelsize=16)

#         axes[i].set_axis_on() # make sure the axis is on
#         axes[i].grid(False) # make sure the grid is off

#         #axes[i].set_xticks([0.1, 0.15, 0.2, 0.25, 0.3])
#         #axes[i].set_xticklabels([r'$0.1$',r'$0.15$',r'$0.2$',r'$0.25$',r'$0.3$'])

#         axes[i].set_xlabel(r'$D/J$', fontsize=18)
#         axes[i].set_ylabel(r'$B/B_s$', fontsize=18)
#     plt.show()

# # %% savefig
# fig.savefig('figures/chern_numbers/canted_chern_number.pdf', bbox_inches='tight', dpi=300)


# %%
with plt.style.context(['science', 'ieee']):
    
    fig, axes = panel(figsize=(4,3), nrows=1, ncols=1, width_ratios=[1], height_ratios=[1], hspace=0.1, wspace=0.5)

    fig.subplots_adjust(top=0.95, bottom=0.15, right=0.99)

    for i in range(1):
        pc = axes.pcolormesh(px, py, Cherns[0], cmap="jet")

        clb = fig.colorbar(pc, ax=axes, shrink=0.9)
        clb.ax.set_title(color_bar_title_FM[0], loc='left', fontsize=18, pad=pads[1])
        clb.ax.tick_params(labelsize=16)

        axes.set_axis_on() # make sure the axis is on
        axes.grid(False) # make sure the grid is off

        #axes[i].set_xticks([0.1, 0.15, 0.2, 0.25, 0.3])
        #axes[i].set_xticklabels([r'$0.1$',r'$0.15$',r'$0.2$',r'$0.25$',r'$0.3$'])

        axes.set_xlabel(r'$D/J$', fontsize=18)
        axes.set_ylabel(r'$B/B_s$', fontsize=18)
    plt.show()
    fig.savefig('figures/chern_numbers/canted_Chern_number_upper_RCD_FM.png', bbox_inches='tight', dpi=300)
# %%
