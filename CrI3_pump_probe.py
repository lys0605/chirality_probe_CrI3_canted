# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from plot_utils import plot, letter_annotation, panel
from mathfuntion import Im, Re, is_invertible
from honeycomb_lattice import *
from CrI3_raman_scattering import get_RCD, get_energy
from CrI3_curvature import get_FM_berry_curvature
import scienceplots
# %%
data = np.load('chi_FM_computed.npz')
chi_FM_lower_T = data['chi_FM_lower_T']
chi_FM_upper_T = data['chi_FM_upper_T']
chi_FM_lower_D = data['chi_FM_lower_D']
chi_FM_upper_D = data['chi_FM_upper_D']
temperatures   = data['temperatures']
D_array        = data['D_array']
w              = data['w']

# %%
def gaussian_function(x, x0=0, width=1e-3):
    """
    1D Gaussian function
    Parameters
    ----------
    x : array_like
        Input values.
    x0 : float, optional
        Center of the Gaussian. The default is 0.
    width : float, optional
        Width of the Gaussian. The default is 1e-3.
    Returns
    ----------
    """
    epsilon = width
    return 1/(epsilon*np.sqrt(2*np.pi))*np.exp(-0.5*((x-x0)/width)**2)

def lorentzian_function(x, x0=0, width=1e-3):
    """
    1D Lorentzian function
    Parameters
    ----------
    x : array_like
        Input values.
    x0 : float, optional
        Center of the Lorentzian. The default is 0.
    width : float, optional
        Width of the Lorentzian. The default is 1e-3.
    Returns
    ----------
    """
    epsilon = width
    return (epsilon/np.pi)/((x-x0)**2+epsilon**2)

def bose_einstein_distribution(E, T=0, mu=0):
    """
    Bose-Einstein distribution function
    Parameters
    ----------
    E : array_like
        Energy values.
    T : float, optional
        Temperature in Kelvin. The default is 0.
    mu : float, optional
        Chemical potential in eV. The default is 0.
    Returns
    ----------
    """
    k_B = 8.617333262145e-2 # meV/K
    return 1/(np.exp((E-mu)/(k_B*T))-1)

def boltzmann_factor(E, T=0, mu=0):
    """
    Boltzmann factor
    Parameters
    ----------
    E : array_like
        Energy values.
    T : float, optional
        Temperature in Kelvin. The default is 0.
    mu : float, optional
        Chemical potential in eV. The default is 0.
    Returns
    ----------
    """
    k_B = 8.617333262145e-2 # meV/K
    return np.exp(-(E-mu)/(k_B*T)) if T!=0 else 0*E

def occupation_function(E, T=0, mu=0):
    """
    Occupation function
    Parameters
    ----------
    E : array_like
        Energy values.
    T : float, optional
        Temperature in Kelvin. The default is 0.
    mu : float, optional
        Chemical potential in eV. The default is 0.
    Returns
    ----------
    """
    return 1/(1-boltzmann_factor(E, T=T, mu=mu))

def partition_function(energy_array, T=0, mu=0):
    """
    Partition function of bosonic oscillators ; GCE
    Z = prod_k sum_alpha occupation_(k,alpha)
    -> ln Z = sum_k ln(sum_alpha occupation_(k,alpha)) ; sum_k ---> int d^2k/(2pi)^2 (V->1 by a -> 1)
    -> Z = exp(ln Z) = exp(sum_k ln(sum_alpha occupation_(k,alpha)))
    Parameters
    ----------
    energy_array : array_like (energy values computed in BZ; the accuracy depends on the resolution of the defined BZ (k-points))
    T : float, optional
        Temperature in Kelvin. The default is 0.
    mu : float, optional
        Chemical potential in meV. The default is 0.
    Returns
    ----------
    """
    return np.exp(bz_integration_honeycomb(np.log(1/(1-boltzmann_factor(energy_array, T=T, mu=mu))))/(2*np.pi)**2)


def plot_frequency_resolved_RCD(ax, w, chi, s_values, ls='-', **kwarg):
    """
    plot the frequency resolved_RCD of corresponding process for different magnetic field
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    w : array_like
        Frequency values.
    chi : array_like
        RCD values.
    s_values : array_like
        Magnetic field values.
    ls : str, optional
        Line style. The default is '-'.
    kwarg : dict, optional
        Additional keyword arguments for plotting.
    Returns
    ----------
    """
    if kwarg['plot_length'] != 1:
        for j in range(kwarg['plot_length']):
            ax.plot(w, chi[j], ls=ls, color=kwarg['color'][j], label=kwarg['label']+fr' $B={s_values[j]}B_s$')
    else:
        ax.plot(w, chi, ls=ls, color=kwarg['color'], label=kwarg['label'])

def plot_frequency_temperature_resolved_RCD(ax, w, chi, temperatures, ls='-', **kwarg):
    """
    plot the frequency resolved_RCD of corresponding process for different magnetic field
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    w : array_like
        Frequency values.
    chi : array_like
        RCD values.
    s_values : array_like
        Magnetic field values.
    ls : str, optional
        Line style. The default is '-'.
    kwarg : dict, optional
        Additional keyword arguments for plotting.
    Returns
    ----------
    """
    if kwarg['plot_length'] != 1:
        for j in range(kwarg['plot_length']):
            ax.plot(w, chi[j], ls=ls, color=kwarg['color'][j], label=kwarg['label']+fr' $T={temperatures[j]}K$')
    else:
        ax.plot(w, chi, ls=ls, color=kwarg['color'], label=kwarg['label'])
# %%
def plot_frequency_temperature_resolved_RCD_D(ax, w, chi, D_array, ls='-', **kwarg):
    """
    plot the frequency resolved_RCD of corresponding process for different magnetic field
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    w : array_like
        Frequency values.
    chi : array_like
        RCD values.
    s_values : array_like
        Magnetic field values.
    ls : str, optional
        Line style. The default is '-'.
    kwarg : dict, optional
        Additional keyword arguments for plotting.
    Returns
    ----------
    """
    if kwarg['plot_length'] != 1:
        for j in range(kwarg['plot_length']):
            ax.plot(w, chi[j], ls=ls, color=kwarg['color'][j], label=kwarg['label']+fr' $D={D_array[j]}$')
    else:
        ax.plot(w, chi, ls=ls, color=kwarg['color'], label=kwarg['label'])
#%% CrI3 parameters from Chen's paper, Phys. Rev. X. 2018
J1 = 2.01 # n.n Heisenberg coupling meV
J2 = 0.16 # n.n.n Heisenberg coupling meV
J3 = -0.08 # n.n.n.n Heisenberg coupling meV
D = 0.31 # DMI meV
Az = 0.49 # anisotropy
S = 3/2 # spin number

#%% frequency reange
w = np.linspace(0,20,300)
len_w = len(w)
width = (w[1]-w[0])

# %%
def get_TRCD_vary_D_T_35(D, J1=2.01, J2=0.16, J3=-0.08, Az=0.49, S=3/2):
    """
    Compute the temperature-resolved RCD for varying D at T=35K.
    """
    w = np.linspace(0,20,300)
    len_w = len(w)
    width = (w[1]-w[0])
    RCD_array = get_RCD(J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S)
    berry_array = get_FM_berry_curvature(J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S)
    energy_array = get_energy(J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S)
    return  RCD_array, berry_array, energy_array
# %%    
D_array = np.linspace(-0.31, 0.31, 101)

len_D = len(D_array)

chi_FM_lower_D = np.zeros((len_D, len_w))
chi_FM_upper_D = np.zeros((len_D, len_w))
for j in range(len_D):
    RCD_array = get_RCD(J1=J1, J2=J2, J3=J3, D=D_array[j], Az=Az, S=S)
    energy_array = get_energy(J1=J1, J2=J2, J3=J3, D=D_array[j], Az=Az, S=S)
    partition_array = partition_function(energy_array[0], T=35)*partition_function(energy_array[1], T=35)
    for i in range(len_w):
        chi_FM_lower_D[j][i] = bz_integration_honeycomb( boltzmann_factor(energy_array[1], T=35)*RCD_array * gaussian_function(w[i], x0= np.abs(energy_array[1]-energy_array[0]), width=width))/partition_array
        chi_FM_upper_D[j][i] = bz_integration_honeycomb( boltzmann_factor(energy_array[1], T=35)*RCD_array * gaussian_function(w[i], x0= np.abs(energy_array[1]-energy_array[0]), width=width))/partition_array
# %%

# %%
RCD_array = get_RCD(J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S)
berry_array = get_FM_berry_curvature(J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S)
energy_array = get_energy(J1=J1, J2=J2, J3=J3, D=D, Az=Az, S=S)

# %%


# %% finite temperature with initial state as one magnon occupied at lowest band
temperatures = np.linspace(0, 48, 49)
#temperatures = np.linspace(0, 45, 120)
temp = temperatures[5]
partition_array = np.array([partition_function(energy_array[0], T=T)*partition_function(energy_array[1], T=T) for T in temperatures])

# %% CrI3
len_T = len(temperatures)
chi_FM_lower_T = np.zeros((len_T, len_w))
chi_FM_upper_T = np.zeros((len_T, len_w))
for j in range(len_T):
    for i in range(len_w):
        chi_FM_lower_T[j][i] = bz_integration_honeycomb( boltzmann_factor(energy_array[1], T=temperatures[j])*RCD_array * gaussian_function(w[i], x0= np.abs(energy_array[1]-energy_array[0]), width=width))/partition_array[j]
        chi_FM_upper_T[j][i] = bz_integration_honeycomb( boltzmann_factor(energy_array[1], T=temperatures[j])*RCD_array * gaussian_function(w[i], x0= np.abs(energy_array[1]-energy_array[0]), width=width))/partition_array[j]

# %% save computed chi arrays
np.savez('chi_FM_computed.npz',
         chi_FM_lower_T=chi_FM_lower_T,
         chi_FM_upper_T=chi_FM_upper_T,
         chi_FM_lower_D=chi_FM_lower_D,
         chi_FM_upper_D=chi_FM_upper_D,
         temperatures=temperatures,
         D_array=D_array,
         w=w)

# %% dos weighted CrI3 at zero T
dos_weighted_FM= np.zeros(len_w)

for i in range(len_w):
    dos_weighted_FM[i] = bz_integration_honeycomb(berry_array[1]*lorentzian_function(w[i], x0=np.abs(energy_array[1]-energy_array[0]), width=width))

#%%
print(np.max(energy_array[1]))
print(partition_array)
print(boltzmann_factor(energy_array[1][67,123],T=temperatures[-1]) /partition_array[-1])
print(boltzmann_factor(energy_array[1][67,123],T=temperatures[4]) /partition_array[4])
print(temperatures[4])
# %%

colors_vacuum = ['#00215d', '#0071bc' , '#8fd0ff']
colors_one_magnon = ['#b6000f' , '#ff814b', '#ffc883']
colors_temperature = ['#1a1a1a',  '#192225' , '#143349' , '#245074', '#2f6fa4', '#3989d6', '#76a1ff', '#aaceff', '#e1f0ff']
colors_D = ['#800000', '#b34747', '#cc8080', '#e6b3b3', '#ffcccc', '#cce6ff', '#99ccff', '#66b3ff', '#3385ff', '#0066ff', '#0047b3']

with plt.style.context(['science', 'ieee']):
    fig, ax = plt.subplots(figsize=(4, 3))

    # 1. SETUP COLORMAP FOR PARAMETER D
    # This maps the min/max of your temperatures to colors
    norm = mpl.colors.Normalize(vmin=temperatures.min(), vmax=temperatures.max())
    cmap = plt.get_cmap('Blues') # or 'viridis'

    # 2. PLOT LOOP (Converting 2D mesh to 1D lines)
    # We zip D_array with the rows of your Chi matrix
    # Assuming chi_FM_lower_D shape is (len(D_array), len(w))
    for i, T in enumerate(temperatures):
        
        # Extract the single line (cut) for this specific D
        y_values = chi_FM_lower_T[i] 
        
        # Plot with the specific color for this D
        ax.plot(w, y_values, 
                color=cmap(norm(T)), 
                linewidth=1, 
                alpha=0.9,
                ls='-') # Alpha helps if lines overlap

    # 3. CREATE THE COLORBAR FOR D
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) 
    clb = fig.colorbar(sm, ax=ax, shrink=0.9)
    
    # Label the colorbar as D, not Chi
    clb.ax.set_title(r"$T$", loc='left', fontsize=22)
    clb.ax.tick_params(labelsize=20)
    clb.set_label(r'$(K)$', fontsize=20)

    # 4. AXIS LABELS
    # Now Y-axis is Chi (Function Value)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel(r'$\hslash\omega (meV)$', fontsize=22)
    ax.set_ylabel(rf'$\chi(\omega, T)$', fontsize=22)
    plt.show()
    fig.savefig('figures/thermal_RCD/CrI3_TRCD_vary_T_more.png', dpi=300 ,bbox_inches='tight')

# %%
with plt.style.context(['science','ieee']):
    fig, ax = plt.subplots(figsize=(3.5,3))

    ax.plot(temperatures, gamma_boltzmann/partition_array, color='black')
    #plot_frequency_resolved_RCD(ax, w, dos_weighted_FM, 1, plot_length=1, ls='--', color=colors_one_magnon[1], label=r'weighted DOS')
    #plt.axvline(w[np.argmin(chi_FM_upper[1])])
    #plt.axvline(w[np.argmax(chi_two_magnons_upper[1])])
    #plt.axvline(w[np.argmin(chi_two_magnons_lower[1])])
    ax.set_xlabel(r'$T$', 

#%%   
with plt.style.context(['science','ieee']):
    fig, axes = panel(figsize=(4,3), 
                     nrows=1, ncols=1, 
                     width_ratios=[1], height_ratios=[1])

    fig.subplots_adjust(top=0.95, bottom=0.15, right=0.99)

    x,y = np.meshgrid(w, D_array)
    # Define the colormap (e.g., 'RdBu')
    cmap = plt.get_cmap('RdBu')

    pc = axes.pcolormesh(x, y, chi_FM_lower_D)

    clb = fig.colorbar(pc, ax=axes, shrink=0.9)
    clb.ax.set_title(r"$\chi$", loc='left', fontsize=16, pad=5)
    clb.ax.tick_params(labelsize=16)

    axes.set_axis_on() # make sure the axis is on
    axes.grid(False) # make sure the grid is off

    axes.set_xlabel(r'$\hslash\omega (meV)$', fontsize=18)
    axes.set_ylabel(r'$D (meV)$', fontsize=18)
    #axes.set_title('RCD of '+r"$CrI_3$", fontsize=18)
    plt.show()    

# %%
gamma_boltzmann = np.array([boltzmann_factor(energy_array[1][67,123],T=temperatures[j]) for j in range(len(temperatures))])

# %%
with plt.style.context(['science', 'ieee']):
    fig, ax = plt.subplots(figsize=(4, 3))

    # 1. SETUP COLORMAP FOR PARAMETER D
    # This maps the min/max of your D_array to colors
    norm = mpl.colors.Normalize(vmin=D_array.min(), vmax=D_array.max())
    cmap = plt.get_cmap('RdBu') # or 'viridis'

    # 2. PLOT LOOP (Converting 2D mesh to 1D lines)
    # We zip D_array with the rows of your Chi matrix
    # Assuming chi_FM_lower_D shape is (len(D_array), len(w))
    for i, d_val in enumerate(D_array):
        
        # Extract the single line (cut) for this specific D
        y_values = chi_FM_lower_D[i] 
        
        # Plot with the specific color for this D
        ax.plot(w, y_values, 
                color=cmap(norm(d_val)), 
                linewidth=1, 
                alpha=0.9,
                ls='-') # Alpha helps if lines overlap

    # 3. CREATE THE COLORBAR FOR D
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) 
    clb = fig.colorbar(sm, ax=ax, shrink=0.9)
    
    # Label the colorbar as D, not Chi
    clb.ax.set_title(r"$D$", loc='left', fontsize=22)
    clb.ax.tick_params(labelsize=20)
    clb.set_label(r'$(meV)$', fontsize=20)

    # 4. AXIS LABELS
    # Now Y-axis is Chi (Function Value)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel(r'$\hslash\omega (meV)$', fontsize=22) # The function value is now the geometry
    ax.set_ylabel(r'$\chi(\omega,D)$', fontsize=22)
    plt.show()
    fig.savefig('figures/thermal_RCD/CrI3_TRCD_vary_D_more.png', dpi=300 ,bbox_inches='tight')
# %%
np.linspace(0, 48, 49)

# %%