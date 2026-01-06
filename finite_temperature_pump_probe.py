# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from plot_utils import plot, letter_annotation, panel
from mathfuntion import Im, Re, is_invertible
from honeycomb_lattice import *
from canted_raman_cross_section import *
import scienceplots
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
# %% canting
J = 1
D = 0.1
S = 5/2
s_values = [0, 0.25, 0.5, 0.75, 1]

raman_array = [get_raman_cross_section_exact(J=J, D=D, S=S, B0=s) for s in s_values]

#%% CrI3
J = 2.01
D = 0.31
S = 3 # since we divid it by 2 in the function
s_values = [0, 0.25, 0.5, 0.75, 1]

raman_array_FM = get_raman_cross_section_exact(J=J, D=D, S=S, B0=s_values[4])

#%%
len_s = len(s_values)
cross_section_array = np.array([raman_array[i][0] for i in range(len_s)])
berry_array = np.array([raman_array[i][1] for i in range(len_s)])
berry_rcd_array = np.array([raman_array[i][2] for i in range(len_s)])
energy_array = np.array([raman_array[i][3] for i in range(len_s)])

#%% CrI3
len_s = len(s_values)
cross_section_array = np.array(raman_array_FM[0])
berry_array = np.array(raman_array_FM[1])
berry_rcd_array = np.array(raman_array_FM[2])
energy_array = np.array(raman_array_FM[3])

# %%
w = np.linspace(0,37.5,300)
len_w = len(w)
width = (w[1]-w[0])*2

# %% finite temperature with initial state as one magnon occupied at lowest band
temperatures = [0, 2, 10, 18, 26, 34, 42, 50, 58]
temp = temperatures[5]
partition_array = np.array([partition_function(energy_array[0], T=T)*partition_function(energy_array[1], T=T) for T in temperatures])
# %%
chi_two_magnons_lower_T = np.zeros((len_s, len_w))
chi_AFM_lower_T = np.zeros((len_s, len_w))
chi_FM_lower_T = np.zeros((len_s, len_w))
chi_two_magnons_upper_T = np.zeros((len_s, len_w))
chi_AFM_upper_T = np.zeros((len_s, len_w))
chi_FM_upper_T = np.zeros((len_s, len_w))

for j in range(len_s):
    for i in range(len_w):
        chi_two_magnons_lower_T[j][i] = bz_integration_honeycomb( boltzmann_factor(energy_array[j][1],T=temp)*cross_section_array[j][1] * gaussian_function(w[i], x0= 2*energy_array[j][1], width=width))
        chi_two_magnons_upper_T[j][i] = bz_integration_honeycomb( boltzmann_factor(energy_array[j][1],T=temp)*cross_section_array[j][0] * gaussian_function(w[i], x0= 2*energy_array[j][0], width=width))
        chi_AFM_lower_T[j][i] = bz_integration_honeycomb( boltzmann_factor(energy_array[j][1],T=temp)*cross_section_array[j][5] * gaussian_function(w[i], x0= energy_array[j][1]+energy_array[j][0], width=width))
        chi_AFM_upper_T[j][i] = bz_integration_honeycomb( boltzmann_factor(energy_array[j][1],T=temp)*cross_section_array[j][4] * gaussian_function(w[i], x0= energy_array[j][1]+energy_array[j][0], width=width))
        chi_FM_lower_T[j][i] = bz_integration_honeycomb( boltzmann_factor(energy_array[j][1],T=temp)*cross_section_array[j][3] * gaussian_function(w[i], x0= np.abs(energy_array[j][1]-energy_array[j][0]), width=width))
        chi_FM_upper_T[j][i] =bz_integration_honeycomb( boltzmann_factor(energy_array[j][1],T=temp)*cross_section_array[j][2] * gaussian_function(w[i], x0= np.abs(energy_array[j][1]-energy_array[j][0]), width=width))

# %% dos
dos_two_magnons_lower = np.zeros((len_s, len_w))
dos_two_magnons_upper = np.zeros((len_s, len_w))
dos_AFM = np.zeros((len_s, len_w))
dos_FM = np.zeros((len_s, len_w)) # FM interband

for j in range(len_s):
    for i in range(len_w):
        dos_two_magnons_lower[j][i] = bz_integration_honeycomb(gaussian_function(w[i], x0=2*energy_array[j][1], width=width))
        dos_two_magnons_upper[j][i] = bz_integration_honeycomb(gaussian_function(w[i], x0=2*energy_array[j][0], width=width))
        dos_AFM[j][i] = bz_integration_honeycomb(gaussian_function(w[i], x0=energy_array[j][1]+energy_array[j][0], width=width))
        dos_FM[j][i] = bz_integration_honeycomb(gaussian_function(w[i], x0=np.abs(energy_array[j][1]-energy_array[j][0]), width=width))

# %% dos weighted
dos_weighted_two_magnons_lower = np.zeros((len_s, len_w))
dos_weighted_two_magnons_upper = np.zeros((len_s, len_w))
dos_weighted_AFM_upper = np.zeros((len_s, len_w))
dos_weighted_AFM_lower = np.zeros((len_s, len_w))
dos_weighted_FM= np.zeros((len_s, len_w))

for j in range(len_s):
    for i in range(len_w):
        dos_weighted_two_magnons_lower[j][i] = bz_integration_honeycomb(berry_array[j][1]*gaussian_function(w[i], x0=2*energy_array[j][1], width=width))
        dos_weighted_two_magnons_upper[j][i] = bz_integration_honeycomb(berry_array[j][0]*gaussian_function(w[i], x0=2*energy_array[j][0], width=width))
        dos_weighted_AFM_upper[j][i] = bz_integration_honeycomb(berry_array[j][0]*gaussian_function(w[i], x0=energy_array[j][1]+energy_array[j][0], width=width))
        dos_weighted_AFM_lower[j][i] = bz_integration_honeycomb(berry_array[j][1]*gaussian_function(w[i], x0=energy_array[j][1]+energy_array[j][0], width=width))
        dos_weighted_FM[j][i] = bz_integration_honeycomb(berry_array[j][1]*gaussian_function(w[i], x0=np.abs(energy_array[j][1]-energy_array[j][0]), width=width))

# %% dos weighted CrI3
dos_weighted_FM= np.zeros( len_w)

for i in range(len_w):
    dos_weighted_FM[i] = bz_integration_honeycomb(berry_array[1]*gaussian_function(w[i], x0=np.abs(energy_array[1]-energy_array[0]), width=width))

#%%
print(np.max(energy_array[1]))
print(np.max(boltzmann_factor(11, T=61)/partition_array))
print(len_T)
# %%
rcd_label = [r'$\chi_{\alpha^{\prime}\bar{\alpha}^\prime}$',r'$\chi_{\beta^{\prime}\bar{\beta}^\prime}$',
             r'$\chi_{\alpha^{\prime}\beta}$', r'$\chi_{\bar{\beta}^{\prime}\bar{\alpha}}$',
             r'$\chi_{\alpha^{\prime}\bar{\beta}^\prime}$', r'$\chi_{\beta^{\prime}\alpha^\prime}$'
            ]
# chi_vacuum = chi_two_magnons_lower+chi_two_magnons_upper+chi_AFM_lower+chi_AFM_upper
# chi_one_magnon = 2*chi_two_magnons_lower+chi_two_magnons_upper+2*chi_AFM_lower+chi_AFM_upper+chi_FM_upper
# chi_vacuum_finite_T = chi_vacuum/partition_array[0]
# chi_one_magnon_finite_T = (2*chi_two_magnons_lower_T+chi_two_magnons_upper_T+2*chi_AFM_lower_T+chi_AFM_upper_T+chi_FM_upper_T)

# dos_weighted_vacuum = dos_weighted_two_magnons_lower+dos_weighted_two_magnons_upper+dos_weighted_AFM_lower+dos_weighted_AFM_upper
# dos_weighted_one_magnon = 2*dos_weighted_two_magnons_lower+dos_weighted_two_magnons_upper+2*dos_weighted_AFM_lower+dos_weighted_AFM_upper+dos_weighted_FM

colors_vacuum = ['#00215d', '#0071bc' , '#8fd0ff']
colors_one_magnon = ['#b6000f' , '#ff814b', '#ffc883']
colors_temperature = ['#1a1a1a',  '#192225' , '#143349' , '#245074', '#2f6fa4', '#3989d6', '#76a1ff', '#aaceff', '#e1f0ff']

with plt.style.context(['science','ieee']):
    fig, ax = plt.subplots(figsize=(4,3))
    #plot_frequency_resolved_RCD(w, chi_vacuum[1:4], s_values[1:4], plot_length=3, color=colors_vacuum, label=r'')
    # plot_frequency_resolved_RCD(ax, w, chi_vacuum_finite_T[1:4], s_values[0:3], plot_length=3, color=colors_vacuum[0:3], label=r'')
    # plot_frequency_resolved_RCD(ax, w, dos_weighted_one_magnon[3], s_values[0], ls='--', plot_length=1, color='black', label=r'$\Lambda^\prime$')
    plot_frequency_temperature_resolved_RCD(ax, w, chi_FM_lower_T, temperatures, plot_length=len_T, color=colors_temperature, label=r'')
    plot_frequency_resolved_RCD(ax, w, dos_weighted_FM, s_values[-1], plot_length=1, ls='--', color=colors_one_magnon[1], label=r'weighted DOS')
    #plt.axvline(w[np.argmin(chi_FM_upper[1])])
    #plt.axvline(w[np.argmax(chi_two_magnons_upper[1])])
    #plt.axvline(w[np.argmin(chi_two_magnons_lower[1])])
    ax.set_xlabel(r'$\hslash\omega/J$', fontsize=13)
    ax.set_ylabel(rf'$\chi(\omega, T)$', fontsize=13)
    ax.legend(loc="lower right", bbox_to_anchor=(1.7, -0.2) , fontsize=13)
    plt.show()
    #fig.savefig('frequency_resolved_RCD_one_magnon.png', dpi=600, bbox_inches='tight')

#%%
with plt.style.context(['science','ieee']):
    fig = plt.figure(figsize=(4,6))
    gs = fig.add_gridspec(2, hspace=0)
    axes = gs.subplots(sharex=True)
    for ax in axes:
        ax.set_box_aspect(0.75)
    
    plot_frequency_resolved_RCD(axes[0], w, chi_vacuum[1:4], s_values[0:3], plot_length=3, color=colors_vacuum, label=r'')
    plot_frequency_resolved_RCD(axes[1], w, chi_one_magnon[1:4], s_values[0:3], plot_length=3, color=colors_one_magnon, label=r'')

    #plt.axvline(w[np.argmax(chi_two_magnons_upper[1])])
    #plt.axvline(w[np.argmin(chi_two_magnons_lower[1])])
    axes[1].set_xlabel(r'$\hslash\omega/J$', fontsize=15)
    axes[0].set_ylabel(r'$\chi^{(0)}(\omega)$', fontsize=15)
    axes[1].set_ylabel(r'$\chi^{(1)}(\omega)$', fontsize=15)
    axes[0].legend(loc="lower left", fontsize=13)
    axes[1].legend(loc="lower left", fontsize=13)
    plt.show()
    #fig.savefig('figures/pump_probe/frequency_resolved_RCD_vacuum_one_magnon.png', dpi=300, bbox_inches='tight')

# %%
def normalize(x):
    return x/np.max(np.abs(x))
with plt.style.context(['science','ieee']):
    fig = plt.figure(figsize=(4,6))
    gs = fig.add_gridspec(2, hspace=0)
    axes = gs.subplots(sharex=True)

    for ax in axes:
        ax.set_box_aspect(0.75)
    
    plot_frequency_resolved_RCD(axes[0], w, normalize(chi_vacuum[1]), s_values[1], plot_length=1, color=colors_vacuum[-1], label=r'$\chi^{(0)}(\omega)$')
    plot_frequency_resolved_RCD(axes[0], w, normalize(chi_one_magnon[1]), s_values[1], plot_length=1, color=colors_one_magnon[-1], label=r'$\chi^{(1)}(\omega)$')
    plot_frequency_resolved_RCD(axes[0], w, normalize(dos_weighted_one_magnon[1]), s_values[1], ls='--', plot_length=1, color='grey', label=r'$\Lambda$')

    plot_frequency_resolved_RCD(axes[1], w, normalize(chi_vacuum[3]), s_values[3], plot_length=1, color=colors_vacuum[-1], label=r'$\chi^{(0)}(\omega)$')
    plot_frequency_resolved_RCD(axes[1], w, normalize(chi_one_magnon[3]), s_values[3], plot_length=1, color=colors_one_magnon[-1], label=r'$\chi^{(1)}(\omega)$')
    plot_frequency_resolved_RCD(axes[1], w, normalize(dos_weighted_one_magnon[3]), s_values[3], ls='--', plot_length=1, color='black', label=r'$\Lambda^\prime$')
    #plt.axvline(w[np.argmax(chi_two_magnons_upper[1])])
    #plt.axvline(w[np.argmin(chi_two_magnons_lower[1])])
    axes[1].set_xlabel(r'$\hslash\omega/J$', fontsize=15)
    axes[0].set_ylabel(r'', fontsize=15)
    axes[1].set_ylabel(r'', fontsize=15)
    axes[0].legend(loc="lower left", fontsize=13)
    axes[1].legend(loc="lower left", fontsize=13)
    plt.show()
    #fig.savefig('figures/pump_probe/frequency_resolved_RCD_vacuum_2dos.png', dpi=300, bbox_inches='tight')
# %%


#%% distance and gap
gap = np.array([np.min(energy_array[i][0]-energy_array[i][1]) for i in range(4)])
gap[1] = energy_array[1][0][67,123]-energy_array[1][1][67,123]
gap[2] = energy_array[2][0][67,123]-energy_array[2][1][67,123]
gap[3] = energy_array[3][0][67,123]-energy_array[3][1][67,123]
gap = np.append(gap, (energy_array[4][0][67,123]-energy_array[4][1][67,123]))

fm_peak = np.array([w[np.argmax(chi_FM_upper[i])] for i in range(5)])
fm_peak[1] = w[np.argmin(chi_FM_upper[1])]

distance = np.array([(w[np.argmax(chi_two_magnons_upper[i])]-w[np.argmin(chi_two_magnons_lower[i])])/2 for i in range(5)])

with plt.style.context(['science','ieee']):
    fig, ax = plt.subplots(figsize=(4,3))

    #ax.plot([0,0.25,0.5,0.75],distance, '-o', color='#b5b5b8', label='distance')
    ax.plot([0,0.25,0.5,0.75, 1], gap , '-o', color='#b5b5b8', label=r'$\Delta_g$')
    ax.plot([0,0.25,0.5,0.75, 1], distance , '-o', color='#0071bc', label=r'$\Delta_{\text{RCD}}$')
    ax.plot([0,0.25,0.5,0.75, 1], fm_peak , '-o', color='#ff5050', label=r'$\xi_{\text{FM}}$')
   
    ax.set_ylabel(r'$\hslash\omega/J$', fontsize=12)
    ax.set_xlabel(r'$B/B_s$', fontsize=12)
    ax.set_xticks([0,0.25,0.5,0.75, 1])
    ax.legend(fontsize=12)
    plt.show()
    #fig.savefig('figures/pump_probe/gap_distance_fm_peak.png', dpi=300, bbox_inches='tight')



# %%
honeycomb_bz_x, honeycomb_bz_y = honeycomb_bz()

kx,ky = bzmesh(m=2)

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
        pc = axes[i].pcolormesh(kx, ky, cross_section_array[i], cmap="jet")
        
        plot(honeycomb_bz_x, honeycomb_bz_y, ax=axes[i], linestyle='-', linewidth=1, color='k')

        clb = fig.colorbar(pc, ax=axes[i], shrink=0.9)
        clb.ax.set_title(r'$\chi_{\bf k}$', loc='left', fontsize=16, pad=pads[i])
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
