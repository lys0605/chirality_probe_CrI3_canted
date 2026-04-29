# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from common.plot_utils import plot, letter_annotation, panel, plot_frequency_resolved_RCD
from common.math_utils import gaussian_function, lorentzian_function, normalize
from common.bose_statistics import bose_einstein, boltzmann_factor
from common.honeycomb_lattice import *
from canted.canted_raman_cross_section import *
import scienceplots
# %%

J = 1
D = 0.1
S = 5/2
s_values = [0, 0.25, 0.5, 0.75, 1]
temperatures = [0, 2, 4, 6, 24, 58]

raman_array = [get_raman_cross_section_exact(J=J, D=D, S=S, B0=s) for s in s_values]

#%%
len_s = len(s_values)
cross_section_array = np.array([raman_array[i][0] for i in range(len_s)])
berry_array = np.array([raman_array[i][1] for i in range(len_s)])
berry_rcd_array = np.array([raman_array[i][2] for i in range(len_s)])
energy_array = np.array([raman_array[i][3] for i in range(len_s)])


w = np.linspace(0,25,200)
len_w = len(w)
width = (w[1]-w[0])*2

# %%
chi_two_magnons_lower = np.zeros((len_s, len_w))
chi_AFM_lower = np.zeros((len_s, len_w))
chi_FM_lower = np.zeros((len_s, len_w))
chi_two_magnons_upper = np.zeros((len_s, len_w))
chi_AFM_upper = np.zeros((len_s, len_w))
chi_FM_upper = np.zeros((len_s, len_w))

for j in range(len_s):
    for i in range(len_w):
        chi_two_magnons_lower[j][i] = bz_integration_honeycomb( cross_section_array[j][1] * gaussian_function(w[i], x0= 2*energy_array[j][1], width=width))
        chi_two_magnons_upper[j][i] = bz_integration_honeycomb( cross_section_array[j][0] * gaussian_function(w[i], x0= 2*energy_array[j][0], width=width))
        chi_AFM_lower[j][i] = bz_integration_honeycomb( cross_section_array[j][5] * gaussian_function(w[i], x0= energy_array[j][1]+energy_array[j][0], width=width))
        chi_AFM_upper[j][i] = bz_integration_honeycomb( cross_section_array[j][4] * gaussian_function(w[i], x0= energy_array[j][1]+energy_array[j][0], width=width))
        chi_FM_lower[j][i] = bz_integration_honeycomb( cross_section_array[j][3] * gaussian_function(w[i], x0= np.abs(energy_array[j][1]-energy_array[j][0]), width=width))
        chi_FM_upper[j][i] =bz_integration_honeycomb( cross_section_array[j][2] * gaussian_function(w[i], x0= np.abs(energy_array[j][1]-energy_array[j][0]), width=width))


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


# %%
rcd_label = [r'$\chi_{\alpha^{\prime}\bar{\alpha}^\prime}$',r'$\chi_{\beta^{\prime}\bar{\beta}^\prime}$',
             r'$\chi_{\alpha^{\prime}\beta}$', r'$\chi_{\bar{\beta}^{\prime}\bar{\alpha}}$',
             r'$\chi_{\alpha^{\prime}\bar{\beta}^\prime}$', r'$\chi_{\beta^{\prime}\alpha^\prime}$'
            ]
chi_vacuum = chi_two_magnons_lower+chi_two_magnons_upper+chi_AFM_lower+chi_AFM_upper
chi_one_magnon = 2*chi_two_magnons_lower+chi_two_magnons_upper+2*chi_AFM_lower+chi_AFM_upper+chi_FM_upper
# two_dos = dos_two_magnons_lower+dos_two_magnons_upper
# dos_vacuum = dos_two_magnons_lower+dos_two_magnons_upper+dos_AFM_upper+dos_AFM_lower
# dos_one_magnon = 2*dos_two_magnons_lower+dos_two_magnons_upper+2*dos_AFM_lower+dos_AFM_upper+dos_FM 
dos_weighted_vacuum = dos_weighted_two_magnons_lower+dos_weighted_two_magnons_upper+dos_weighted_AFM_lower+dos_weighted_AFM_upper
dos_weighted_one_magnon = 2*dos_weighted_two_magnons_lower+dos_weighted_two_magnons_upper+2*dos_weighted_AFM_lower+dos_weighted_AFM_upper+dos_weighted_FM

colors_vacuum = ['#00215d', '#0071bc' , '#8fd0ff']
colors_one_magnon = ['#b6000f' , '#ff814b', '#ffc883']

with plt.style.context(['science','ieee']):
    fig, ax = plt.subplots(figsize=(4,3))
    #plot_frequency_resolved_RCD(w, chi_vacuum[1:4], s_values[1:4], plot_length=3, color=colors_vacuum, label=r'')
    plot_frequency_resolved_RCD(ax, w, chi_vacuum[1:4], s_values[0:3], plot_length=3, color=colors_vacuum[0:3], label=r'')
    #plot_frequency_resolved_RCD(ax, w, dos_weighted_one_magnon[3], s_values[0], ls='--', plot_length=1, color='black', label=r'$\Lambda^\prime$')
    #plt.axvline(w[np.argmin(chi_FM_upper[1])])
    #plt.axvline(w[np.argmax(chi_two_magnons_upper[1])])
    #plt.axvline(w[np.argmin(chi_two_magnons_lower[1])])
    ax.set_xlabel(r'$\hslash\omega/J$', fontsize=13)
    ax.set_ylabel(r'$\chi^{(1)}(\omega)$', fontsize=13)
    ax.legend(loc="lower left", fontsize=13)
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
