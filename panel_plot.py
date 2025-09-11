# %%
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.physics.quantum.dagger import Dagger
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from labellines import labelLine, labelLines
from scipy.signal import savgol_filter
import math
from PIL import Image
from plot_utils import plot, letter_annotation, panel, panel_unequal
import scienceplots
# %%

# default plotting parameters unless specified
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['axes.labelsize'] = 16

#%% 1x3; figure 1
img_setup = Image.open('figures/concept_maps/setup.png')
img_config = Image.open('figures/concept_maps/configuration-canted.png')
img_dmi = Image.open('figures/concept_maps/DMI_honeycomb.png')
#%% 
Nr = 1
Nc = 3
fig, axes = panel(figsize=(18,6), 
                  nrows=Nr, ncols=Nc, 
                  width_ratios=[1, 1.2, 1.1], height_ratios=[1], 
                  hspace=0, wspace=0)
axes[0].imshow(img_dmi)
axes[1].imshow(img_config)
axes[2].imshow(img_setup)
for n in range(Nc):
    axes[n].set_axis_off()
letter_annotation(axes[0],0,1.0,r'$\mathrm{(a)}$',size=20)
letter_annotation(axes[1],-0.1,1.0,r'$\mathrm{(b)}$',size=20)
letter_annotation(axes[2],-0.05,0.92,r'$\mathrm{(c)}$',size=20)
fig.subplots_adjust(top=0.99,bottom=0.01,right=0.99,left=0.01)

plt.show()
fig.savefig('figures/panel_plots/figure1_canted_afm.png', dpi=300, bbox_inches='tight')

#%% 1 + 2x2; figure 2
img_band = Image.open('figures/canted_energy_bands/canted_afm_band_structure.png')
img_berry_curvature_upper = Image.open('figures/berry_curvatures/canted_berry_curvature_upper_exact.png')
img_berry_curvature_lower = Image.open('figures/berry_curvatures/canted_berry_curvature_lower_exact.png')
img_chern_number_upper = Image.open('figures/chern_numbers/canted_Chern_upper.png')
img_chern_number_lower = Image.open('figures/chern_numbers/canted_Chern_lower.png')

#%%  unequal size
Nr = 2
Nc = 4
fig, gs = panel_unequal(figsize=(24,9), 
                  nrows=Nr, ncols=Nc, 
                  width_ratios=[1, 1, 1, 1], height_ratios=[1, 1], 
                  hspace=0, wspace=0)

ax1 = fig.add_subplot(gs[0:, 0:2])  # Band structure
ax2 = fig.add_subplot(gs[0, 2])  # Berry curvature lower
ax3 = fig.add_subplot(gs[0, 3])  # Berry curvature upper
ax4 = fig.add_subplot(gs[1, 2])  # Chern number lower
ax5 = fig.add_subplot(gs[1, 3])  # Chern number upper
axes = [ax1, ax2, ax3, ax4, ax5]
ax1.imshow(img_band)
ax2.imshow(img_berry_curvature_lower)
ax3.imshow(img_berry_curvature_upper)
ax4.imshow(img_chern_number_lower)
ax5.imshow(img_chern_number_upper)
for n in range(5):
    axes[n].set_axis_off()
letter_annotation(axes[0],0,0.97,r'$\mathrm{(a)}$',size=24)
letter_annotation(axes[1],-0.035,0.933,r'$\mathrm{(b)}$',size=24)
letter_annotation(axes[2],-0.035,0.943,r'$\mathrm{(c)}$',size=24)
letter_annotation(axes[3],-0.04,0.97,r'$\mathrm{(d)}$',size=24)
letter_annotation(axes[4],-0.02,0.97,r'$\mathrm{(e)}$',size=24)
fig.subplots_adjust(top=0.99,bottom=0.01,right=0.99,left=0.01)

plt.show()
fig.savefig('figures/panel_plots/figure2_canted_AFM_topology.png', dpi=300, bbox_inches='tight')

# %%
# 2x3
img_raman_cross_section_two_magnon_upper = Image.open('figures/raman_scattering/raman_cross_section_RL_two-magnon_upper.png')
img_raman_cross_section_AFM_upper = Image.open('figures/raman_scattering/raman_cross_section_RL_AFM_upper.png')
img_raman_cross_section_FM_upper = Image.open('figures/raman_scattering/raman_cross_section_RL_FM_upper.png')
img_raman_scattering_two_magnon_upper = Image.open('figures/raman_scattering/two-magnon_process_RL.png')
img_raman_scattering_AFM_upper = Image.open('figures/raman_scattering/AFM_process_RL.png')
img_raman_scattering_FM_upper = Image.open('figures/raman_scattering/FM_process_RL.png')

# %%
Nr = 2
Nc = 3
fig, axes = plt.subplots(Nr,Nc,figsize=(20,8),gridspec_kw={'width_ratios': [1, 1,1], 'height_ratios':[1.5,1]})
fig.subplots_adjust(wspace=0.01,hspace=0)
axes[0][0].imshow(img_raman_cross_section_two_magnon_upper)
axes[0][1].imshow(img_raman_cross_section_AFM_upper)
axes[0][2].imshow(img_raman_cross_section_FM_upper)

axes[1][0].imshow(img_raman_scattering_two_magnon_upper)
axes[1][1].imshow(img_raman_scattering_AFM_upper)
axes[1][2].imshow(img_raman_scattering_FM_upper)
for m in range(Nr):
    for n in range(Nc):
        axes[m][n].set_axis_off()
letter_annotation(axes[0][0],0,0.95,r'$\mathrm{(a)}$',size=20)
letter_annotation(axes[0][1],0,0.95,r'$\mathrm{(b)}$',size=20)
letter_annotation(axes[0][2],0,0.95,r'$\mathrm{(c)}$',size=20)

letter_annotation(axes[1][0],-0.085,0.99,r'$\mathrm{(d)}$',size=20)
letter_annotation(axes[1][1],-0.105,0.99,r'$\mathrm{(e)}$',size=20)
letter_annotation(axes[1][2],-0.08,0.99,r'$\mathrm{(f)}$',size=20)

# letter_annotation(axes[0][0],0.44,0.97,r'$\mathrm{2M}$',size=25)
# letter_annotation(axes[0][1],0.44,0.97,r'$\mathrm{AFM}$',size=25)
# letter_annotation(axes[0][2],0.44,0.97,r'$\mathrm{FM}$',size=25)

fig.subplots_adjust(top=0.98,bottom=0.00,right=1.0,left=0)
#fig.tight_layout()

plt.show()

fig.savefig("figures/panel_plots/figure3_raman_process_and_cross_section_RL.png", dpi=300)
# %%
