# %%
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from common.plot_utils import plot, letter_annotation, panel, panel_unequal
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
# 2x2
img_canted_berry_rcd_lower = Image.open('figures/berry_curvatures/canted_berry_curvature_lower_RCD.png')
img_canted_berry_rcd_upper = Image.open('figures/berry_curvatures/canted_berry_curvature_upper_RCD.png')
img_chern_rcd_lower = Image.open('figures/chern_numbers/canted_Chern_RCD_lower.png')
img_chern_rcd_upper = Image.open('figures/chern_numbers/canted_Chern_RCD_upper.png')

Nr = 2
Nc = 2
fig, axes = plt.subplots(Nr,Nc,figsize=(12.5,9),gridspec_kw={'width_ratios': [1, 1]})

plt.tight_layout() # fix the layout first
fig.subplots_adjust(wspace=-0.05, hspace=-0.01) # fix the spacing

# first row
axes[0][0].imshow(img_canted_berry_rcd_lower)
pos_00_1 = axes[0][0].get_position() # get the original position 
pos_00_2 = [pos_00_1.x0+0.0055, pos_00_1.y0 , pos_00_1.width, pos_00_1.height]
axes[0][0].set_position(pos_00_2) # set a new position

axes[0][1].imshow(img_canted_berry_rcd_upper)
pos_01_1 = axes[0][1].get_position() # get the original position 
pos_01_2 = [pos_01_1.x0+0.0015 , pos_01_1.y0 , pos_01_1.width, pos_01_1.height] 
axes[0][1].set_position(pos_01_2) # set a new position

# second row
axes[1][0].imshow(img_chern_rcd_lower)
axes[1][1].imshow(img_chern_rcd_upper)

for m in range(Nr):
    for n in range(Nc):
        axes[m][n].set_axis_off()
letter_annotation(axes[0][0],0,0.97,r'$\mathrm{(a)}$',size=20)
letter_annotation(axes[0][1], 0, 0.97,r'$\mathrm{(b)}$',size=20)
letter_annotation(axes[1][0], 0.001, 0.97,r'$\mathrm{(c)}$',size=20)
letter_annotation(axes[1][1], -0.005, 0.97,r'$\mathrm{(d)}$',size=20)


plt.show()
fig.savefig("figures/panel_plots/figure5_berry_curvature_rcd", dpi=300, bbox_inches='tight')
# %%
# 2x4
img_canted_berry_noFM_lower = Image.open('figures/berry_curvatures/canted_berry_curvature_lower_noFM.png')
img_canted_berry_noFM_upper = Image.open('figures/berry_curvatures/canted_berry_curvature_upper_noFM.png')
img_canted_berry_FM_only_lower = Image.open('figures/berry_curvatures/canted_berry_curvature_lower_FM.png')
img_canted_berry_FM_only_upper = Image.open('figures/berry_curvatures/canted_berry_curvature_upper_FM.png')

img_chern_no_FM_lower = Image.open('figures/chern_numbers/canted_Chern_number_lower_RCD_noFM.png')
img_chern_no_FM_upper = Image.open('figures/chern_numbers/canted_Chern_number_upper_RCD_noFM.png')
img_chern_FM_only_lower = Image.open('figures/chern_numbers/canted_Chern_number_lower_RCD_FM.png')
img_chern_FM_only_upper = Image.open('figures/chern_numbers/canted_Chern_number_upper_RCD_FM.png')

Nr = 2
Nc = 4
fig, axes = plt.subplots(Nr,Nc,figsize=(24,9),gridspec_kw={'width_ratios': [1, 1, 1, 1]})

plt.tight_layout() # fix the layout first
fig.subplots_adjust(wspace=-0.01,hspace=-0.01) # fix the spacing

axes[0][0].imshow(img_canted_berry_noFM_lower)
pos_00_1 = axes[0][0].get_position() # get the original position 
pos_00_2 = [pos_00_1.x0+0.001, pos_00_1.y0, pos_00_1.width, pos_00_1.height] 
axes[0][0].set_position(pos_00_2) # set a new position

axes[0][1].imshow(img_canted_berry_noFM_upper)
pos_01_1 = axes[0][1].get_position() # get the original position 
pos_01_2 = [pos_01_1.x0+0.0055, pos_01_1.y0, pos_01_1.width, pos_01_1.height] 
axes[0][1].set_position(pos_01_2) # set a new position

axes[0][2].imshow(img_canted_berry_FM_only_lower)
pos_02_1 = axes[0][2].get_position() # get the original position 
pos_02_2 = [pos_02_1.x0+0.0055, pos_02_1.y0, pos_02_1.width, pos_02_1.height] 
axes[0][2].set_position(pos_02_2) # set a new position

axes[0][3].imshow(img_canted_berry_FM_only_upper)
pos_03_1 = axes[0][3].get_position() # get the original position 
pos_03_2 = [pos_03_1.x0+0.0015, pos_03_1.y0, pos_03_1.width, pos_03_1.height] 
axes[0][3].set_position(pos_03_2) # set a new position

# axes[1][0].imshow(img_chern_rcd_only_lower)
# axes[1][1].imshow(img_chern_rcd_only_upper)
axes[1][0].imshow(img_chern_no_FM_lower)
axes[1][1].imshow(img_chern_no_FM_upper)
axes[1][2].imshow(img_chern_FM_only_lower)
axes[1][3].imshow(img_chern_FM_only_upper)
for m in range(Nr):
    for n in range(Nc):
        axes[m][n].set_axis_off()
letter_annotation(axes[0][0],0,0.97,r'$\mathrm{(a)}$',size=24)
letter_annotation(axes[0][1],0,0.97,r'$\mathrm{(b)}$',size=24)
letter_annotation(axes[0][2],0,0.97,r'$\mathrm{(c)}$',size=24)
letter_annotation(axes[0][3],0,0.97,r'$\mathrm{(d)}$',size=24)
letter_annotation(axes[1][0],0.023,0.97,r'$\mathrm{(e)}$',size=24)
letter_annotation(axes[1][1],0.024,0.975,r'$\mathrm{(f)}$',size=24)
letter_annotation(axes[1][2],0.023,0.98,r'$\mathrm{(g)}$',size=24)
letter_annotation(axes[1][3],0.024,0.97,r'$\mathrm{(h)}$',size=24)


plt.show()
fig.savefig("figures/panel_plots/figure6_chern_curvature_noFM_FM.png",dpi=300, bbox_inches = "tight")
# %%
# 2x2 CrI3
img_set_up_CrI3 = Image.open('figures/concept_maps/setup_3D.png')
img_CrI3_bands = Image.open('figures/CrI3_bands/CrI3_band_structure.png')
img_CrI3_berry_curvature_upper = Image.open('figures/CrI3_berry_curvatures/CrI3_berry_curvature_upper.png')
img_CrI3_berry_curvature_lower = Image.open('figures/CrI3_berry_curvatures/CrI3_berry_curvature_lower.png')

Nr = 2
Nc = 2
fig, axes = panel(figsize=(8,6), 
                  nrows=Nr, ncols=Nc, 
                  width_ratios=[1, 1], height_ratios=[1, 1], 
                  hspace=0, wspace=0)
plt.tight_layout() # fix the layout first
fig.subplots_adjust(wspace=0.05, hspace=-0.01) # fix the spacing
# ax1 = fig.add_subplot(gs[0, 0])  # setup
# ax2 = fig.add_subplot(gs[0, 1])  # bands
# ax3 = fig.add_subplot(gs[1, :])  # berry curvatures

# first row
axes[0,0].imshow(img_set_up_CrI3)
pos_00_1 = axes[0,0].get_position() # get the original position 
pos_00_2 = [pos_00_1.x0-0.0055, pos_00_1.y0 , pos_00_1.width, pos_00_1.height]
axes[0][0].set_position(pos_00_2) # set a new position

axes[0,1].imshow(img_CrI3_bands)
pos_01_1 = axes[0,1].get_position() # get the original position 
pos_01_2 = [pos_01_1.x0+0.0015 , pos_01_1.y0 , pos_01_1.width, pos_01_1.height] 
axes[0,1].set_position(pos_01_2) # set a new position

# second row
axes[1,0].imshow(img_CrI3_berry_curvature_upper)
pos_10_1 = axes[1,0].get_position() # get the original position
pos_10_2 = [pos_10_1.x0-0.01, pos_10_1.y0 , pos_10_1.width, pos_10_1.height]
axes[1,0].set_position(pos_10_2) # set a new position

axes[1,1].imshow(img_CrI3_berry_curvature_lower)
pos_11_1 = axes[1,1].get_position() # get the original position
pos_11_2 = [pos_11_1.x0 , pos_11_1.y0 , pos_11_1.width, pos_11_1.height]
axes[1,1].set_position(pos_11_2) # set a new position

for m in range(Nr):
    for n in range(Nc):
        axes[m][n].set_axis_off()

letter_annotation(axes[0][0],0,1.17,r'$\mathrm{(a)}$',size=18)
letter_annotation(axes[0][1], -0.022, 0.97,r'$\mathrm{(b)}$',size=18)
letter_annotation(axes[1][0], -0.0523, 0.97,r'$\mathrm{(c)}$',size=18)
letter_annotation(axes[1][1], -0.0523, 0.97,r'$\mathrm{(d)}$',size=18)

plt.show()
fig.savefig("figures/panel_plots/figure1_CrI3_set_up", dpi=300, bbox_inches='tight')

#%% 1x2; figure 2 in short paper
img_TRCD_vary_T = Image.open('figures/thermal_RCD/CrI3_TRCD_vary_T_more.png')
img_TRCD_vary_D = Image.open('figures/thermal_RCD/CrI3_TRCD_vary_D_more.png')
#%% 
Nr = 1
Nc = 2
fig, axes = panel(figsize=(18,6), 
                  nrows=Nr, ncols=Nc, 
                  width_ratios=[1, 1], height_ratios=[1], 
                  hspace=0, wspace=-0.1)
axes[0].imshow(img_TRCD_vary_T)
axes[1].imshow(img_TRCD_vary_D)
for n in range(Nc):
    axes[n].set_axis_off()
letter_annotation(axes[0],0,1.0,r'$\mathrm{(a)}$',size=40)
letter_annotation(axes[1],0,1.0,r'$\mathrm{(b)}$',size=40)
fig.subplots_adjust(top=0.99,bottom=0.01,right=0.99,left=0.01)

plt.show()
fig.savefig('figures/panel_plots/figure2_thermal_RCD_more.png', dpi=300, bbox_inches='tight')
# %%
