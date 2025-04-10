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
from plot_utils import plot, letter_annotation, panel
import scienceplots
# %%

# default plotting parameters unless specified
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['axes.labelsize'] = 16

def letter_annotation(ax, xoffset, yoffset, letter,size=12):
 ax.text(xoffset, yoffset, letter, transform=ax.transAxes,
         size=size)

#%% 1x3; figure 1
img_setup = Image.open('figures/concept_maps/setup.png')
img_config = Image.open('figures/concept_maps/configuration-canted.png')
img_dmi = Image.open('figures/concept_maps/DMI_honeycomb.png')
#%% 
Nr = 1
Nc = 3
fig, axes = plt.subplots(Nr,Nc,figsize=(18,6),gridspec_kw={'width_ratios': [1, 1.2, 1.1]})
fig.subplots_adjust(wspace=0,hspace=0)
axes[0].imshow(img_dmi)
axes[1].imshow(img_config)
axes[2].imshow(img_setup)
for n in range(Nc):
    axes[n].set_axis_off()
letter_annotation(axes[0],0,1.0,r'$\mathrm{(a)}$',size=20)
letter_annotation(axes[1],-0.1,1.0,r'$\mathrm{(b)}$',size=20)
letter_annotation(axes[2],-0.1,0.8,r'$\mathrm{(c)}$',size=20)
fig.subplots_adjust(top=0.99,bottom=0.01,right=0.99,left=0.01)

plt.show()
plt.savefig('figures/panel_plots/figure1_canted_afm.png', dpi=300, bbox_inches='tight')

#%%