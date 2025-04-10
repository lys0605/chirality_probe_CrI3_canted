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
    k_B = 8.617333262145e-5 # eV/K
    return 1/(np.exp((E-mu)/(k_B*T))-1)

J = 1
S = 5/2
s_values = [0, 0.25, 0.5, 0.75, 1]

raman_array = [get_raman_cross_section_exact(J=J, D=D, S=S, B0=s) for s in s_values]

#%%
len_s = len(s_values)
cross_section_array = np.array([raman_array[i][0] for i in range(len_s)])
berry_array = np.array([raman_array[i][1] for i in range(len_s)])
berry_rcd_array = np.array([raman_array[i][2] for i in range(len_s)])
energy_array = np.array([raman_array[i][3] for i in range(len_s)])

#%%
w = np.linspace(0,25,400)
len_w = len(w)
width = (w[1]-w[0])

