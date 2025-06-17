from scipy.constants import c
import numpy as np
import pandas as pd

import os

XPS_first_stage_loss_dB   = 0.07
XPS_second_stage_loss_dB  = 0.07
XPS_third_stage_loss_dB   = 0.07
SiN_propagation_loss_dB_m = 40
SiN_directional_coupler_loss_dB = 0.01

# gf45clo_sin_neff_ng_variability = pd.read_csv(
#     os.getcwd()+"/friendly-system-lmi/pcm/congo/interleaver/gf45clo_sin_neff_ng_variability.csv")
gf45clo_sin_neff_ng_variability = pd.read_csv("gf45clo_sin_neff_ng_variability.csv")

sin_group_index_median = np.median(gf45clo_sin_neff_ng_variability['ng_freq0'])
sin_effective_index_median = np.median(gf45clo_sin_neff_ng_variability['neff_freq0'])

SiN_group_index = sin_group_index_median
SiN_effective_index = sin_effective_index_median
center_wavelength = 1310e-9
freq = c/(center_wavelength)
FSR_wavelength = 1600e9 * (center_wavelength)**2/(c)
channel_spacing = FSR_wavelength/8
channel1_wavelength = center_wavelength - 3.5*channel_spacing
channel2_wavelength = center_wavelength - 2.5*channel_spacing
channel3_wavelength = center_wavelength - 1.5*channel_spacing
channel4_wavelength = center_wavelength - 0.5*channel_spacing
channel5_wavelength = center_wavelength + 0.5*channel_spacing
channel6_wavelength = center_wavelength + 1.5*channel_spacing
channel7_wavelength = center_wavelength + 2.5*channel_spacing
channel8_wavelength = center_wavelength + 3.5*channel_spacing
channel_wavelength_array = [
    channel1_wavelength,
    channel2_wavelength,
    channel3_wavelength,
    channel4_wavelength,
    channel5_wavelength,
    channel6_wavelength,
    channel7_wavelength,
    channel8_wavelength,
]
dL_1600Ghz_FSR = (center_wavelength)**2/(SiN_group_index * FSR_wavelength)
dL_800Ghz_FSR = (center_wavelength)**2/(SiN_group_index * FSR_wavelength/2)
dL_400Ghz_FSR = (center_wavelength)**2/(SiN_group_index * FSR_wavelength/4)

optimization_wavelength_sweep = np.linspace(1295e-9,1325e-9,num=4000)
process_variation_num_samples = 2000
random_variable_sigma = 0.04
cost_function_plot_alpha = 0.3
bounds_epsilon = 5e-3