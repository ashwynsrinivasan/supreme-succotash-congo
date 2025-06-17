import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import sys

from lmphoton import OptElement, OptColumn, OptRow, OptNetwork
from lmphoton.transforms import reflect
from lmphoton.models import Laser, Detector, BeamSplitter, DirectionalCoupler, Absorber, Waveguide, LossElement
from lmphoton.simulation import current_simulation as sim

from scipy.optimize import minimize
from scipy.signal import find_peaks
from functools import partial

from interleaver_parameters import *

import warnings
warnings.filterwarnings("ignore")
plt.style.use("plot_style.mplstyle")

um = 1e-6

from scipy.signal import find_peaks

def calculate_spectrum_bandwidth(wav, port_amp, peak_distance = 400, total_channels = 8, IL_channel_offset = [0e9, 10e9, 20e9, 30e9, 40e9, 50e9], bandwidth=1):
    channel_bandwidth_1dB = []
    channel_min_IL_bandwidth = []
    channel_center_freq = []
    channel_fsr = []
    min_IL_dB = []
    average_IL_dB = []
    ripple_IL_dB = []
    IL_dB = []
    
    port_amp_dB = 10*np.log10(port_amp)
    peaks, _ = find_peaks(port_amp, height=0.5, distance = 200)
    
    if len(peaks) < 1:
        peaks, _ = find_peaks(port_amp, distance = 200)
    center_index = np.argmin(np.abs(wav[peaks]-center_wavelength))
    
    
    fsr = np.diff(peaks)
    
    for i in range(-4, 4):
        center_index_min = np.argwhere(wav == wav[peaks][center_index+i]).flatten()[0] - int(fsr[center_index-1+i]/2)
        center_index_max = np.argwhere(wav == wav[peaks][center_index+i]).flatten()[0] + int(fsr[center_index+i]/2)
    
        port_amp_dB_local_normalized = port_amp_dB[center_index_min:center_index_max] - np.max(port_amp_dB[center_index_min:center_index_max])
        wav_local = wav[center_index_min:center_index_max]
        
        peak_local, _ = find_peaks(port_amp_dB_local_normalized)
        
        if len(peak_local) > 2:
            if peak_local[0] < 50:
                peak_local = peak_local[1:3]
            else:
                peak_local = peak_local[0:2]
        elif len(peak_local) == 1:
            peak_local = [peak_local[0], peak_local[0]]
        elif len(peak_local) == 2:
            if peak_local[0] < 50:
                peak_local = [peak_local[1], peak_local[1]]
            
        higher_freq_index = peak_local[1]+ np.argmin(np.abs(port_amp_dB_local_normalized[peak_local[1]:-1] + bandwidth))
        lower_freq_index = np.argmin(np.abs(port_amp_dB_local_normalized[0:peak_local[0]] + bandwidth))
        
        higher_freq = c/wav_local[higher_freq_index]
        lower_freq = c/wav_local[lower_freq_index]

        peaks_2,_ = find_peaks(port_amp_dB[center_index_min:center_index_max][lower_freq_index:higher_freq_index])

        if len(peaks_2) > 1:
            craters_2,_ = find_peaks(-port_amp_dB[center_index_min:center_index_max][lower_freq_index:higher_freq_index])

        if len(peaks_2) > 1:
            craters_2,_ = find_peaks(-port_amp_dB[center_index_min:center_index_max][lower_freq_index:higher_freq_index])
            average_IL_dB.append(np.average(port_amp_dB[center_index_min:center_index_max][lower_freq_index:higher_freq_index][peaks_2[0]:peaks_2[1]]))
            min_IL_dB.append(np.average(port_amp_dB[center_index_min:center_index_max][lower_freq_index:higher_freq_index][peaks_2]))
            ripple_IL_dB.append(np.average(port_amp_dB[center_index_min:center_index_max][lower_freq_index:higher_freq_index][craters_2]))
            channel_min_IL_bandwidth.append(((c/wav_local[lower_freq_index:higher_freq_index][peaks_2[0]] - c/wav_local[lower_freq_index:higher_freq_index][peaks_2[1]])))
        elif len(peaks_2) == 1:
            average_IL_dB.append(port_amp_dB[center_index_min:center_index_max][lower_freq_index:higher_freq_index][peaks_2][0])            
            min_IL_dB.append(port_amp_dB[center_index_min:center_index_max][lower_freq_index:higher_freq_index][peaks_2][0])
            ripple_IL_dB.append(0)
            channel_min_IL_bandwidth.append(0)
        
        
        channel_center_freq.append(lower_freq/2 + higher_freq/2)
        channel_bandwidth_1dB.append(((lower_freq - higher_freq)))
        
        ## calculating the RF signal spectrum        
        if i == 0:
            for offset in IL_channel_offset:
                diff_freq = c/wav_local - (lower_freq/2 + higher_freq/2) + offset
                Tb = 1/(56e9)
                rf_data_before_interleaver = np.array(
                    [0.5 * Tb * (np.sin(np.pi*f*Tb)/(np.pi*f*Tb))**2 for f in diff_freq])
                rf_data_before_interleaver /= np.max(rf_data_before_interleaver)

                rf_data_after_interleaver = rf_data_before_interleaver * port_amp[center_index_min:center_index_max]

                IL_linear = np.sum(rf_data_after_interleaver) / np.sum(rf_data_before_interleaver)
                IL_dB.append(10*np.log10(IL_linear))
    channel_fsr = np.diff(channel_center_freq)
    
    spectrum_fom = {
        "1dB_bandwidth": np.array(channel_bandwidth_1dB[0:total_channels]),
        "channel_center_frequency": np.array(channel_center_freq[0:total_channels]),         
        "min_dc_il_bandwidth": np.array(channel_min_IL_bandwidth[0:total_channels]),
        "rf_il_dB": np.array(IL_dB[0:total_channels]), 
        "min_dc_il_dB": np.array(min_IL_dB[0:total_channels]), 
        "ave_dc_il_dB": np.array(average_IL_dB[0:total_channels]), 
        "ripple_dc_il_dB": np.array(ripple_IL_dB[0:total_channels]), 
    }
    
    return spectrum_fom

class interleaver_3_stage_calc(OptRow):
    def __init__(self,
                L = 1*um,
                dL1 = 0.0,
                dL2 = 0.0,
                dL3 = 0.0,
                dc0 = 0.5,
                dc1 = 0.5,
                dc2 = 0.5,
                effective_index = SiN_effective_index,
                group_index = SiN_group_index,
                ports = ['1l PORT1', '2r PORT2', '3r PORT3', '4l PORT4'],
                name = 'INTERLEAVER_3STAGE'):
        
        self.L = L
        self.channel_spacing = FSR_wavelength/8

        self.wg0 = Waveguide(length=L+dL1,index=effective_index,group_index=group_index, loss_rate = 0.01*SiN_propagation_loss_dB_m)
        self.wg1 = Waveguide(length=L,    index=effective_index,group_index=group_index, loss_rate = 0.01*SiN_propagation_loss_dB_m)

        self.XPS0 = LossElement(loss = XPS_first_stage_loss_dB)
        self.XPS1 = LossElement(loss = XPS_first_stage_loss_dB)

        self.wg2 = Waveguide(length=L+dL2,index=effective_index,group_index=group_index, loss_rate = 0.01*SiN_propagation_loss_dB_m)
        self.wg3 = Waveguide(length=L,    index=effective_index,group_index=group_index, loss_rate = 0.01*SiN_propagation_loss_dB_m)

        self.XPS2 = LossElement(loss = XPS_first_stage_loss_dB)
        self.XPS3 = LossElement(loss = XPS_first_stage_loss_dB)

        self.wg4 = Waveguide(length=L+dL3,index=effective_index,group_index=group_index, loss_rate = 0.01*SiN_propagation_loss_dB_m)
        self.wg5 = Waveguide(length=L,    index=effective_index,group_index=group_index, loss_rate = 0.01*SiN_propagation_loss_dB_m)

        self.XPS4 = LossElement(loss = XPS_first_stage_loss_dB)
        self.XPS5 = LossElement(loss = XPS_first_stage_loss_dB)

        self.dc0 = DirectionalCoupler(splitratio=dc0, loss = SiN_directional_coupler_loss_dB)
        self.dc1 = DirectionalCoupler(splitratio=dc1, loss = SiN_directional_coupler_loss_dB)
        self.dc2 = DirectionalCoupler(splitratio=dc1, loss = SiN_directional_coupler_loss_dB)
        self.dc3 = DirectionalCoupler(splitratio=dc2, loss = SiN_directional_coupler_loss_dB)

        # Construct network:
        network = [
            self.dc0,
            OptColumn(
                [
                    self.wg0,
                    self.wg1,
                ]
            ),
            OptColumn(
                [
                    self.XPS0,
                    self.XPS1,
                ]
            ),
            self.dc1,
            OptColumn(
                [
                    self.wg2,
                    self.wg3,
                ]
            ),
            OptColumn(
                [
                    self.XPS2,
                    self.XPS3,
                ]
            ),
            self.dc2,
            OptColumn(
                [
                    self.wg4,
                    self.wg5,
                ]
            ),
            OptColumn(
                [
                    self.XPS4,
                    self.XPS5,
                ]
            ),
            self.dc3,
        ]
        
        super().__init__(
            network,
            ports=ports, 
            name=name
        )

    @property
    def phase_shift_1_top(self):
        return self.wg0._init_phase

    @phase_shift_1_top.setter
    def phase_shift_1_top(self, x):
        self.wg0._init_phase = x

    @property
    def phase_shift_2_top(self):
        return self.wg2._init_phase

    @phase_shift_2_top.setter
    def phase_shift_2_top(self, x):
        self.wg2._init_phase = x

    @property
    def phase_shift_3_top(self):
        return self.wg4._init_phase
    
    @phase_shift_3_top.setter
    def phase_shift_3_top(self,x):
        self.wg4._init_phase = x

    def top_input_bar_port_transmission(self):
        return np.real(self.smatrix[2,0] * np.conj(self.smatrix[2,0]))
    
    def bottom_input_bar_port_transmission(self):
        return np.real(self.smatrix[3,1] * np.conj(self.smatrix[3,1]))

    def top_input_cross_port_transmission(self):
        return np.real(self.smatrix[3,0] * np.conj(self.smatrix[3,0]))
    
    def bottom_input_cross_port_transmission(self):
        return np.real(self.smatrix[2,1] * np.conj(self.smatrix[2,1]))
    
    # Create optimal transfer function (square wave in frequency space)
    def calculate_optimal_transfer_function(self):
        optimal_transfer_function = np.array(
            [
                int(1*(i- 1310e-9 - self.channel_spacing/2)/self.channel_spacing % 2) for i in optimization_wavelength_sweep
            ]
        )
        return optimal_transfer_function
    
    def calculate_cost_function(self,param_list, info, printy=False):
        dc_split_0,dc_split_1, dc_split_2, phi1, phi2, phi3 = param_list
        _, bar_port_response, cross_port_response = self.calculate_spectrum(dc_split_0,dc_split_1,dc_split_2, phi1,phi2, phi3)
        cost = np.sum(np.abs(self.calculate_optimal_transfer_function() - bar_port_response)) 
        cost += np.sum(np.abs(1 - self.calculate_optimal_transfer_function() - cross_port_response)) # 1 - is to flip the response for the cross port. #I'm summing both the bar and cross response to give equally weighting to pass and stop bands

        # display information
        if (info['Nfeval'] == 0) and (printy == True):
            print("   #    DCSplit0    DCSplit1    DCSplit2    phi1        phi2      phi3       Cost")
        if (info['Nfeval']%10 == 0) and (printy == True):
            print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}  {5: 3.6f} {6: 3.6f}  {7: 3.6f}'.format(
                info['Nfeval'],
                dc_split_0,
                dc_split_1,
                dc_split_2,
                phi1,
                phi2,
                phi3,
                cost
            )
        )
        info['Nfeval'] += 1
        return cost
        
    def minimize_cost_function(
            self,
            random_dc=True,
            random_phase=True,
            printy = False
        ):
        initial_condition = []
        bounds = []
        if random_dc == True:
            for _ in range(3):
                initial_condition.append(np.random.rand())
                bounds.append((0,1))
        elif random_dc == False:
            initial_condition.append(self.dc0._splitratio)
            initial_condition.append(self.dc1._splitratio)
            initial_condition.append(self.dc3._splitratio)
            bounds.append((self.dc0._splitratio - bounds_epsilon,self.dc0._splitratio + bounds_epsilon))
            bounds.append((self.dc1._splitratio - bounds_epsilon,self.dc1._splitratio + bounds_epsilon))
            bounds.append((self.dc3._splitratio - bounds_epsilon,self.dc3._splitratio + bounds_epsilon))

        if random_phase == True:
            for _ in range(3):
                initial_condition.append(np.pi*(2*np.random.rand()-1))
                bounds.append((-2*np.pi,2*np.pi))
        elif random_phase == False:
            initial_condition.append(self.phase_shift_1_top)
            initial_condition.append(self.phase_shift_2_top)
            initial_condition.append(self.phase_shift_3_top)
            bounds.append((self.phase_shift_1_top - bounds_epsilon,self.phase_shift_1_top + bounds_epsilon))
            bounds.append((self.phase_shift_2_top - bounds_epsilon,self.phase_shift_2_top + bounds_epsilon))
            bounds.append((self.phase_shift_3_top - bounds_epsilon,self.phase_shift_3_top + bounds_epsilon))

        result = minimize(
            partial(self.calculate_cost_function,printy=printy),
            initial_condition,
            method='Nelder-Mead',
            bounds = bounds,
            args=({
                'Nfeval':0
            },), 
            options = {
                "maxiter":1000, 
                'xatol':1e-3,
                'fatol':1e-3,
            },
        )
        return result
    
    def calculate_spectrum(
            self,
            dc_split_0,
            dc_split_1,
            dc_split_2,
            phi1,
            phi2,
            phi3
        ):
        self.dc0._splitratio = dc_split_0
        self.dc1._splitratio = dc_split_1
        self.dc2._splitratio = dc_split_1
        self.dc3._splitratio = dc_split_2
        
        self.phase_shift_1_top = phi1
        self.phase_shift_2_top = phi2
        self.phase_shift_3_top = phi3

        cross_storage = []
        bar_storage = []
        for wav in optimization_wavelength_sweep:
            sim().wavelength = wav
            self.recursive_update()
            cross_storage.append(self.top_input_cross_port_transmission())
            bar_storage.append(self.top_input_bar_port_transmission())

        cross_storage = np.array(cross_storage)
        bar_storage   = np.array(bar_storage)

        return optimization_wavelength_sweep, bar_storage, cross_storage
                
    def RF_spectrum_56G(self):
        frequency_sweep = c/optimization_wavelength_sweep
        diff_freq = freq - frequency_sweep
        Tb = 1/(56e9)
        tp = np.array(
            [0.5 * Tb * (np.sin(np.pi*f*Tb)/(np.pi*f*Tb))**2 for f in diff_freq])
        tp /= np.max(tp)
        return tp

    def calculate_spectrum_IL_and_crosstalk_from_sweep(self, bar_storage, cross_storage):
        #First, we calculate the RF spectrum in the optical domain for 56G encoded data
        rf_spectrum_before_interleaver = self.RF_spectrum_56G()

        rf_data_after_interleaver = rf_spectrum_before_interleaver * bar_storage
        rf_data_crosstalk_after_interleaver = rf_spectrum_before_interleaver * cross_storage

        IL_linear = np.sum(rf_data_after_interleaver) / \
            np.sum(rf_spectrum_before_interleaver)
        CT_linear = np.sum(rf_data_crosstalk_after_interleaver) / \
            np.sum(rf_spectrum_before_interleaver)

        return IL_linear, CT_linear
            

    def plot_spectrum(self, result):
        wav, bar_port_amp, cross_port_amp = self.calculate_spectrum(*result.x)
        cost_sweep = self.calculate_optimal_transfer_function()

        plt.figure()
        plt.plot(wav*1e9, bar_port_amp)
        plt.plot(wav*1e9, cross_port_amp)
        plt.plot(wav*1e9, bar_port_amp + cross_port_amp)
        plt.plot(wav*1e9, cost_sweep, alpha=cost_function_plot_alpha)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Transmission")
        plt.title("2 Stage Filter after optimization")
        plt.legend(['Bar', 'Cross', 'Total', 'Cost function'])
        plt.show()

        plt.figure()
        plt.plot(wav*1e9, 10*np.log10(bar_port_amp))
        plt.plot(wav*1e9, 10*np.log10(cross_port_amp))
        plt.plot(wav*1e9, 10*np.log10(bar_port_amp + cross_port_amp))
        plt.plot(wav*1e9, 10*np.log10(cost_sweep+1e-6),
                 zorder=-100, alpha=cost_function_plot_alpha)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Transmission (dB)")
        plt.title("Filter after optimization")
        plt.legend(["Bar", "Cross", 'Total', 'Cost function'])
        plt.ylim(-31, 1)
        plt.show()

    def save_result(self,result):
        with open("bidi_sweep_results_3_stage.csv",'a') as f:
            dc_split_0,dc_split_1,dc_split_2,phi1,phi2,phi3 = result.x
            final_cost_function_value = result.fun

            df = pd.DataFrame(
                {
                    "DC0":dc_split_0,
                    "DC1":dc_split_1,
                    "DC2":dc_split_2,
                    "PHI1":phi1,
                    "PHI2":phi2,
                    "PHI3":phi3,
                    "Final_cost":final_cost_function_value,
                 },
                 index = [0]
            )

            df.to_csv(f, header=False)
