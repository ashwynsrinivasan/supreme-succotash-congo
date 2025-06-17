from lmphoton import OptColumn, OptRow
from lmphoton.models import DirectionalCoupler, Waveguide
from lmphoton.simulation import current_simulation as sim 

from lmtimesim.components.Phase_shifters.hps_phase_shifter import HPS_PhaseShifter
from lmtimesim.components.Passives.tap import Tap
import scipy.constants as const

import numpy as np

import warnings
warnings.filterwarnings('ignore')

um = 1e-6

class Interleaver(OptRow):
    def __init__(
        self,
        L = 350*um, 
        dL1 = 484.135*um,         
        dL2 = 50.8*um,
        dL3 = 968.27*um, 
        dL4 = 101.6*um,
        dc1 = 0.5, 
        dc2 = 0.63,
        dc3 = 0.9,
        tap_ratio = 0.01,
        # ports = ('1l PORT_1L_2ST','2r TAP_1ST','3r TAP_2ST_A', '4r PORT_2R_2ST', '5r TAP_2ST_B', '6r PORT_3R_2ST','7l PORT_4L_2ST'),
        name = 'Interleaver_2Stage'
        ):

        self.FSR_wavelength = 1600e9 * 1310e-9**2/const.c

        self.sin_effective_index = 1.587
        self.sin_group_index = 1.94
        
        self.si_effective_index = 2.4
        self.si_group_index = 4.34
        
        self.sin_propagation_loss_dB_m = 30
        self.HPS_insertion_loss_dB = 0.01
        self.sin_coupler_loss_dB = 0.12
        
        self.L = L
        self.channel_spacing = self.FSR_wavelength / 8.0

        self.wg0 = Waveguide(length=L+dL1,index=self.sin_effective_index,group_index=self.sin_group_index, loss_rate = 0.01*self.sin_propagation_loss_dB_m)
        self.wg1 = Waveguide(length=L,    index=self.sin_effective_index,group_index=self.sin_group_index, loss_rate = 0.01*self.sin_propagation_loss_dB_m)
        self.wg2 = Waveguide(length=dL2,index=self.sin_effective_index,group_index=self.sin_group_index, loss_rate = 0.01*self.sin_propagation_loss_dB_m)

        self.HPS0 = HPS_PhaseShifter()
        self.HPS1 = HPS_PhaseShifter()

        self.wg3 = Waveguide(length=L+dL3, index=self.sin_effective_index, group_index=self.sin_group_index, loss_rate = 0.01*self.sin_propagation_loss_dB_m)
        self.wg4 = Waveguide(length=L,     index=self.sin_effective_index, group_index=self.sin_group_index, loss_rate = 0.01*self.sin_propagation_loss_dB_m)
        self.wg5 = Waveguide(length=dL4,   index=self.sin_effective_index, group_index=self.sin_group_index, loss_rate = 0.01*self.sin_propagation_loss_dB_m)

        self.HPS2 = HPS_PhaseShifter()
        self.HPS3 = HPS_PhaseShifter()

        self.dc0 = DirectionalCoupler(splitratio=dc1, loss = self.sin_coupler_loss_dB)
        self.dc1 = DirectionalCoupler(splitratio=dc2, loss = self.sin_coupler_loss_dB)
        self.dc2 = DirectionalCoupler(splitratio=dc3, loss = self.sin_coupler_loss_dB)

        ########## Time domain parameters ###############
        self.HPS0.heater_voltage = 0.0
        self.HPS1.heater_voltage = 0.0
        self.HPS2.heater_voltage = 0.0
        self.HPS3.heater_voltage = 0.0

        ########## Enable aggressors ###############
        self._enable_aggressors = False

        self._rc_filter_bool = False
        self._time_ref = sim().time

        # Construct network:
        network = [
            self.dc0,
            OptColumn(
                [
                    self.wg0,
                    OptRow([
                        self.wg1,
                        self.wg2])
                ]
            ),
            OptColumn(
                [
                    self.HPS0,
                    self.HPS1,
                ]
            ),
            self.dc1,
            OptColumn(
                [
                    self.wg3,
                    OptRow([
                        self.wg4,
                        self.wg5
                    ])
                ]
            ),
            OptColumn(
                [
                    self.HPS2,
                    self.HPS3,
                ]
            ),
            self.dc2
        ]
        
        super().__init__(
            network,
            # ports=ports, 
            name=name
        )

    @property
    def rc_filter_bool(self):
        return self._rc_filter_bool
        
    @rc_filter_bool.setter
    def rc_filter_bool(self,new_rc_filter_bool):
        self._rc_filter_bool = new_rc_filter_bool
        self.HPS0.rc_filter_bool = self.rc_filter_bool
        self.HPS1.rc_filter_bool = self.rc_filter_bool
        self.HPS2.rc_filter_bool = self.rc_filter_bool
        self.HPS3.rc_filter_bool = self.rc_filter_bool
    
    def get_bar_port_transmission(self):
       return np.real(self.smatrix * np.conj(self.smatrix))[2,0]
    
    def get_cross_port_transmission(self):
       return np.real(self.smatrix * np.conj(self.smatrix))[3,0]
