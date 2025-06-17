import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from lmphoton import OptElement, OptColumn, OptRow, OptNetwork
from lmphoton.transforms import reflect
from lmphoton.models import Laser, Detector, BeamSplitter, DirectionalCoupler, Absorber, Waveguide, LossElement
from lmphoton.simulation import current_simulation as sim 

from xps_phase_shifter import XPS_PhaseShifter
from tap import Tap

from scipy.optimize import minimize
from scipy.signal import find_peaks
import scipy.constants as const
from functools import partial 

import warnings
warnings.filterwarnings('ignore')

um = 1e-6

class Interleaver(OptRow):
  def __init__(
      self,
      L = 350*um, 
      dL1 = 385.27*um, 
      dL2 = 385.27*2*um, 
      dc1 = 0.5, 
      dc2 = 0.63,
      dc3 = 0.9,
      tap_ratio = 0.01,
      tap_length = 30*um,
      name = 'Interleaver_2Stage'
      ):
    
    self.FSR_wavelength = 1600e9 * 1310e-9**2/const.c

    self.effective_index = 1.587
    self.group_index = 1.94
    self.sin_propagation_loss_dB_m = 40
    self.xps_insertion_loss_dB = 0.01
    self.sin_coupler_loss_dB = 0.02
    
    self.L = L
    self.channel_spacing = self.FSR_wavelength / 8.0

    self.wg0 = Waveguide(length=L+dL1,index=self.effective_index,group_index=self.group_index, loss_rate = 0.01*self.sin_propagation_loss_dB_m)
    self.wg1 = Waveguide(length=L,    index=self.effective_index,group_index=self.group_index, loss_rate = 0.01*self.sin_propagation_loss_dB_m)

    self.XPS0 = XPS_PhaseShifter()
    self.XPS1 = XPS_PhaseShifter()

    self.wg2 = Waveguide(length=L+dL2,index=self.effective_index,group_index=self.group_index, loss_rate = 0.01*self.sin_propagation_loss_dB_m)
    self.wg3 = Waveguide(length=L,    index=self.effective_index,group_index=self.group_index, loss_rate = 0.01*self.sin_propagation_loss_dB_m)

    self.XPS2 = XPS_PhaseShifter()
    self.XPS3 = XPS_PhaseShifter()

    self.dc0 = DirectionalCoupler(splitratio=dc1, loss = self.sin_coupler_loss_dB)
    self.dc1 = DirectionalCoupler(splitratio=dc2, loss = self.sin_coupler_loss_dB)
    self.dc2 = DirectionalCoupler(splitratio=dc3, loss = self.sin_coupler_loss_dB)

    self.dc_tap0 = Tap(tap_ratio=tap_ratio, name='TAP0')
    self.dc_tap1 = Tap(tap_ratio=tap_ratio, name='TAP1')
    self.dc_tap2 = Tap(tap_ratio=tap_ratio, name='TAP2')
    
    self.wg_tap0 = Waveguide(length=tap_length,index=self.effective_index,group_index=self.group_index, loss_rate = 0.01*self.sin_propagation_loss_dB_m, name='TAP0_WG')
    self.wg_tap1 = Waveguide(length=tap_length,index=self.effective_index,group_index=self.group_index, loss_rate = 0.01*self.sin_propagation_loss_dB_m, name='TAP1_WG')
    self.wg_tap2 = Waveguide(length=tap_length,index=self.effective_index,group_index=self.group_index, loss_rate = 0.01*self.sin_propagation_loss_dB_m, name='TAP2_WG')
    self.wg_tap3 = Waveguide(length=tap_length,index=self.effective_index,group_index=self.group_index, loss_rate = 0.01*self.sin_propagation_loss_dB_m, name='TAP3_WG')
    self.wg_tap4 = Waveguide(length=tap_length,index=self.effective_index,group_index=self.group_index, loss_rate = 0.01*self.sin_propagation_loss_dB_m, name='TAP4_WG')

    ########## Time domain parameters ###############
    self.XPS0.heater_voltage = 0.0
    self.XPS1.heater_voltage = 0.0
    self.XPS2.heater_voltage = 0.0
    self.XPS3.heater_voltage = 0.0

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
              self.dc_tap0,
              self.wg_tap0
            ]
        ),
        OptColumn(
            [
                self.wg_tap1,
                self.wg2,
                self.wg3,
            ]
        ),
        OptColumn(
            [
                self.wg_tap2,
                self.XPS2,
                self.XPS3,
            ]
        ),
        OptColumn( 
            [
                self.wg_tap3,
                self.dc2,
            ]
        ),
        OptColumn([
           self.wg_tap4,
           self.dc_tap1,
           self.dc_tap2,
        ])
    ]
    
    super().__init__(
        network,
        name=name
    )

    @property
    def rc_filter_bool(self):
        return self._rc_filter_bool
    
    @rc_filter_bool.setter
    def rc_filter_bool(self,new_rc_filter_bool):
        self._rc_filter_bool = new_rc_filter_bool
        self.XPS0.rc_filter_bool = self.rc_filter_bool
        self.XPS1.rc_filter_bool = self.rc_filter_bool
        self.XPS2.rc_filter_bool = self.rc_filter_bool
        self.XPS3.rc_filter_bool = self.rc_filter_bool     