from typing import List, Union, Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys, os
import pathlib

from lmphoton import OptColumn, OptElement, OptRow, OptNetwork
from lmphoton.transforms import reflect 
from lmphoton.models import DirectionalCoupler, Waveguide, LossElement, BeamSplitter, Absorber
from lmphoton.simulation import current_simulation as sim
from lmphoton.helpers import db2mag, dbloss2alpha
from lmphoton.constants import dndT_Si

PATH = pathlib.Path(__file__).parent.absolute()
from functools import partial

import warnings
warnings.filterwarnings("ignore")

um = 1e-6

class ph_ht_tr(OptElement):
  def __init__(
      self,
    index: Union[float, Callable] = 3.2,
    group_index: Union[float, Callable] = 3.93,
    waveguide_loss_rate: Union[float, Callable] = 33,
    length: float = 100.0e-6,
    init_phase: Union[float, Callable] = 0.0,
    resistance: Union[float, Callable] = 57.0,
    dmW: float = 0.0,
    dpi_dmW: float = 1.0/43.63,
    ports: Tuple[str, str] = ('1l', '2r'),
    name: str = 'phase_shifter',
  ):

    self._length = self._genvar(length)
    self._index = self._genvar(index)
    self._group_index = self._genvar(group_index)
    self._init_phase = self._genvar(init_phase)
    self._loss_rate = self._genvar(waveguide_loss_rate)
    self._loss_dB = self._loss_rate * self._length
    self._alpha = dbloss2alpha(self._loss_dB + 1e-9)
    self._resistance = self._genvar(resistance)
    self.dpi_dmW = dpi_dmW
    self.dmW = dmW
    self.dn_dlambda = (self._index - self._group_index)/(1.31e-6)

    super().__init__(
      ports=ports,
      name=name
    )


  def _iv_pht_ht_tr(self):
    self._i_ph_ht_tr = np.sqrt(self.dmW*1e-3/self._resistance)
    self._v_ph_ht_tr = self._i_ph_ht_tr * self._resistance

  @property
  def length(self):
      """Waveguide length (m)."""
      return self._length

  @property
  def index(self):
      """Guided mode index."""
      return self._index + dndT_Si*(self.temp-300.0) + (self.wavelength - 1.31e-6)*self.dn_dlambda 

  @property
  def wavenumber(self):
      """Guided mode wavenumber."""
      beta = self.index*2*np.pi/self.wavelength
      return beta - 1j*self._alpha

  @property
  def complex_phase(self):
      return self.wavenumber*self.length + self._init_phase + self.dpi_dmW * self.dmW * np.pi

  @property
  def phase(self):
      return np.real(self.complex_phase)

  def _construct_smatrix(self):
      s21 = np.exp(-1j*self.complex_phase)
      return [[0.0, s21],
              [s21, 0.0]]