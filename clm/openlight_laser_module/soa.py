from typing import List, Union, Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys, os
import pathlib

from scipy.optimize import newton

from lmphoton import OptColumn, OptElement, OptRow, OptNetwork
from lmphoton.transforms import reflect 
from lmphoton.models import DirectionalCoupler, Waveguide, LossElement, BeamSplitter, Absorber
from lmphoton.simulation import current_simulation as sim
from lmphoton.helpers import db2mag 

PATH = pathlib.Path(__file__).parent.absolute()
from functools import partial

import warnings
warnings.filterwarnings("ignore")

um = 1e-6

class SOA(OptElement):
  """
    Model of a simple semiconductor optical amplifier.
    Args:
        width: SOA width (m).
        length: SOA length (m).
        current_density: SOA current density (kA/cm^2).
        temperature: SOA temperature (C).
        input_power: Input power (dBm).
        init_phase (optional): Initial phase (rad).
        ports (optional): S-matrix ports mapping.
        name (optional): String name of the element.
  """
  def __init__(self,
               width: Union[float, Callable] = 2.7e-6,
               length: Union[float, Callable] = 0.44e-3,
               current_density: Union[float, Callable] = 3.0,
               temperature: Union[float, Callable] = 70.0,
               input_power: Union[float, Callable] = 5.0,
               init_phase: Union[float, Callable] = 0.0,
               index: Union[float, Callable] = 3.5,
               group_index: Union[float, Callable] = 4.2,
               ports: Union[str, str] = ['1l', '2r'],
               name: str = 'SOA'
               ):
    
    self.name = name
    self._index = self._genvar(index)
    self._group_index = self._genvar(group_index)
    self._length = self._genvar(length)
    self._width = self._genvar(width)
    self._current_density = self._genvar(current_density)
    self._temperature = self._genvar(temperature)
    self._init_phase = self._genvar(init_phase)
    self._input_power = self._genvar(input_power)

    super().__init__(
        ports=ports,
        name=name
    )

  def _check_inputs(self):
    T_ranges = [35, 80]  # Temperature ranges in C
    J_ranges = [3, 7]  # Current ranges in kA/cm^2
    L_ranges = [40, 440]  # Gain length ranges in µm
    wav_ranges = [1304 * um, 1318 * um]  # wavelength ranges in nm
    if self._temperature:
      assert np.all(self._temperature >= min(T_ranges)) and np.all(self._temperature <= max(T_ranges))
    if self.J:
      assert np.all(self._current_density >= min(J_ranges)) and np.all(self._current_density <= max(J_ranges))
    if self.L:
      assert np.all(self._length >= min(L_ranges)) and np.all(self._length <= max(L_ranges))
    if self.wl:
      assert np.all(self.wavelength >= min(wav_ranges)) and np.all(self.wavelength <= max(wav_ranges))
    pass
     
  def _gain_peak(self):
    T = self._temperature
    J = self._current_density
    L = self._length * 1e6

    output = 4.678 -0.0729* T + 10.098* np.log(J)- 0.001380 *(L+460) \
    -0.00024 *(T - 60) *(T - 60) - 0.0081*np.log(J) *(T - 60) - 2.158* np.log(J)* np.log(J) \
    -0.0001589 *(T - 60) *(L - 240) + 0.02311 *np.log(J) *(L - 240) \
    -0.000001886* (T -60)* (T-60) *(L - 240) \
    -0.00002088 *np.log(J)* (T- 60)* (L - 240) \
    -0.005336* np.log(J) *np.log(J) *(L - 240)

    self._gain_peak_dB = output

  def _wavelength_peak(self):
    T = self._temperature
    J = self._current_density
    L = self._length * 1e6

    output = 1273.73 + 0.6817* T - 28.73* np.log(J)+ 0.01362 *(L + 460) \
        + 0.004585 *(T - 60)* (T - 60) - 0.1076 *np.log(J)* (T - 60) + 8.787* np.log(J)* np.log(J) \
        + 0.00004185 *(T - 60)* (L - 240) - 0.02367* np.log(J) *(L - 240) \
        - 0.0000002230 *(T - 60)* (T - 60)* (L - 240) \
        + 0.000136* np.log(J)* (T - 60) *(L - 240) + 0.004894 *np.log(J)* np.log(J) *(L - 240)
    self._wavelength_peak = output * 1e-9
  
  def _FWHM(self):
    T = self._temperature
    J = self._current_density
    L = self._length *1e6

    output = 120.15 - 0.08555* T + 0.3837* np.log(J) - 0.07255 *(L + 460) \
    + 0.00007784 *(T - 60) *(T - 60) + 0.2386 *np.log(J) *(T - 60) + 2.759 *np.log(J)* np.log(J) \
    - 0.0004342* (T - 60)* (L - 240) + 0.003947* np.log(J)* (L - 240) \
    +0.00002085*(T - 60) *(T - 60) *(L - 240) \
    +0.000009466 *np.log(J) *(T - 60) *(L - 240) \
    -0.0007991*np.log(J) *np.log(J)* (L - 240)

    self._FWHM = output * 1e-9

  def _Pos_3dB(self):
    # Output power saturation (Pos) in dBm
    # outputs 3dB saturated power in dBm

    wav = self.wavelength * 1e9
    T = self._temperature
    J = self._current_density

    output = -74.08+ 0.06226*wav - 0.008877*T + 0.994*J + \
    -0.08721*(J - 4.571)* (J - 4.571) + 0.01752*(wav - 1310.8)* (wav - 1310.8) \
    -0.00002341*(T - 60.07)*(T - 60.07) - 0.001266*(wav - 1310.8)*(T- 60.07) \
    -0.001763* (T - 60.07)*(J- 4.571) - 0.008584*(wav - 1310.8)*(J- 4.571)

    self._Pos_3dB = output

  def Lorentzian(self,x, x0, fwhm):
      denom = (x - x0)**2 + (fwhm/2) **2
      return fwhm / denom # Outputs are in units of 1/nm

  def _g0(self):
      self._wavelength_peak()
      self._FWHM()
      self._gain_peak()
      f = self.Lorentzian(self.wavelength * 1e9, self._wavelength_peak * 1e9, self._FWHM * 1e9)
      self._g0 = f * 10**(self._gain_peak_dB/10) / (4 / self._FWHM / 1e9) # Linear, unitless output
  
  def _noise_figure(self):
    # outputs noise figure in dB
    wav = self.wavelength * 1e9
    T = self.T
    J = self.J
    L = self.L * 1e6

    output = 131.58 + -0.09959*wav + 0.08972*T -5.0895*np.log(J) \
    + 2.7334*np.log(J)*np.log(J) + 0.0009195 *(wav - 1306.38)* (wav - 1306.38) \
    + 0.0007484*(T - 60)*(T - 60) -0.001299*(wav - 1306.38)*(T - 60) \
    - 0.07995*(T - 60)*np.log(J) + 0.103*(wav-1306.38)*np.log(J) \
    + 0.0005740*(wav - 1306.38)*(T - 60)*np.log(J) \
    + 0.0197*np.log(J)*np.log(J)*(T - 60) - 0.02785*np.log(J)*np.log(J)*(wav - 1306.38) \
    -0.0003141*(T - 60)*(T - 60)*np.log(J) -0.00001095*(T - 60)*(T - 60)*(wav - 1306.38) \
    -0.0002678*(wav - 1306.38)*(wav-1306.38)*np.log(J) \
    + 0.000003281*(wav-1306.38)*(wav - 1306.38)*(T - 60) \
    -0.4606*np.log(J)*np.log(J)*np.log(J) - 0.000002634*(wav - 1306.38)*(wav - 1306.38)*(wav - 1306.38)

    self._NF = output

  def _Psat(self): 
    self._Pos_3dB()
    Ps_3dB_ = 10**(self._Pos_3dB/10)  # in mW
    g0_ = self._g0
    self.Psat = Ps_3dB_ * (g0_-2) / (g0_*np.log(2)) 
    self.Psat = self.Psat * 1e-3 #Psat in W

  def _gain(self):
    self._g0()
    self._Psat()
    Pin = 10**(self._input_power/10) * 1e-3

    def f(g):
        return g - self._g0 * np.exp( (1-g) * Pin/self.Psat)

    def fprime(g):
        z = Pin/self.Psat
        return 1 + self._g0 * z * np.exp(z*(1-g))

    self._gain = newton(f, self._g0, fprime=fprime, maxiter=10000)
    self._output_power = 10*np.log10(self._gain) + 10*np.log10(Pin/1e-3)

    self._alpha = - self._gain / self._length / 1e2
  
  def _vdiode(self):
    J = self._current_density
    L_SOA = self._length * 1e6
    W = self._width * 1e6
    self._i_diode = J *1e3*  W * (L_SOA + 460) * 1e-8 # [A]
    self._v_diode = 0.98 + 2.8*self._i_diode * (1100/(L_SOA+460))

  @property
  def length(self):
    """Waveguide length (m)."""
    return self._length

  @property
  def index(self):
    """Guided mode index."""
    return self._index

  @property
  def wavenumber(self):
    """Guided mode wavenumber."""
    self._gain()
    beta = self.index*2*np.pi/self.wavelength
    return beta - 1j*self._alpha

  @property
  def complex_phase(self):
    return self.wavenumber*self.length + self._init_phase

  @property
  def phase(self):
    return np.real(self.complex_phase)

  def _construct_smatrix(self):
    s21 = np.exp(-1j*self.complex_phase)
    return [[0.0, s21],
            [s21, 0.0]]