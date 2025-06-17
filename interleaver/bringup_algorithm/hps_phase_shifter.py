from typing import Callable, List, Tuple, Union
import numpy as np
from scipy.optimize import fmin

from lmphoton import OptElement
from lmphoton.helpers import wrap2
from lmphoton.constants import dndT_Si
from lmphoton.helpers import dbloss2alpha, db2mag

from lmphoton.simulation import current_simulation as sim

from phase_shifter import PhaseShifter

import matplotlib.pyplot as plt

um = 1e-6


class HPS_PhaseShifter(OptElement):
    """
    Model of HPS Phase Shifter.
    Args:
        ports (optional): S-matrix ports mapping.
        name (optional): String name of the element.
        voltage (optional): Voltage (V).
    """

    def __init__(
        self,
        ports: Tuple[str, str] = ('1l', '2r'),
        name: str = 'HPS',
        length: float = 300 * um,
        index: Union[float, Callable] = 2.4, 
        group_index: Union[float, Callable] = 4.385,
        loss: Union[float, Callable] = 0.08336, # Taken from  
    ):
        #Physical constants
        self._length = length
        self._index = index
        self.ng = group_index
        self.dn_dlambda = (self._index - self.ng) / (1.31e-6)
        self._alpha = db2mag(self._genvar(loss))
        self._time_ref = sim().time
        self.rc_filter_bool = False

        #Thermal properties
        self._heater_voltage = 0.0
        self.thermal_time_constant1 = 6.15e-6
        self.thermal_time_constant2 = 56e-6
        self.thermal_weight1 = 0.67
        self.thermal_weight2 = 0.33
        self.heater_coeff_A = -2.19042885e-5
        self.heater_coeff_B = -2.12241092e-5
        self.heater_coeff_C = 1.12634676e-3
        self.thermal_resistance = 5012 #Degrees C per Watt
        self._waveguide_temperature = 0
        super().__init__(ports=ports, name=name)

    @property
    def length(self):
        """Waveguide length (m)."""
        return self._length

    @property
    def heater_voltage(self):
        return self._heater_voltage

    @heater_voltage.setter
    def heater_voltage(self, new_voltage):
        self._heater_voltage = new_voltage
        self._time_ref = sim().time
        self.update()

    @property
    def heater_power(self):
        heater_current = self.heater_coeff_A * np.power(self._heater_voltage,3) + self.heater_coeff_B * np.power(self._heater_voltage,2) + self.heater_coeff_C * self._heater_voltage
        return heater_current * self._heater_voltage

    @property
    def waveguide_temperature(self):
        new_temp = self.thermal_resistance * self.heater_power + self._temp
        if self.rc_filter_bool == True:
            t0 = self._time_ref
            t1 = sim().time
            self._waveguide_temperature = self._waveguide_temperature + (
                new_temp - self._waveguide_temperature) * (
                    1 - self.thermal_weight1 * np.exp(-(t1 - t0) / self.thermal_time_constant1) -
                    self.thermal_weight2 * np.exp(-(t1 - t0) / self.thermal_time_constant2))
            self._time_ref = sim().time
        elif self.rc_filter_bool == False:
            self._waveguide_temperature = new_temp
        return self._waveguide_temperature

    @property
    def index(self):
        """Guided mode index."""
        return self._index + dndT_Si * (self.waveguide_temperature -
                                        300.0) + (self._wavelength - 1.31e-6) * self.dn_dlambda
        # return self._index + dndT_Si*(self.waveguide_temperature-300.0)

    @property
    def wavenumber(self):
        """Guided mode wavenumber."""
        beta = self.index * 2 * np.pi / self._wavelength
        return beta - 1j * self._alpha

    @property
    def complex_phase(self):
        return self.wavenumber * self.length

    def _construct_smatrix(self):
        s21 = np.exp(-1j * self.complex_phase)
        return [[0.0, s21], [s21, 0.0]]

    def check_children_timesteps(self):
        if abs(self._time_ref - sim().time) < sim().time_step:
            return True
        else:
            print(
                f"Component {self.name} has time reference {self._time_ref} but simulation time is different at {sim().time}"
            )
