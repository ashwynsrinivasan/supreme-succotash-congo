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


class XPS_PhaseShifter(OptElement):
    """
    Model of XPS Phase Shifter.
    Args:
        ports (optional): S-matrix ports mapping.
        name (optional): String name of the element.
        voltage (optional): Voltage (V).
    """

    def __init__(
        self,
        ports: Tuple[str, str] = ('1l', '2r'),
        name: str = 'XPS',
        length: float = 34 * um,
        index: Union[float, Callable] = 2.4, 
        group_index: Union[float, Callable] = 3.313, 
        loss: Union[float, Callable] = 0.08336, # Taken from https://github.com/lightmatter-ai/lmpdk/blob/main/gf45clo/models/spectre/xps/xps_model.va 
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
        self.thermal_time_constant1 = 17.7e-6
        self.thermal_weight1 = 1.0
        self.heater_coeff_A = 0.00188176
        self.heater_coeff_B = 0.11290256
        self.heater_coeff_C = 1.62103404
        self.thermal_resistance = 12185.1 #Degrees C per Watt
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
        heater_current = self.heater_coeff_A * self._heater_voltage/(1 + self.heater_coeff_B * np.power(self._heater_voltage,self.heater_coeff_C))
        return heater_current * self._heater_voltage

    @property
    def waveguide_temperature(self):
        new_temp = self.thermal_resistance * self.heater_power + self._temp
        if self.rc_filter_bool == True:
            t0 = self._time_ref
            t1 = sim().time
            self._waveguide_temperature = self._waveguide_temperature + (
                new_temp - self._waveguide_temperature) * (
                    1 - self.thermal_weight1 * np.exp(-(t1 - t0) / self.thermal_time_constant1)
                    )
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
