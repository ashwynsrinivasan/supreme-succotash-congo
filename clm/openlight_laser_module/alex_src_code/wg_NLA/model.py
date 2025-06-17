"""
Copyright Lightmatter, Inc. 2020
Author: Mykhailo Tymchenko
Email: mykhailo@lightmatter.co

"""
from typing import Callable, Union
import numpy as np

from lmphoton import OptElement
from lmphoton.constants import dndT_Si
from lmphoton.helpers import dbloss2alpha

class Waveguide_NLA(OptElement):
    """Model of a simple optical waveguide.

    Args:
        length: Waveguide length (m).
        index (optional): Optical refractive index.
        loss_rate (optional): Waveguide loss rate (dB/cm).
        init_phase (optional): Initial phase (rad).
        ports (optional): S-matrix ports mapping.
        name (optional): String name of the element.

    """
    def __init__(self,
                 length: float,
                 index: Union[float, Callable] = 2.45,
                 loss_rate: Union[float, Callable] = 1.0,
                 init_phase: Union[float, Callable] = 0.0,
                 ports: Union[str, str] = ['1l', '2r'],
                 name: str = 'WG'):

        assert length > 0, 'Cannot create waveguide with 0 length.'

        self._length = length
        self._index = self._genvar(index)
        self._init_phase = self._genvar(init_phase)
        self._alpha = dbloss2alpha(self._genvar(loss_rate)+1e-9)
        self.dn_dlambda = (2.45 - 4.385)/(1.31e-6)

        super().__init__(
            ports=ports,
            name=name)

    @property
    def length(self):
        """Waveguide length (m)."""

        return self._length

    @property
    def index(self):
        """Guided mode index."""
        return self._index + dndT_Si*(self.temp-300.0) + (self.wavelength - 1.31e-6) * self.dn_dlambda
        # return self._index + dndT_Si*(self.temp-300.0)

    @property
    def wavenumber(self):
        """Guided mode wavenumber."""

        beta = self.index*2*np.pi/self.wavelength
        return beta - 1j*self._alpha

    @property
    def complex_phase(self):
        return self.wavenumber*self.length + self._init_phase

    @property
    def phase(self):
        return np.real(self.complex_phase)

    def _construct_smatrix(self):
        # print(self.ports[0].voltage)
        # print(self.ports[1].voltage)
        input_power = np.real(self.ports[0].voltage * np.conj(self.ports[0].voltage))
        if input_power == 0:
            input_power = 1e-9
        P_temp = input_power
        kappa1 = 17.6
        kappa2 = 4172
        kappa3 = 1.11e6
        step_size = 1e-7
        num_steps = int(self.length//step_size)
        for i in range(num_steps):
            P_temp -= kappa1*P_temp*step_size + kappa2 * P_temp**2 * step_size + kappa3 * P_temp**3 * step_size
        power_loss = P_temp/input_power
        field_loss = np.sqrt(power_loss)

        beta = self.index*2*np.pi/self.wavelength
        alpha = kappa1 + kappa2 * input_power + kappa3 * input_power**2
        # wavenumber = beta - 1j*alpha
        wavenumber = beta

        complex_phase = wavenumber*self.length + self._init_phase

        s21 = np.exp(-1j*complex_phase) * field_loss
        return [[0.0, s21],
                [s21, 0.0]]