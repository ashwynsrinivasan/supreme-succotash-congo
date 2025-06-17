from typing import Callable, List, Tuple
import numpy as np
from scipy.optimize import fmin

from lmphoton import OptElement
from lmphoton.helpers import wrap2
from lmphoton.helpers import dbloss2alpha

import matplotlib.pyplot as plt

class PhaseShifter(OptElement):
    """Model of an abstract optical phase shifter.
    Args:
        ports (optional): S-matrix ports mapping.
        name (optional): String name of the element.
        voltage (optional): Voltage (V).
    """
    def __init__(self,
                 ports: Tuple[str, str] = ('1l', '2r'),
                 name: str = 'PS'):
        self._voltage = 0.0
        super().__init__(ports=ports, name=name)

    def set_voltage(self,new_voltage):
        self._voltage = new_voltage
        self._smatrix_ready = False
    
    def _construct_smatrix(self):
        #Linear mapping from voltage to phase shift
        self.phase_shift = 2*np.pi*self._voltage
        #self.phase_shift is the phase shift in radians
        s21 = np.exp(-1j*self.phase_shift)
        print(s21)
        return [[0.0, s21],
                [s21, 0.0]]

