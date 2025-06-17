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
from lmphoton.helpers import db2mag 

PATH = pathlib.Path(__file__).parent.absolute()
from functools import partial

import warnings
warnings.filterwarnings("ignore")

um = 1e-6

class Ring(OptNetwork):
  def __init__(
      self,
      name = "ring",
      splitratio1 = 0.2,
      splitratio2 = 0.2,
      radius = 25,
      waveguide_loss_rate = 0.33,
      index = 1,
  ):
    self.name = name
    self.index = index
    self.waveguide_loss_rate = waveguide_loss_rate
    self.L_circum = 2 * np.pi * radius * um

    self.kappa1 = np.sqrt(splitratio1)
    self.kappa2 = np.sqrt(splitratio2)

    self.dc_1 = DirectionalCoupler(name=f"dc_1_index_{self.index}",splitratio=self.kappa1**2)
    self.dc_2 = DirectionalCoupler(name=f"dc_2_index_{self.index}",splitratio=self.kappa2**2)
    self.wg1 = Waveguide(name=f"wg_1_index_{self.index}",length=self.L_circum/2, loss_rate=self.waveguide_loss_rate)
    self.wg2 = Waveguide(name=f"wg_2_index_{self.index}",length=self.L_circum/2, loss_rate=self.waveguide_loss_rate)

    self.absorber1 = Absorber(name=f'absorber_1_index_{self.index}')
    self.absorber2 = Absorber(name=f'absorber_2_index_{self.index}')

    children = {
      f"dc_1_index_{self.index}": self.dc_1,
      f"dc_2_index_{self.index}": self.dc_2,
      f"wg_1_index_{self.index}": self.wg1,
      f"wg_2_index_{self.index}": self.wg2,
      f"absorber_1_index_{self.index}": self.absorber1,
      f"absorber_2_index_{self.index}": self.absorber2,
    }

    links = [
      (f'dc_1_index_{self.index}:PORT3', f'wg_1_index_{self.index}:PORT1'),
      (f'wg_1_index_{self.index}:PORT2', f'dc_2_index_{self.index}:PORT2'),
      (f'dc_2_index_{self.index}:PORT1', f'wg_2_index_{self.index}:PORT1'),
      (f'wg_2_index_{self.index}:PORT2', f'dc_1_index_{self.index}:PORT4'),
      (f'dc_1_index_{self.index}:PORT2', f'absorber_1_index_{self.index}:PORT1'),
      (f'dc_2_index_{self.index}:PORT3', f'absorber_2_index_{self.index}:PORT1'),
    ]

    super().__init__(
      name=name,
      children=children, 
      links=links
      )
    
  def get_effective_length(self):
    return self.L_circum * (1 - self.kappa1**2) / (self.kappa2**2)
