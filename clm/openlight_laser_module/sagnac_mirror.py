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

class SagnacRing(OptNetwork):
  def __init__(
      self,
      name = "sagnac_ring",
      loss = 0.33,
      splitratio = 0.5,
      radius = 40.0,
      waveguide_loss_rate = 0.33,
      index = 1,
  ):
    self.name = name
    self.loss = loss
    self.waveguide_length = 2 * np.pi * radius * um

    self.index = index
    
    self.dc = DirectionalCoupler(name=f"dc_sr_{self.index}",splitratio=splitratio, loss = self.loss)
    self.wg = Waveguide(name=f"wg_sr_{self.index}",length=self.waveguide_length, loss_rate=waveguide_loss_rate)

    children = {
      f"dc_sr_{self.index}": self.dc,
      f"wg_sr_{self.index}": self.wg,
    }

    links = [
      (f'dc_sr_{self.index}:PORT2', f'wg_sr_{self.index}:PORT1'),
      (f'wg_sr_{self.index}:PORT2', f'dc_sr_{self.index}:PORT3'),
    
    ]

    super().__init__(
      name=name,
      children=children, 
      links=links
      )
  
  def get_effective_length(self):
    return self.waveguide_length