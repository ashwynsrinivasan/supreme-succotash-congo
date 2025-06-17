from typing import List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import sys, os 
import pathlib
from typing import Callable, List, Tuple, Union

from lmphoton import OptColumn, OptElement, OptRow, OptNetwork 
from lmphoton.transforms import reflect
from lmphoton.models import Laser, Detector, BeamSplitter, DirectionalCoupler, Absorber, Waveguide, Crossing, LossElement
from lmphoton.simulation import current_simulation as sim 
from lmphoton.helpers import db2mag

PATH = pathlib.Path(__file__).parent.absolute()

from functools import partial 

import warnings
warnings.filterwarnings("ignore")

um = 1e-6

class Xing(OptElement):
    """Model of a waveguide Xing.

    Args:
        crosstalk (optional): Crosstalk (dB).
        loss (optional): Insertion loss (dB).
        ports (optional): S-matrix ports mapping.
        name (optional): String name of the element.

    """
    color = [90, 90, 90]
    size = [100, 100]
    icon_path = os.path.join(PATH, 'icon.svg')

    def __init__(self,
                 crosstalk: Union[float, Callable] = -100.0,
                 loss: Union[float, Callable] = 0.0,
                 ports: Tuple[str, str, str, str] = ('1l', '2r', '3r', '4l'),
                 name: str = 'XING'):

        self._crosstalk = self._genvar(crosstalk)
        self._loss = self._genvar(loss)+1e-9

        super().__init__(
            ports=ports,
            name=name)

    def _construct_smatrix(self):
        crt = db2mag(self._crosstalk)
        los = db2mag(-self._loss)
        return [[0.0, crt, los, crt],
                [crt, 0.0, crt, los],
                [los, crt, 0.0, crt],
                [crt, los, crt, 0.0]]

class PassiveSplitterCombinerTree(OptRow):
  def __init__(
      self,
      SiN_directional_coupler_splitratio_list,
      SiN_directional_coupler_loss_dB_list,
      SiN_crossing_loss_dB_list,
      SiN_crossing_crosstalk_dB_list,
      L = 50*um,
      SiN_propagation_loss_dB_m = 40,
      effective_index = 1.67,
      group_index = 1.95,
      ports = ['1l', '2r','3r','4r','5r','6r','7r','8r','9r','10l', '11l', '12l','13l','14l','15l','16l'],
      name = 'PassiveSplitterCombinerTree'
  ):
    
    self.SiN_directional_coupler_splitratio_list = SiN_directional_coupler_splitratio_list
    self.SiN_crossing_loss_dB_list = SiN_crossing_loss_dB_list
    self.SiN_crossing_crosstalk_dB_list = SiN_crossing_crosstalk_dB_list
    self.SiN_directional_coupler_loss_dB_list = SiN_directional_coupler_loss_dB_list
    self.SiN_propagation_loss_dB_m = SiN_propagation_loss_dB_m
    self.effective_index = effective_index
    self.group_index = group_index
    self.L = L

    self.dc1 = DirectionalCoupler(splitratio=self.SiN_directional_coupler_splitratio_list[0], loss = self.SiN_directional_coupler_loss_dB_list[0])
    self.dc2 = DirectionalCoupler(splitratio=self.SiN_directional_coupler_splitratio_list[1], loss = self.SiN_directional_coupler_loss_dB_list[1])
    self.dc3 = DirectionalCoupler(splitratio=self.SiN_directional_coupler_splitratio_list[2], loss = self.SiN_directional_coupler_loss_dB_list[2])
    self.dc4 = DirectionalCoupler(splitratio=self.SiN_directional_coupler_splitratio_list[3], loss = self.SiN_directional_coupler_loss_dB_list[3])
    self.dc5 = DirectionalCoupler(splitratio=self.SiN_directional_coupler_splitratio_list[4], loss = self.SiN_directional_coupler_loss_dB_list[4])
    self.dc6 = DirectionalCoupler(splitratio=self.SiN_directional_coupler_splitratio_list[5], loss = self.SiN_directional_coupler_loss_dB_list[5])
    self.dc7 = DirectionalCoupler(splitratio=self.SiN_directional_coupler_splitratio_list[6], loss = self.SiN_directional_coupler_loss_dB_list[6])
    self.dc8 = DirectionalCoupler(splitratio=self.SiN_directional_coupler_splitratio_list[7], loss = self.SiN_directional_coupler_loss_dB_list[7])
    self.dc9 = DirectionalCoupler(splitratio=self.SiN_directional_coupler_splitratio_list[8], loss = self.SiN_directional_coupler_loss_dB_list[8])
    self.dc10 = DirectionalCoupler(splitratio=self.SiN_directional_coupler_splitratio_list[9], loss = self.SiN_directional_coupler_loss_dB_list[9])
    self.dc11 = DirectionalCoupler(splitratio=self.SiN_directional_coupler_splitratio_list[10], loss = self.SiN_directional_coupler_loss_dB_list[10])
    self.dc12 = DirectionalCoupler(splitratio=self.SiN_directional_coupler_splitratio_list[11], loss = self.SiN_directional_coupler_loss_dB_list[11])

    self.crossing1 = Xing(crosstalk=self.SiN_crossing_crosstalk_dB_list[0], loss=self.SiN_crossing_loss_dB_list[0])
    self.crossing2 = Xing(crosstalk=self.SiN_crossing_crosstalk_dB_list[1], loss=self.SiN_crossing_loss_dB_list[1])
    self.crossing3 = Xing(crosstalk=self.SiN_crossing_crosstalk_dB_list[2], loss=self.SiN_crossing_loss_dB_list[2])
    self.crossing4 = Xing(crosstalk=self.SiN_crossing_crosstalk_dB_list[3], loss=self.SiN_crossing_loss_dB_list[3])
    self.crossing5 = Xing(crosstalk=self.SiN_crossing_crosstalk_dB_list[4], loss=self.SiN_crossing_loss_dB_list[4])
    self.crossing6 = Xing(crosstalk=self.SiN_crossing_crosstalk_dB_list[5], loss=self.SiN_crossing_loss_dB_list[5])
    self.crossing7 = Xing(crosstalk=self.SiN_crossing_crosstalk_dB_list[6], loss=self.SiN_crossing_loss_dB_list[6])
    self.crossing8 = Xing(crosstalk=self.SiN_crossing_crosstalk_dB_list[7], loss=self.SiN_crossing_loss_dB_list[7])

    self.wg1 = Waveguide(length=self.L, index = self.effective_index, group_index = self.group_index, loss_rate = 0.01*self.SiN_propagation_loss_dB_m)
    self.wg2 = Waveguide(length=self.L, index = self.effective_index, group_index = self.group_index, loss_rate = 0.01*self.SiN_propagation_loss_dB_m)
    self.wg3 = Waveguide(length=self.L, index = self.effective_index, group_index = self.group_index, loss_rate = 0.01*self.SiN_propagation_loss_dB_m)
    self.wg4 = Waveguide(length=self.L, index = self.effective_index, group_index = self.group_index, loss_rate = 0.01*self.SiN_propagation_loss_dB_m)
    self.wg5 = Waveguide(length=self.L, index = self.effective_index, group_index = self.group_index, loss_rate = 0.01*self.SiN_propagation_loss_dB_m)
    self.wg6 = Waveguide(length=self.L, index = self.effective_index, group_index = self.group_index, loss_rate = 0.01*self.SiN_propagation_loss_dB_m)
    self.wg7 = Waveguide(length=self.L, index = self.effective_index, group_index = self.group_index, loss_rate = 0.01*self.SiN_propagation_loss_dB_m)
    self.wg8 = Waveguide(length=self.L, index = self.effective_index, group_index = self.group_index, loss_rate = 0.01*self.SiN_propagation_loss_dB_m)
    self.wg9 = Waveguide(length=self.L, index = self.effective_index, group_index = self.group_index, loss_rate = 0.01*self.SiN_propagation_loss_dB_m)
    self.wg10 = Waveguide(length=self.L, index = self.effective_index, group_index = self.group_index, loss_rate = 0.01*self.SiN_propagation_loss_dB_m)
    self.wg11 = Waveguide(length=self.L, index = self.effective_index, group_index = self.group_index, loss_rate = 0.01*self.SiN_propagation_loss_dB_m)
    self.wg12 = Waveguide(length=self.L, index = self.effective_index, group_index = self.group_index, loss_rate = 0.01*self.SiN_propagation_loss_dB_m)
    self.wg13 = Waveguide(length=self.L, index = self.effective_index, group_index = self.group_index, loss_rate = 0.01*self.SiN_propagation_loss_dB_m)
    self.wg14 = Waveguide(length=self.L, index = self.effective_index, group_index = self.group_index, loss_rate = 0.01*self.SiN_propagation_loss_dB_m)
    self.wg15 = Waveguide(length=self.L, index = self.effective_index, group_index = self.group_index, loss_rate = 0.01*self.SiN_propagation_loss_dB_m)
    self.wg16 = Waveguide(length=self.L, index = self.effective_index, group_index = self.group_index, loss_rate = 0.01*self.SiN_propagation_loss_dB_m)

    OptColumn1 = OptColumn(
      [
        self.dc1,
        self.dc2,
        self.dc3,
        self.dc4,
      ]
    )

    OptColumn2 = OptColumn(
      [
        self.wg1,
        self.crossing1,
        self.wg2,
        self.wg3,
        self.crossing2,
        self.wg4
      ]
    )

    OptColumn3 = OptColumn(
      [
        self.dc5,
        self.dc6,
        self.dc7,
        self.dc8
      ]
    )

    OptColumn4 = OptColumn(
      [
        self.wg5,
        self.wg6,
        self.wg7,
        self.crossing3,
        self.wg8,
        self.wg9,
        self.wg10
      ]
    )

    OptColumn5 = OptColumn(
      [
        self.wg11,
        self.wg12,
        self.crossing4,
        self.crossing5,
        self.wg13,
        self.wg14,
      ]
    )

    OptColumn6 = OptColumn(
      [
        self.wg15,
        self.crossing6,
        self.crossing7,
        self.crossing8,
        self.wg16,

      ]
    )

    OptColumn7 = OptColumn(
      [
        self.dc9,
        self.dc10,
        self.dc11,
        self.dc12
      ]
    )

    network = [
      OptColumn1,
      OptColumn2,
      OptColumn3,
      OptColumn4,
      OptColumn5,
      OptColumn6,
      OptColumn7
    ]

    super().__init__(
      network,
      ports=ports,
      name=name
    )