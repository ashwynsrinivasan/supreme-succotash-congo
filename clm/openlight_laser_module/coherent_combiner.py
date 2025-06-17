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

from soa import SOA
from pcm.congo.vernier_laser.vernier_ring import PhaseShifter

PATH = pathlib.Path(__file__).parent.absolute()
from functools import partial

import warnings
warnings.filterwarnings("ignore")

um = 1e-6

class CoherentCombiner(OptNetwork):
  def __init__(
      self,
      name = "coherent_combiner",

  ):
    self.name = name
    self.cc0_splitter0 = BeamSplitter(name=f"cc0_splitter0")

    self.cc0_htr0 = PhaseShifter(name=f"cc0_htr0")
    self.cc0_htr1 = PhaseShifter(name=f"cc0_htr1")

    self.cc1_splitter0 = BeamSplitter(name=f"cc1_splitter0")
    self.cc1_htr0 = PhaseShifter(name=f"cc1_htr0")
    self.cc1_htr1 = PhaseShifter(name=f"cc1_htr1")
    self.cc1_soa0 = LossElement(name=f"cc1_soa0")
    self.cc1_soa1 = LossElement(name=f"cc1_soa1")

    self.cc2_splitter0 = BeamSplitter(name=f"cc2_splitter0")
    self.cc2_htr0 = PhaseShifter(name=f"cc2_htr0")
    self.cc2_htr1 = PhaseShifter(name=f"cc2_htr1")
    self.cc2_soa0 = LossElement(name=f"cc2_soa0")
    self.cc2_soa1 = LossElement(name=f"cc2_soa1")

    self.cc0_splitter1 = BeamSplitter(name=f"cc0_splitter1")
    self.cc1_splitter1 = BeamSplitter(name=f"cc2_splitter1")
    self.cc2_splitter1 = BeamSplitter(name=f"cc2_splitter1")

    children = {
      f"cc0_splitter0": self.cc0_splitter0,
      f"cc0_htr0": self.cc0_htr0,
      f"cc0_htr1": self.cc0_htr1,
      f"cc1_splitter0": self.cc1_splitter0,
      f"cc1_htr0": self.cc1_htr0,
      f"cc1_htr1": self.cc1_htr1,
      f"cc1_soa0": self.cc1_soa0,
      f"cc1_soa1": self.cc1_soa1,
      f"cc2_splitter0": self.cc2_splitter0,
      f"cc2_htr0": self.cc2_htr0,
      f"cc2_htr1": self.cc2_htr1,
      f"cc2_soa0": self.cc2_soa0,
      f"cc2_soa1": self.cc2_soa1,
      f"cc0_splitter1": self.cc0_splitter1,
      f"cc1_splitter1": self.cc1_splitter1,
      f"cc2_splitter1": self.cc2_splitter1,
    }

    links = [
      (f"cc0_splitter0:PORT2", f"cc0_htr0:PORT1"),
      (f"cc0_splitter0:PORT3", f"cc0_htr1:PORT1"),
      (f"cc0_htr0:PORT2", f"cc1_splitter0:PORT1"),
      (f"cc0_htr1:PORT2", f"cc2_splitter0:PORT1"),

      (f"cc1_splitter0:PORT2", f"cc1_htr0:PORT1"),
      (f"cc1_splitter0:PORT3", f"cc1_htr1:PORT1"),
      (f"cc1_htr0:PORT2", f"cc1_soa0:PORT1"),
      (f"cc1_htr1:PORT2", f"cc1_soa1:PORT1"),
      (f"cc1_splitter1:PORT4", f"cc1_soa0:PORT2"),
      (f"cc1_splitter1:PORT1", f"cc1_soa1:PORT2"),

      (f"cc2_splitter0:PORT2", f"cc2_htr0:PORT1"),
      (f"cc2_splitter0:PORT3", f"cc2_htr1:PORT1"),
      (f"cc2_htr0:PORT2", f"cc2_soa0:PORT1"),
      (f"cc2_htr1:PORT2", f"cc2_soa1:PORT2"),
      (f"cc2_splitter1:PORT4", f"cc2_soa0:PORT2"),
      (f"cc2_splitter1:PORT1", f"cc2_soa1:PORT2"),

      (f"cc1_splitter1:PORT2", f"cc0_splitter1:PORT4"),
      (f"cc2_splitter1:PORT2", f"cc0_splitter1:PORT1"),
    ]

    super().__init__(
      name=name,
      children=children, 
      links=links
      )