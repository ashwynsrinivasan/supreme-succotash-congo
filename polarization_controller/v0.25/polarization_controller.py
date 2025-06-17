from typing import List, Union, Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys, os
import pathlib


from lmphoton import OptColumn, OptElement, OptRow, OptNetwork
from lmphoton.transforms import reflect
from lmphoton.models import Laser, Detector, BeamSplitter, DirectionalCoupler, Absorber, Waveguide, Crossing, LossElement
from lmphoton.simulation import current_simulation as sim
from lmphoton.helpers import db2mag

from xps_phase_shifter import XPS_PhaseShifter

PATH = pathlib.Path(__file__).parent.absolute()

from functools import partial

import warnings
warnings.filterwarnings("ignore")

um = 1e-6

def rotation_rx(angle):
  rx = np.array(
      [
        [np.cos(angle/2), -1j*np.sin(angle/2)], 
        [-1j*np.sin(angle/2), np.cos(angle/2)]
        ]
      )
  return rx

def rotation_ry(angle):
  ry = np.array(
      [
        [np.cos(angle/2), -np.sin(angle/2)], 
        [np.sin(angle/2), np.cos(angle/2)]
        ]
      )
  return ry

def rotation_rz(angle):
  rz = np.array(
      [
        [np.exp(-1j*angle/2), 0], 
        [0, np.exp(1j*angle/2)]
        ]
      )
  return rz

def rotation_delta(angle):
  return np.exp(1j*angle)

def jones_matrix(_rotation):
    _jones_matrix_rx = rotation_rx(_rotation[0])
    _jones_matrix_ry = rotation_ry(_rotation[1])
    _jones_matrix_rz = rotation_rz(_rotation[2])
    _jones_matrix_global_phase = rotation_delta(_rotation[3])
    jones_matrix = _jones_matrix_global_phase * _jones_matrix_rx @ _jones_matrix_ry @ _jones_matrix_rz 
    return jones_matrix

class fiber_shuffle(OptElement):
  """
  Model of a fiber shuffle.
  Args:
      loss: Insertion loss (dB). 
      return_loss: Return loss (dB).
      ports: S-matrix ports mapping.
      name: String name of the element.
  """
  def __init__(
    self,
    loss: Union[float, Callable] = 1.0,
    return_loss: Union[float, Callable] = -100.0,
    ports: Tuple[str, str, str, str] = ('1l FS', '2r FS', '3r FS', '4l FS'),
    name: str = 'FIBER SHUFFLE'
  ):
    self._te_loss = self._genvar(loss)
    self._tm_loss = self._genvar(loss)
    self._return_loss = self._genvar(return_loss)
    super().__init__(
      ports=ports,
      name=name
    )

  def _construct_smatrix(self):
    te_loss = db2mag(-self._te_loss)
    tm_loss = db2mag(-self._tm_loss)
    return_loss = db2mag(self._return_loss)

    if te_loss + return_loss > 1:
      te_loss = 1 - return_loss
    if tm_loss + return_loss > 1:
      tm_loss = 1 - return_loss

    return [
      [return_loss, te_loss, 0.0, 0.0],
      [te_loss, return_loss, 0.0, 0.0],
      [0.0, 0.0, return_loss, tm_loss],
      [0.0, 0.0, tm_loss, return_loss]
      ]

class optical_fiber(OptElement):
  """
  Modelling the Jones matrix of an optical fiber.
  Args:
      rotation: Tuple of floats (rx, ry, rz, global_phase) representing the rotation in the Bloch and Poincare sphere.
      length: Length of the fiber (meters).
      loss: Loss of the fiber (dB).
      name: String name of the element.
  """
  def __init__(
    self,
    rotation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    length: Union[float, Callable] = 16.6,
    loss: Union[float, Callable] = 35e-4,
    ports: Tuple[str, str, str, str] = ('1l', '2r', '3r', '4l'),
    name: str = 'FIBER'
  ):

    self._rotation = self._genvar(rotation)
    self._loss = self._genvar(loss)
    self._length = self._genvar(length)
    self._jones_matrix = jones_matrix(self._rotation)
    super().__init__(
      ports=ports,
      name=name
    )

  def _construct_smatrix(self):
    loss = db2mag(-self._loss * self._length)
    self._jones_matrix = jones_matrix(self._rotation)

    s_matrix = [
      [0, loss * self._jones_matrix[0,0], loss * self._jones_matrix[1,0], 0],
      [loss * self._jones_matrix[0,0], 0, 0, loss * self._jones_matrix[0,1]],
      [loss * self._jones_matrix[1,0], 0, 0, loss * self._jones_matrix[1,1]],
      [0, loss * self._jones_matrix[0,1], loss * self._jones_matrix[1,1], 0]
    ]
    return s_matrix
  
class optical_connector(OptElement):
  """
  Modelling the Jones matrix of an optical connector.
  Args:
      loss: Loss of the connector (dB).
      name: String name of the element.
  """

  def __init__(
    self,
    loss: Union[float, Callable] = 0.5,
    return_loss: Union[float, Callable] = -20,
    ports: Tuple[str, str, str, str] = ('1l', '2r', '3r', '4l'),
    name: str = 'CONNECTOR'
  ):

    self._loss = self._genvar(loss)
    self._return_loss = self._genvar(return_loss)
    super().__init__(
      ports=ports,
      name=name
    )

  def _construct_smatrix(self):
    loss = db2mag(-self._loss)
    return_loss = db2mag(self._return_loss)

    if loss + return_loss > 1:
      loss = 1 - return_loss

    return [
      [return_loss, loss, 0.0, 0.0],
      [loss, return_loss, 0.0 , 0.0],
      [0.0, 0.0 , return_loss, loss],
      [0.0 , 0.0, loss, return_loss]
    ]

class edge_coupler(OptElement):
  """
  Model of a waveguide edge coupler.
  Args:
      loss: Insertion loss (dB). 
      return_loss: Return loss (dB).
      ports: S-matrix ports mapping.
      name: String name of the element.
  """
  def __init__(
    self,
    TE_loss: Union[float, Callable] = 2.1,
    TM_loss: Union[float, Callable] = 2.5,
    return_loss: Union[float, Callable] = -100.0,
    ports: Tuple[str, str, str, str] = ('1l', '2r', '3r', '4l'),
    name: str = 'EDGE COUPLER'
  ):
    self._te_loss = self._genvar(TE_loss)
    self._tm_loss = self._genvar(TM_loss)
    self._return_loss = self._genvar(return_loss)

    super().__init__(
      ports=ports,
      name=name
    )

  def _construct_smatrix(self):
    te_loss = db2mag(-self._te_loss)
    tm_loss = db2mag(-self._tm_loss)
    return_loss = db2mag(self._return_loss)
    if te_loss + return_loss > 1:
      te_loss = 1 - return_loss
    if tm_loss + return_loss > 1:
      tm_loss = 1 - return_loss
    
    return [
      [return_loss, te_loss, 0.0, 0.0],
      [te_loss, return_loss, 0.0, 0.0],
      [0.0, 0.0, return_loss, tm_loss],
      [0.0, 0.0, tm_loss, return_loss]
      ]

class polarization_splitter_rotator(OptElement):
  """
  Model of a waveguide PSR.
  Args:
      loss (optional): Insertion loss (dB). 
      PER (optional): Polarization extinction ratio (dB).
      ports (optional): S-matrix ports mapping.
      name (optional): String name of the element.
  """
  def __init__(
    self,
    TE_loss: Union[float, Callable] = 1.0,
    TM_loss: Union[float, Callable] = 1.0,
    TE_per: Union[float, Callable] = -100.0,
    TM_per: Union[float, Callable] = -100.0,
    TE_phase: Union[float, Callable] = 0.0,
    TM_phase: Union[float, Callable] =0.0,
    ports: Tuple[str, str, str, str] = ('1l', '2r', '3r', '4l'),
    name: str = 'PSR'
  ):
    self._te_loss = self._genvar(TE_loss)
    self._tm_loss = self._genvar(TM_loss)
    self._te_per = self._genvar(TE_per)
    self._tm_per = self._genvar(TM_per)
    self._te_phase = self._genvar(TE_phase)
    self._tm_phase = self._genvar(TM_phase)
    super().__init__(
      ports=ports,
      name=name
    )

  def _construct_smatrix(self):
    te_loss = db2mag(-self._te_loss)
    tm_loss = db2mag(-self._tm_loss)
    te_per = db2mag(self._te_per)
    tm_per = db2mag(self._tm_per)

    if te_loss + te_per > 1:
      te_loss = 1 - te_per
    if tm_loss + tm_per > 1:
      tm_loss = 1 - tm_per

    te_phase = np.exp(1j * self._te_phase)
    tm_phase = np.exp(1j * self._tm_phase)

    return [
      [0.0, te_loss * te_phase, tm_per * tm_phase, 0.0],
      [te_loss * te_phase, 0.0, 0.0, te_per * te_phase],
      [tm_per * tm_phase, 0.0, 0.0, tm_loss * tm_phase],
      [0.0, te_per * te_phase, tm_loss *tm_phase, 0.0]
      ]
  
class phase_aligner(OptColumn):
  def __init__(
    self,
    L_input = 100*um,
    effective_index = 1.58,
    group_index = 1.94,
    XPS_loss_dB = 0.07,
    name = 'PA',
  ):
    self.Linput = L_input

    self.wg1 = Waveguide(length = self.Linput, index = effective_index, group_index = group_index)
    self.wg2 = Waveguide(length = self.Linput, index = effective_index, group_index = group_index)

    self.XPS1 = XPS_PhaseShifter()
    self.XPS2 = XPS_PhaseShifter()
    
    optical_column_1 = OptColumn(
      [
        OptRow([
          self.wg1,
          self.XPS1,
        ]),
        OptRow([
          self.wg2,
          self.XPS2,
        ])
      ]
    )

    network = [
      optical_column_1
    ]

    super().__init__(
      network,
      name = name
    )

class mzi(OptRow):
  def __init__(
    self,
    L_mzi= 150*um,
    dc1 = 0.5,
    dc2 = 0.5,
    SiN_directional_coupler_loss_dB = 0.02,
    effective_index = 1.58,
    group_index = 1.94,
    XPS_loss_dB = 0.07,
    name = 'MZI_2STAGE',
  ):
    
    self.Lmzi = L_mzi

    self.dc1 = DirectionalCoupler(splitratio = dc1,loss = SiN_directional_coupler_loss_dB)
    self.dc2 = DirectionalCoupler(splitratio = dc2,loss = SiN_directional_coupler_loss_dB)

    self.XPS1 = XPS_PhaseShifter()
    self.XPS2 = XPS_PhaseShifter()

    self.wg3 = Waveguide(length = self.Lmzi, index = effective_index, group_index = group_index)
    self.wg4 = Waveguide(length = self.Lmzi, index = effective_index, group_index = group_index)


    optical_column_1 = OptColumn(
      [
          OptRow([
            self.XPS1,
            self.wg3
            ]),
          OptRow([
            self.XPS2,
            self.wg4
            ]),
        ]
    )

    network = [
      self.dc1, 
      optical_column_1, 
      self.dc2
    ]
    
    super().__init__(
      network, 
      name = name
      )

class bidi_tx(OptRow):
  def __init__(
    self,
    num_mzi_stages = 1,
    name = 'bidi_tx',
  ):
    self.mzi_1 = mzi()
    self.mzi_2 = mzi()
    self.pa = phase_aligner()
    self.psr = polarization_splitter_rotator()
    self.edge_coupler = edge_coupler()

    if num_mzi_stages == 1:
      network = [
        OptRow([
          self.mzi_1,
          self.pa,
          self.psr,
          self.edge_coupler
        ])
      ]
    elif num_mzi_stages == 2:
      network = [
        OptRow([
          self.mzi_1,
          self.mzi_2,
          self.pa,
          self.psr,
          self.edge_coupler
        ])
      ]
    
    super().__init__(
      network, 
      name = name
      )

class bidi_rx(OptRow):
  def __init__(
    self,
    psr_bool = True,
    mzi_bool = False,
    num_mzi_stages = 2,
    name = 'bidi_rx',
  ):
    self.edge_coupler = edge_coupler()
    self.psr = polarization_splitter_rotator()
    self.mzi_1 = mzi()
    self.mzi_2 = mzi()

    if psr_bool == True and mzi_bool == False:
      network = [
        OptRow(
          [  
          self.edge_coupler.reflect('h'),
          self.psr.reflect('h')
        ]
        )
      ]
    elif psr_bool == True and mzi_bool == True:
      if num_mzi_stages == 1:
        network = [
          OptRow(
            [  
            self.edge_coupler.reflect('h'),
            self.psr.reflect('h'),
            self.mzi_1.reflect('h')
          ]
          )
        ]
      elif num_mzi_stages == 2:
        network = [
          OptRow(
            [  
            self.edge_coupler.reflect('h'),
            self.psr.reflect('h'),
            self.mzi_1.reflect('h'),
            self.mzi_2.reflect('h')
          ]
          )
        ]
    elif psr_bool == False and mzi_bool == False:
      network = [
        OptRow(
          [  
          self.edge_coupler.reflect('h')
        ]
        )
      ]
    
    super().__init__(
      network, 
      name = name
      )

class polarization_bidi_single_fiber(OptRow):
  def __init__(
    self, 
    psr_bool = True,
    mzi_bool = False,
    tx_num_mzi_stages = 2,
    rx_num_mzi_stages = 1,
    name = 'POLARIZATION_BIDI_SINGLE_FIBER',
  ):
    self.bidi_tx = bidi_tx(num_mzi_stages = tx_num_mzi_stages)
    self.fiber_1 = optical_fiber()
    self.bidi_rx = bidi_rx(num_mzi_stages = rx_num_mzi_stages, psr_bool = psr_bool, mzi_bool = mzi_bool)

    network = [
      OptRow(
        [
        self.bidi_tx,
        self.fiber_1,
        self.bidi_rx
        ]
        )
    ]
    
    super().__init__(
      network, 
      name = name
      )

class polarization_bidi_double_fiber(OptRow):
  def __init__(
    self,
    psr_bool = True,
    mzi_bool = False,
    tx_num_mzi_stages = 2,
    rx_num_mzi_stages = 1,
    name = 'POLARIZATION_BIDI_DOUBLE_FIBER',
  ):
    self.bidi_tx = bidi_tx(num_mzi_stages = tx_num_mzi_stages)
    self.fiber_1 = optical_fiber()
    self.optical_connector = optical_connector()
    self.fiber_2 = optical_fiber()
    self.bidi_rx = bidi_rx(num_mzi_stages = rx_num_mzi_stages, psr_bool = psr_bool, mzi_bool = mzi_bool)

    network = [
      OptRow([
        self.bidi_tx,
        self.fiber_1,
        self.optical_connector,
        self.fiber_2,
        self.bidi_rx
      ])
    ]
    
    super().__init__(
      network, 
      name = name
      )

class polarization_bidi(OptRow):
  def __init__(
    self,
    psr_bool = True,
    mzi_bool = False,
    tx_num_mzi_stages = 2,
    rx_num_mzi_stages = 1,
    name = 'POLARIZATION_BIDI',
  ):
    self.bidi_tx = bidi_tx(num_mzi_stages = tx_num_mzi_stages)
    self.fiber_1 = optical_fiber()
    self.oc_1 = optical_connector()
    self.fiber_2 = optical_fiber()
    self.oc_2 = optical_connector()
    self.fiber_3 = optical_fiber()
    self.oc_3 = optical_connector()
    self.fiber_4 = optical_fiber()
    self.oc_4 = optical_connector()
    self.fiber_5 = optical_fiber()
    self.oc_5 = optical_connector()
    self.fiber_6 = optical_fiber()
    self.oc_6 = optical_connector()
    self.fiber_7 = optical_fiber()
    self.bidi_rx = bidi_rx(num_mzi_stages = rx_num_mzi_stages, psr_bool = psr_bool, mzi_bool = mzi_bool)

    network = [
      OptRow([
        self.bidi_tx,
        self.fiber_1,
        self.oc_1,
        self.fiber_2,
        self.oc_2,
        self.fiber_3,
        self.oc_3,
        self.fiber_4,
        self.oc_4,
        self.fiber_5,
        self.oc_5,
        self.fiber_6,
        self.oc_6,
        self.fiber_7,
        self.bidi_rx
      ])
    ]

    super().__init__(
      network, 
      name = name
      )