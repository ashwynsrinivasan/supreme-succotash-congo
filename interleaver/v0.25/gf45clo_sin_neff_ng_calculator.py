import numpy as np
import matplotlib.pylab as plt

import tidy3d as td
from tidy3d.constants import C_0
from tidy3d.plugins.mode import ModeSolver

import warnings
warnings.filterwarnings("ignore")


def mode_data_calculator(wg_width, wg_height, sidewall_angle, wg_n, wg_k, freqs, freq0, Lx = 10, Ly=10, Lz=10, dl=0.05, min_steps_per_wvl = 20):
  Lx, Ly, Lz = Lx, Ly, Lz

  wg_medium = td.Medium.from_nk(n=wg_n, k=0, freq=freq0)
  background_medium = td.Medium.from_nk(n=1.45, k=0, freq=freq0)

  wvl_um = C_0/freq0

  vertices = np.array([(-1.0, -wg_width/2.0), (1.0, -wg_width/2.0),
                      (1.0, wg_width/2.0), (-1.0, wg_width/2.0)])
  wg = td.Structure(
      geometry=td.PolySlab(
          vertices=vertices,
          axis=2,
          slab_bounds=(-wg_height/2.0, wg_height/2.0),
          sidewall_angle=sidewall_angle,
          reference_plane="bottom"
      ),
      medium=wg_medium
  )

  # automatic grid specification
  grid_spec = td.GridSpec.auto(min_steps_per_wvl=20, wavelength=wvl_um)
  run_time = 1e-12

  sim = td.Simulation(
      size=(Lx, Ly, Lz),
      grid_spec=grid_spec,
      structures=[wg],
      run_time=run_time,
      medium=background_medium,
      boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
      
  )

  plane = td.Box(
      center=(0, 0, 0),
      size=(0, Ly*0.75, Lz*0.75)
  )

  mode_spec = td.ModeSpec(
      num_modes=1,
      target_neff=wg_n,
      group_index_step=True
  )

  mode_solver = ModeSolver(
      simulation=sim,
      plane=plane,
      mode_spec=mode_spec,
      freqs=freqs,
  )

  mode_data = mode_solver.solve()

  return (mode_solver, mode_data.to_dataframe())