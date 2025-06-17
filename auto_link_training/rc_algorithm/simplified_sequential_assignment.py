import numpy as np
from copy import deepcopy

def device_resonances_heated_func(device_resonances_iter, heat_applied):
  heat_device_resonances_iter = deepcopy(device_resonances_iter)
  for device_no in range(device_resonances_iter.shape[0]):
    heat_device_resonances_iter[device_no,:] = device_resonances_iter[device_no,:] + heat_applied[device_no]
    
  return heat_device_resonances_iter

def device_laser_heated_lock_func(device_resonances_iter, device_no, laser_frequency, resonance_laser_error, laser_assigned, device_assigned, heat_assigned, heat_applied, resonance_assigned, step_sweep = 200, debug = 0):
  device_locked = False
  comp_check = - resonance_laser_error
  for laser_no, laser_val in enumerate(laser_frequency):
    if not device_locked:
      resonance_location = device_resonances_iter[device_no,:] - laser_val + step_sweep * 2
      for res_no in range(device_resonances_iter.shape[1]):
        if (resonance_location[res_no] >= -resonance_laser_error) and (resonance_location[res_no] <= step_sweep*2+resonance_laser_error):
          if comp_check == - resonance_laser_error:
            comp_check = resonance_location[res_no]
            device_assigned.append(device_no)
            laser_assigned.append(laser_no)
            heat_applied.append(laser_frequency[laser_no]- device_resonances_iter[device_no,res_no])
            heat_assigned.append(laser_frequency[laser_no]- device_resonances_iter[device_no,res_no])
            resonance_assigned.append(res_no)
          if resonance_location[res_no]> comp_check:
            comp_check = resonance_location[res_no]
            device_assigned[-1] = device_no
            laser_assigned[-1] = laser_no
            heat_applied[-1] = laser_frequency[laser_no]- device_resonances_iter[device_no,res_no]
            heat_assigned[-1] = laser_frequency[laser_no]- device_resonances_iter[device_no,res_no]
            resonance_assigned[-1] = res_no          
          break
  device_locked = True
  
  return device_assigned, laser_assigned, heat_assigned, heat_applied, resonance_assigned

def adjacent_device_search_func(device_resonances_iter, device_no, laser_frequency, resonance_laser_error, laser_assigned, heat_applied, step_sweep, debug=0):
  device_found = False
  for step in [6, -2]:
    if not device_found:
      laser_search = laser_frequency[laser_assigned[-1]]
      resonance_location = device_resonances_iter[device_no,:] - laser_search + step_sweep * step
      for res_no in range(device_resonances_iter.shape[1]):
        if debug:
          print("\nlaser_search:", laser_search, "\nstep_sweep:", step_sweep*step, "\nres_no:", res_no, "\ndevice_resonances:", device_resonances_iter[device_no,res_no], "\nresonance_location:", resonance_location[res_no])
        if (step>0) and (resonance_location[res_no] >= -resonance_laser_error) and (resonance_location[res_no] <= step_sweep*step+resonance_laser_error):
          device_found = True
        if (step<0) and (resonance_location[res_no] <= resonance_laser_error) and (resonance_location[res_no] >= step_sweep*step-resonance_laser_error):
          device_found = True
        if device_found:
          heat_applied.append(laser_frequency[laser_assigned[-1]] - device_resonances_iter[device_no,res_no])
          break
  
  return device_found, laser_assigned, heat_applied, res_no

def device_knockoff_func(device_resonances_iter, device_no, laser_frequency, device_assigned, laser_assigned, heat_assigned, heat_applied, resonance_assigned, resonance_laser_error, step_sweep = 200, debug = 0):
  if debug:
    print("\nDevice_no:", device_no, "Searching for device:", device_no+1,"\n")
    
  device_found, laser_assigned, heat_applied, res_no = adjacent_device_search_func(device_resonances_iter, device_no, laser_frequency, resonance_laser_error, laser_assigned, heat_applied, step_sweep, debug=debug)
  
  if debug:
    print("Found device:", device_no+1, "\nlaser_search:", laser_frequency[laser_assigned[-1]], "\ndevice_resonances_iter:", device_resonances_iter[device_no,res_no], "\ndevice_resonances_iter:", device_resonances_iter[device_no,:],"\ndevice_resonances_iter + heat_applied:",device_resonances_iter[device_no,:] + heat_applied[-1],"\nheat_applied:", heat_applied[-1])
  
  if device_found:
    if debug:
      print("Device:", device_no, " found device", device_no + 1)
    device_locked = False
    for jump_step in [-400, -800]:
      if not device_locked:
        device_heat_applied = heat_applied[-1] + jump_step
        device_assigned, laser_assigned, heat_assigned, resonance_assigned, device_locked = device_laser_local_lock_func(device_resonances_iter, device_no, laser_frequency, resonance_laser_error, laser_assigned, device_assigned, heat_assigned, resonance_assigned, device_locked, device_heat_applied, step_sweep, debug = debug)
  
  return device_assigned, laser_assigned, heat_assigned, heat_applied, resonance_assigned

def device_laser_local_lock_func(device_resonances_iter, device_no, laser_frequency, resonance_laser_error, laser_assigned, device_assigned, heat_assigned, resonance_assigned, device_locked, device_heat_applied, step_sweep, debug = 0):
  device_locked = False
  for step in [0.5, -0.5]:
    if not device_locked:
      for laser_no, laser_val in enumerate(laser_frequency):
        if laser_no not in laser_assigned:
          resonance_location = device_resonances_iter[device_no,:] - laser_val + step_sweep * step + device_heat_applied #heat_applied[-1] + jump_step
          for res_no in range(device_resonances_iter.shape[1]):
            if (step == 0.5) and (resonance_location[res_no] >= -resonance_laser_error) and (resonance_location[res_no] <= step_sweep*step+resonance_laser_error):
              device_locked = True
            if (step == -0.5) and (resonance_location[res_no] <= resonance_laser_error) and (resonance_location[res_no] >= step_sweep*step-resonance_laser_error):
              device_locked = True
            if device_locked:
              device_assigned.append(device_no)
              laser_assigned.append(laser_no)
              heat_assigned.append(laser_frequency[laser_no]- device_resonances_iter[device_no,res_no])
              resonance_assigned.append(res_no)
              break
        if device_locked:
          if debug:
            print("Device", device_no," is locked.")
          break
  return device_assigned, laser_assigned, heat_assigned, resonance_assigned, device_locked

def sequential_assignment(device_resonances_iter, laser_frequency, device_optical_bandwidth, step_sweep = 200, debug = 0):
  device_assigned = []
  laser_assigned = []
  heat_applied = []
  heat_assigned = []
  resonance_assigned = []
  
  device_no = 7
  resonance_laser_error = device_optical_bandwidth/2
  # resonance_laser_error = 0.0
  device_assigned, laser_assigned, heat_assigned, heat_applied, resonance_assigned = device_laser_heated_lock_func(device_resonances_iter, device_no, laser_frequency, resonance_laser_error, laser_assigned, device_assigned, heat_assigned, heat_applied, resonance_assigned, step_sweep = step_sweep, debug = debug)
  
  if debug:
    print("\nDevice 7 locked ", "\ndevice_assigned:",device_assigned,"\nlaser_assigned",laser_assigned, "\nheat_assigned", heat_assigned,"\nresonance_assigned",resonance_assigned)
  
  for device_no in np.linspace(6, 0, 7, dtype=int):
    device_assigned, laser_assigned, heat_assigned, heat_applied, resonance_assigned = device_knockoff_func(device_resonances_iter, device_no, laser_frequency, device_assigned, laser_assigned, heat_assigned, heat_applied, resonance_assigned, resonance_laser_error, step_sweep = step_sweep, debug = debug)
  
  device_assigned = np.array(device_assigned)
  laser_assigned = np.array(laser_assigned)
  heat_assigned = np.array(heat_assigned)
  heat_applied = np.array(heat_applied)
  resonance_assigned = np.array(resonance_assigned)
    
  if debug:
    print("Device_assigned:", device_assigned, "\nLaser_assigned:", laser_assigned, "\nHeat_assigned:", heat_assigned, "\nHeat_applied:", heat_applied, "\nResonance_assigned:", resonance_assigned)
  
  device_dict = {
    "device_assigned": device_assigned,
    "laser_assigned":laser_assigned,
    "heat_assigned":heat_assigned,
    "heat_applied":heat_applied,
    "resonance_assigned":resonance_assigned,
  }
  
  return device_dict


def optimal_sequential_assignment(device_resonances_iter, laser_frequency, device_optical_bandwidth, step_sweep = 200, debug = 0, optimization_bool = True, ):
  device_dict = sequential_assignment(device_resonances_iter, laser_frequency, device_optical_bandwidth, step_sweep = step_sweep, debug = debug)
  optimized_heat = deepcopy(device_dict["heat_assigned"])
  og_min_heat = np.min(optimized_heat)
  
  if og_min_heat > 0:
    class_type = 0
    
  if og_min_heat >= -200 and og_min_heat<=0:
    class_type = 1
    
  if og_min_heat >= -400 and og_min_heat<=-200:
    class_type = 2
    
  else:
    class_type = 3
    
  heat_cycling = 0
  cool_cycling = 0
  
  if optimization_bool:
    if np.min(optimized_heat) < -400.0:
      heat_cycling += 1
      heated_device_resonances_iter = device_resonances_heated_func(device_resonances_iter, ((heat_cycling + cool_cycling)*step_sweep)*np.ones(device_resonances_iter.shape[0]))
      device_dict = sequential_assignment(heated_device_resonances_iter, laser_frequency, device_optical_bandwidth, step_sweep = step_sweep, debug = debug)
      optimized_heat = deepcopy(device_dict["heat_assigned"]) + (heat_cycling + cool_cycling)*step_sweep*np.ones(device_dict["heat_assigned"].shape[0])
   
  
  device_dict = {
    "device_assigned": device_dict["device_assigned"],
    "laser_assigned":device_dict["laser_assigned"],
    "heat_assigned":device_dict["heat_assigned"],
    "heat_applied":device_dict["heat_applied"],
    "resonance_assigned":device_dict["resonance_assigned"],
    "heat_cycling": heat_cycling,
    "cool_cycling": cool_cycling
  }
          
  return device_dict