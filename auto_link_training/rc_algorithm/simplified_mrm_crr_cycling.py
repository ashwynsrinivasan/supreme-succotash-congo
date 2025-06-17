import numpy as np
from simplified_sequential_assignment import sequential_assignment, device_resonances_heated_func
from copy import deepcopy
        
def mrm_cycling_func(device_resonances_iter, laser_frequency, device_optical_bandwidth, heat_assigned, mrm_heat_cycling, step_sweep = 200, debug = 0):
  optimized_heat = deepcopy(heat_assigned)
  heat_cycling = mrm_heat_cycling
  
  heated_device_resonances_iter = device_resonances_heated_func(device_resonances_iter, (heat_cycling*step_sweep)*np.ones(device_resonances_iter.shape[0]))
  device_dict = sequential_assignment(heated_device_resonances_iter, laser_frequency, device_optical_bandwidth, step_sweep = step_sweep, debug = debug)
  optimized_heat = deepcopy(device_dict["heat_assigned"]) + heat_cycling*step_sweep*np.ones(device_dict["heat_assigned"].shape[0])
        
  cycled_mrm_dict = {
    "cycled_mrm_assigned":device_dict["device_assigned"], 
    "cycled_mrm_laser_assigned": device_dict["laser_assigned"], 
    "cycled_mrm_heat_assigned": device_dict["heat_assigned"], 
    "cycled_mrm_heat_applied": device_dict["heat_applied"],
    "cycled_mrm_resonance_assigned": device_dict["resonance_assigned"], 
    "cycled_mrm_heat_cycling": mrm_heat_cycling
  }
        
  return cycled_mrm_dict

def crr_locking_func(crr_resonances_iter, crr_optical_bandwidth, mrm_resonances_iter, laser_frequency, mrm_optical_bandwidth, no_mrm_cycling_steps, no_crr_cycling_steps, mrm_assigned, mrm_laser_assigned, mrm_heat_assigned, mrm_heat_cycling, step_sweep = 200, debug = 0):
  crr_assigned = []
  crr_laser_assigned = []
  crr_heat_assigned = []
  crr_resonance_assigned = []
  crr_cycling_assigned = []
  mrm_cycling_assigned = []
  crr0_locked_bool = False
  
  resonance_laser_error = crr_optical_bandwidth/2
  
  step_range = [1, -1]
  
  
  if debug:
    print("\n")
    print("Locking CRRs")
    
  for crr_cycling_no in range(no_crr_cycling_steps):    
    for mrm_cycling_no in range(no_mrm_cycling_steps):
      if debug:
        print("\n")
        print("Lockking CRR0, crr_cycling_no:", crr_cycling_no,"mrm_cycling_no:", mrm_cycling_no)
      if not crr0_locked_bool:
        total_mrm_heat_cycling = mrm_cycling_no + mrm_heat_cycling
        cycled_mrm_dict = mrm_cycling_func(mrm_resonances_iter, laser_frequency, mrm_optical_bandwidth, mrm_heat_assigned, total_mrm_heat_cycling, step_sweep = step_sweep, debug = 0)
        if debug:
          print("cycled_mrm_dict:", cycled_mrm_dict)
          print("total_mrm_heat_cycling:", total_mrm_heat_cycling)
          print("cycle_mrm_assigned:", cycled_mrm_dict["cycled_mrm_assigned"])
        for mrm_idd, mrm_no in enumerate(cycled_mrm_dict["cycled_mrm_assigned"]):
          if mrm_no == 0:
            laser_search = laser_frequency[int(cycled_mrm_dict["cycled_mrm_laser_assigned"][mrm_idd])]
            laser_id = int(cycled_mrm_dict["cycled_mrm_laser_assigned"][mrm_idd])
            if debug:
              print("mrm_no:", mrm_no, "mrm_idd:", mrm_idd, "laser_search:", laser_search, "laser_id:", laser_id)
            break
        for step in step_range:
          if not crr0_locked_bool:
            resonances_location = crr_resonances_iter[0,:] - laser_search + crr_cycling_no  * step_sweep + step_sweep * step
            if debug:
              print("crr_no:0, step:", step)
              print("crr_resonances_iter", crr_resonances_iter[0,:])
              print("resonances_location:", resonances_location, "step_sweep:", step_sweep, "step:", step)
            for res_no in range(crr_resonances_iter.shape[1]):
              if (step == np.max(step_range)) and ((resonances_location[res_no] >= - resonance_laser_error/2) and (resonances_location[res_no] <  step_sweep * step + resonance_laser_error/2)):
                crr0_locked_bool = True
              if (step == np.min(step_range)) and ((resonances_location[res_no] <= resonance_laser_error/2) and (resonances_location[res_no] > step_sweep * step - resonance_laser_error/2)):
                crr0_locked_bool = True
              if crr0_locked_bool:
                if debug:
                  print("crr0_locked_bool:", crr0_locked_bool)
                  print("CRR0 locked, mrm_cycling_no:", mrm_cycling_no, "crr_cycling_no:",crr_cycling_no*2, "step:", step)
                  print("CRR0 locked to :",laser_search)
                  print("CRR0 resonances:", crr_resonances_iter[0,res_no])
                  print("resonances_locations:", resonances_location[res_no])
                  print("step_sweep",step_sweep * step)
                  print("resonance_id:", res_no)
                  print("heat_applied:",laser_search - crr_resonances_iter[0,res_no])
                crr_assigned.append(0)
                crr_laser_assigned.append(laser_id)
                crr_heat_assigned.append(laser_search - crr_resonances_iter[0,res_no])
                crr_cycling_assigned.append(crr_cycling_no)
                mrm_cycling_assigned.append(mrm_cycling_no)
                crr_resonance_assigned.append(res_no)
                break
      if crr0_locked_bool:
        break
    if crr0_locked_bool:
      break

  step_range = [7, -2]
  
  for crr_no in range(1, crr_resonances_iter.shape[0]):
    if debug:
      print("\n")
    if crr0_locked_bool:
      crrx_locked_bool = False
      for step in step_range:
        if not crrx_locked_bool:
          for mrm_idd, mrm_no in enumerate(mrm_assigned):
            if mrm_no == crr_no:
              break
          laser_id = int(mrm_laser_assigned[mrm_idd])
          laser_search = laser_frequency[laser_id]
          resonances_location = crr_resonances_iter[crr_no,:] - laser_search + crr_cycling_assigned[-1] * step_sweep + step_sweep * step
          if debug:
            print("crr_no:", crr_no, "step:", step)
            print("laser_search:", laser_search, "mrm_idd:", mrm_idd, "mrm_no:", mrm_no, "laser_id:", laser_id)
            print("crr_resonances_iter", crr_resonances_iter[crr_no,:])
            print("resonances_location:", resonances_location, "step_sweep:", step_sweep, "step:", step)

          for res_no in range(crr_resonances_iter.shape[1]):
            if (step == np.max(step_range)) and ((resonances_location[res_no] >= - resonance_laser_error/2) and (resonances_location[res_no] < step_sweep * step + resonance_laser_error/2)):
              if debug:
                print("Entering max step check", crrx_locked_bool)
                print("Current resonance location:", resonances_location[res_no], "res_no:", res_no)
              crrx_locked_bool = True
            if (step == np.min(step_range)) and ((resonances_location[res_no] <= resonance_laser_error/2) and (resonances_location[res_no] > step_sweep * step - resonance_laser_error/2)):
              if debug:
                print("Entering min step check", crrx_locked_bool)
              crrx_locked_bool = True
            if crrx_locked_bool:
              crr_assigned.append(crr_no)
              crr_laser_assigned.append(laser_id)
              # if step == np.max(step_range):
              crr_heat_assigned.append(laser_search - crr_resonances_iter[crr_no,res_no])
              # else:
                # crr_heat_assigned.append(crr_resonances_iter[crr_no,res_no] - laser_search)
              crr_cycling_assigned.append(crr_cycling_assigned[-1])
              mrm_cycling_assigned.append(mrm_cycling_assigned[-1])
              crr_resonance_assigned.append(res_no)
              break
  
  if debug:
    print("crr_assigned:", crr_assigned)
    print("crr_laser_assigned:", crr_laser_assigned)
    print("crr_heat_assigned:", crr_heat_assigned)
    print("crr_resonance_assigned:", crr_resonance_assigned)
    print("crr_cycling_assigned:", crr_cycling_assigned)
    print("mrm_cycling_assigned:", mrm_cycling_assigned)
            
  cycled_crr_dict = {
    "crr_assigned": crr_assigned, 
    "crr_laser_assigned": crr_laser_assigned, 
    "crr_heat_assigned": crr_heat_assigned, 
    "crr_resonance_assigned": crr_resonance_assigned, 
    "crr_cycling_assigned": crr_cycling_assigned, 
    "mrm_cycling_assigned": mrm_cycling_assigned
  }
  
  return cycled_mrm_dict, cycled_crr_dict