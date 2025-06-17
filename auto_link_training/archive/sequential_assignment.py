import numpy as np
from copy import deepcopy

def device_heating(device_resonance_iter, additional_heating, step_sweep = 200):
  device_resonance_cycling = deepcopy(device_resonance_iter)
  for device_no in range(device_resonance_iter.shape[0]):
    device_resonance_cycling[device_no,:] = device_resonance_iter[device_no,:] + additional_heating * step_sweep * 2
  return device_resonance_cycling

def sequential_assignment(device_resonances_iter, laser_frequency, count_idx_iter, device_optical_bandwidth, step_sweep = 200, debug = 0):
  additional_heating = count_idx_iter[1]
  device_focus = int(count_idx_iter[2])
  device_preheated = device_heating(device_resonances_iter, additional_heating)
  
  
  device_assigned = []
  laser_assigned = []
  heat_assigned = []
  resonance_assigned = []
  
  resonance_laser_error = device_optical_bandwidth/2
  resonance_laser_error = 0.0
  
  comp_check = -resonance_laser_error
  
  ## Sequential Alignment Phase - 0 : Locking Reference Device to laser line 4
  ## Blind search first reference MRM within 400 GHz to find L4
  for laser_no, laser_val in enumerate(laser_frequency):
    resonance_location = device_preheated[device_focus,:] - laser_val + step_sweep * 2
    for res_no in range(device_resonances_iter.shape[1]):
      if (resonance_location[res_no] >= -resonance_laser_error) and (resonance_location[res_no] <= step_sweep*2+resonance_laser_error):
        if comp_check == 0:
          comp_check = resonance_location[res_no]
          device_assigned.append(device_focus)
          laser_assigned.append(laser_no)
          heat_assigned.append(laser_frequency[laser_no]- device_preheated[device_focus,res_no])#+ additional_heating * step_sweep * 2 ) # used for debugging
          resonance_assigned.append(res_no)
        if resonance_location[res_no] > comp_check:
          comp_check = resonance_location[res_no]
          laser_assigned[0] = laser_no
          heat_assigned[0] = laser_frequency[laser_no] - device_preheated[device_focus,res_no] #+ additional_heating * step_sweep * 2 # used for debugging
          # heat_assigned[0] = 0 # used for debugging
          resonance_assigned[0] = res_no
  
  ## Sequential Alignment phase - 1 : Locking Reference Device from laser line 3 to 0
  ## Uses knocking off the following MRMs locked to a laser line and then cooling by 400 GHz
  for device_no in np.linspace(device_focus-1, device_focus-4, 3, dtype=int):
    device_locked = False
    laser_search = laser_frequency[laser_assigned[-1]]
    step = 5
    resonance_location = device_preheated[device_no,:] - laser_search + step_sweep * step
    for res_no in range(device_resonances_iter.shape[1]):
      if (resonance_location[res_no] >= -resonance_laser_error) and (resonance_location[res_no] <= step_sweep*step + resonance_laser_error):
        device_assigned.append(device_no)
        laser_assigned.append(laser_assigned[-1]-1)     # This is the code references that we cool by 400GHz and lock to the previous laser 1 before the current knocked off device # used for debugging
        heat_assigned.append(laser_frequency[laser_assigned[-1]-1] - device_preheated[device_no,res_no])# + additional_heating * step_sweep * 2)
        # heat_assigned.append(0) # used for debugging
        resonance_assigned.append(res_no)
        break
          
  ## Sequential Alignment phase - 2: Locking Reference Device from laser 7  
  if device_focus - 5 >= 0:
    device_no = device_focus - 5
    ## Phase 2a
    ## Uses knocking off the following MRMs locked to a laser line, but will require cooling by 800 GHz
    device_locked = False
    laser_search = laser_frequency[laser_assigned[-1]]
    step = 5
    resonance_location = device_preheated[device_no,:] - laser_search + step_sweep * step
    for res_no in range(device_resonances_iter.shape[1]):
      if (resonance_location[res_no] >= -resonance_laser_error) and (resonance_location[res_no] <= step_sweep*step + resonance_laser_error):
        device_assigned.append(device_no)
        laser_assigned.append(laser_assigned[-1]-1) # This is the code references that we cool by 800 GHz
        heat_assigned.append(laser_frequency[laser_assigned[-1]-1] - device_preheated[device_no,res_no+2])# + additional_heating * step_sweep * 2) # used for debugging
        # heat_assigned.append(0) # used for debugging
        resonance_assigned.append(res_no+2) # This shows that we are locking with a higher order resonance
        break
    device_no = device_no - 1
      
    if device_no <0:
      device_no = 7
    else:
      ## Phase 2b
      ## Uses knocking off the following devices locked to a laser line, but will require cooling by 400 GHz
      for device_no in np.linspace(device_focus-6, 0, device_focus-6+1, dtype=int):
        device_locked = False
        laser_search = laser_frequency[laser_assigned[-1]]
        for step in [6, -3]:
          if not device_locked:
            resonance_location = device_preheated[device_no,:] - laser_search + step_sweep * step
            for res_no in range(device_resonances_iter.shape[1]):
              if (step > 0) and (resonance_location[res_no] >= -resonance_laser_error) and (resonance_location[res_no] <= step_sweep*step + resonance_laser_error):
                device_locked = True
              if (step < 0) and (resonance_location[res_no] < resonance_laser_error) and (resonance_location[res_no] >= step_sweep*step - resonance_laser_error):
                device_locked = True
              if device_locked:
                device_assigned.append(device_no)
                laser_assigned.append(laser_assigned[-1]-1) # This is the code references that we cool by 400 GHz
                heat_assigned.append(laser_frequency[laser_assigned[-1]-1] - device_preheated[device_no,res_no])# + additional_heating * step_sweep * 2) # used for debugging
                # heat_assigned.append(0) # used for debugging
                resonance_assigned.append(res_no)
                break
  else:
    device_no = 7
  
  ### Phase 2c
  ### Locking for device7 if device focus is not 7
  ## Blind search for device7 till it finds a laser => heating up to 600 GHz needed
  if (device_no == 7) and (device_focus!= 7):
    device_locked = False
    for step in [3, -2]:
      if not device_locked:
        comp_check = 0
        for laser_no, laser_val in enumerate(laser_assigned):
          if laser_val < 0:
            laser_assigned[laser_no] = laser_val + 8
        for laser_no, laser_val in enumerate(laser_frequency):
          if laser_no not in laser_assigned:
            resonance_location = device_preheated[device_no,:] - laser_val + step_sweep * step
            if debug:
              print("device_preheated: ", device_preheated[device_no,:])
              print("resonance_location: ", resonance_location, step_sweep * step) 
              print("device_no: ", device_no, ", laser_no: ", laser_no, ", step: ", step, )
              print(laser_assigned, device_assigned, heat_assigned, resonance_assigned)
            for res_no in range(device_resonances_iter.shape[1]):
              if (step>0) and (resonance_location[res_no] >=-resonance_laser_error) and (resonance_location[res_no] <= step_sweep*step+resonance_laser_error):
                device_locked = True
              if (step<0) and (resonance_location[res_no] < resonance_laser_error) and (resonance_location[res_no] >= step_sweep*step-resonance_laser_error):
                device_locked = True
              if device_locked:
                if comp_check == 0:
                  comp_check = resonance_location[res_no]
                  device_assigned.append(device_no)
                  laser_assigned.append(laser_no)
                  # heat_assigned.append(laser_val - device_preheated[device_no,res_no] + additional_heating * step_sweep * 2)
                  heat_assigned.append(0)
                  resonance_assigned.append(res_no)
                else:
                  comp_check = resonance_location[res_no]
                  laser_assigned[-1] = laser_no
                  # heat_assigned[-1] = laser_val - device_preheated[device_no,res_no] + additional_heating * step_sweep * 2
                  heat_assigned[-1] = 0
                  resonance_assigned[-1] = res_no
    ### Phase 2d: Locking remaining devices      
    device_no = device_no - 1
    if device_no > device_focus:
      for device_no in np.linspace(6, device_focus+1, 6-device_focus, dtype=int):
        device_locked=False
        laser_search = laser_frequency[laser_assigned[-1]]
        step = 5
        resonance_location = device_preheated[device_no,:] - laser_search + step_sweep * step
        for res_no in range(device_resonances_iter.shape[1]):
          if (resonance_location[res_no] >= -resonance_laser_error) and (resonance_location[res_no] <= step_sweep*step + resonance_laser_error):
            device_assigned.append(device_no)
            laser_assigned.append(laser_assigned[-1]-1)     # This is the code references that we cool by 400GHz and lock to the previous laser 1 before the current knocked off device # used for debugging
            heat_assigned.append(laser_frequency[laser_assigned[-1]-1] - device_preheated[device_no,res_no])# + additional_heating * step_sweep * 2)
            # heat_assigned.append(0) # used for debugging
            resonance_assigned.append(res_no)
            break
        
  # for device_no in np.linspace(device_focus-1, 0, device_focus, dtype=int):
  #   device_locked = False
  #   laser_search = laser_frequency[laser_assigned[-1]]
    
  #   for step in [6, -3]:
  #     if not device_locked:
  #       resonance_location = device_preheated[device_no,:] - laser_search + step_sweep * step
  #       for res_no in range(device_resonances_iter.shape[1]):
  #         if (step > 0) and (resonance_location[res_no] >= -resonance_laser_error) and (resonance_location[res_no] <= step_sweep*step + resonance_laser_error):
  #           device_locked = True
  #         if (step < 0) and (resonance_location[res_no] < resonance_laser_error) and (resonance_location[res_no] >= step_sweep*step - resonance_laser_error):
  #           device_locked = True
  #         if device_locked:
  #           device_assigned.append(device_no)
  #           laser_assigned.append(laser_assigned[-1]-1)
  #           # if (device_focus > 4) and (device_no <= device_focus-6):
  #           if laser_assigned[-1] -1 < 0:
  #             heat_assigned.append(laser_frequency[laser_assigned[-1]-1] - device_preheated[device_no,res_no+1])# + additional_heating * step_sweep * 2)
  #             # heat_assigned.append(0)
  #           else:
  #             heat_assigned.append(laser_frequency[laser_assigned[-1]-1] - device_preheated[device_no,res_no])# + additional_heating * step_sweep * 2)
  #             # heat_assigned.append(0)
  #           resonance_assigned.append(res_no)
  #           break
  # for device_no in np.linspace(7, device_focus+1, 7-device_focus, dtype=int):
  #   device_locked=False
  #   if debug:
  #     print("Phase-3")
  #   for step in [4, -2]:
  #     if not device_locked:
  #       comp_check = 0
  #       for laser_no, laser_val in enumerate(laser_assigned):
  #         if laser_val < 0:
  #           laser_assigned[laser_no] = laser_val + 8
  #       for laser_no, laser_val in enumerate(laser_frequency):
  #         if laser_no not in laser_assigned:
  #           resonance_location = device_preheated[device_no,:] - laser_val + step_sweep * step
  #           if debug:
  #             print("device_preheated: ", device_preheated[device_no,:])
  #             print("resonance_location: ", resonance_location, step_sweep * step) 
  #             print("device_no: ", device_no, ", laser_no: ", laser_no, ", step: ", step, )
  #             print(laser_assigned, device_assigned, heat_assigned, resonance_assigned)
  #           for res_no in range(device_resonances_iter.shape[1]):
  #             if (step>0) and (resonance_location[res_no] >=-resonance_laser_error) and (resonance_location[res_no] <= step_sweep*step+resonance_laser_error):
  #               device_locked = True
  #             if (step<0) and (resonance_location[res_no] < resonance_laser_error) and (resonance_location[res_no] >= step_sweep*step-resonance_laser_error):
  #               device_locked = True
  #             if device_locked:
  #               if comp_check == 0:
  #                 comp_check = resonance_location[res_no]
  #                 device_assigned.append(device_no)
  #                 laser_assigned.append(laser_no)
  #                 # heat_assigned.append(laser_val - device_preheated[device_no,res_no] + additional_heating * step_sweep * 2)
  #                 heat_assigned.append(0)
  #                 resonance_assigned.append(res_no)
  #               else:
  #                 comp_check = resonance_location[res_no]
  #                 laser_assigned[-1] = laser_no
  #                 # heat_assigned[-1] = laser_val - device_preheated[device_no,res_no] + additional_heating * step_sweep * 2
  #                 heat_assigned[-1] = 0
  #                 resonance_assigned[-1] = res_no
        
    if debug:
      print("device_assigned: ", device_assigned)
      print("laser_assigned: ", laser_assigned)
      print("resonance_assigned: ", resonance_assigned)
      print("heat_assigned: ", heat_assigned)
    
  return np.array(device_assigned), np.array(laser_assigned), np.array(heat_assigned), np.array(resonance_assigned)

def device_array_decision(device_assigned, no_devices, iter_no, preheat=0):
  count = np.bincount(device_assigned)
  count_idx = []
  if np.min(count) == 2:
    count_idx.append((iter_no, 1, -1, 2))
  else:
    for idx in range(no_devices):
      if count[no_devices - idx - 1] == 1:
        if (no_devices - idx - 1) < 4:
          count_idx.append((iter_no, preheat + 1, no_devices - idx - 1, 1))
        if (no_devices - idx - 1) >= 4:
          count_idx.append((iter_no, preheat + 0, no_devices - idx - 1, 1))
        break
          
  return count_idx