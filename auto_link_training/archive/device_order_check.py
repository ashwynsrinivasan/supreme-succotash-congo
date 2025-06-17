import numpy as np
from copy import deepcopy

def device_order_check(device_resonances_iter, laser_frequency, step_sweep=200, mrm_optical_bandwidth=0, debug = 0):
  device_assigned = []
  laser_assigned = []
  step_assigned = []
  resonance_assigned = []
  device_resonances_copy = deepcopy(device_resonances_iter)
  
  no_devices = device_resonances_iter.shape[0]
  no_resonances = device_resonances_iter.shape[1]
  no_laser_wavelengths = len(laser_frequency)
  
  for device_no in range(no_devices): # sweeping for mrm order
    for step_no in [-1, 1]:
        for laser_no in range(no_laser_wavelengths):
              confirmed = False
              resonances_location = device_resonances_copy[device_no,:] - laser_frequency[laser_no] + step_no * step_sweep
              if debug:
                print("device_no: ", device_no, ", step_no: ", step_no, ", laser_no: ", laser_no, ", laser_frequency:", laser_frequency[laser_no],", device_resonance_iter:", device_resonances_copy[device_no,:], ", resonances_location: ", resonances_location, ", heating/cooling:", step_no * step_sweep)
              if step_no == -1:
                for res_no in range(no_resonances):
                  if ((resonances_location[res_no]) <= mrm_optical_bandwidth/2) and ((resonances_location[res_no]) >= - step_sweep - mrm_optical_bandwidth/2):
                    if debug:
                      print("Negative Step Entered")
                    confirmed = True
              else:
                for res_no in range(no_resonances):
                  if ((resonances_location[res_no]  >= -mrm_optical_bandwidth/2) and ((resonances_location[res_no]) <= step_sweep + mrm_optical_bandwidth/2)):
                    if debug:
                      print(" Positive Step Entered")
                    confirmed = True
              if confirmed:
                device_assigned.append(device_no)
                laser_assigned.append(laser_no)
                step_assigned.append(step_no)
                resonance_assigned.append(np.argmin(resonances_location * step_no))
                break
    
  return np.array(device_assigned), np.array(laser_assigned), np.array(step_assigned), np.array(resonance_assigned)