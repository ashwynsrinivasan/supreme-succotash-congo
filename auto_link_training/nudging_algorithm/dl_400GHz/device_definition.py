import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt


def device_definition(no_iterations=int(1e4+1), no_mrms = 8, no_crrs = 8, no_resonances = 6, center_resonance = 3, no_laser_wavelengths = 8, centroid_die_variation_bool = False, laser_distribution_uniform=True):
  np.random.seed(123)
  mrm_resonances = np.arange(0, 3200, 400)
  crr_resonances = np.arange(0, 3200, 400)
  laser_frequency = np.arange(0, 3200, 400)

  mrm_fsr = np.arange(1780, 1820, 5)
  mrm_fabrication_variation = 80
  mrm_optical_bandwidth = 100
  crr_fsr = np.arange(1780, 1820, 5)
  crr_fabrication_variation = mrm_fabrication_variation
  crr_optical_bandwidth = 300
  laser_variation = 40/4
  
  mrm_resonances_die_wafer_variation = np.zeros((no_mrms,no_resonances,no_iterations))
  mrm_resonances_die_variation = np.zeros((no_mrms,no_resonances,no_iterations))
  mrm_resonances_wafer_variation = np.zeros((no_mrms,no_resonances,no_iterations))
  mrm_resonances_ideal = np.zeros((no_mrms,no_resonances,no_iterations))
  
  mrm_wafer_variation = np.random.uniform(0, np.mean(mrm_fsr), no_iterations) # Start variation of the MRM resonances
  if laser_distribution_uniform:
    if not centroid_die_variation_bool:
      mrm_die_variation = np.random.uniform(-mrm_fabrication_variation, mrm_fabrication_variation, size = (no_laser_wavelengths,no_iterations)) + np.random.uniform(-laser_variation*4, laser_variation*4, size=(no_laser_wavelengths, no_iterations)) # Fabrication variation of the MRM resonances  
    else:
      mrm_die_variation = np.zeros((no_laser_wavelengths,no_iterations))
      mrm_die_variation_max = np.random.uniform(0, mrm_fabrication_variation, size=(no_iterations))
      for iter_no in range(no_iterations):
        mrm_die_variation[:,iter_no] = np.linspace(-mrm_die_variation_max[iter_no], mrm_die_variation_max[iter_no], no_laser_wavelengths)
  else:
    if not centroid_die_variation_bool:
      mrm_die_variation = np.random.uniform(-mrm_fabrication_variation, mrm_fabrication_variation, size = (no_laser_wavelengths,no_iterations)) + np.random.normal(0, laser_variation, size = (no_laser_wavelengths,no_iterations))# Fabrication variation of the MRM resonances  
    else:
      mrm_die_variation = np.zeros((no_laser_wavelengths,no_iterations))
      mrm_die_variation_max = np.random.uniform(0, mrm_fabrication_variation, size=(no_iterations))
      for iter_no in range(no_iterations):
        mrm_die_variation[:,iter_no] = np.linspace(-mrm_die_variation_max[iter_no], mrm_die_variation_max[iter_no], no_laser_wavelengths) + np.random.normal(0, laser_variation, no_laser_wavelengths)

  for iter_no in range(no_iterations):
    for res_no in range(no_resonances):
      mrm_resonances_die_wafer_variation[:,res_no,iter_no] = mrm_resonances + (res_no-center_resonance)*mrm_fsr + mrm_die_variation[:,iter_no] + mrm_wafer_variation[iter_no]
      mrm_resonances_die_variation[:,res_no,iter_no] = mrm_resonances + (res_no-center_resonance)*mrm_fsr + mrm_die_variation[:,iter_no]
      mrm_resonances_wafer_variation[:,res_no,iter_no] = mrm_resonances + (res_no-center_resonance)*mrm_fsr + mrm_wafer_variation[iter_no]
      mrm_resonances_ideal[:,res_no,iter_no] = mrm_resonances + (res_no-center_resonance)*mrm_fsr
      
  np.random.seed(456)
  crr_resonances_die_wafer_variation = np.zeros((no_crrs,no_resonances,no_iterations))
  crr_resonances_die_variation = np.zeros((no_crrs,no_resonances,no_iterations))
  crr_resonances_wafer_variation = np.zeros((no_crrs,no_resonances,no_iterations))
  crr_resonances_ideal = np.zeros((no_crrs,no_resonances,no_iterations))

  crr_wafer_variation = np.random.uniform(0, np.mean(crr_fsr), no_iterations) # Start variation of the CRR resonances
  
  if not centroid_die_variation_bool:
    crr_die_variation = np.random.uniform(-crr_fabrication_variation, crr_fabrication_variation, size = (no_laser_wavelengths,no_iterations)) # Fabrication variation of the CRR resonances
  else:
    crr_die_variation = np.zeros((no_laser_wavelengths,no_iterations))
    crr_die_variation_max = np.random.uniform(0, crr_fabrication_variation, size=(no_iterations))
    for iter_no in range(no_iterations):
      crr_die_variation[:,iter_no] = np.linspace(-crr_die_variation_max[iter_no], crr_die_variation_max[iter_no], no_laser_wavelengths)

  for iter_no in range(no_iterations):
    for res_no in range(no_resonances):
      crr_resonances_die_wafer_variation[:,res_no,iter_no] = crr_resonances + (res_no-center_resonance)*crr_fsr + crr_die_variation[:,iter_no] + crr_wafer_variation[iter_no]
      crr_resonances_die_variation[:,res_no,iter_no] = crr_resonances + (res_no-center_resonance)*crr_fsr + crr_die_variation[:,iter_no] 
      crr_resonances_wafer_variation[:,res_no,iter_no] = crr_resonances + (res_no-center_resonance)*crr_fsr + crr_wafer_variation[iter_no]
      crr_resonances_ideal[:,res_no,iter_no] = crr_resonances + (res_no-center_resonance)*crr_fsr
  
  tx_bank = {
    "mrm_resonances_die_wafer_variation": mrm_resonances_die_wafer_variation,
    "mrm_resonances_die_variation": mrm_resonances_die_variation,
    "mrm_resonances_wafer_variation": mrm_resonances_wafer_variation,
    "ideal_resonances": mrm_resonances_ideal,
    "mrm_optical_bandwidth": mrm_optical_bandwidth,
  }
  
  rx_bank = {
    "crr_resonances_die_wafer_variation": crr_resonances_die_wafer_variation,
    "crr_resonances_die_variation": crr_resonances_die_variation,
    "crr_resonances_wafer_variation": crr_resonances_wafer_variation,
    "ideal_resonances": crr_resonances_ideal,
    "crr_optical_bandwidth": crr_optical_bandwidth,
  }
  
  laser = {
    "laser_frequency": laser_frequency,
  }
  
  return tx_bank, rx_bank, laser


def plotting_device_resonances(device_resonance, idd_iter=0):
  no_mrms = device_resonance.shape[0]
  for res_no in [1, 2,3,4]:  
    plt.plot(device_resonance[:,res_no,idd_iter], np.ones(no_mrms)*(res_no-1), 'o', label="Resonance # %d" % res_no)
    plt.legend()
    plt.grid(True)
    plt.xlim([-800, 3600])
    plt.xticks(np.arange(-800, 4000, 400))
    plt.ylim([0, 4.0])
    plt.title("Distribution of MRM ")
    plt.xlabel("Frequency [GHz]")

def plotting_device_resonances_iter(device_resonance):
  no_mrms = device_resonance.shape[0]
  for res_no in [1,2,3,4]:  
    plt.plot(device_resonance[:,res_no], np.ones(no_mrms)*(res_no-1), 'o', label="Resonance # %d" % res_no)
    plt.legend()
    plt.grid(True)
    plt.xlim([-800, 3600])
    plt.xticks(np.arange(-800, 4000, 400))
    plt.ylim([0, 4.0])
    plt.title("Distribution of MRM ")
    plt.xlabel("Frequency [GHz]")