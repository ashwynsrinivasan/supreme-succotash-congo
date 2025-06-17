import numpy as np
import matplotlib.pyplot as plt
from endlesspc import endlesspc

# Import voltage to phase mapping 
filename = 'phase_x_vheater.csv'
heater_data = np.genfromtxt(filename, delimiter=',', skip_header=1)
# ------------------------------- 

# DAC specs
dac_res = 13
dac_vmax = 1.65
dac_vmin = 0.15
ps_n = 2
# ---------

# IADC specs
adc_res = np.array([6, 7, 8, 9, 10, 11, 12])
adc_iref = 1.01
# ----------

# Algorithm settings -------------
nstages = 2
dither = 2 / 128 
update_mode = 1 #(1 - linear limit function, 2 - erfc limit function, 3 - hard limit, 4 - no limit)
fclk = 16e3
mu1 = 1.5
mu2 = 0.15
# --------------------------------

# Sweep settings -----------------
slope = 100 # rad/s
u_ind = 2
# --------------------------------

N = adc_res.size
dims = int(300e-3 / slope * ps_n * 100 * fclk)
maxloss = np.zeros(N)

phi_u = np.zeros((3, dims))
time = np.arange(0, dims / fclk, 1 / fclk)
phi_u[u_ind, ] = slope * time
max_maxloss = -100

for ind in range(N):
    print(f'Simulating... adc_res = {ind} / {N}')

    # Linearly increasing inputs


    code_ctrl, iloss, iadc_code = endlesspc(heater_data, ps_n, dac_res, dac_vmax, dac_vmin, adc_res[ind], adc_iref, nstages, dither, update_mode, mu1, mu2, phi_u)
    maxloss[ind] = np.min(iloss[int(dims/2):])

plt.figure()
plt.plot(adc_res, maxloss)
plt.xlabel('ADC resolution (bits)')
plt.ylabel('Loss (dB)')
plt.title(f'Loss vs. ADC resolution, mu1 = {mu1} mu2 = {mu2}')
plt.grid()
plt.show()
