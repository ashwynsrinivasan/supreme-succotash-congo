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
adc_res = 9
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
slope = np.logspace(1,3,20) # rad/s
u_ind = 2
# --------------------------------

slopeN = slope.size
dims = int(300e-3 / slope[0] * 100 * fclk)
maxloss = np.zeros(slopeN)

phi_u = np.zeros((3, dims))
time = np.arange(0, dims / fclk, 1 / fclk)

max_maxloss = -100

for slope_ind in range(slopeN):
    print(f'Simulating... slope = {slope_ind} / {slopeN}')

    # Linearly increasing inputs
    phi_u[u_ind, ] = slope[slope_ind] * time

    code_ctrl, iloss, iadc_code = endlesspc(heater_data, ps_n, dac_res, dac_vmax, dac_vmin, adc_res, adc_iref, nstages, dither, update_mode, mu1, mu2, phi_u)
    maxloss[slope_ind] = np.min(iloss[int(dims/2):])

plt.figure()
plt.semilogx(slope, maxloss)
plt.xlabel('Slope (rad/s)')
plt.ylabel('Loss (dB)')
plt.title(f'Loss vs. Aggressor slope, mu1 = {mu1} mu2 = {mu2}')
plt.grid()
plt.show()
