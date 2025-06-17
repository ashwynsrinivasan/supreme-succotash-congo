import numpy as np
import matplotlib.pyplot as plt
from endlesspc import endlesspc

filename = 'phase_x_vheater.csv'
heater_data = np.genfromtxt(filename, delimiter=',', skip_header=1)

# DAC specs
dac_res = 13
dac_vmax = 1.65
dac_vmin = 0.15
ps_n = 2
# ---------

# IADC specs
adc_res = 9
adc_iref = 1.01 # proportion to input power
# ----------

# Algorithm settings -------------
nstages = 2
code_update = 2
update_mode = 1 #(1 - linear limit function, 1.1 - sin, 1.2 - tan, 1.3 - cube
fclk = 16e3
# Parameter for mode (1)
mu1 = 1.5
mu2 = 0.15
# --------------------------------

# Sweep settings -----------------
slope = 100  # rad/s
dither = 2 / 128
dc_phase = 0 # rad
u_ind = 2
# dims = int(80000 * 10 / slope)
dims = int(300e-3 / slope * 100 * fclk)
#dims = 64
# --------------------------------

phi_u = np.zeros((3, dims))
time = np.arange(0, dims / fclk, 1 / fclk)

# Linearly increasing inputs
phi_u[u_ind, ] = slope * time + dc_phase

code_ctrl, iloss, iadc_code = endlesspc(heater_data, ps_n, dac_res, dac_vmax, dac_vmin, adc_res, adc_iref, nstages, dither, update_mode, mu1, mu2, phi_u)

maxloss = np.min(iloss[int(iloss.size/2):])

fig, ax = plt.subplots(3, 1)
ax[0].plot(time, phi_u[0,], label='Phi U1')
ax[0].plot(time, phi_u[1,], label='Phi U2')
ax[0].plot(time, phi_u[2,], label='Phi U3')
ax[0].legend()
ax[0].set_title(f'Input rotation ({slope} rad/s)')
ax[0].set_ylabel('Rads')

for ind in range(nstages):
    ax[1].plot(time, code_ctrl[ind,], label=f'code_ctrl{ind+1}')
ax[1].legend()
ax[1].set_title('Phase shifter HDAC ctrl code')
ax[1].set_ylabel('Code')
#ax[1].set_ylim(-2**(dac_res+0.5), 2**(dac_res+0.5))
#ax[1].axhline(y=-2**dac_res, color='r', linestyle='--')

for ind in range(nstages):
    print(f'Min code {ind+1} = {np.min(code_ctrl[ind,int(dims/2):])}')
    print(f'Max code {ind+1} = {np.max(code_ctrl[ind,int(dims/2):])}')


ax[2].plot(time, iloss, label='insertion loss')
ax[2].legend()
ax[2].set_title(f'Insertion Loss (max ss loss = {maxloss:.2f} dB)')
ax[2].set_ylabel('dB')
ax[2].set_xlabel('second')
#ax[2].set_ylim(-2, 0)
plt.subplots_adjust(hspace=1)
plt.show()

print('End')





