import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from endlesspc import endlesspc

# Import voltage to phase mapping 
filename = 'phase_x_vheater.csv'
heater_data = np.genfromtxt(filename, delimiter=',', skip_header=1)
# ------------------------------- 

# DAC specs
dac_res = 13
dac_vmax = 1.65
dac_vmin = 0.15
ps_n = 1
# ---------

# IADC specs
adc_res = 9
adc_iref = 1.01
# ----------

# Algorithm settings -------------
nstages = 4
dither =  2 / 128 # need to figure out a good way to set this
update_mode = 1 #(1 - linear limit function, 2 - erfc limit function, 3 - hard limit, 4 - no limit)
fclk = 16e3
mu1 = np.logspace(-1, 1, 20)
mu2 = np.logspace(-2, 0, 20)
# print(mu1_unscaled)
# print(mu2_unscaled)
# --------------------------------

# Sweep settings -----------------
slope = 100 # rad/s
u_ind = 1
# --------------------------------

mu1N = mu1.size
mu2N = mu2.size
dims = int(300e-3 / slope * ps_n * 100 * fclk)
maxloss = np.zeros((mu1N, mu2N))

phi_u = np.zeros((3, dims))
time = np.arange(0, dims / fclk, 1 / fclk)
# Linearly increasing inputs
phi_u[u_ind, ] = slope * time

max_maxloss = -100

for mu1_ind in range(mu1N):
    for mu2_ind in range(mu2N):
        print(f'Simulating... mu1_ind = {mu1_ind} / {mu1N} and mu2_ind = {mu2_ind} / {mu2N}')
        code_ctrl, iloss, iadc_code = endlesspc(heater_data, ps_n, dac_res, dac_vmax, dac_vmin, adc_res, adc_iref, nstages, dither, update_mode, mu1[mu1_ind], mu2[mu2_ind], phi_u)
        maxloss[mu1_ind, mu2_ind] = np.min(iloss[int(dims/2):])
        if maxloss[mu1_ind, mu2_ind] > max_maxloss:
            max_maxloss = maxloss[mu1_ind, mu2_ind]
            code_ctrl_best = code_ctrl
            iloss_best = iloss
            mu1_best = mu1[mu1_ind]
            mu2_best = mu2[mu2_ind]
        

#plotting best case convergence
fig, ax = plt.subplots(3, 1)
ax[0].plot(time, phi_u[0,], label='Phi U1')
ax[0].plot(time, phi_u[1,], label='Phi U2')
ax[0].plot(time, phi_u[2,], label='Phi U3')
ax[0].legend()
ax[0].set_title(f'Input rotation ({slope} rad/s)')
ax[0].set_ylabel('Rads')

for ind in range(nstages):
    ax[1].plot(time, code_ctrl_best[ind,], label=f'code_ctrl{ind}')
ax[1].legend()
ax[1].set_title(f'PS HDAC code (mu1 = {mu1_best:.2f}, mu2 = {mu2_best:.2f})')
ax[1].set_ylabel('Code')
ax[1].set_ylim(-2**(dac_res+0.5), 2**(dac_res+0.5))
ax[1].axhline(y=2**dac_res, color='r', linestyle='--')
ax[1].axhline(y=-2**dac_res, color='r', linestyle='--')

ax[2].plot(time, iloss_best, label='Loss')
ax[2].legend()
ax[2].set_title(f'Insertion Losst (max ss loss = {max_maxloss:.2f} dB)')
ax[2].set_ylabel('dB')
ax[2].set_xlabel('second')
plt.subplots_adjust(hspace=1)  

# plotting contour plot of all maxlosses
mu1_mesh, mu2_mesh = np.meshgrid(mu1, mu2, indexing='ij')

plt.figure()
contour_levels=[0.1, 0.25, 0.5, 1, 2.5, 5, 10]
contour_plot = plt.contour(mu1_mesh, mu2_mesh, -maxloss, norm=colors.LogNorm(), levels=contour_levels, cmap='viridis')
print(contour_plot.levels)
plt.scatter(mu1_best, mu2_best, color='red', marker='o', s=100, label=f'Best loss = {max_maxloss:.2f} dB')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('mu1')
plt.ylabel('mu2')
plt.title(f'Max loss contour plot for agressor (U{u_ind+1}, Slope = {slope} rad/s)')
plt.clabel(contour_plot, inline=True, fontsize=10)
plt.grid(which='both')
plt.show()


