import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.interpolate import interp1d

# Import voltage to phase mapping 
dirname = '/Users/adam/Library/CloudStorage/GoogleDrive-adam@lightmatter.co/My Drive/Cadence_Data/Sim/ph45spclo_lmpdk_tb/hps_tb/' # this is hps
filename = 'phase_x_vheater.csv'
data = np.genfromtxt(dirname + filename, delimiter=',', skip_header=1)
vheat = data[:, 0]
phase = data[:, 1] / 180 * np.pi
vheat2phase = interp1d(vheat, phase, kind='linear')
# -------------------------------

# HDAC resolution
hdac_res = 13
code_range = 2**hdac_res
vmax = 1.65
vmin = 0.15
# ----------------

dims = 10000
input_te_pow = np.ones(dims)
input_te_angle = np.zeros(dims)
input_tm_pow = 0 * np.ones(dims)
input_tm_angle = np.zeros(dims)

input_te = input_te_pow * (np.cos(input_te_angle) + np.sin(input_te_angle) * 1j)
input_tm = input_tm_pow * (np.cos(input_tm_angle) + np.sin(input_tm_angle) * 1j)

# Linearly increasing inputs
phi_u = np.array([0 * np.ones(dims), 
                  0 * np.ones(dims),
                  np.concatenate((np.linspace(0,2*np.pi,int(dims/2)) * np.ones(int(dims/2)), 2*np.pi*np.ones(int(dims/2))))])

code_ps = 0 * np.ones((2, dims)) # HDAC code
voltage_ps = 0 * np.ones((2, 2, dims)) # heater voltage. Dimensions: (upper/lower branch, stage, time)
phi_ps = 0 * np.ones((2, 2, dims)) # phase setting. Dimensions: (upper/lower branch, stage, time)
uout_te = np.zeros((3,dims), dtype=complex)
uout_tm = np.zeros((3,dims), dtype=complex)

psout_up = np.zeros((2, dims), dtype=complex)
psout_down = np.zeros((2, dims), dtype=complex)
dcoupout_up = np.zeros((2, dims), dtype=complex)
dcoupout_down = np.zeros((2, dims), dtype=complex)
det_up = np.zeros(dims)
det_down = np.zeros(dims)

# Algorithm settings

state = 1 #state
dcode = 1 # need to figure out a good way to set this
code_update = 0
update_mode = 1 #(1 - linear limit function, 2 - no limit)

# Parameter for mode (1)
a1 = 0
a2 = 0

# Parameter for mode (2)
a1_2 = 100e-3

for ind in range(dims):

    for snum in range(2):
        if state % 2 == 1:
            code_ps[snum, ind] = code_ps[snum, ind - 2]
        else:
            code_ps[snum, ind] = code_ps[snum, ind - 1]
    if state == 1:
        if np.abs(code_ps[1, ind-2] + code_update) + dcode < code_range:
            code_ps[1, ind] = code_ps[1, ind-2] + code_update
        else:
            code_ps[1, ind] = code_ps[1, ind-2]
    if state == 2:
        code_ps[0, ind] = code_ps[0, ind-1] + dcode
    if state == 3:
        if np.abs(code_ps[0, ind-2] + code_update) + dcode < code_range:
            code_ps[0, ind] = code_ps[0, ind-2] + code_update
        else:
            code_ps[0, ind] = code_ps[0, ind-2]
    if state == 4:
        code_ps[1, ind] = code_ps[1, ind-1] + dcode
    
    # u1 rotation
    uout_te[0, ind] = input_te[ind] * np.exp(-1j * phi_u[0, ind] / 2)
    uout_tm[0, ind] = input_tm[ind] * np.exp(1j * phi_u[0, ind] / 2)
    #u2 rotation
    uout_te[1, ind] = uout_te[0, ind] * np.cos(phi_u[1, ind] / 2) - uout_tm[0, ind] * 1j * np.sin(phi_u[1, ind] / 2)
    uout_tm[1, ind] = -uout_te[0, ind] * 1j * np.sin(phi_u[1, ind] / 2) + uout_tm[0, ind] * np.cos(phi_u[1, ind] / 2)
    # u3 rotation
    uout_te[2, ind] = uout_te[1, ind] * np.cos(phi_u[2, ind] / 2) - uout_tm[1, ind] * np.sin(phi_u[2, ind] / 2)
    uout_tm[2, ind] = uout_te[1, ind] * np.sin(phi_u[2, ind] / 2) + uout_tm[1, ind] * np.cos(phi_u[2, ind] / 2)

    # Mapping DAC code to vheater and phase
    for ind2 in range(2):
        if code_ps[ind2, ind] >= 0: # positive code (and 0) activates the upper branch phase shifter
            voltage_ps[0, ind2, ind] = code_ps[ind2, ind] / code_range * (vmax - vmin) + vmin
            voltage_ps[1, ind2, ind] = 0
        else:                       # negative code activates the lower branch phase shifter
            voltage_ps[0, ind2, ind] = 0
            voltage_ps[1, ind2, ind] = (np.abs(code_ps[ind2, ind]) - 1) / code_range * (vmax - vmin) + vmin
        phi_ps[0, ind2, ind] = vheat2phase(voltage_ps[0, ind2, ind])
        phi_ps[1, ind2, ind] = vheat2phase(voltage_ps[1, ind2, ind])
    # ------------------------------------
    
    # 1st stage phase shifters
    psout_up[0, ind] = uout_tm[2, ind] * np.exp(1j * phi_ps[0, 0, ind])
    psout_down[0, ind] = uout_te[2, ind] * np.exp(1j * phi_ps[1, 0, ind])
    # 1st stage dcoupler
    dcoupout_up[0, ind] = 1 / np.sqrt(2) * (psout_up[0, ind] + 1j * psout_down[0, ind])
    dcoupout_down[0, ind] =  1 / np.sqrt(2) * (1j * psout_up[0, ind] + psout_down[0, ind])

    for s in range(1,2): 
        # phase shifters
        psout_up[s, ind] = dcoupout_up[s-1, ind] * np.exp(1j * phi_ps[0, s, ind])
        psout_down[s, ind] = dcoupout_down[s-1, ind] * np.exp(1j * phi_ps[1, s, ind])
        # dcoupler
        dcoupout_up[s, ind] = 1 / np.sqrt(2) * (psout_up[s, ind] + 1j * psout_down[s, ind])
        dcoupout_down[s, ind] =  1 / np.sqrt(2) * (1j * psout_up[s, ind] + psout_down[s, ind])

    det_up[ind] = dcoupout_up[1, ind] * np.conj(dcoupout_up[1, ind])
    det_down[ind] = dcoupout_down[1, ind] * np.conj(dcoupout_down[1, ind])

    if state % 2 == 0:
        if update_mode == 1:
            code_update = - a1 * np.sign(det_down[ind] - det_down[ind-1]) * det_down[ind-1] - a2 * code_ps[int(state/2)-1, ind-1]
        elif update_mode == 1.1:
            code_update = - a1 * np.sign(det_down[ind] - det_down[ind-1]) * det_down[ind-1] - a2 * np.sin(0.5 * code_ps[int(state/2)-1, ind-1])
        elif update_mode == 1.2:
            code_update = - a1 * np.sign(det_down[ind] - det_down[ind-1]) * det_down[ind-1] - a2 * np.tan(0.5 * code_ps[int(state/2)-1, ind-1])
        elif update_mode == 1.3:
            code_update = - a1 * np.sign(det_down[ind] - det_down[ind-1]) * det_down[ind-1] - a2 * code_ps[int(state/2)-1, ind-1]**3
        elif update_mode == 2:
            code_update = - a1_2 * np.sign(det_down[ind] - det_down[ind-1]) 
    code_update = np.round(code_update)
    print(code_update)

        
    if state < 4:
        state = state + 1
    else:
        state = 1

maxloss = 10 * np.log10(np.min(det_up[int(det_up.size/2):]))

'''
fig, ax = plt.subplots(2, 1)
ax[0].plot(np.real(psout_up[0,]), label='Real up branch')
ax[0].plot(np.imag(psout_up[0,]), label='Imag up branch')
ax[0].legend()
ax[1].plot(np.real(psout_down[0,]), label='Real down branch')
ax[1].plot(np.imag(psout_down[0,]), label='Imag down branch')
ax[1].legend()

fig, ax = plt.subplots(2, 1)
ax[0].plot(np.real(dcoupout_up[0,]), label='Real up branch')
ax[0].plot(np.imag(dcoupout_up[0,]), label='Imag up branch')
ax[0].legend()
ax[1].plot(np.real(dcoupout_down[0,]), label='Real down branch')
ax[1].plot(np.imag(dcoupout_down[0,]), label='Imag down branch')
ax[1].legend()

fig, ax = plt.subplots(2, 1)
ax[0].plot(np.real(psout_up[1,]), label='Real up branch')
ax[0].plot(np.imag(psout_up[1,]), label='Imag up branch')
ax[0].legend()
ax[1].plot(np.real(psout_down[1,]), label='Real down branch')
ax[1].plot(np.imag(psout_down[1,]), label='Imag down branch')
ax[1].legend()

fig, ax = plt.subplots(2, 1)
ax[0].plot(np.real(dcoupout_up[1,]), label='Real up branch')
ax[0].plot(np.imag(dcoupout_up[1,]), label='Imag up branch')
ax[0].legend()
ax[1].plot(np.real(dcoupout_down[1,]), label='Real down branch')
ax[1].plot(np.imag(dcoupout_down[1,]), label='Imag down branch')
ax[1].legend()
'''

fig, ax = plt.subplots(5, 1)
ax[0].plot(phi_u[0,], label='Phi U1')
ax[0].plot(phi_u[1,], label='Phi U2')
ax[0].plot(phi_u[2,], label='Phi U3')
ax[0].legend()
ax[0].set_title('Input rotation')
ax[0].set_ylabel('Rads')

ax[1].plot(code_ps[0,], label='PS code stage 1')
ax[1].plot(code_ps[1,], label='PS code stage 2')
ax[1].legend()
ax[1].set_title('HDAC codes')
ax[1].set_ylabel('bit')

ax[2].plot(voltage_ps[0, 0,], label='Upper voltage_ps1')
ax[2].plot(voltage_ps[1, 0,], label='Lower voltage_ps1')
ax[2].plot(voltage_ps[0, 1,], label='Upper voltage_ps2')
ax[2].plot(voltage_ps[1, 1,], label='Lower voltage_ps2')
ax[2].legend()
ax[2].set_title('HDAC voltage')
ax[2].set_ylabel('V')

ax[3].plot(phi_ps[0, 0,], label='Upper phi_ps1')
ax[3].plot(phi_ps[1, 0,], label='Lower phi_ps1')
ax[3].plot(phi_ps[0, 1,], label='Upper phi_ps2')
ax[3].plot(phi_ps[1, 1,], label='Lower phi_ps2')
ax[3].legend()
ax[3].set_title('Phase shifter phases')
ax[3].set_ylabel('Rads')
# ax[3].set_ylim(-np.pi,np.pi)



ax[4].plot(det_down, label='Det to minimize')
ax[4].plot(det_up, label='Det data')
ax[4].legend()
ax[4].set_title(f'Detector current (max ss loss = {maxloss:.2f} dB)')
ax[4].set_ylabel('A')
ax[4].set_xlabel('Cycle')

'''
fig, ax = plt.subplots(5, 1)
ax[0].plot(code_ps[0,])
ax[1].plot(voltage_ps[0, 0,])
ax[2].plot(voltage_ps[1, 0,])
ax[3].plot(phi_ps[0, 0,])
ax[4].plot(phi_ps[1, 0,])
'''
plt.subplots_adjust(hspace=1)

plt.show()




