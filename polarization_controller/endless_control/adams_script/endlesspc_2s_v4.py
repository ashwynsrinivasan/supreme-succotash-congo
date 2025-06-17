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
vheat2phase = interp1d(vheat, phase, kind='linear', fill_value='extrapolate')
# -------------------------------

# DAC specs
dacres = 13
# --------

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
                  np.linspace(0,4*np.pi,dims) * np.ones(dims) * np.ones(dims)])

code_ctrl = 0 * np.ones((2, dims))
code_ps = 0 * np.ones((2, 2, dims))
voltage_ps = 0 * np.ones((2, 2, dims))
phi_ps = 0 * np.ones((2, 2, dims)) # (upper/lower branch ps, stage, time)

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
dcode = np.pi / 128 / np.pi * (2**dacres)# need to figure out a good way to set this
code_update = 0
update_mode = 1 #(1 - linear limit function, 2 - erfc limit function, 3 - hard limit, 4 - no limit)

# Parameter for mode (1)
mu1 = 5 / np.pi / 2 * (2**dacres)
mu2 = 0.1

for ind in range(dims):
    for snum in range(2):
        if state % 2 == 1:
            code_ctrl[snum, ind] = code_ctrl[snum, ind - 2]
        else:
            code_ctrl[snum, ind] = code_ctrl[snum, ind - 1]
    if state == 1:
        code_ctrl[1, ind] = code_ctrl[1, ind-2] + code_update
    if state == 2:
        code_ctrl[0, ind] = code_ctrl[0, ind-1] + dcode
    if state == 3:
        code_ctrl[0, ind] = code_ctrl[0, ind-2] + code_update
    if state == 4:
        code_ctrl[1, ind] = code_ctrl[1, ind-1] + dcode
    
    # u1 rotation
    uout_te[0, ind] = input_te[ind] * np.exp(-1j * phi_u[0, ind] / 2)
    uout_tm[0, ind] = input_tm[ind] * np.exp(1j * phi_u[0, ind] / 2)
    #u2 rotation
    uout_te[1, ind] = uout_te[0, ind] * np.cos(phi_u[1, ind] / 2) - uout_tm[0, ind] * 1j * np.sin(phi_u[1, ind] / 2)
    uout_tm[1, ind] = -uout_te[0, ind] * 1j * np.sin(phi_u[1, ind] / 2) + uout_tm[0, ind] * np.cos(phi_u[1, ind] / 2)
    # u3 rotation
    uout_te[2, ind] = uout_te[1, ind] * np.cos(phi_u[2, ind] / 2) - uout_tm[1, ind] * np.sin(phi_u[2, ind] / 2)
    uout_tm[2, ind] = uout_te[1, ind] * np.sin(phi_u[2, ind] / 2) + uout_tm[1, ind] * np.cos(phi_u[2, ind] / 2)

    # convert code_ctrl to phi_ps
    for sn in range(2):
        if code_ctrl[sn, ind] >= 0:
            code_ps[0, sn, ind] = code_ctrl[sn, ind]
            code_ps[1, sn, ind] = 0
            voltage_ps[0, sn, ind] = code_ps[0, sn, ind] / 2**dacres * 1.65
            voltage_ps[1, sn, ind] = code_ps[1, sn, ind] / 2**dacres * 1.65
            phi_ps[0, sn, ind] = vheat2phase(voltage_ps[0, sn, ind])
            phi_ps[1, sn, ind] = vheat2phase(voltage_ps[1, sn, ind])
        else:
            code_ps[0, sn, ind] = 0
            code_ps[1, sn, ind] = -code_ctrl[sn, ind]
            voltage_ps[0, sn, ind] = code_ps[0, sn, ind] / 2**dacres * 1.65
            voltage_ps[1, sn, ind] = code_ps[1, sn, ind] / 2**dacres * 1.65
            phi_ps[0, sn, ind] = vheat2phase(voltage_ps[0, sn, ind])
            phi_ps[1, sn, ind] = vheat2phase(voltage_ps[1, sn, ind])


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
            code_update = - mu1 * np.sign(det_down[ind] - det_down[ind-1]) * det_down[ind-1] - mu2 / np.pi * code_ctrl[int(state/2)-1, ind-1]
        elif update_mode == 1.1:
            code_update = - mu1 * np.sign(det_down[ind] - det_down[ind-1]) * det_down[ind-1] - mu2 * np.sin(1/1.65 * code_ctrl[int(state/2)-1, ind-1])
        elif update_mode == 1.2:
            code_update = - mu1 * np.sign(det_down[ind] - det_down[ind-1]) * det_down[ind-1] - mu2 * np.tan(1/1.65 * code_ctrl[int(state/2)-1, ind-1])
        elif update_mode == 1.3:
            code_update = - mu1 * np.sign(det_down[ind] - det_down[ind-1]) * det_down[ind-1] - mu2 / np.pi**3 * code_ctrl[int(state/2)-1, ind-1]**3
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

fig, ax = plt.subplots(4, 1)
ax[0].plot(phi_u[0,], label='Phi U1')
ax[0].plot(phi_u[1,], label='Phi U2')
ax[0].plot(phi_u[2,], label='Phi U3')
ax[0].legend()
ax[0].set_title('Input rotation')
ax[0].set_ylabel('Rads')

ax[1].plot(np.real(uout_te[2, ]), label='Real TE')
ax[1].plot(np.imag(uout_te[2, ]), label='Imag TE')
ax[1].plot(np.real(uout_tm[2, ]), label='Real TM')
ax[1].plot(np.imag(uout_tm[2, ]), label='Imag TM')
ax[1].legend()
ax[1].set_title('Input TE and TM components')
ax[1].set_ylabel('V/m')

ax[2].plot(code_ctrl[0,], label='code_ctrl1')
ax[2].plot(code_ctrl[1,], label='code_ctrl2')
ax[2].legend()
ax[2].set_title('Phase shifter HDAC ctrl code')
ax[2].set_ylabel('Code')
ax[2].set_ylim(-2**dacres, 2**dacres)

print(f'Min phase 1 = {np.min(code_ctrl[0,int(dims/2):])}')
print(f'Max phase 1 = {np.max(code_ctrl[0,int(dims/2):])}')
print(f'Min phase 2 = {np.min(code_ctrl[1,int(dims/2):])}')
print(f'Max phase 2 = {np.max(code_ctrl[1,int(dims/2):])}')

ax[3].plot(det_down, label='Det to minimize')
ax[3].plot(det_up, label='Det data')
ax[3].legend()
ax[3].set_title(f'Detector current (max ss loss = {maxloss:.2f} dB)')
ax[3].set_ylabel('A')
ax[3].set_xlabel('Cycle')
plt.subplots_adjust(hspace=1)

plt.show()




