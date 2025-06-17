import numpy as np
import matplotlib.pyplot as plt
from scipy import special


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

phi_ctrl = 0 * np.ones((2, dims))
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
dphase = np.pi / 128 # need to figure out a good way to set this
phase_update = 0
update_mode = 1 #(1 - linear limit function, 2 - erfc limit function, 3 - hard limit, 4 - no limit)

# Parameter for mode (1)
mu1 = 5
mu2 = 0.1

for ind in range(dims):

    for snum in range(2):
        if state % 2 == 1:
            phi_ctrl[snum, ind] = phi_ctrl[snum, ind - 2]
        else:
            phi_ctrl[snum, ind] = phi_ctrl[snum, ind - 1]
    if state == 1:
        phi_ctrl[1, ind] = phi_ctrl[1, ind-2] + phase_update
    if state == 2:
        phi_ctrl[0, ind] = phi_ctrl[0, ind-1] + dphase
    if state == 3:
        phi_ctrl[0, ind] = phi_ctrl[0, ind-2] + phase_update
    if state == 4:
        phi_ctrl[1, ind] = phi_ctrl[1, ind-1] + dphase
    
    # u1 rotation
    uout_te[0, ind] = input_te[ind] * np.exp(-1j * phi_u[0, ind] / 2)
    uout_tm[0, ind] = input_tm[ind] * np.exp(1j * phi_u[0, ind] / 2)
    #u2 rotation
    uout_te[1, ind] = uout_te[0, ind] * np.cos(phi_u[1, ind] / 2) - uout_tm[0, ind] * 1j * np.sin(phi_u[1, ind] / 2)
    uout_tm[1, ind] = -uout_te[0, ind] * 1j * np.sin(phi_u[1, ind] / 2) + uout_tm[0, ind] * np.cos(phi_u[1, ind] / 2)
    # u3 rotation
    uout_te[2, ind] = uout_te[1, ind] * np.cos(phi_u[2, ind] / 2) - uout_tm[1, ind] * np.sin(phi_u[2, ind] / 2)
    uout_tm[2, ind] = uout_te[1, ind] * np.sin(phi_u[2, ind] / 2) + uout_tm[1, ind] * np.cos(phi_u[2, ind] / 2)

    # convert phi_ctrl to phi_ps
    for sn in range(2):
        if phi_ctrl[sn, ind] >= 0:
            phi_ps[0, sn, ind] = phi_ctrl[sn, ind]
            phi_ps[1, sn, ind] = 0
        else:
            phi_ps[0, sn, ind] = 0
            phi_ps[1, sn, ind] = phi_ctrl[sn, ind]


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
            phase_update = - mu1 * np.sign(det_down[ind] - det_down[ind-1]) * det_down[ind-1] - mu2 / np.pi * phi_ctrl[int(state/2)-1, ind-1]
        elif update_mode == 1.1:
            phase_update = - mu1 * np.sign(det_down[ind] - det_down[ind-1]) * det_down[ind-1] - mu2 * np.sin(0.5 * phi_ctrl[int(state/2)-1, ind-1])
        elif update_mode == 1.2:
            phase_update = - mu1 * np.sign(det_down[ind] - det_down[ind-1]) * det_down[ind-1] - mu2 * np.tan(0.5 * phi_ctrl[int(state/2)-1, ind-1])
        elif update_mode == 1.3:
            phase_update = - mu1 * np.sign(det_down[ind] - det_down[ind-1]) * det_down[ind-1] - mu2 / np.pi**3 * phi_ctrl[int(state/2)-1, ind-1]**3
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

ax[2].plot(phi_ctrl[0,], label='phi_ctrl1')
ax[2].plot(phi_ctrl[1,], label='phi_ctrl2')
ax[2].legend()
ax[2].set_title('Phase shifter phases')
ax[2].set_ylabel('Rads')
ax[2].set_ylim(-2*np.pi,2*np.pi)

print(f'Min phase 1 = {np.min(phi_ctrl[0,int(dims/2):])}')
print(f'Max phase 1 = {np.max(phi_ctrl[0,int(dims/2):])}')
print(f'Min phase 2 = {np.min(phi_ctrl[1,int(dims/2):])}')
print(f'Max phase 2 = {np.max(phi_ctrl[1,int(dims/2):])}')

ax[3].plot(det_down, label='Det to minimize')
ax[3].plot(det_up, label='Det data')
ax[3].legend()
ax[3].set_title(f'Detector current (max ss loss = {maxloss:.2f} dB)')
ax[3].set_ylabel('A')
ax[3].set_xlabel('Cycle')
plt.subplots_adjust(hspace=1)

plt.show()




