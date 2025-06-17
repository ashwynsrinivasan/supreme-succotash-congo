import numpy as np
import matplotlib.pyplot as plt

dims = (10000)
input_te_pow = np.ones(dims)
input_te_angle = np.zeros(dims)
input_tm_pow = np.zeros(dims)
input_tm_angle = np.zeros(dims)

input_te = input_te_pow * (np.cos(input_te_angle) + np.sin(input_te_angle) * 1j)
input_tm = input_tm_pow * (np.cos(input_tm_angle) + np.sin(input_tm_angle) * 1j)

phi_u1 = np.linspace(0,4*np.pi,dims) * np.ones(dims)
phi_u2 = 0 * np.ones(dims)
phi_u3 = np.linspace(0,16*np.pi,dims) * np.ones(dims)
phi_ps1 = 0 * np.ones(dims)
phi_ps2 = 0 * np.ones(dims)
phi_ps3 = 0 * np.ones(dims)
phi_ps4 = 0 * np.ones(dims)


u1out_te = np.zeros(dims)+0j
u1out_tm = np.zeros(dims)+0j
u2out_te = np.zeros(dims)+0j
u2out_tm = np.zeros(dims)+0j
u3out_te = np.zeros(dims)+0j
u3out_tm = np.zeros(dims)+0j
ps1out_up = np.zeros(dims)+0j
ps1out_down = np.zeros(dims)+0j
dcoup1out_up = np.zeros(dims)+0j
dcoup1out_down = np.zeros(dims)+0j
ps2out_up = np.zeros(dims)+0j
ps2out_down = np.zeros(dims)+0j
dcoup2out_up = np.zeros(dims)+0j
dcoup2out_down = np.zeros(dims)+0j
ps3out_up = np.zeros(dims)+0j
ps3out_down = np.zeros(dims)+0j
dcoup3out_up = np.zeros(dims)+0j
dcoup3out_down = np.zeros(dims)+0j
ps4out_up = np.zeros(dims)+0j
ps4out_down = np.zeros(dims)+0j
dcoup4out_up = np.zeros(dims)+0j
dcoup4out_down = np.zeros(dims)+0j
det_up = np.zeros(dims)
det_down = np.zeros(dims)

state = 1 #state
dphase = np.pi / 128 # need to figure out a good way to set this
phase_update = 0
a1 = 1000e-3
a2 = 50e-3

for ind in range(dims):

    if state == 1:
        phi_ps1[ind] = phi_ps1[ind-2]
        phi_ps2[ind] = phi_ps2[ind-2]
        phi_ps3[ind] = phi_ps3[ind-2]
        phi_ps4[ind] = phi_ps4[ind-2] + phase_update
    if state == 2:
        phi_ps1[ind] = phi_ps1[ind-1] + dphase
        phi_ps2[ind] = phi_ps2[ind-1]
        phi_ps3[ind] = phi_ps3[ind-1]
        phi_ps4[ind] = phi_ps4[ind-1]
    if state == 3:
        phi_ps1[ind] = phi_ps1[ind-2] + phase_update
        phi_ps2[ind] = phi_ps2[ind-2]
        phi_ps3[ind] = phi_ps3[ind-2]
        phi_ps4[ind] = phi_ps4[ind-2]
    if state == 4:
        phi_ps1[ind] = phi_ps1[ind-1]
        phi_ps2[ind] = phi_ps2[ind-1] + dphase
        phi_ps3[ind] = phi_ps3[ind-1]
        phi_ps4[ind] = phi_ps4[ind-1]
    if state == 5:
        phi_ps1[ind] = phi_ps1[ind-2]
        phi_ps2[ind] = phi_ps2[ind-2] + phase_update
        phi_ps3[ind] = phi_ps3[ind-2]
        phi_ps4[ind] = phi_ps4[ind-2]
    if state == 6:
        phi_ps1[ind] = phi_ps1[ind-1]
        phi_ps2[ind] = phi_ps2[ind-1]
        phi_ps3[ind] = phi_ps3[ind-1] + dphase
        phi_ps4[ind] = phi_ps4[ind-1]
    if state == 7:
        phi_ps1[ind] = phi_ps1[ind-2]
        phi_ps2[ind] = phi_ps2[ind-2] 
        phi_ps3[ind] = phi_ps3[ind-2] + phase_update
        phi_ps4[ind] = phi_ps4[ind-2]       
    if state == 8:
        phi_ps1[ind] = phi_ps1[ind-1]
        phi_ps2[ind] = phi_ps2[ind-1]
        phi_ps3[ind] = phi_ps3[ind-1]
        phi_ps4[ind] = phi_ps4[ind-1] + dphase

    # u1 rotation
    u1out_te[ind] = input_te[ind] * np.exp(-1j * phi_u1[ind] / 2)
    u1out_tm[ind] = input_tm[ind] * np.exp(1j * phi_u1[ind] / 2)
    # u2 rotation
    u2out_te[ind] = u1out_te[ind] * np.cos(phi_u2[ind] / 2) - u1out_tm[ind] * 1j * np.sin(phi_u2[ind] / 2)
    u2out_tm[ind] = - u1out_te[ind] * 1j * np.sin(phi_u2[ind] / 2) + u1out_tm[ind] * np.cos(phi_u2[ind] / 2)
    # u3 rotation
    u3out_te[ind] = u2out_te[ind] * np.cos(phi_u3[ind] / 2) - u2out_tm[ind] * np.sin(phi_u3[ind] / 2)
    u3out_tm[ind] = u2out_te[ind] * np.sin(phi_u3[ind] / 2) + u2out_tm[ind] * np.cos(phi_u3[ind] / 2)

    # 1st set of phase shifters
    ps1out_up[ind] = u3out_tm[ind] * np.exp(1j * np.max((0, phi_ps1[ind])))
    ps1out_down[ind] = u3out_te[ind] * np.exp(1j * np.max((-phi_ps1[ind], 0)))
    print(np.max((-phi_ps1[ind], 0)))
    # dcoupler1
    dcoup1out_up[ind] = 1 / np.sqrt(2) * (ps1out_up[ind] + 1j * ps1out_down[ind])
    dcoup1out_down[ind] =  1 / np.sqrt(2) * (1j * ps1out_up[ind] + ps1out_down[ind])
    # 2nd set of phase shifters
    ps2out_up[ind] = dcoup1out_up[ind] * np.exp(1j * np.max((0, phi_ps2[ind])))
    ps2out_down[ind] = dcoup1out_down[ind] * np.exp(1j * np.max((-phi_ps2[ind], 0)))
    # dcoupler2
    dcoup2out_up[ind] = 1 / np.sqrt(2) * (ps2out_up[ind] + 1j * ps2out_down[ind])
    dcoup2out_down[ind] =  1 / np.sqrt(2) * (1j * ps2out_up[ind] + ps2out_down[ind])
    # 3rd set of phase shifters
    ps3out_up[ind] = dcoup2out_up[ind] * np.exp(1j * np.max((0, phi_ps3[ind])))
    ps3out_down[ind] = dcoup2out_down[ind] * np.exp(1j * np.max((-phi_ps3[ind], 0)))
    # dcoupler2
    dcoup3out_up[ind] = 1 / np.sqrt(2) * (ps3out_up[ind] + 1j * ps3out_down[ind])
    dcoup3out_down[ind] =  1 / np.sqrt(2) * (1j * ps3out_up[ind] + ps3out_down[ind])
    # 4th set of phase shifters
    ps4out_up[ind] = dcoup3out_up[ind] * np.exp(1j * np.max((0, phi_ps4[ind])))
    ps4out_down[ind] = dcoup3out_down[ind] * np.exp(1j * np.max((-phi_ps4[ind], 0)))
    # dcoupler4
    dcoup4out_up[ind] = 1 / np.sqrt(2) * (ps4out_up[ind] + 1j * ps4out_down[ind])
    dcoup4out_down[ind] =  1 / np.sqrt(2) * (1j * ps4out_up[ind] + ps4out_down[ind])

    det_up[ind] = dcoup4out_up[ind] * np.conj(dcoup4out_up[ind])
    det_down[ind] = dcoup4out_down[ind] * np.conj(dcoup4out_down[ind])

    if state == 2:
        phase_update = - a1 * np.sign(det_up[ind] - det_up[ind-1]) * det_up[ind-1] - a2 * phi_ps1[ind-1]
    if state == 4:
        phase_update = - a1 * np.sign(det_up[ind] - det_up[ind-1]) * det_up[ind-1] - a2 * phi_ps2[ind-1]
    if state == 6:
        phase_update = - a1 * np.sign(det_up[ind] - det_up[ind-1]) * det_up[ind-1] - a2 * phi_ps3[ind-1]
    if state == 8:
        phase_update = - a1 * np.sign(det_up[ind] - det_up[ind-1]) * det_up[ind-1] - a2 * phi_ps4[ind-1]


    if state < 8:
        state = state + 1
    else:
        state = 1



test = np.max((0,-1))

#fig, ax = plt.subplots(2, 1)


#fig, ax = plt.subplots(2, 1)
#ax[0].plot(np.real(ps1out_up), label='Real up branch')
#ax[0].plot(np.imag(ps1out_up), label='Imag up branch')
#ax[0].legend()
#ax[1].plot(np.real(ps1out_down), label='Real down branch')
#ax[1].plot(np.imag(ps1out_down), label='Imag down branch')
#ax[1].legend()

#fig, ax = plt.subplots(2, 1)
#ax[0].plot(np.real(dcoup4out_up), label='Real up branch')
#ax[0].plot(np.imag(dcoup4out_up), label='Imag up branch')
#ax[0].legend()
#ax[1].plot(np.real(dcoup4out_down), label='Real down branch')
#ax[1].plot(np.imag(dcoup4out_down), label='Imag down branch')
#ax[1].legend()

fig, ax = plt.subplots(4, 1)
ax[0].plot(phi_u1, label='Phi U1')
ax[0].plot(phi_u2, label='Phi U2')
ax[0].plot(phi_u3, label='Phi U3')
ax[0].legend()
ax[0].set_title('Input rotation')
ax[0].set_ylabel('Rads')

ax[1].plot(np.real(u3out_te), label='Real TE')
ax[1].plot(np.imag(u3out_te), label='Imag TE')
ax[1].plot(np.real(u3out_tm), label='Real TM')
ax[1].plot(np.imag(u3out_tm), label='Imag TM')
ax[1].legend()
ax[1].set_title('Input TE and TM components')
ax[1].set_ylabel('V/m')

ax[2].plot(phi_ps1, label='phi_ps1')
ax[2].plot(phi_ps2, label='phi_ps2')
ax[2].plot(phi_ps3, label='phi_ps3')
ax[2].plot(phi_ps4, label='phi_ps4')
ax[2].legend()
ax[2].set_title('Phase shifter phases')
ax[2].set_ylabel('Rads')

ax[3].plot(det_up, label='Det to minimize')
ax[3].plot(det_down, label='Det data')
ax[3].legend()
ax[3].set_title('Detector current')
ax[3].set_ylabel('A')
ax[3].set_xlabel('Cycle')
plt.subplots_adjust(hspace=0.5)

plt.show()


