import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.interpolate import interp1d

def endlesspc(ps_file, numOfPSperStage, dacres, vmax, vmin, adcres, iref, nstages, dither, update_mode, mu1, mu2, phi_u):
    # Import voltage to phase mapping 
    # dirname = '/Users/adam/Library/CloudStorage/GoogleDrive-adam@lightmatter.co/My Drive/Cadence_Data/Sim/ph45spclo_lmpdk_tb/hps_tb/' # this is hps
    ps_file = 'phase_x_vheater.csv'
    data = np.genfromtxt(ps_file, delimiter=',', skip_header=1)
    vheat = data[:, 0]
    #    ------------------------------- 

    # TODO - throw an error if phi_u doesn't have (3, N) dimensions

    # Algorithm setup -------------
    state = 1 #state
    dcode =  dither * int(np.pi * vmax / np.pi * (2**dacres) / (vmax - vmin)) # need to figure out a good way to set this
    code_update = 0
    print(f'dcode = {dcode}')
    dims = phi_u.shape[1]
    phase = data[:, 1] / 180 * np.pi * numOfPSperStage # add extra factor of 2 to double the range
    vheat2phase = interp1d(vheat, phase, kind='linear', fill_value='extrapolate')

    mu1 = mu1 * (vmax / np.pi * (2**dacres) / (vmax - vmin) / 2**adcres)
    mu2 = mu2 

    input_te_pow = np.ones(dims)
    input_te_angle = np.zeros(dims)
    input_tm_pow = 0 * np.ones(dims)
    input_tm_angle = np.zeros(dims)
    input_te = input_te_pow * (np.cos(input_te_angle) + np.sin(input_te_angle) * 1j)
    input_tm = input_tm_pow * (np.cos(input_tm_angle) + np.sin(input_tm_angle) * 1j)

    code_ctrl = 0 * np.ones((nstages, dims))
    code_ps = 0 * np.ones((2, nstages, dims))
    voltage_ps = 0 * np.ones((2, nstages, dims))
    phi_ps = 0 * np.ones((2, nstages, dims)) # (upper/lower branch ps, stage, time)

    uout_te = np.zeros((3,dims), dtype=complex)
    uout_tm = np.zeros((3,dims), dtype=complex)

    psout_up = np.zeros((nstages, dims), dtype=complex)
    psout_down = np.zeros((nstages, dims), dtype=complex)
    dcoupout_up = np.zeros((nstages, dims), dtype=complex)
    dcoupout_down = np.zeros((nstages, dims), dtype=complex)
    det_up = np.zeros(dims)
    det_down = np.zeros(dims)
    min_det = np.zeros(dims)
    max_det = np.zeros(dims)
    iadc_code = np.zeros(dims)

    for ind in range(dims):
    
        for snum in range(nstages):
            if state % 2 == 1:
                code_ctrl[snum, ind] = code_ctrl[snum, ind - 2]
            else:
                code_ctrl[snum, ind] = code_ctrl[snum, ind - 1]

        if state == 1:
            code_ctrl[nstages - 1, ind] = np.minimum(2**dacres - dcode, np.maximum(code_ctrl[nstages - 1, ind-2] + code_update, -2**dacres))
        elif state % 2 == 0:
            code_ctrl[state // 2 - 1, ind] = code_ctrl[state // 2 - 1, ind-1] + dcode
        else:
            code_ctrl[state // 2 - 1, ind] = np.minimum(2**dacres - dcode, np.maximum(code_ctrl[state // 2 - 1, ind-2] + code_update, -2**dacres))
    
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
        for sn in range(nstages):
            if code_ctrl[sn, ind] >= 0:
                code_ps[0, sn, ind] = code_ctrl[sn, ind]
                code_ps[1, sn, ind] = 0
            else:
                code_ps[0, sn, ind] = 0
                code_ps[1, sn, ind] = -code_ctrl[sn, ind]

            voltage_ps[0, sn, ind] = code_ps[0, sn, ind] / 2**dacres * (vmax - vmin) + vmin
            voltage_ps[1, sn, ind] = code_ps[1, sn, ind] / 2**dacres * (vmax - vmin) + vmin
            phi_ps[0, sn, ind] = vheat2phase(voltage_ps[0, sn, ind])
            phi_ps[1, sn, ind] = vheat2phase(voltage_ps[1, sn, ind])

        # 1st stage phase shifters
        psout_up[0, ind] = uout_tm[2, ind] * np.exp(1j * phi_ps[0, 0, ind])
        psout_down[0, ind] = uout_te[2, ind] * np.exp(1j * phi_ps[1, 0, ind])
        # 1st stage dcoupler
        dcoupout_up[0, ind] = 1 / np.sqrt(2) * (psout_up[0, ind] + 1j * psout_down[0, ind])
        dcoupout_down[0, ind] =  1 / np.sqrt(2) * (1j * psout_up[0, ind] + psout_down[0, ind])

        for s in range(1, nstages): 
            # phase shifters
            psout_up[s, ind] = dcoupout_up[s-1, ind] * np.exp(1j * phi_ps[0, s, ind])
            psout_down[s, ind] = dcoupout_down[s-1, ind] * np.exp(1j * phi_ps[1, s, ind])
            # dcoupler
            dcoupout_up[s, ind] = 1 / np.sqrt(2) * (psout_up[s, ind] + 1j * psout_down[s, ind])
            dcoupout_down[s, ind] =  1 / np.sqrt(2) * (1j * psout_up[s, ind] + psout_down[s, ind])

        det_up[ind] = dcoupout_up[nstages - 1, ind] * np.conj(dcoupout_up[nstages - 1, ind])
        det_down[ind] = dcoupout_down[nstages - 1, ind] * np.conj(dcoupout_down[nstages - 1, ind])

        # iadc conversion
        if nstages == 2:
            min_det[ind] = det_down[ind]
            max_det[ind] = det_up[ind]
        else:
            min_det[ind] = det_up[ind]
            max_det[ind] = det_down[ind]

        iadc_code[ind] = int(min_det[ind] / iref * 2**adcres)

        if state % 2 == 0:
            if update_mode == 1:
                code_update = - mu1 * np.sign(iadc_code[ind] - iadc_code[ind-1] + 0.1) * iadc_code[ind-1] - mu2  / np.pi * code_ctrl[int(state/2)-1, ind-1]
            elif update_mode == 1.1:
                code_update = - mu1 * np.sign(iadc_code[ind] - iadc_code[ind-1] + 0.1) * iadc_code[ind-1] - mu2 * np.sin(1/1.65 * code_ctrl[int(state/2)-1, ind-1])
            elif update_mode == 1.2:
                code_update = - mu1  * np.sign(iadc_code[ind] - iadc_code[ind-1] + 0.1) * iadc_code[ind-1] - mu2 * np.tan(1/1.65 * code_ctrl[int(state/2)-1, ind-1])
            elif update_mode == 1.3:
                code_update = - mu1  * np.sign(iadc_code[ind] - iadc_code[ind-1] + 0.1) * iadc_code[ind-1] - mu2 / np.pi**3 * code_ctrl[int(state/2)-1, ind-1]**3
            code_update = int(code_update)
        if state < nstages * 2:
            state = state + 1
        else:
            state = 1

    iloss = 10 * np.log10(max_det)

    return code_update, iloss
