# Test out some questions related to MRM

import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

c = 299792458

lambda0 = 1311e-9
neff0 = 2.5755
ng0 = 3.97

#MRM parameters
dL = 0.5*60e-9
L0 = 2*np.pi*6.6e-6
L = L0+2*dL

dfrac = 0.5
sigma1 = np.sqrt(0.9)
sigma2 = np.sqrt(0.97)
a = np.sqrt(0.9)

afrac = a**dfrac

#Excitation parameters
Ein = np.sqrt(1e-3)
lambda_sweep = np.arange(1300e-9,1320e-9,1e-12)

##########################################

def find_center_FSR(dips_wavel: list, lambdaTarget: float) -> tuple[float, int]:
  id_l = 0
  for i in range(len(dips_wavel)-1):
    if dips_wavel[i] <= lambdaTarget and dips_wavel[i+1] > lambdaTarget:
        id_l = i
        break

  return (dips_wavel[id_l+1] - dips_wavel[id_l], id_l)

##########################################

dndlambda = (neff0-ng0)/lambda0
neff = neff0+dndlambda*(lambda_sweep-lambda0)

phi = 2*np.pi/lambda_sweep*neff*L
phifrac = phi*dfrac

Ethru = Ein*np.exp(1j*(np.pi+phi))*(sigma2*a-sigma1*np.exp(-1j*phi))/(1-sigma1*sigma2*a*np.exp(1j*phi))
Edrop = Ein*np.sqrt(1-sigma1**2)*np.sqrt(1-sigma2**2)*afrac*np.exp(1j*phifrac)/(1-sigma1*sigma2*a*np.exp(1j*phi))

Pin = np.abs(Ein)**2
Pthru = np.abs(Ethru)**2
Pdrop = np.abs(Edrop)**2
Pcavity_max = (1-sigma1**2)/(1+sigma1**2*sigma2**2*a**2-2*sigma1*sigma2*a*np.cos(phi))*Pin
Pcavity_min = (1-sigma1**2)*sigma2**2*a**2/(1+sigma1**2*sigma2**2*a**2-2*sigma1*sigma2*a*np.cos(phi))*Pin
Pcavityavg = (1-sigma1**2)*(afrac**2-1+sigma2*a**2-sigma2*afrac**2)/(1+sigma1**2*sigma2**2*a**2-2*sigma1*sigma2*a*np.cos(phi))/(2*np.log(a))*Pin

Tthru = Pthru/Pin
Tdrop = Pdrop/Pin

Tbuildup = Pcavityavg/Pin

#Tthru = (sigma1**2+sigma2**2*a**2-2*sigma1*sigma2*a*np.cos(phi))/(1-2*sigma1*sigma2*a*np.cos(phi)+sigma1**2*sigma2**2*a**2)
#Tdrop = (1-sigma1**2)*(1-sigma2**2)*a/(1-2*sigma1*sigma2*a*np.cos(phi)+sigma1**2*sigma2**2*a**2)

##########################################

T = Tthru
filter_thres = (np.max(T) + np.min(T))/2
dips_idx, _ = find_peaks(np.negative(T))
dips_wavel = lambda_sweep[dips_idx]

(FSR,dip_idx) = find_center_FSR(dips_wavel,lambda0)
FSR_lambda = (lambda_sweep[dips_idx[dip_idx]]+lambda_sweep[dips_idx[dip_idx+1]])/2
FSR_Hz = FSR*c/FSR_lambda**2
idx_offres = (dips_idx[dip_idx]+dips_idx[dip_idx+1])//2

lambda_res = lambda_sweep[dips_idx[dip_idx]]

Tmax = T[idx_offres]
Tmin = T[dips_idx[dip_idx]]
ER_dB = 10*np.log10(Tmax/Tmin)

Thalf = (Tmin+Tmax)/2
lambdaTruncate = lambda_sweep[dips_idx[dip_idx]:idx_offres]
Ttruncate = T[dips_idx[dip_idx]:idx_offres]

FWHM = 2*(np.interp(Thalf,Ttruncate,lambdaTruncate)-lambda_res)

T = Tdrop
Tmax = T[dips_idx[dip_idx]]
Tmin = T[idx_offres]
ERdrop_dB = 10*np.log10(Tmax/Tmin)

Thalf = (Tmin+Tmax)/2
print(Thalf)

lambdaTruncate = lambda_sweep[dips_idx[dip_idx]:idx_offres]
Ttruncate = T[dips_idx[dip_idx]:idx_offres]

FWHMdrop= 2*(np.interp(-Thalf,-Ttruncate,lambdaTruncate)-lambda_res)
FWHM_calc = (1-sigma1*sigma2*a)*lambda_res**2/(np.pi*ng0*L*np.sqrt(sigma1*sigma2*a))

Finesse = FSR/FWHM
Qfactor = lambda_res/FWHM

print(FSR_Hz*1e-9)

print(Finesse/(np.pi))
print(Qfactor)

print(lambda_res*1e9)

##########################################

figure1 = plt.figure(1,(10,5))
plt.plot(lambda_sweep*1e9,Tthru,label="Thru",linewidth=2.5)
plt.plot(lambda_sweep*1e9,Tdrop,label="Drop",linewidth=2.5)
plt.xlabel('Wavelength (nm)',fontsize=16)
plt.ylabel('Normalized Power', fontsize=16)
plt.title('Drop Port Ring Transmission',fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid()
plt.legend()
plt.savefig("figs/Spectrum.png",dpi=600)

figure2 = plt.figure(2,(10,5))
plt.plot(lambdaTruncate*1e9,Ttruncate,label="Thru",linewidth=2.5)
plt.xlabel('Wavelength (nm)',fontsize=16)
plt.ylabel('Normalized Power', fontsize=16)
plt.title('Drop Port Ring Transmission',fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid()
plt.legend()
plt.savefig("figs/SpectrumTruncate.png",dpi=600)

figure3 = plt.figure(3,(10,5))
plt.plot(lambda_sweep*1e9,Tbuildup,label="Buildup avg",linewidth=2.5)
plt.plot(lambda_sweep*1e9,Pcavity_max/Pin,label="Buildup max",linewidth=2.5)
plt.plot(lambda_sweep*1e9,Pcavity_min/Pin,label="Buildup min",linewidth=2.5)
plt.xlabel('Wavelength (nm)',fontsize=16)
plt.ylabel('Normalized Power', fontsize=16)
plt.title('Cavity Buildup Factor',fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid()
plt.legend()
plt.savefig("figs/SpectrumBuildup.png",dpi=600)

#plt.show()

Tau = 0.95
TO = 1.86e-4

print(lambda0/(Tau*TO*L))

Rbar = 10
Pin = 4e-3
PC = Rbar*Pin*8*2*np.pi*6.7e-6
print(PC)
