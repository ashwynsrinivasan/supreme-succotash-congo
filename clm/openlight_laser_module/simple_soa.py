import numpy as np
from scipy.optimize import newton

def dB2val(x):
  return 10**(x/10)

def val2dB(x):
  return 10*np.log10(x)

class SOA():
  def __init__(self,T, J, L, wl):
    self.T = T
    self.J = J
    self.L = L
    self.wl = wl

    # self._check_inputs()
    self._gain_peak()
    self._wavelength_peak()
    self._FWHM()
    self._Pos_3dB()
    self._g0()
    self._Psat()

  def Vdiode(W=2.7):
    J = self.J
    L_SOA = self.L
    Idiode = J *1e3*  W * (L_SOA + 460) * 1e-8 # [A]
    return (0.98 + 2.8*Idiode * (1100/(L_SOA+460)), Idiode)

  def _check_inputs(self, wavelength=None):
    T_ranges = [35, 80]  # Temperature ranges in C
    J_ranges = [3, 7]  # Current ranges in kA/cm^2
    L_ranges = [40, 440]  # Gain length ranges in µm
    wav_ranges = [1304, 1318]  # wavelength ranges in nm
    if self.T:
      assert np.all(self.T >= min(T_ranges)) and np.all(self.T <= max(T_ranges))
    if self.J:
      assert np.all(self.J >= min(J_ranges)) and np.all(self.J <= max(J_ranges))
    if self.L:
      assert np.all(self.L >= min(L_ranges)) and np.all(self.L <= max(L_ranges))
    if self.wl:
      assert np.all(self.wl >= min(wav_ranges)) and np.all(self.wl <= max(wav_ranges))
    pass

  def _gain_peak(self):
    # outputs peak gain in dB
    wav = self.wl
    T = self.T
    J = self.J
    L = self.L

    output = 4.678 -0.0729* T + 10.098* np.log(J)- 0.001380 *(L+460) \
    -0.00024 *(T - 60) *(T - 60) - 0.0081*np.log(J) *(T - 60) - 2.158* np.log(J)* np.log(J) \
    -0.0001589 *(T - 60) *(L - 240) + 0.02311 *np.log(J) *(L - 240) \
    -0.000001886* (T -60)* (T-60) *(L - 240) \
    -0.00002088 *np.log(J)* (T- 60)* (L - 240) \
    -0.005336* np.log(J) *np.log(J) *(L - 240)

    self.gain_peak = output

  def _wavelength_peak(self):
    # outputs peak wavelength in nm
    T = self.T
    J = self.J
    L = self.L

    output = 1273.73 + 0.6817* T - 28.73* np.log(J)+ 0.01362 *(L + 460) \
        + 0.004585 *(T - 60)* (T - 60) - 0.1076 *np.log(J)* (T - 60) + 8.787* np.log(J)* np.log(J) \
        + 0.00004185 *(T - 60)* (L - 240) - 0.02367* np.log(J) *(L - 240) \
        - 0.0000002230 *(T - 60)* (T - 60)* (L - 240) \
        + 0.000136* np.log(J)* (T - 60) *(L - 240) + 0.004894 *np.log(J)* np.log(J) *(L - 240)

    self.wavelength_peak = output
  
  def _FWHM(self):
    # output FWHM in nm
    T = self.T
    J = self.J
    L = self.L

    output = 120.15 - 0.08555* T + 0.3837* np.log(J) - 0.07255 *(L + 460) \
    + 0.00007784 *(T - 60) *(T - 60) + 0.2386 *np.log(J) *(T - 60) + 2.759 *np.log(J)* np.log(J) \
    - 0.0004342* (T - 60)* (L - 240) + 0.003947* np.log(J)* (L - 240) \
    +0.00002085*(T - 60) *(T - 60) *(L - 240) \
    +0.000009466 *np.log(J) *(T - 60) *(L - 240) \
    -0.0007991*np.log(J) *np.log(J)* (L - 240)

    self.FWHM = output

  
  def _Pos_3dB(self):
    # Output power saturation (Pos) in dBm
    # outputs 3dB saturated power in dBm
    wav = self.wl
    T = self.T
    J = self.J
    L = self.L

    output = -74.08+ 0.06226*wav - 0.008877*T + 0.994*J + \
    -0.08721*(J - 4.571)* (J - 4.571) + 0.01752*(wav - 1310.8)* (wav - 1310.8) \
    -0.00002341*(T - 60.07)*(T - 60.07) - 0.001266*(wav - 1310.8)*(T- 60.07) \
    -0.001763* (T - 60.07)*(J- 4.571) - 0.008584*(wav - 1310.8)*(J- 4.571)

    self.Pos_3dB = output

  def Lorentzian(self,x, x0, fwhm):
      denom = (x - x0)**2 + (fwhm/2) **2
      return fwhm / denom # Outputs are in units of 1/nm

  def _g0(self):
      f = self.Lorentzian(self.wl, self.wavelength_peak, self.FWHM)
      self.g0 = f * 10**(self.gain_peak/10) / (4 / self.FWHM) # Linear, unitless output
  
  def noise_figure(self):
    # outputs noise figure in dB
    wav = self.wl
    T = self.T
    J = self.J
    L = self.L

    output = 131.58 + -0.09959*wav + 0.08972*T -5.0895*np.log(J) \
    + 2.7334*np.log(J)*np.log(J) + 0.0009195 *(wav - 1306.38)* (wav - 1306.38) \
    + 0.0007484*(T - 60)*(T - 60) -0.001299*(wav - 1306.38)*(T - 60) \
    - 0.07995*(T - 60)*np.log(J) + 0.103*(wav-1306.38)*np.log(J) \
    + 0.0005740*(wav - 1306.38)*(T - 60)*np.log(J) \
    + 0.0197*np.log(J)*np.log(J)*(T - 60) - 0.02785*np.log(J)*np.log(J)*(wav - 1306.38) \
    -0.0003141*(T - 60)*(T - 60)*np.log(J) -0.00001095*(T - 60)*(T - 60)*(wav - 1306.38) \
    -0.0002678*(wav - 1306.38)*(wav-1306.38)*np.log(J) \
    + 0.000003281*(wav-1306.38)*(wav - 1306.38)*(T - 60) \
    -0.4606*np.log(J)*np.log(J)*np.log(J) - 0.000002634*(wav - 1306.38)*(wav - 1306.38)*(wav - 1306.38)

    self.NF = output

  def _Psat(self): #Calculates Ps on page 6 of tower's document
      # Ps_3dB is the output power 3dB saturation in dBm
      # g0 is the unsaturated gain, linear + unitless
      Ps_3dB_ = dB2val(self.Pos_3dB)  # in mW
      # g0_ = dB2val(g0)  # unitless
      g0_ = self.g0
      self.Psat = Ps_3dB_ * (g0_-2) / (g0_*np.log(2)) #in mW
      self.Psat = self.Psat * 1e-3 #Psat in W
      # return val2dB()  # back to dBm

  def gain(self, Pin):
      # g0 is unsaturated gain, linear and unitless
      # Pin is input laser power in W
      # Ps is the saturation power value, in W
      # g0_ = dB2val(g0)
      # g0_ = self._g0()
      # Pin_ = dB2val(Pin)
      # Ps_ = self.dB2val(self.Psat)
      # Ps_ = 
      def f(g):
          return g - self.g0 * np.exp( (1-g) * Pin/self.Psat)

      def fprime(g):
          z = Pin/self.Psat
          return 1 + self.g0 * z * np.exp(z*(1-g))

      return newton(f, self.g0, fprime=fprime, maxiter=10000)
