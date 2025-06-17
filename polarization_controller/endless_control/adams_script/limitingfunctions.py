import numpy as np
import matplotlib.pyplot as plt
from scipy import special

phase = np.linspace(-np.pi,np.pi,100)

a2 = 1

# function1 = special.erfc(2 * (phase + alpha) / beta) + special.erfc(- 2 * (phase - alpha) / beta)
function1 = - a2 / np.pi * phase
function2 = - a2 * np.sin(0.5 * phase)
function3 = - a2 * np.tan(0.5 * phase)
function4 = - a2 / np.pi**3 * phase**3

fig, ax = plt.subplots(1, 1)
ax.plot(phase, function1, label='linear')
ax.plot(phase, function2, label='sine')
ax.plot(phase, function3, label='tangent')
ax.plot(phase, function4, label='cube')
ax.set_xlabel('Phase')
ax.set_ylabel('Normalized Error')
ax.set_ylim((-2, 2))
ax.legend()

plt.show()