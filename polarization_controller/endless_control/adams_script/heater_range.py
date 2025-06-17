import numpy as np
import matplotlib.pyplot as plt

N = 13
vref = 1.65
offset = 0.15

code = np.arange(2**13)
voltage = np.sqrt(code) / 2**(N / 2) * vref

fig, ax = plt.subplots(1, 1)
ax.plot(code, voltage)