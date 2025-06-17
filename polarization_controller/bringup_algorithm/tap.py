import numpy as np

from lmphoton import OptElement

class Tap(OptElement):
    def __init__(self, tap_ratio = 0.01, ports=['1l', '2r', '3r'], name='TAP'):
        self.tap_ratio = self._genvar(tap_ratio)
        super().__init__(ports, name=name)

    def _construct_smatrix(self):
        tap = np.sqrt(self.tap_ratio)
        through = np.sqrt(1-self.tap_ratio)
        return [[0.0, tap, through],
                [tap, 0.0, 0.0],
                [through, 0.0, 0.0]]