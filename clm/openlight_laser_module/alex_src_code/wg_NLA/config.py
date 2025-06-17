import inspect

from .model import Waveguide_NLA

params = inspect.signature(Waveguide_NLA.__init__).parameters


class DefaultConfig:
    index = params['index'].default
    loss_rate = params['loss_rate'].default
    init_phase = params['init_phase'].default
