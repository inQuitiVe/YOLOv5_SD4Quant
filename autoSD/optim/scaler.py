import torch
from collections import OrderedDict
import sys

__all__ = ["LossScaler"]


class LossScaler(object):
    def __init__(self,
                 dynamic=True,
                 init_scale=2**15,
                 scale_factor=2.,
                 scale_window=1000,
                 max_scale=2**31,
                 tolerance=0.):
        self._dynamic = dynamic
        self._loss_scale = init_scale
        self._scale_window = scale_window
        self._scale_factor = scale_factor
        self._max_loss_scale = max_scale
        self._has_overflow = False
        self._tolerance = tolerance
        self._iter = 0
        self._last_overflow_iter = -1
        self._last_rescale_iter = -1
        self._overflows_since_rescale = 0

        assert (tolerance >= 0 and tolerance < 1), "Tolerance should be between 0 and 1."

    def unscale(self, grads):
        for grad in grads:
            if grad is not None:
                self._has_overflow = self._scale_check_overflow_python(grad, 1./self._loss_scale, self._dynamic)
                if self._has_overflow and self._dynamic:
                    break

    def _scale_check_overflow_python(self, grad, scale, check_overflow=False):
        # Exception handling for 18.04 compatibility
        if check_overflow:
            cpu_sum = float(grad.float().sum())
            if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
                return True

        if scale != 1.0:
            grad.mul_(scale)

        return False

    def update_scale(self, update=True):
        if (self._has_overflow):
            should_skip = True
            if (self._dynamic and update):
                iter_since_rescale = self._iter - self._last_rescale_iter
                self._last_overflow_iter = self._iter
                self._overflows_since_rescale += 1
                overflow_pct = self._overflows_since_rescale / float(iter_since_rescale)
                if (overflow_pct >= self._tolerance): #decrease scaling factor
                    self._loss_scale = max(1, self._loss_scale / self._scale_factor)
                    print('Overflow! New scale = {}.'.format(self._loss_scale))
                    self._last_rescale_iter = self._iter
                    self._overflows_since_rescale = 0
        else:
            should_skip = False
            if (self._dynamic and update):
                if ((self._iter - self._last_overflow_iter) % self._scale_window == 0): #increase scaling factor
                    new_scale = min(self._max_loss_scale, self._loss_scale * self._scale_factor)
                    if (new_scale != self._loss_scale):
                        self._loss_scale = new_scale
                        print('New scale = {}.'.format(self._loss_scale))
                    self._last_rescale_iter = self._iter

        if (update):
            self._iter += 1

        if (self._loss_scale == 1.):
            print('Loss scaling factor error.')
            sys.exit(1)

        return should_skip

    def loss_scale(self):
        return self._loss_scale

    def state_dict(self, destination=None):
        if destination is None:
            destination = OrderedDict()
        destination['loss_scale'] = self._loss_scale
        destination['iter'] = self._iter
        destination['last_overflow_iter'] = self._last_overflow_iter
        destination['last_rescale_iter'] = self._last_rescale_iter
        destination['overflows_since_rescale'] = self._overflows_since_rescale
        return destination

    def load_state_dict(self, state_dict):
        state_dict = state_dict.copy()
        self._loss_scale = state_dict['loss_scale']
        self._iter = state_dict['iter']
        self._last_overflow_iter = state_dict['last_overflow_iter']
        self._last_rescale_iter = state_dict['last_rescale_iter']
        self._overflows_since_rescale = state_dict['overflows_since_rescale']

    def __repr__(self):
        return ("Loss scaler:\n"
                "   dynamic loss scaling: {}\n"
                "   scale factor: {}\n"
                "   scale window: {}\n"
                "   maximum scale: {}\n"
                "   tolerance: {}"
                ).format(self._dynamic, self._scale_factor, self._scale_window, self._max_loss_scale, self._tolerance)

    def __str__(self):
        return ("Loss scaler:\n"
                "   dynamic loss scaling: {}\n"
                "   scale factor: {}\n"
                "   scale window: {}\n"
                "   maximum scale: {}\n"
                "   tolerance: {}"
                ).format(self._dynamic, self._scale_factor, self._scale_window, self._max_loss_scale, self._tolerance)