from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch.nn import Module
from .observer import MovingAverageMinMaxObserver, HistogramObserver, MovingAveragePerChannelMinMaxObserver, PercentileObserver, _with_args
from .quant import Log_Preprocess_cpu, Log_Preprocess_gpu

class FakeQuantize(Module):
    r""" Simulate the quantize and dequantize operations in training time.
    The output of this module is given by

    x_out = (clamp(round(x/scale + zero_point), quant_min, quant_max)-zero_point)*scale



    * :attr:`scale` defines the scale factor used for quantization.

    * :attr:`zero_point` specifies the quantized value to which 0 in floating point maps to

    * :attr:`quant_min` specifies the minimum allowable quantized value.

    * :attr:`quant_max` specifies the maximum allowable quantized value.

    * :attr:`fake_quant_enable` controls the application of fake quantization on tensors, note that
      statistics can still be updated.

    * :attr:`observer_enable` controls statistics collection on tensors

    * :attr:`dtype` specifies the quantized dtype that is being emulated with fake-quantization,
                    allowable values are torch.qint8 and torch.quint8. The values of quant_min and
                    quant_max should be chosen to be consistent with the dtype


    Args:
        observer (module): Module for observing statistics on input tensors and calculating scale
                           and zero-point.
        quant_min (int): The minimum allowable quantized value.
        quant_max (int): The maximum allowable quantized value.
        observer_kwargs (optional): Arguments for the observer module

    Attributes:
        observer (Module): User provided module that collects statistics on the input tensor and
                           provides a method to calculate scale and zero-point.

    """
    def __init__(self, observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255, **observer_kwargs):
        super(FakeQuantize, self).__init__()
        assert quant_min <= quant_max, \
            'quant_min must be less than or equal to quant_max'
        self.quant_min = quant_min
        self.quant_max = quant_max
        # fake_quant_enabled and observer_enabled are buffers to support their
        # replication in DDP. Data type is uint8 because NCCL does not support
        # bool tensors.
        self.register_buffer('fake_quant_enabled', torch.tensor([1], dtype=torch.uint8))
        self.register_buffer('observer_enabled', torch.tensor([1], dtype=torch.uint8))
        self.activation_post_process = observer(**observer_kwargs)
        self.log_scale = self.activation_post_process.log_scale
        #assert torch.iinfo(self.activation_post_process.dtype).min <= quant_min, 'quant_min out of bound'
        #assert quant_max <= torch.iinfo(self.activation_post_process.dtype).max, 'quant_max out of bound'
        self.register_buffer('scale', torch.tensor([1.0]))
        self.register_buffer('zero_point', torch.tensor([0]))
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.ch_axis = self.activation_post_process.ch_axis \
            if hasattr(self.activation_post_process, 'ch_axis') else -1

    @torch.jit.export
    def enable_fake_quant(self, enabled=True):
        # type: (bool) -> FakeQuantize
        self.fake_quant_enabled[0] = 1 if enabled else 0
        return self

    @torch.jit.export
    def disable_fake_quant(self):
        return self.enable_fake_quant(False)

    @torch.jit.export
    def enable_observer(self, enabled=True):
        # type: (bool) -> FakeQuantize
        #print(enabled)
        self.observer_enabled[0] = 1 if enabled else 0
        return self

    @torch.jit.export
    def disable_observer(self):
        return self.enable_observer(False)

    @torch.jit.export
    def calculate_qparams(self):
        return self.activation_post_process.calculate_qparams()

    @torch.jit.ignore
    def sync_observer(self):
        if (torch.distributed.is_initialized()):
            torch.distributed.broadcast(self.scale, 0)
            torch.distributed.broadcast(self.zero_point, 0)

    def forward(self, X):
        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach()) #update observer statistics
            _scale, _zero_point = self.calculate_qparams()
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            self.scale.resize_(_scale.shape)
            self.scale.copy_(_scale)
            self.zero_point.resize_(_zero_point.shape)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled[0] == 1:
            if (self.log_scale):
                X = X.float()
                # X = Log_Preprocess_gpu(X, self.scale)
                X = Log_Preprocess_cpu(X, self.scale)
                X = X.half()

            if self.qscheme == torch.per_channel_symmetric or self.qscheme == torch.per_channel_affine:
                X = X.float()
                X = torch.fake_quantize_per_channel_affine(X, self.scale, self.zero_point,
                                                           self.ch_axis, self.quant_min, self.quant_max)
                X = X.half()
            else:
                X = X.float()
                X = torch.fake_quantize_per_tensor_affine(X, float(self.scale),
                                                          int(self.zero_point), self.quant_min,
                                                          self.quant_max)
                X = X.half()
        return X

    with_args = classmethod(_with_args)

    @torch.jit.export
    def extra_repr(self):
        return 'fake_quant_enabled={}, observer_enabled={},\
            scale={}, zero_point={}'.format(
            self.fake_quant_enabled, self.observer_enabled,
            self.scale, self.zero_point)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # We cannot currently register scalar values as buffers, so need to manually
        # specify serialization here.
        super(FakeQuantize, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = self.scale
        destination[prefix + 'zero_point'] = self.zero_point

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Removing this function throws an error that the the size of the loaded tensor does not match the original size
        # i.e., These buffers start out with numel 0 and become numel 1 once they have their first forward pass.
        local_state = ['scale', 'zero_point']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                setattr(self, name, val)
            elif strict:
                missing_keys.append(key)
        super(FakeQuantize, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                        missing_keys, unexpected_keys, error_msgs)

default_fake_quant = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=254,
                                            dtype=torch.quint8, qscheme=torch.per_tensor_symmetric, bitwidth=8)

default_int4_fake_quant = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=14,
                                                 dtype=torch.quint8, qscheme=torch.per_tensor_affine, bitwidth=4)

default_log4_fake_quant = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=128,
                                                 dtype=torch.quint8, qscheme=torch.per_tensor_symmetric, bitwidth=4, log_scale=True)

default_weight_fake_quant = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=-7, quant_max=7,
                                                   dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, bitwidth=4)

default_per_channel_weight_fake_quant = FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver,
                                                               quant_min=-7,
                                                               quant_max=7,
                                                               dtype=torch.qint8,
                                                               qscheme=torch.per_channel_symmetric,
                                                               bitwidth=4,
                                                               ch_axis=0)

default_histogram_fake_quant = FakeQuantize.with_args(observer=HistogramObserver,
                                                      quant_min=0,
                                                      quant_max=126,
                                                      dtype=torch.quint8,
                                                      qscheme=torch.per_tensor_affine,
                                                      bitwidth=7)

default_percentile_fake_quant = FakeQuantize.with_args(observer=PercentileObserver, quant_min=0, quant_max=254,
                                                       dtype=torch.quint8, qscheme=torch.per_tensor_symmetric, bitwidth=8)

default_int4_percentile_fake_quant = FakeQuantize.with_args(observer=PercentileObserver, quant_min=0, quant_max=14, percentile=0.999,
                                                            dtype=torch.quint8, qscheme=torch.per_tensor_affine, bitwidth=4)

default_log_weight_fake_quant = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=-64, quant_max=64,
                                                       dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, bitwidth=4, log_scale=True)

default_per_channel_log_weight_fake_quant = FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver,
                                                                   quant_min=-64,
                                                                   quant_max=64,
                                                                   dtype=torch.qint8,
                                                                   qscheme=torch.per_channel_symmetric,
                                                                   bitwidth=4,
                                                                   log_scale=True,
                                                                   ch_axis=0)

def disable_fake_quant(mod):
    if type(mod) in set([FakeQuantize, torch.quantization.FakeQuantize]):
        mod.disable_fake_quant()

def enable_fake_quant(mod):
    if type(mod) in set([FakeQuantize, torch.quantization.FakeQuantize]):
        mod.enable_fake_quant()

def disable_observer(mod):
    if type(mod) in set([FakeQuantize, torch.quantization.FakeQuantize]):
        mod.disable_observer()

def enable_observer(mod):
    if type(mod) in set([FakeQuantize, torch.quantization.FakeQuantize]):
        mod.enable_observer()

def sync_observer(mod):
    if type(mod) in set([FakeQuantize]):
        mod.sync_observer()

def disable_act_fake_quant(mod):
    if type(mod) in set([FakeQuantize, torch.quantization.FakeQuantize]):
        if (mod.activation_post_process.dtype == torch.quint8):
            mod.disable_fake_quant()

def enable_act_fake_quant(mod):
    if type(mod) in set([FakeQuantize, torch.quantization.FakeQuantize]):
        if (mod.activation_post_process.dtype == torch.quint8):
            mod.enable_fake_quant()

def disable_act_observer(mod):
    if type(mod) in set([FakeQuantize, torch.quantization.FakeQuantize]):
        if (mod.activation_post_process.dtype == torch.quint8):
            mod.disable_observer()

def enable_act_observer(mod):
    if type(mod) in set([FakeQuantize, torch.quantization.FakeQuantize]):
        if (mod.activation_post_process.dtype == torch.quint8):
            mod.enable_observer()

def disable_weight_fake_quant(mod):
    if type(mod) in set([FakeQuantize, torch.quantization.FakeQuantize]):
        if (mod.activation_post_process.dtype == torch.qint8):
            mod.disable_fake_quant()

def enable_weight_fake_quant(mod):
    if type(mod) in set([FakeQuantize, torch.quantization.FakeQuantize]):
        if (mod.activation_post_process.dtype == torch.qint8):
            mod.enable_fake_quant()

def disable_weight_observer(mod):
    if type(mod) in set([FakeQuantize, torch.quantization.FakeQuantize]):
        if (mod.activation_post_process.dtype == torch.qint8):
            mod.disable_observer()

def enable_weight_observer(mod):
    if type(mod) in set([FakeQuantize, torch.quantization.FakeQuantize]):
        if (mod.activation_post_process.dtype == torch.qint8):
            mod.enable_observer()
