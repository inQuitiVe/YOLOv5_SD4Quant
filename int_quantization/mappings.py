
from torch import nn

import torch.nn.intrinsic as nni
import torch.nn.intrinsic.quantized as nniq
import torch.nn.intrinsic.qat as nniqat
import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
import torch.nn.qat as nnqat
import int_quantization.nn as lognn

from torch.quantization.stubs import QuantStub, DeQuantStub

# Map for swapping float module to log quantization supported ones
LOG_MODULE_MAPPING = {
    nn.Linear: lognn.Linear,
    nn.ReLU: nnq.ReLU,
    nn.ReLU6: nnq.ReLU6,
    nn.Hardswish: nnq.Hardswish,
    nn.ELU: nnq.ELU,
    nn.Conv1d: lognn.Conv1d,
    nn.Conv2d: lognn.Conv2d,
    nn.Conv3d: lognn.Conv3d,
    nn.BatchNorm2d: nnq.BatchNorm2d,
    nn.BatchNorm3d: nnq.BatchNorm3d,
    nn.LayerNorm: nnq.LayerNorm,
    nn.GroupNorm: nnq.GroupNorm,
    nn.InstanceNorm1d: nnq.InstanceNorm1d,
    nn.InstanceNorm2d: nnq.InstanceNorm2d,
    nn.InstanceNorm3d: nnq.InstanceNorm3d,
    QuantStub: nnq.Quantize,
    DeQuantStub: nnq.DeQuantize,
    # Wrapper Modules:
    nnq.FloatFunctional: nnq.QFunctional,
    # Intrinsic modules:
    nni.ConvReLU1d: lognn.ConvReLU1d,
    nni.ConvReLU2d: lognn.ConvReLU2d,
    nni.ConvReLU3d: lognn.ConvReLU3d,
    nni.LinearReLU: nniq.LinearReLU,
    nni.BNReLU2d: nniq.BNReLU2d,
    nni.BNReLU3d: nniq.BNReLU3d,
    nniqat.ConvReLU2d: lognn.ConvReLU2d,
    nniqat.LinearReLU: nniq.LinearReLU,
    nniqat.ConvBn2d: lognn.Conv2d,
    nniqat.ConvBnReLU2d: lognn.ConvReLU2d,
    # QAT modules:
    nnqat.Linear: lognn.Linear,
    nnqat.Conv2d: lognn.Conv2d,
}
