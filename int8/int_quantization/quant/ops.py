import torch
from torch.utils.cpp_extension import load
import os

current_path = os.path.dirname(os.path.realpath(__file__))

if torch.cuda.is_available():
    quant_cuda = load(
            name='ops_cuda',
            sources=[
                os.path.join(current_path, "quant_cuda/quant_cuda.cpp"),
                os.path.join(current_path, "quant_cuda/bit_helper.cu"),
                os.path.join(current_path, "quant_cuda/float_kernel.cu"),
                os.path.join(current_path, "quant_cuda/quant.cu"),
            ]
    )

__all__ = ["Log_Preprocess_cpu", "Log_Preprocess_gpu"]

class Log_Preprocess(torch.autograd.Function):
    #only support channel_axis = 0
    @staticmethod
    def forward(ctx, input, scale): #w' = s * 2**(round(log2(w/s)))
        expand_shape = [-1]
        dims = len(input.shape)
        expand_shape += [1] * (dims-1)
        expand_scale = scale.view(*expand_shape)
        
        sign = torch.sign(input)
        input_new = input.abs() / expand_scale
        idx_0 = (input_new == 0)
        input_new[idx_0] = 1 #tmp value to prevent log(0)
        input_new = 2**(torch.round(input_new.log2())) * expand_scale
        input_new[idx_0] = 0
        input_new = input_new * sign

        return input_new

    @staticmethod
    def backward(ctx, grad_output): #straight through estimator
        grad_input = grad_output.clone()
        return grad_input, None

class Log_Preprocess_cuda(torch.autograd.Function):
    #only support channel_axis = 0
    @staticmethod
    def forward(ctx, input, scale): #w' = s * 2**(round(log2(w/s)))
        expand_shape = [-1]
        dims = len(input.shape)
        expand_shape += [1] * (dims-1)
        expand_scale = scale.view(*expand_shape)

        return quant_cuda.fake_log_quantization(input/expand_scale) * expand_scale

    @staticmethod
    def backward(ctx, grad_output): #straight through estimator
        grad_input = grad_output.clone()
        return grad_input, None

class Squeeze_clip(torch.autograd.Function):
    #only support channel_axis = 0
    @staticmethod
    def forward(ctx, input, scale): #w' = s * 2**(round(log2(w/s)))
        expand_shape = [-1]
        dims = len(input.shape)
        expand_shape += [1] * (dims-1)
        expand_scale = scale.view(*expand_shape)

        threshold = expand_scale * 32 #squeeze = 0.5

        upper_idx = input > threshold
        lower_idx = input < (-1*threshold)
        clip_idx = ~(upper_idx | lower_idx)

        input[clip_idx] = 0
        ctx.save_for_backward(clip_idx)
        return input

    @staticmethod
    def backward(ctx, grad_output): #straight through estimator
        clip_idx, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[clip_idx] = 0
        return grad_input, None

Log_Preprocess_cpu = Log_Preprocess.apply
Log_Preprocess_gpu = Log_Preprocess_cuda.apply
Squeeze_clipping = Squeeze_clip.apply