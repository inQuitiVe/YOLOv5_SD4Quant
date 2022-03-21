import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.cpp_extension import load
import os

current_path = os.path.dirname(os.path.realpath(__file__))

if torch.cuda.is_available():
    quant_cuda = load(
            name='quant_cuda',
            sources=[
                os.path.join(current_path, "quant_cuda/quant_cuda.cpp"),
                os.path.join(current_path, "quant_cuda/bit_helper.cu"),
                os.path.join(current_path, "quant_cuda/float_kernel.cu"),
                os.path.join(current_path, "quant_cuda/quant.cu"),
            ]
    )

__all__ = ["quantizer"]

def quantizer(forward_rounding="identity", backward_rounding="identity", 
              forward_fp_offset=24, backward_fp_offset=24, 
              forward_fp_structure=[4, 3], backward_fp_structure=[5, 2], 
              verbose=True, adaptive_offset=False, adaptive_structure=False, 
              channel_wise=False):

    for rounding in [forward_rounding, backward_rounding]:
        assert rounding in ["identity", "floatsd8", "floatsd4", "floatsd4_ex", "fp"], "Invalid rounding type {:s}".format(rounding)

    assert (torch.cuda.is_available()), "Only support cuda quantizer."

    for rounding, offset, fp_struct in zip([forward_rounding, backward_rounding], 
                                           [forward_fp_offset, backward_fp_offset],
                                           [forward_fp_structure, backward_fp_structure]):
        if (rounding == "fp"):
            exp, mantissa = fp_struct
            assert (8 >= exp >= 0), "Exponent should be in [0,8]."
            assert (23 >= mantissa >= 0), "Mantissa should in [0,23]."
            assert (exp+mantissa > 0), "Exp + mantissa should > 0."

    fwd_message = None
    bwd_message = None
    exp_fwd, mantissa_fwd = forward_fp_structure
    exp_bwd, mantissa_bwd = backward_fp_structure
    #forward
    if (forward_rounding == "floatsd8"): #floatsd8
        fwd_message = "  Forward: floatsd8 dynamic quantizer"
        forward_quant = lambda x, bias: quant_cuda.float_quantize_floatsd8_dynamic(x, bias)

    elif (forward_rounding == "floatsd4"): #floatsd4 (13 possible value)
        fwd_message = "  Forward: floatsd4 dynamic quantizer"
        forward_quant = lambda x, bias: quant_cuda.float_quantize_floatsd4_dynamic(x, bias)

    elif (forward_rounding == "floatsd4_ex"): #floatsd4_ex (15 possible value)
        if (channel_wise):
            fwd_message = '  Forward: floatsd4_ex channel-wise dynamic quantizer'
            forward_quant = lambda x, bias: quant_cuda.float_quantize_floatsd4_ex_cwise_dynamic(x, bias)
        else:
            fwd_message = '  Forward: floatsd4_ex dynamic quantizer'
            forward_quant = lambda x, bias: quant_cuda.float_quantize_floatsd4_ex_dynamic(x, bias)

    elif (forward_rounding == "fp"): #fp => 'clip' version of quantization function is used
        if (channel_wise):
            if (adaptive_structure):
                fwd_message = '  Forward: fp(adaptive) channel-wise dynamic quantizer'.format(exp_fwd, mantissa_fwd)
                if (adaptive_offset):
                    forward_quant = lambda x, bias, exp, mantissa: quant_cuda.float_quantize_fp_clip_cwise_dynamic(x, bias, exp_fwd, mantissa_fwd)
                else:
                    forward_quant = lambda x, exp, mantissa: quant_cuda.float_quantize_fp_clip_cwise_dynamic(x, forward_fp_offset, exp_fwd, mantissa_fwd)
            else:
                fwd_message = '  Forward: fp(e:{}/m:{}) channel-wise dynamic quantizer'.format(exp_fwd, mantissa_fwd)
                if (adaptive_offset):
                    forward_quant = lambda x, bias: quant_cuda.float_quantize_fp_clip_cwise_dynamic(x, bias, exp_fwd, mantissa_fwd)
                else:
                    forward_quant = lambda x: quant_cuda.float_quantize_fp_clip_cwise_dynamic(x, forward_fp_offset, exp_fwd, mantissa_fwd)
        #not channel-wise
        elif (adaptive_structure):
            fwd_message = '  Forward: fp(adaptive) dynamic quantizer'.format(exp_fwd, mantissa_fwd)
            if (adaptive_offset):
                forward_quant = lambda x, bias, exp, mantissa: quant_cuda.float_quantize_fp_clip_dynamic(x, bias, exp, mantissa)
            else:
                forward_quant = lambda x, exp, mantissa: quant_cuda.float_quantize_fp_clip_dynamic(x, forward_fp_offset, exp, mantissa)
        #for [5,2] / [4,3] / [3,4], exclusive cuda function is used (faster)
        elif (forward_fp_structure == [5,2]):
            fwd_message = '  Forward: fp8(152) dynamic quantizer'
            if (adaptive_offset):
                forward_quant = lambda x, bias: quant_cuda.float_quantize_fp8_152_clip_nearest_dynamic(x, bias)
            else:
                forward_quant = lambda x: quant_cuda.float_quantize_fp8_152_clip_nearest_dynamic(x, forward_fp_offset)

        elif (forward_fp_structure == [4,3]):
            fwd_message = '  Forward: fp8(143) dynamic quantizer'
            if (adaptive_offset):
                forward_quant = lambda x, bias: quant_cuda.float_quantize_fp8_143_clip_nearest_dynamic(x, bias)
            else:
                forward_quant = lambda x: quant_cuda.float_quantize_fp8_143_clip_nearest_dynamic(x, forward_fp_offset)

        elif (forward_fp_structure == [3,4]):
            fwd_message = '  Forward: fp8(134) dynamic quantizer'
            if (adaptive_offset):
                forward_quant = lambda x, bias: quant_cuda.float_quantize_fp8_134_clip_nearest_dynamic(x, bias)
            else:
                forward_quant = lambda x: quant_cuda.float_quantize_fp8_134_clip_nearest_dynamic(x, forward_fp_offset)
        
        else:
            fwd_message = '  Forward: fp(e:{}/m:{}) dynamic quantizer'.format(exp_fwd, mantissa_fwd)
            if (adaptive_offset):
                forward_quant = lambda x, bias: quant_cuda.float_quantize_fp_clip_dynamic(x, bias, exp_fwd, mantissa_fwd)
            else:
                forward_quant = lambda x: quant_cuda.float_quantize_fp_clip_dynamic(x, forward_fp_offset, exp_fwd, mantissa_fwd)

    else: #identity
        fwd_message = '  Forward: identity'
        forward_quant = lambda x: x
        
    #backward
    if (backward_rounding == "floatsd8"): #floatsd8
        bwd_message = '  Backward: floatsd8 dynamic quantizer'
        backward_quant = lambda x: quant_cuda.float_quantize_floatsd8_dynamic(x, backward_fp_offset)

    elif (backward_rounding == "floatsd4"): #floatsd4
        bwd_message = '  Backward: floatsd4 dynamic quantizer'
        backward_quant = lambda x: quant_cuda.float_quantize_floatsd4_dynamic(x, backward_fp_offset)

    elif (backward_rounding == "floatsd4_ex"): #floatsd4_ex
        bwd_message = '  Backward: floatsd4_ex dynamic quantizer'
        backward_quant = lambda x: quant_cuda.float_quantize_floatsd4_ex_dynamic(x, backward_fp_offset)

    elif (backward_rounding == "fp"): #fp => 'non clip' version is used to detect loss scaling overflow
        if (backward_fp_structure == [5,2]):
            bwd_message = '  Backward: fp8(152) dynamic quantizer'
            backward_quant = lambda x: quant_cuda.float_quantize_fp8_152_nearest_dynamic(x, backward_fp_offset)

        elif (backward_fp_structure == [4,3]):
            bwd_message = '  Backward: fp8(143) dynamic quantizer'
            backward_quant = lambda x: quant_cuda.float_quantize_fp8_143_nearest_dynamic(x, backward_fp_offset)

        elif (backward_fp_structure == [3,4]):
            bwd_message = '  Backward: fp8(134) dynamic quantizer'
            backward_quant = lambda x: quant_cuda.float_quantize_fp8_134_nearest_dynamic(x, backward_fp_offset)
        
        else:
            bwd_message = '  Backward: fp(e:{}/m:{}) dynamic quantizer'.format(exp_bwd, mantissa_bwd)
            backward_quant = lambda x: quant_cuda.float_quantize_fp_dynamic(x, backward_fp_offset, exp_bwd, mantissa_bwd)

    else: #identity
        bwd_message = '  Backward: identity'
        backward_quant = lambda x: x

    #print message
    if (verbose):
        print(fwd_message)
        print(bwd_message)

    #create autograd function
    if (adaptive_structure and forward_rounding == "fp"):
        if (adaptive_offset):
            class Rounding(torch.autograd.Function):
                @staticmethod
                def forward(self, x, bias, exp, mantissa):                   
                    out = forward_quant(x.contiguous(), bias, exp, mantissa)
                    return out

                @staticmethod
                def backward(self, grad_output):
                    if self.needs_input_grad[0]:
                        grad_input = backward_quant(grad_output.contiguous())
                    else:
                        grad_input = None
                    return grad_input, None, None, None
        else:
            class Rounding(torch.autograd.Function):
                @staticmethod
                def forward(self, x, exp, mantissa):                   
                    out = forward_quant(x.contiguous(), exp, mantissa)
                    return out

                @staticmethod
                def backward(self, grad_output):
                    if self.needs_input_grad[0]:
                        grad_input = backward_quant(grad_output.contiguous())
                    else:
                        grad_input = None
                    return grad_input, None, None

    elif ((forward_rounding not in ["fp", "identity"]) or (forward_rounding == "fp" and adaptive_offset)): #sd4 / sd4_ex / sd8 / fp with adaptive offset
        class Rounding(torch.autograd.Function):
            @staticmethod
            def forward(self, x, bias):                   
                out = forward_quant(x.contiguous(), bias)
                return out

            @staticmethod
            def backward(self, grad_output):
                if self.needs_input_grad[0]:
                    grad_input = backward_quant(grad_output.contiguous())
                else:
                    grad_input = None
                return grad_input, None

    else: #normal => fixed offset / structure
        class Rounding(torch.autograd.Function):
            @staticmethod
            def forward(self, x):
                out = forward_quant(x.contiguous())
                return out

            @staticmethod
            def backward(self, grad_output):
                if self.needs_input_grad[0]:
                    grad_input = backward_quant(grad_output.contiguous())
                else:
                    grad_input = None
                return grad_input

    return Rounding.apply