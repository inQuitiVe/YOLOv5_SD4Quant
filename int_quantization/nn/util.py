import torch
from int_quantization.quant import Log_Preprocess_cpu

def _quantize_weight(float_wt, observer): #log scale quantization
    wt_scale, wt_zp = observer.calculate_qparams()
    if (observer.log_scale):
        #preprocess: w' = s * 2**(round(log2(w/s)))
        float_wt = Log_Preprocess_cpu(float_wt, wt_scale)

    if observer.qscheme in [torch.per_tensor_symmetric, torch.per_tensor_affine]:
        qweight = torch.quantize_per_tensor(
            float_wt,
            float(wt_scale), int(wt_zp), torch.qint8)
    elif observer.qscheme in [torch.per_channel_symmetric, torch.per_channel_affine]:
        wt_axis = observer.ch_axis
        qweight = torch.quantize_per_channel(
            float_wt,
            wt_scale.to(torch.double), wt_zp.to(torch.int64), wt_axis, torch.qint8)
    else:
        raise ValueError("Unexpected qscheme " + observer.qscheme)
    return qweight


'''
def _quantize_weight(float_wt, observer): #reverse log scale quantization
    wt_scale, wt_zp = observer.calculate_qparams()
    #preprocess: w' = s * 2**(round(log2(w/s)))
    expand_shape = [-1]
    dims = len(float_wt.shape)
    expand_shape += [1] * (dims-1)
    expand_scale = wt_scale.view(*expand_shape) 

    sign = torch.sign(float_wt)
    float_wt_abs = float_wt.abs() / expand_scale
    maxx = torch.max(float_wt_abs)
    residual = maxx - float_wt_abs
    idx_0 = (residual == 0)
    residual[idx_0] = 1 #tmp value
    residual = 2**(torch.round(residual.log2()))
    residual[idx_0] = 0
    float_wt_abs = maxx - residual
    float_wt_abs = float_wt_abs * sign * expand_scale

    if observer.qscheme in [torch.per_tensor_symmetric, torch.per_tensor_affine]:
        qweight = torch.quantize_per_tensor(
            float_wt_abs,
            float(wt_scale), int(wt_zp), torch.qint8)
    elif observer.qscheme in [torch.per_channel_symmetric, torch.per_channel_affine]:
        wt_axis = observer.ch_axis
        qweight = torch.quantize_per_channel(
            float_wt_abs,
            wt_scale.to(torch.double), wt_zp.to(torch.int64), wt_axis, torch.qint8)
    else:
        raise ValueError("Unexpected qscheme " + observer.qscheme)
    return qweight
'''