import torch
from torch._six import string_classes
import collections.abc as container_abcs
import numpy as np

def applier(value, fn):
	if isinstance(value, torch.Tensor):
		return fn(value)
	elif isinstance(value, string_classes):
		return value
	elif isinstance(value, np.ndarray):
		return value
	elif hasattr(value, "to"): # Allow handling of custom batch classes
		return fn(value)
	elif isinstance(value, container_abcs.Mapping):
		return {applier(k, fn) : applier(v, fn) for k, v in value.items()}
	elif isinstance(value, container_abcs.Iterable):
		return type(value)(applier(v, fn) for v in value)
	else:
		return value

#patch float quantize function to half quantize function
def patch_float_to_half(old_fwd):
	to_fp32 = lambda x: x.float()
	to_fp16 = lambda x: x.half()
	def new_fwd(*args, **kwargs):
		output = old_fwd(*applier(args, to_fp32), **applier(kwargs, to_fp32))
		output = applier(output, to_fp16)
		return output
	return new_fwd

#count percentage of quantized parameter
def count_param(optimizer):
	total_param = 0
	quant_param = 0
	non_quant_param = 0
	for group in optimizer.param_groups:
		for p, quant in zip(group['params'], group['quant']):
			if (not p.requires_grad):
				continue
			num = p.numel()
			total_param += num
			if (quant):
				quant_param += num
			else:
				non_quant_param += num

	q_percent = quant_param * 100 / total_param
	print("Total parameters: {}".format(total_param))
	print("Quantized parameters: {}".format(quant_param))
	print("Un-quantized parameters: {}".format(non_quant_param))
	print("Quantization percentage(%): {}".format(q_percent))
