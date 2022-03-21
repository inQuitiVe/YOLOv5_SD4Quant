import torch
import torch.nn.init as init
import math
from .stu.stu_function import random_init, SDtoFloat


def _calculate_fan_in_and_fan_out(tensor):
	dimensions = tensor.dim()
	if dimensions < 2:
		raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

	if dimensions == 2:  # Linear
		fan_in = tensor.size(1)
		fan_out = tensor.size(0)
	else:
		num_input_fmaps = tensor.size(1)
		num_output_fmaps = tensor.size(0)
		receptive_field_size = 1
		if tensor.dim() > 2:
			receptive_field_size = tensor[0][0].numel()
		fan_in = num_input_fmaps * receptive_field_size
		fan_out = num_output_fmaps * receptive_field_size

	return fan_in, fan_out

def xavier_uniform_(tensor, gain=1., mode='FloatSD4', movement=0):
	#input: weight tensor
	#input: movement: bigger movement, bigger offset, smaller wegiht upper bound
	#input: cg: computing group
	#output: weight sd group, weight exp, weight offset
	assert (mode in ["FloatSD4", "FloatSD8"]), "Mode should be FloatSD4 or FloatSD8."

	sd_group = torch.zeros_like(tensor, dtype=torch.int32)
	sd_exp = torch.zeros_like(tensor, dtype=torch.int32)

	fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
	std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
	a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

	if (mode == 'FloatSD4'):
		max_v = 0.5714285374
		offset = 24 - math.floor(math.log2(a / max_v)) + movement

	else: #sd8
		max_v = 0.5803918839
		offset = 22 - math.floor(math.log2(a / max_v)) + movement

	#random_init(mode)(sd_group)
	for i in range(8):
		if ((i == 1 or i == 4) and mode == 'FloatSD8'):
			a = torch.randint_like(sd_group.data, low=0, high=5)
		else:
			a = torch.randint_like(sd_group.data, low=0, high=7)  
		sd_group += a * (2**(3*i))

	return sd_group, sd_exp, offset

def _calculate_correct_fan(tensor, mode):
	mode = mode.lower()
	valid_modes = ['fan_in', 'fan_out']
	if mode not in valid_modes:
		raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

	fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
	return fan_in if mode == 'fan_in' else fan_out


def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='relu', movement=0):
	fan = _calculate_correct_fan(tensor, mode)
	gain = init.calculate_gain(nonlinearity, a)
	std = gain / math.sqrt(fan)
	bound = math.sqrt(3.0) * std

	sd_group = torch.zeros_like(tensor, dtype=torch.int32)
	sd_exp = torch.zeros_like(tensor, dtype=torch.int32)
	max_v = 0.5714285374
	offset = 24 - math.floor(math.log2(bound / max_v)) + movement
	for i in range(8):
		a = torch.randint_like(sd_group.data, low=0, high=7)  
		sd_group += a * (2**(3*i))

	return sd_group, sd_exp, offset

def kaiming_bound(tensor, a=0, mode='fan_in', nonlinearity='relu'):
	fan = _calculate_correct_fan(tensor, mode)
	gain = init.calculate_gain(nonlinearity, a)
	std = gain / math.sqrt(fan)
	bound = math.sqrt(3.0) * std

	return bound

def xavier_bound(tensor, gain=1.):
	fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
	std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
	a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

	return a


