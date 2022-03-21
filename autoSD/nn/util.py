import functools
import torch
from torch import nn

def _classify(module, act_quant_level):
	target = False
	layer_type = ''
	if (isinstance(module, torch.nn.modules.linear.Linear)):
		target = True
		layer_type = 'FC'
	elif (isinstance(module, torch.nn.modules.conv._ConvNd)):
		target = True
		layer_type = 'CONV'
	elif (act_quant_level == 2):
		if (isinstance(module, torch.nn.modules.batchnorm._BatchNorm)):
			target = True
			layer_type = 'BN'
		elif (isinstance(module, torch.nn.modules.dropout._DropoutNd)):
			target = True
			layer_type = 'DP'
		elif (isinstance(module, torch.nn.modules.pooling._AvgPoolNd)):
			target = True
			layer_type = 'AVG_POOL'
	return target, layer_type

def _rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(_rgetattr(obj, pre) if pre else obj, post, val)

def _rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def _replace_layer(module, quant_in, quant_out, insert_mode='output'):
	if (insert_mode == 'output'):
		layers = [module, quant_out()]
		layers[1].level = 'output'
	elif (insert_mode == 'input'):
		layers = [quant_in(), module]
		layers[0].level = 'output'
	else: #inout
		layers = [quant_in(), module, quant_out()]
		layers[0].level = 'input'
		layers[2].level = 'output'
	return nn.Sequential(*layers)

def _should_quant(num, patch_num, placement, exclude, chain, name, black_list,
				  quant_in_redundant, quant_out_redundant):
	q = True
	insert_mode = 'inout'

	ex_flag = (name in black_list)

	if ((patch_num == 1) and (exclude in ['input', 'inout'])): #input quant exclude
		q = False
	elif (ex_flag):
		q = False
	elif ((patch_num == num) and (exclude in ['output', 'inout'])): #output quant exclude
		q = False
	elif (placement == "inout" and chain):
		insert_mode = 'output'

	#Redundant handling
	if (quant_in_redundant and quant_out_redundant): #no need to Quant if all identity
		q = False
	elif (quant_in_redundant):
		insert_mode = 'output'
	elif (quant_out_redundant):
		insert_mode = 'input'
	#there is impossible to have placement==inout && chain && quant_in_redundant,
	#since for placement==inout, quant_in_redundant == quant_out_redundant

	return q, insert_mode