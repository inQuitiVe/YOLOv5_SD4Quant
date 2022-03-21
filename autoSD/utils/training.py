import torch
import autoSD.nn

#fused loss scale + backward + unscale + update scale
def Scale_Backward(loss, scaler, optimizer, update=True):
	scale = scaler.loss_scale()
	loss *= scale
	loss.backward()

	grads = []
	for group in optimizer.param_groups:
		for p in group['params']:
			if (p.grad is None):
				continue
			grads.append(p.grad)

	scaler.unscale(grads)
	skip = scaler.update_scale(update=update)

	return skip

#set BN to eval mode to prevent statistics update
def freeze_bn_statistics(model):
	for module in model.modules():
		if (isinstance(module, torch.nn.modules.batchnorm._BatchNorm)):
			module.eval()

def _in_blacklist(name, black_list):
	for l_name in black_list:
		w_name = l_name + '.weight'
		if (w_name == name):
			print('Note: {} are not quantized'.format(name))
			return True
	return False

#create param_groups for optimizer
def split_parameters(model, quant_bias=False, black_list=[], split_bias=False):
	#split model parameters to different groups
	#Parameters:
	#	model: model which parameters would be splitted
	#   quant_bias: whether to quantize bias or batchnorm weight/bias
	#   black_list: List of param name, param in black list would not be quant
	param_quant = []
	param_no_quant = []
	quant_quant = []
	quant_no_quant = []
	param_bn = []
	quant_bn = []
	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue
		if (split_bias and (len(param.shape) == 1)):
			param_bn.append(param)
			if (not quant_bias):
				quant_bn.append(False)
			else:
				quant_bn.append(True)
		else:
			if (not split_bias and (len(param.shape) == 1) and not quant_bias):
				param_no_quant.append(param)
				quant_no_quant.append(False)
			elif (_in_blacklist(name, black_list)):
				param_no_quant.append(param)
				quant_no_quant.append(False)
			else:
				param_quant.append(param)
				quant_quant.append(True)

	group_no_quant = {'params': param_no_quant, 'quant': quant_no_quant}
	group_quant = {'params': param_quant, 'quant': quant_quant}
	group_bn = {'params': param_bn, 'quant': quant_bn}

	if (split_bias):
		return [group_no_quant, group_quant, group_bn]
	else:
		return [group_no_quant, group_quant]

#set no-quant to all param in group
def set_no_quant(group): #group: {'params': [], 'weight_decay': xxx, ....}
	quant = [False] * len(group['params'])
	group['quant'] = quant
	return group

#set quant to param in group
def set_quant(group, quant_bias=False): #group: {'params': [], 'weight_decay': xxx, ....}
	quant = []
	for param in group['params']:
		if not param.requires_grad:
			quant.append(False)
		elif (len(param.shape) == 1):
			if (not quant_bias):
				quant.append(False)
			else:
				quant.append(True)
		else:
			quant.append(True)
	group['quant'] = quant
	return group

#reconfigure input quantization settings (forward of input Quantizer & backward of output Quantizer)
def reconfigure_input(model, rounding=None, fp_structure=None, fp_offset=None):
	for name, module in model.named_modules():
		if (isinstance(module, autoSD.nn.Quantizer)):
			if (module.level == 'input'): #input Quantizer
				module.reconfigure(forward_rounding=rounding, 
								   forward_fp_structure=fp_structure,
								   forward_fp_offset=fp_offset)
			else: #output Quantizer
				module.reconfigure(backward_rounding=rounding, 
								   backward_fp_structure=fp_structure,
								   backward_fp_offset=fp_offset)

#reconfigure output quantization settings (backward of input Quantizer & forward of output Quantizer)
def reconfigure_output(model, rounding=None, fp_structure=None, fp_offset=None):
	for name, module in model.named_modules():
		if (isinstance(module, autoSD.nn.Quantizer)):
			if (module.level == 'output'): #output Quantizer
				module.reconfigure(forward_rounding=rounding, 
								   forward_fp_structure=fp_structure,
								   forward_fp_offset=fp_offset)
			else: #input Quantizer
				module.reconfigure(backward_rounding=rounding, 
								   backward_fp_structure=fp_structure,
								   backward_fp_offset=fp_offset)