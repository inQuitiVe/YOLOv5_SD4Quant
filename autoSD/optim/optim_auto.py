from .optim_stu import OptimLP_STU, SGD_STU, Adam_STU
from .optim_quant import OptimLP_Q, SGD_Q, Adam_Q
import torch.optim as optim

def autoOptimizer(params, optimizer='SGD', method='quant', weight_rounding='floatsd4_ex', weight_offset=None, weight_structure=[5,2],
				  grad_rounding='fp', grad_structure=[5,2], grad_offset=24, fp16_training=True, fp32_mastercopy=False, 
				  quant_bias=False, verbose=True, channel_wise=False, add_param_group=[],
				  **optim_kargs):
	if (method == 'quant'):
		assert (weight_rounding in ['floatsd4', 'floatsd4_ex', 'floatsd8', 'fp']), "Invalid weight rounding mode."
	elif (method == 'stu'):
		assert (weight_rounding in ['floatsd4', 'floatsd8']), "STU only supports floatsd4 or floatsd8, use quant method instead."
	assert (method in ['quant', 'stu', 'fp32']), "Method should be quant or stu or fp32."
	assert (optimizer in ['SGD', 'Adam']), "Only SGD/Adam is supported now."
	assert (grad_rounding in ['floatsd4', 'floatsd4_ex', 'floatsd8', 'fp', 'identity']), "Invalid gradient rounding mode."

	if (method == 'quant'):
			if (optimizer == 'SGD'):
				base_optimizer = SGD_Q(params, quant_bias=quant_bias, fp32_mastercopy=fp32_mastercopy, add_param_group=add_param_group, **optim_kargs)
			else:
				base_optimizer = Adam_Q(params, quant_bias=quant_bias, fp32_mastercopy=fp32_mastercopy, add_param_group=add_param_group, **optim_kargs)

			wrap_optimizer = OptimLP_Q(base_optimizer, 
								   grad_rounding=grad_rounding,
								   grad_structure=grad_structure, 
				 				   grad_offset=grad_offset,
				 				   fp16_training=fp16_training, #use when fp16 acc
				 				   fp32_mastercopy=fp32_mastercopy,
				 				   weight_rounding=weight_rounding,
				 				   weight_offset=weight_offset,
				 				   weight_structure=weight_structure,
				 				   verbose=verbose,
				 				   channel_wise=channel_wise)

	elif (method == 'stu'):
		if (optimizer == 'SGD'):
			base_optimizer = SGD_STU(params, quant_bias=quant_bias, **optim_kargs)
		else:
			base_optimizer = Adam_STU(params, quant_bias=quant_bias)
		wrap_optimizer = OptimLP_STU(base_optimizer, 
								   	 grad_rounding=grad_rounding,
								   	 grad_structure=grad_structure, 
				 				   	 grad_offset=grad_offset,
				 				   	 fp16_training=fp16_training, #use when fp16 acc
				 				   	 mode=weight_rounding,
				 				   	 verbose=verbose)

	else:
		if (optimizer == 'SGD'):
			wrap_optimizer = optim.SGD(params, **optim_kargs)
		else:
			wrap_optimizer = optim.Adam(params, **optim_kargs)

	return wrap_optimizer