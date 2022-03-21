import torch
import autoSD.nn

#=== Post training
#disable all input Quantizer / set collect_hist to True
def collect_forward_hist(model):
	for module in model.modules():
		if (isinstance(module, autoSD.nn.Quantizer)):
			if (module.level == 'input'):
				module.collect_hist = True
				module.adaptive = False
				module.forward_rounding = 'identity'
				module.rebuild()

#adpative structure / adaptive offset
def forward_calibration_post(model, verbose=False):
	for name, module in model.named_modules():
		if (isinstance(module, autoSD.nn.Quantizer)):
			if (module.level == 'input'):
				module.verbose = verbose
				module.forward_rounding = 'fp'
				module.post_forward_calibrate(name)
				module.collect_hist = False

#disable all input Quantizer / set collect_max to True
def collect_forward_hist_cwise(model):
	for module in model.modules():
		if (isinstance(module, autoSD.nn.Quantizer)):
			if (module.level == 'input'):
				module.collect_max = True
				module.adaptive = False
				module.forward_rounding = 'identity'
				module.rebuild()

#fixed structure / adaptive c-wise offset
def forward_calibration_post_cwise(model, verbose=False):
	for name, module in model.named_modules():
		if (isinstance(module, autoSD.nn.Quantizer)):
			if (module.level == 'input'):
				module.verbose = verbose
				module.adaptive = True
				module.forward_rounding = 'fp'
				module.post_forward_calibrate_cwise(name)
				module.collect_max = False

#collect mean for bias correction
def collect_forward_mean(model, phase='gt'):
	if (phase == 'q'):
		assert isinstance(model, autoSD.nn.Quantizer), "Model should be an Quantizer in phase q."
		model.collect_mean_gt = False
		model.collect_mean_q = True
	else:
		for module in model.modules():
			if (isinstance(module, autoSD.nn.Quantizer)):
				if (module.level == 'output'): #target output layer
					if (phase == 'gt'):
						module.collect_mean_gt = True
						module.collect_mean_q = False
					else: #disable collection
						module.collect_mean_gt = False
						module.collect_mean_q = False

#get output Quantizer in model
def get_Quantizer(model, level='input'):
	assert level in ['input', 'output', 'all'], "Level should be in ['input', 'output', 'all']"
	target_list = []
	for module in model.modules():
		if (isinstance(module, autoSD.nn.Quantizer)):
			if (level == 'all' or (module.level == level)):
				target_list += [module]
	return target_list

#calculate correction term / turn on bias correct mode
def bias_correction(model):
	for module in model.modules():
		if (isinstance(module, autoSD.nn.Quantizer)):
			if (module.level == 'output'): #target output layer
				module.bias_correct()

def collect_backward_hist(model):
	for module in model.modules():
		if (isinstance(module, autoSD.nn.Quantizer)):
			if (module.backward_rounding != 'identity'):
				module.collect_bwd_hist = True
				module.backward_rounding = 'identity'
				module.rebuild()
				module.register_calibration_hook_post_train()

def backward_calibration_post(model):
	for name, module in model.named_modules():
		if (isinstance(module, autoSD.nn.Quantizer)):
			if (module.collect_bwd_hist):
				module.backward_rounding = 'fp'
				module.post_backward_calibrate(name)
				module.collect_bwd_hist = False
				if (module.handle is not None):
					module.handle.remove()