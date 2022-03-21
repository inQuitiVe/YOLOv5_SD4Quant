import torch
import autoSD.nn

#=== Training
#collect fwd act for calibration / must be called before forward
def collect_forward_act(model, srange=1, metric="mse"):
	for module in model.modules():
		if (isinstance(module, autoSD.nn.Quantizer)):
			if (module.forward_rounding != 'identity'):
				module.collect_fwd = True
				module.forward_metric = metric
				module.forward_srange = srange

#perform fwd calibration / turn off fwd collection
def forward_calibration(model, verbose=False):
	if (verbose):
		print('============================================')
		print('Begin forward pass calibration...')
	for name, module in model.named_modules():
		if (isinstance(module, autoSD.nn.Quantizer)):
			if (module.collect_fwd):
				module.verbose = verbose
				module.forward_calibrate(name)
				module.verbose = False
				module.collect_fwd = False
	if (verbose):
		print('============================================')

#register hook for every Quantizer, must be called before forward
def collect_backward_act(model, srange=1, metric="mse"):
	for module in model.modules():
		if (isinstance(module, autoSD.nn.Quantizer)):
			if (module.backward_rounding != 'identity'):
				module.backward_metric = metric
				module.backward_srange = srange
				module.register_calibration_hook_train()

#perform bwd calibration / remove hook handle
def backward_calibration(model, verbose=False):
	if (verbose):
		print('============================================')
		print('Begin backward pass calibration...')
	for name, module in model.named_modules():
		if (isinstance(module, autoSD.nn.Quantizer)):
			if (module.backward_rounding != 'identity'):
				module.verbose = verbose
				module.backward_calibrate(name)
				if (module.handle is not None):
					module.handle.remove()
				module.verbose = False
	if (verbose):
		print('============================================')