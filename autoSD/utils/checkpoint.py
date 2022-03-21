import torch
from collections import OrderedDict
import autoSD.nn

#for checkpoint
def _struct2str(struct):
    tmp = " ".join([str(i) for i in struct])
    return tmp

def _str2struct(strr):
    tmp = [int(i) for i in strr.split()]
    return tmp

#get state_dict of all Quantizer
def get_state_dict(model, destination=None):
	if destination is None:
		destination = OrderedDict()

	for name, module in model.named_modules():
		if (isinstance(module, autoSD.nn.Quantizer)):
			tmp = {}
			tmp['forward_rounding'] = module.forward_rounding
			tmp['forward_fp_offset'] = module.forward_fp_offset
			tmp['forward_fp_structure'] = _struct2str(module.forward_fp_structure)
			tmp['backward_rounding'] = module.backward_rounding
			tmp['backward_fp_offset'] = module.backward_fp_offset
			tmp['backward_fp_structure'] = _struct2str(module.backward_fp_structure)
			destination[name] = tmp
	return destination

#load state_dict of all Quantizer
def load_state_dict(model, state_dict):
	state_dict = state_dict.copy()
	for name, module in model.named_modules():
		if (isinstance(module, autoSD.nn.Quantizer)):
			module.forward_rounding = state_dict[name]['forward_rounding']
			module.forward_fp_offset = state_dict[name]['forward_fp_offset']
			module.forward_fp_structure = _str2struct(state_dict[name]['forward_fp_structure'])
			module.backward_rounding = state_dict[name]['backward_rounding']
			module.backward_fp_offset = state_dict[name]['backward_fp_offset']
			module.backward_fp_structure = _str2struct(state_dict[name]['backward_fp_structure'])
			module.rebuild()

#fused state_dict of model+quantizer+optimizer+scaler
def get_checkpoint(model=None, optimizer=None, scaler=None, destination=None):
	if destination is None:
		destination = {}

	if (model is not None):
		destination['model_state_dict'] = model.state_dict()
		destination['quantizer_state_dict'] = get_state_dict(model)
	if (optimizer is not None):
		destination['optimizer_state_dict'] = optimizer.state_dict()
	if (scaler is not None):
		destination['scaler_state_dict'] = scaler.state_dict()
	return destination

#fused load_state_dict of model+quantizer+optimizer+scaler
def load_checkpoint(checkpoint, model=None, optimizer=None, scaler=None):
	checkpoint = checkpoint.copy()
	if (model is not None):
		model.load_state_dict(checkpoint['model_state_dict'])
		load_state_dict(model, checkpoint['quantizer_state_dict'])
	if (optimizer is not None):
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	if (scaler is not None):
		scaler.load_state_dict(checkpoint['scaler_state_dict'])