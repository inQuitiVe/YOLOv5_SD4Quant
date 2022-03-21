import torch
import torch.nn as nn

from .util import _classify, _should_quant, _replace_layer, _rsetattr
from .quant_module import Quantizer
from ..utils import applier

def auto_insert(model, act_quant_level=0, act_forward_rounding='fp', act_backward_rounding='fp', act_forward_offset=24, 
				act_backward_offset=24, act_forward_structure=[5,2], act_backward_structure=[5,2], fp16_training=True, 
				keep_bn_fp32=True, placement='input', exclude='inout', verbose=True, black_list=[], patch_bn=False, 
				remove_redundant=True):
	assert (placement in ['output', 'input', 'inout']), 'Placement should be input/output/inout.'
	assert (exclude in ['output', 'input', 'inout', 'no']), 'Exclude should be input/output/inout/no.'
	assert (act_quant_level in [0, 1, 2]), 'Quant level should be 0-2'

	#only print on main rank
	rank_0_flag = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
	verbose = verbose and rank_0_flag
	
	#Switch to half model / keep BN layers in fp32
	if (fp16_training):
		if (verbose):
			print('============================================')
			print('Switching to half model...')
			print('============================================')
		model.half()
		if (keep_bn_fp32):
			for module in model.modules():
				if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
					module.float()
				elif isinstance(module, torch.nn.modules.pooling._AdaptiveAvgPoolNd):
					module.float()

	#get NO. of quant layers
	num = 0
	if (act_quant_level > 0):
		for name, module in model.named_modules():
			if (isinstance(module, torch.nn.modules.linear.Linear) or isinstance(module, torch.nn.modules.conv._ConvNd)):
				num += 1
			elif ((isinstance(module, torch.nn.modules.dropout._DropoutNd) or isinstance(module, torch.nn.modules.pooling._AvgPoolNd))
				   and (act_quant_level == 2)):
				num += 1
			elif ((isinstance(module, torch.nn.modules.batchnorm._BatchNorm)) and (act_quant_level == 2)):
				num += 1

	#Handle activation quant
	to_fp32 = lambda x: x.float()
	to_fp16 = lambda x: x.half()

	if (act_quant_level > 0):
		if (fp16_training):
			patch = True
		else:
			patch = False

		#flag to remove redundant Quantizer, that is fwd = bwd = identity
		quant_in_redundant = False
		quant_out_redundant = False

		if (placement == 'inout'):
			act_quant_in = lambda: Quantizer(forward_rounding=act_forward_rounding,
											 backward_rounding=act_backward_rounding,
											 half=patch,
											 forward_fp_structure=act_forward_structure,
										  	 backward_fp_structure=act_backward_structure, 
										  	 forward_fp_offset=act_forward_offset, 
										  	 backward_fp_offset=act_backward_offset, 
										  	 verbose=False)

			act_quant_out = lambda: Quantizer(forward_rounding=act_forward_rounding,
											  backward_rounding=act_backward_rounding,
											  half=patch,
											  forward_fp_structure=act_forward_structure,
										  	  backward_fp_structure=act_backward_structure, 
										  	  forward_fp_offset=act_forward_offset, 
										  	  backward_fp_offset=act_backward_offset, 
										  	  verbose=False)

			if ((act_forward_rounding == "identity") and (act_backward_rounding == "identity")
				and remove_redundant):
				quant_in_redundant = True
				quant_out_redundant = True

			if (verbose):
				print("Creating input activation quantizer...")
				Quantizer(forward_rounding=act_forward_rounding,
						  backward_rounding=act_backward_rounding,
						  half=patch,
						  forward_fp_structure=act_forward_structure,
					  	  backward_fp_structure=act_backward_structure, 
					  	  forward_fp_offset=act_forward_offset, 
					  	  backward_fp_offset=act_backward_offset, 
					  	  verbose=True)
				print("Creating output activation quantizer...")
				Quantizer(forward_rounding=act_forward_rounding,
						  backward_rounding=act_backward_rounding,
						  half=patch,
						  forward_fp_structure=act_forward_structure,
					  	  backward_fp_structure=act_backward_structure, 
					  	  forward_fp_offset=act_forward_offset, 
					  	  backward_fp_offset=act_backward_offset, 
					  	  verbose=True)


		elif (placement == 'input'):
			act_quant_in = lambda: Quantizer(forward_rounding=act_forward_rounding,
											 backward_rounding="identity",
											 half=patch,
											 forward_fp_structure=act_forward_structure,
										  	 backward_fp_structure=act_backward_structure, 
										  	 forward_fp_offset=act_forward_offset, 
										  	 backward_fp_offset=act_backward_offset, 
										  	 verbose=False)

			act_quant_out = lambda: Quantizer(forward_rounding="identity",
											  backward_rounding=act_backward_rounding,
											  half=patch,
											  forward_fp_structure=act_forward_structure,
										  	  backward_fp_structure=act_backward_structure, 
										  	  forward_fp_offset=act_forward_offset, 
										  	  backward_fp_offset=act_backward_offset, 
										  	  verbose=False)

			if ((act_forward_rounding == "identity") and remove_redundant):
				quant_in_redundant = True

			if ((act_backward_rounding == "identity") and remove_redundant):
				quant_out_redundant = True

			if (verbose):
				print("Creating input activation quantizer...")
				Quantizer(forward_rounding=act_forward_rounding,
						  backward_rounding="identity",
						  half=patch,
						  forward_fp_structure=act_forward_structure,
					  	  backward_fp_structure=act_backward_structure, 
					  	  forward_fp_offset=act_forward_offset, 
					  	  backward_fp_offset=act_backward_offset, 
					  	  verbose=True)
				print("Creating output activation quantizer...")
				Quantizer(forward_rounding="identity",
						  backward_rounding=act_backward_rounding,
						  half=patch,
						  forward_fp_structure=act_forward_structure,
					  	  backward_fp_structure=act_backward_structure, 
					  	  forward_fp_offset=act_forward_offset, 
					  	  backward_fp_offset=act_backward_offset, 
					  	  verbose=True)

		else: #placement = outupt
			act_quant_in = lambda: Quantizer(forward_rounding="identity",
											 backward_rounding=act_backward_rounding,
											 half=patch,
											 forward_fp_structure=act_forward_structure,
										  	 backward_fp_structure=act_backward_structure, 
										  	 forward_fp_offset=act_forward_offset, 
										  	 backward_fp_offset=act_backward_offset, 
										  	 verbose=False)

			act_quant_out = lambda: Quantizer(forward_rounding=act_forward_rounding,
											  backward_rounding="identity",
											  half=patch,
											  forward_fp_structure=act_forward_structure,
										  	  backward_fp_structure=act_backward_structure, 
										  	  forward_fp_offset=act_forward_offset, 
										  	  backward_fp_offset=act_backward_offset, 
										  	  verbose=False)

			if ((act_forward_rounding == "identity") and remove_redundant):
				quant_out_redundant = True

			if ((act_backward_rounding == "identity") and remove_redundant):
				quant_in_redundant = True

			if (verbose):
				print("Creating input activation quantizer...")
				Quantizer(forward_rounding="identity",
						  backward_rounding=act_backward_rounding,
						  half=patch,
						  forward_fp_structure=act_forward_structure,
					  	  backward_fp_structure=act_backward_structure, 
					  	  forward_fp_offset=act_forward_offset, 
					  	  backward_fp_offset=act_backward_offset, 
					  	  verbose=True)
				print("Creating output activation quantizer...")
				Quantizer(forward_rounding=act_forward_rounding,
						  backward_rounding="identity",
						  half=patch,
						  forward_fp_structure=act_forward_structure,
					  	  backward_fp_structure=act_backward_structure, 
					  	  forward_fp_offset=act_forward_offset, 
					  	  backward_fp_offset=act_backward_offset, 
					  	  verbose=True)

		if (verbose):
			print('============================================')
			print('Inserting quantization layer...')
			print('  There are total {} target layers.'.format(num))

		patch_num = 0
		name_list = []
		layer_list = []
		chain = False #prevent consecutive quantization layer (only for placement == 'inout')
		for name, module in model.named_modules(): #patch forward function
			target, layer_type = _classify(module, act_quant_level)
			if (target):
				patch_num += 1
				should_quant, insert_mode = _should_quant(num, patch_num, placement, exclude, chain, 
													 	  name, black_list, quant_in_redundant, quant_out_redundant)
				chain = True
				if (should_quant):
					if (verbose):
						print('  Patching layer {} ({}): {} ({})'.format(patch_num, insert_mode, name, layer_type))
					new_layer = _replace_layer(module, act_quant_in, act_quant_out, insert_mode=insert_mode)
					name_list.append(name)
					layer_list.append(new_layer)
				else:
					if (verbose):
						print('  Excluding layer {}: {} ({})'.format(patch_num, name, layer_type))
					chain = False
			elif (isinstance(module, torch.nn.modules.activation.ReLU) or isinstance(module, torch.nn.modules.pooling._MaxPoolNd)):
				chain = chain #ReLU / Maxpool would not make quantized value un-quantized 
			else:
				chain = False

		if (verbose):
			print('============================================')

		for name, layer in zip(name_list, layer_list):
			_rsetattr(model, name, layer)

	#handle BN input/output
	if (keep_bn_fp32 and fp16_training and patch_bn):
		if (verbose):
			print('============================================')
			print('Handling BN I/O...')
			print('============================================')
		def patch_forward_bn(old_fwd):
			def new_fwd(*args, **kwargs):
				output = old_fwd(*applier(args, to_fp32), **applier(kwargs, to_fp32))
				return applier(output, to_fp16)
			return new_fwd

		for module in model.modules():
			if (isinstance(module, torch.nn.modules.batchnorm._BatchNorm)):
				module.forward = patch_forward_bn(module.forward)
			elif isinstance(module, torch.nn.modules.pooling._AdaptiveAvgPoolNd):
				module.forward = patch_forward_bn(module.forward)

	return model