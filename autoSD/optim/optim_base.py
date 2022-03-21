import torch
from torch.optim import Optimizer
import numpy as np
import math

__all__ = ["OptimLP_base"]


class OptimLP_base(Optimizer):
	def __init__(self, params, defaults, sd_defaults, add_param_group):
		super(OptimLP_base, self).__init__(params, defaults)
		for group in add_param_group:
				self.add_param_group(group)
		self.sd_defaults = sd_defaults
		self._split(sd_defaults)

	def _split(self, sd_defaults):
		#param groups: list
		#group: dict
		#black list not support now
		#print(sd_defaults)
		if (sd_defaults['method'] == 'stu'):
			for group in self.param_groups:
				sd_group = []
				sd_exp = []
				offset = []
				grad_offset = []
				quant = []
				for param in group['params']:
					sd_group.append(torch.zeros_like(param, dtype=torch.int32))
					sd_exp.append(torch.zeros_like(param, dtype=torch.int32))
					offset.append(None)
					grad_offset.append(None)
					if not param.requires_grad:
						quant.append(False)
					elif ((len(param.shape) == 1) and not sd_defaults['quant_bias']):
						quant.append(False)
					else:
						quant.append(True)
				if ('quant' not in group.keys()):
					group['quant'] = quant

				if ('sd_group' not in group.keys()):
					group['sd_group'] = sd_group

				if ('sd_exp' not in group.keys()):
					group['sd_exp'] = sd_exp

				if ('offset' not in group.keys()):
					group['offset'] = offset

				if ('grad_offset' not in group.keys()):
					group['grad_offset'] = grad_offset

		else:
			for group in self.param_groups:
				sd_master = []
				offset = []
				grad_offset = []
				quant = []
				for param in group['params']:
					if (sd_defaults['fp32_mastercopy']):
						sd_master_copy = param.detach().clone().float()
					else:
						sd_master_copy = param.detach().clone().half()
					offset.append(None)
					grad_offset.append(None)
					sd_master.append(sd_master_copy)
					if not param.requires_grad:
						quant.append(False)
					elif ((len(param.shape) == 1) and not sd_defaults['quant_bias']):
						quant.append(False)
					else:
						quant.append(True)
				if ('quant' not in group.keys()):
					group['quant'] = quant

				if ('sd_master_copy' not in group.keys()):
					group['sd_master_copy'] = sd_master

				if ('offset' not in group.keys()):
					group['offset'] = offset

				if ('grad_offset' not in group.keys()):
					group['grad_offset'] = grad_offset



