import torch
from torch.optim import Optimizer
import numpy as np
import math
import random
from ..quant.quant_function import quantizer
from .optim_base import OptimLP_base
from ..utils import calc_best_offset, calc_best_offset_cos, patch_float_to_half, calc_mse, calc_hist, calc_bd_from_hist 
from ..utils import calc_minmax, calc_best_offset_channel_wise


__all__ = ["OptimLP_Q", "SGD_Q", "Adam_Q"]


class OptimLP_Q(Optimizer):
	def __init__(self, optim,
				 grad_rounding=True,
				 grad_structure=[5,2],
				 grad_offset=24,
				 fp16_training=True,  # use when fp16 acc
				 fp32_mastercopy=False,
				 weight_rounding='floatsd4_ex',
				 weight_structure=[5,2],
				 weight_offset=None,
				 channel_wise=False,
				 verbose=True
				 ):
		assert (isinstance(optim, Adam_Q) or isinstance(optim, SGD_Q))
		assert (weight_rounding in ['floatsd4', 'floatsd4_ex', 'floatsd8', 'fp']), "Invalid rounding mode."
		super(OptimLP_Q, self).__init__(optim.param_groups, optim.defaults)  # place holder

		# python dictionary does not copy by default
		self.optim = optim
		self.param_groups = self.optim.param_groups
		self.state = self.optim.state
		
		#hyper params
		self.grad_rounding = grad_rounding
		self.fp16_training = fp16_training
		self.fp32_mastercopy = fp32_mastercopy
		self.grad_structure = grad_structure
		self.grad_offset = grad_offset
		self.channel_wise = channel_wise
		self.weight_rounding = weight_rounding
		self.weight_offset = weight_offset
		self.weight_structure = weight_structure

		#for grad calibration
		self.grad_list = None
		self.iter_collect_grad = 0
		self.grad_mode = "mse"
		self.grad_srange = 0

		#only print on main rank
		rank_0_flag = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
		verbose = verbose and rank_0_flag
		self.verbose = verbose
		if (verbose):
			print("Creating weight quantizer...")

		w_quantizer = quantizer(forward_rounding=weight_rounding, forward_fp_structure=weight_structure,
								verbose=verbose, adaptive_offset=True, adaptive_structure=False, channel_wise=channel_wise)

		if (self.fp32_mastercopy):
			self.weight_quantizer = w_quantizer
		elif (fp16_training):
			self.weight_quantizer = patch_float_to_half(w_quantizer)
		else:
			self.weight_quantizer = w_quantizer

		if (self.grad_rounding != "identity"):
			if (verbose):
				print("Creating gradient quantizer...")
			self.grad_quantizer = quantizer(forward_rounding=grad_rounding, forward_fp_structure=grad_structure,
											forward_fp_offset=grad_offset, verbose=verbose, adaptive_offset=True)
			if (fp16_training):
				self.grad_quantizer = patch_float_to_half(self.grad_quantizer)
		else:
			self.grad_quantizer = None

		self._init_offset()
		self._rebuild()

	#calculate min exp offset form fp struct
	def _calc_min_offset(self, struct):
		exp, mantissa = struct
		if (mantissa > 0):
			min_offset = 2**exp - 16
		else:
			min_offset = 2**exp - 16 - 1
		return min_offset

	#init exp offset for each layer
	def _init_offset(self):
		for group in self.param_groups:
			for i, (p, quant, sd_master_copy) in enumerate(zip(group['params'], group['quant'], group['sd_master_copy'])):
				if (quant):
					if (group['offset'][i] is None):
						#weight offset
						bound = torch.max(sd_master_copy.abs()).log2().ceil()
						bound = int(bound) - 1

						if (self.weight_rounding == 'floatsd8'):  #floatsd8
							offset = 8 - bound
							self.w_offset_min = -7
						elif (self.weight_rounding == 'floatsd4_ex'):
							offset = 6 - bound
							self.w_offset_min = -9
						elif (self.weight_rounding == 'floatsd4'):
							offset = 5 - bound
							self.w_offset_min = -10
						else:
							offset = 2**self.weight_structure[0] - 1 - bound
							self.w_offset_min = self._calc_min_offset(self.weight_structure)

						#max/min offset
						self.w_offset_max = 24

						if (self.fp16_training):
							offset = min(offset, 24)

						channel = p.shape[0]
						if (self.weight_offset is not None):
							if (self.channel_wise):
								group['offset'][i] = np.array([self.weight_offset] * channel)
							else:
								group['offset'][i] = self.weight_offset
						else:
							if (self.channel_wise):
								offset_array = np.array([offset] * channel)
								offset_tensor = torch.IntTensor(offset_array).cuda().contiguous()
								group['offset'][i] = offset_tensor
							else:
								group['offset'][i] = offset

					# add grad offset
					if (group['grad_offset'][i] is None):
						group['grad_offset'][i] = self.grad_offset

	#requantized master copy
	def _rebuild(self):
		for group in self.param_groups:
			for i, (p, sd_master_copy, offset, quant) in enumerate(
					zip(group['params'], group['sd_master_copy'], group['offset'], group['quant'])):
				if quant:  # use quant
					'''tmp = torch.zeros_like(sd_master_copy)
					tmp.data.copy_(self.weight_quantizer(sd_master_copy.data, offset))
					p.data.copy_(tmp.data)'''
					p.data.copy_(self.weight_quantizer(sd_master_copy.data, offset))

	def weight_calibration(self, srange=1, verbose=True, metric="mse"):
		#only print on main rank
		rank_0_flag = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
		verbose = verbose and rank_0_flag
		if (verbose):
			print('============================================')
			print('Begin weight calibration...')
		for group in self.param_groups:
			for i, (p, sd_master_copy, offset, quant) in enumerate(
					zip(group['params'], group['sd_master_copy'], group['offset'], group['quant'])):
				if quant:  # use quant
					channel = p.shape[0]

					maxx = self.w_offset_max
					minn = self.w_offset_min

					if (self.channel_wise):
						maxx = [maxx] * channel
						maxx = np.array(maxx)
						minn = [minn] * channel
						minn = np.array(minn)

					if srange > 0:
						if (self.channel_wise):
							maxx = np.minimum(offset + srange, maxx)
							minn = np.maximum(offset - srange, minn)
						else:
							maxx = min(offset + srange, maxx)
							minn = max(offset - srange, minn)

					if (self.channel_wise):
						offset_new = calc_best_offset_channel_wise(sd_master_copy, self.weight_quantizer, minn, maxx, metric=metric)
						if verbose and (offset_new[0] != offset[0]):
							print('  Weight exp offset (index {} / channel 0) changes from {} to {}.'.format(i, offset[0], offset_new[0]))
					else:
						offset_new, _ = calc_best_offset(sd_master_copy, self.weight_quantizer, minn, maxx, metric=metric)
						if verbose and (offset_new != offset):
							print('  Weight exp offset (index {}) changes from {} to {}.'.format(i, offset, offset_new))

					group['offset'][i] = np.array(offset_new)

					'''tmp = torch.zeros_like(sd_master_copy)
					tmp.data.copy_(self.weight_quantizer(sd_master_copy.data, offset_new))
					p.data.copy_(tmp.data)'''
					p.data.copy_(self.weight_quantizer(sd_master_copy.data, offset_new))

		if (verbose):
			print('============================================')

	def collect_grad(self, srange=1, metric="mse"): #not support mse
		self.grad_metric = metric
		self.grad_srange = srange
		if (self.grad_list is None): #init / grad_list = [param_groups_num*[param_num*[srange]]]
			self.grad_list = []
			for group in self.param_groups:
				tmp = []
				for i, (p, quant) in enumerate(zip(group['params'], group['quant'])):
					tmp.append(None)
				self.grad_list.append(tmp)

		self.iter_collect_grad += 1
		for group, grad_group in zip(self.param_groups, self.grad_list):
			for i, (p, quant, grad_offset) in enumerate(zip(group['params'], group['quant'], group['grad_offset'])):
				if (quant):
					if p.grad is None:
						continue
					grad = p.grad.data
					#range
					minn = self._calc_min_offset(self.grad_structure)
					maxx = 24
					if srange > 0:
						maxx = min(grad_offset + srange, maxx)
						minn = max(grad_offset - srange, minn)
					#create list of metrics for each offset
					if (grad_group[i] is None):
						grad_group[i] = [0 for _ in range(minn, maxx+1)]
					#get metric of each offset
					for ii, o in enumerate(range(minn, maxx+1)):
						if (metric == "mse"):
							mse = calc_mse(grad, self.grad_quantizer, o)
							grad_group[i][ii] += mse
						elif (metric == "minmax"):
							maxx = calc_minmax(grad, self.grad_quantizer, o)
							if (maxx > grad_group[i][ii]):
								grad_group[i][ii] = maxx
						elif (metric == "bd"):
							p_hist, q_hist = calc_hist(grad, self.grad_quantizer, o)
							if (p_hist is not None):
								if (grad_group[i][ii] == 0):
									grad_group[i][ii] = [p_hist, q_hist]
								else:
									grad_group[i][ii][0] += p_hist
									grad_group[i][ii][1] += q_hist

	def grad_calibration(self, verbose=True):
		#only print on main rank
		rank_0_flag = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
		verbose = verbose and rank_0_flag
		if verbose:
			print('============================================')
			print('Begin gradient calibration...')
		for group, grad_group in zip(self.param_groups, self.grad_list):
			for i, (p, quant, grad_offset) in enumerate(zip(group['params'], group['quant'], group['grad_offset'])):
				if quant:  # use quant
					if p.grad is None:
						continue
					grad = p.grad.data
					#range
					minn = self._calc_min_offset(self.grad_structure)
					maxx = 24
					if self.grad_srange > 0:
						maxx = min(grad_offset + self.grad_srange, maxx)
						minn = max(grad_offset - self.grad_srange, minn)

					#find best offset
					dist_min = float('inf')
					offset_new = grad_offset
					for ii, offset_now in enumerate(range(minn, maxx+1)):
						if (self.grad_metric == "mse"):
							mse_now = grad_group[i][ii]
							if (mse_now <= dist_min):
								offset_new = offset_now
								dist_min = mse_now
						elif (self.grad_metric == "minmax"):
							max_now = grad_group[i][ii]
							if (max_now <= dist_min):
								offset_new = offset_now
								dist_min = max_now
						elif (self.grad_metric == "bd"):
							if (grad_group[i][ii] != 0):
								p_hist, q_hist = grad_group[i][ii]
								p_hist = p_hist / self.iter_collect_grad
								q_hist = q_hist / self.iter_collect_grad
								bd_now = calc_bd_from_hist(p_hist, q_hist)
							else:
								bd_now = float('inf')

							if (bd_now <= dist_min):
								offset_new = offset_now
								dist_min = bd_now
					grad_group[i] = None #reset grad_group
					if verbose and (grad_offset != offset_new):
						print('  Gradient exp offset (index {}) changes from {} to {}.'.format(i, grad_offset, offset_new))
					group['grad_offset'][i] = offset_new

		self.iter_collect_grad = 0
		if verbose:
			print('============================================')

	def step(self, closure=None):
		loss = self.optim.step(self.weight_quantizer, self.grad_quantizer, self.fp16_training, self.fp32_mastercopy)
		return loss

	def load_state_dict(self, state_dict):
		self.optim.load_state_dict(state_dict)
		#reference would not be same after load_state_dict
		self.param_groups = self.optim.param_groups
		self.state = self.optim.state

	def __repr__(self):
		optim_type = self.optim.__class__.__name__[:-2]
		return ("Low precision {}:\n"
				"   weight rounding: {} / weight structure: {}\n"
				"   gradient rounding: {} / gradient structure: {}"
				).format(optim_type, self.weight_rounding, self.weight_structure, self.grad_rounding, self.grad_structure)

	def __str__(self):
		optim_type = self.optim.__class__.__name__[:-2]
		return ("Low precision {}:\n"
				"   weight rounding: {} / weight structure: {}\n"
				"   gradient rounding: {} / gradient structure: {}"
				).format(optim_type, self.weight_rounding, self.weight_structure, self.grad_rounding, self.grad_structure)

class Adam_Q(OptimLP_base):
	def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
				 weight_decay=0, amsgrad=False, quant_bias=False, fp32_mastercopy=False):
		if not 0.0 <= lr:
			raise ValueError("Invalid learning rate: {}".format(lr))
		if not 0.0 <= eps:
			raise ValueError("Invalid epsilon value: {}".format(eps))
		if not 0.0 <= betas[0] < 1.0:
			raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
		if not 0.0 <= betas[1] < 1.0:
			raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
		defaults = dict(lr=lr, betas=betas, eps=eps,
						weight_decay=weight_decay, amsgrad=amsgrad)
		sd_defaults = dict(method='quant', quant_bias=quant_bias, fp32_mastercopy=fp32_mastercopy)
		super(Adam_Q, self).__init__(params, defaults, sd_defaults)

	def __setstate__(self, state):
		super(Adam_Q, self).__setstate__(state)
		for group in self.param_groups:
			group.setdefault('amsgrad', False)

	def step(self, weight_quantizer=None, grad_quantizer=None, fp16_training=True, fp32_mastercopy=False, closure=None):
		loss = None
		if closure is not None:
			loss = closure()

		for group in self.param_groups:
			for i, (p, sd_master_copy, offset, quant, grad_offset) in enumerate(
						zip(group['params'], group['sd_master_copy'], group['offset'], group['quant'], group['grad_offset'])):
				if (quant):  # use quant
					if p.grad is None:
						continue
					#grad quant
					if (grad_quantizer is not None):
						grad = grad_quantizer(p.grad.data, grad_offset).float()
					else:
						grad = p.grad.data.float()

					if grad.is_sparse:
						raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
					amsgrad = group['amsgrad']

					p_data_fp32 = sd_master_copy

					state = self.state[p]

					# State initialization
					if len(state) == 0:
						state['step'] = 0
						# Exponential moving average of gradient values
						state['exp_avg'] = torch.zeros_like(p_data_fp32)
						# Exponential moving average of squared gradient values
						state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
						if amsgrad:
							# Maintains max of all exp. moving avg. of sq. grad. values
							state['max_exp_avg_sq'] = torch.zeros_like(p_data_fp32)
					else:
						state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
						state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
						if amsgrad:
							state['max_exp_avg_sq'] = state['max_exp_avg_sq'].type_as(p_data_fp32)

					exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
					if amsgrad:
						max_exp_avg_sq = state['max_exp_avg_sq']
					beta1, beta2 = group['betas']

					state['step'] += 1

					# Decay the first and second moment running average coefficient
					exp_avg.mul_(beta1).add_(1 - beta1, grad)
					exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
					if amsgrad:
						# Maintains the maximum of all 2nd moment running avg. till now
						torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
						# Use the max. for normalizing running avg. of gradient
						denom = max_exp_avg_sq.sqrt().add_(group['eps'])
					else:
						denom = exp_avg_sq.sqrt().add_(group['eps'])

					bias_correction1 = 1 - beta1 ** state['step']
					bias_correction2 = 1 - beta2 ** state['step']
					step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

					if group['weight_decay'] != 0:
						p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

					p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

					# TODO: remove check once pyTorch avoids a copy for this case
					tmp = torch.zeros_like(p_data_fp32)
					tmp.data = weight_quantizer(p_data_fp32.data, offset)
					p.data.copy_(tmp.data)

				else:  # normal weight
					if p.grad is None:
						continue
					grad = p.grad.data.float()
					if grad.is_sparse:
						raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
					amsgrad = group['amsgrad']

					p_data_fp32 = p.data.float()

					state = self.state[p]

					# State initialization
					if len(state) == 0:
						state['step'] = 0
						# Exponential moving average of gradient values
						state['exp_avg'] = torch.zeros_like(p_data_fp32)
						# Exponential moving average of squared gradient values
						state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
						if amsgrad:
							# Maintains max of all exp. moving avg. of sq. grad. values
							state['max_exp_avg_sq'] = torch.zeros_like(p_data_fp32)
					else:
						state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
						state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
						if amsgrad:
							state['max_exp_avg_sq'] = state['max_exp_avg_sq'].type_as(p_data_fp32)

					exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
					if amsgrad:
						max_exp_avg_sq = state['max_exp_avg_sq']
					beta1, beta2 = group['betas']

					state['step'] += 1

					# Decay the first and second moment running average coefficient
					exp_avg.mul_(beta1).add_(1 - beta1, grad)
					exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
					if amsgrad:
						# Maintains the maximum of all 2nd moment running avg. till now
						torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
						# Use the max. for normalizing running avg. of gradient
						denom = max_exp_avg_sq.sqrt().add_(group['eps'])
					else:
						denom = exp_avg_sq.sqrt().add_(group['eps'])

					bias_correction1 = 1 - beta1 ** state['step']
					bias_correction2 = 1 - beta2 ** state['step']
					step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

					if group['weight_decay'] != 0:
						p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

					p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

					# TODO: remove check once pyTorch avoids a copy for this case
					if p.data_ptr() != p_data_fp32.data_ptr():
						p.data.copy_(p_data_fp32)

		return loss

class SGD_Q(OptimLP_base):
	def __init__(self, params, lr=1e-3, momentum=0, dampening=0,
				 weight_decay=0, nesterov=False, quant_bias=False, fp32_mastercopy=False, add_param_group=[]):
		if lr < 0.0:
			raise ValueError("Invalid learning rate: {}".format(lr))
		if momentum < 0.0:
			raise ValueError("Invalid momentum value: {}".format(momentum))
		if weight_decay < 0.0:
			raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

		defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
						weight_decay=weight_decay, nesterov=nesterov)
		sd_defaults = dict(method='quant', quant_bias=quant_bias, fp32_mastercopy=fp32_mastercopy)
		if nesterov and (momentum <= 0 or dampening != 0):
			raise ValueError("Nesterov momentum requires a momentum and zero dampening")
		super(SGD_Q, self).__init__(params, defaults, sd_defaults, add_param_group)

	def __setstate__(self, state):
		super(SGD_Q, self).__setstate__(state)
		for group in self.param_groups:
			group.setdefault('nesterov', False)

	@torch.no_grad()
	def step(self, weight_quantizer=None, grad_quantizer=None, fp16_training=True, fp32_mastercopy=False, 
			 rescale=False, c_group=1, closure=None):
		loss = None
		if closure is not None:
			loss = closure()

		for group in self.param_groups:
			weight_decay = group['weight_decay']
			momentum = group['momentum']
			dampening = group['dampening']
			nesterov = group['nesterov']

			for i, (p, sd_master_copy, offset, quant, grad_offset) in enumerate(
					zip(group['params'], group['sd_master_copy'], group['offset'], group['quant'], group['grad_offset'])):
				if (quant):  # use quant
					if p.grad is None:
						continue
					# grad quant
					if (grad_quantizer is not None):
						d_p = grad_quantizer(p.grad.data, grad_offset).float()
					else:
						d_p = p.grad.data.float()

					if weight_decay != 0:
						d_p = d_p.add(sd_master_copy.data, alpha=weight_decay)

					if momentum != 0:
						param_state = self.state[p]
						if 'momentum_buffer' not in param_state:
							buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
						else:
							buf = param_state['momentum_buffer']
							buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
						if nesterov:
							d_p = d_p.add(buf, alpha=momentum)
						else:
							d_p = buf

					sd_master_copy.data.add_(d_p, alpha=-group['lr'])
					if (rescale):
						tmp = torch.zeros_like(sd_master_copy)
						tmp.data.copy_(weight_quantizer(sd_master_copy.data, offset))
						shape = tmp.size()
						channel = shape[0]
						max_m, _ = torch.max(torch.abs(sd_master_copy.view(channel, -1)), dim=1)
						max_q, _ = torch.max(torch.abs(tmp.view(channel, -1)), dim=1)
						scale = max_m / max_q #(channel,)
						tmp = tmp.view(channel, -1) * scale.view(channel, -1)
						tmp = tmp.view(shape)
						p.data.copy_(tmp.data)
					elif (c_group == 2):
						bound1 = torch.max(sd_master_copy.abs()).log2().ceil().item()
						offset1 = 6 - int(bound1)
						tmp = weight_quantizer(sd_master_copy.data, offset1)
						residual = sd_master_copy - tmp

						bound2 = torch.max(residual.abs()).log2().ceil().item()
						offset2 = 6 - int(bound2)
						p.data.copy_(tmp.data + weight_quantizer(residual.data, offset2))
					elif (c_group == 1):
						bound1 = torch.max(sd_master_copy.abs()).log2().round().item()
						offset1 = 6 - int(bound1)
						p.data.copy_(weight_quantizer(sd_master_copy.data, offset1))
					else:
						p.data.copy_(weight_quantizer(sd_master_copy.data, offset))

				else:  # normal weight
					if p.grad is None:
						continue
					d_p = p.grad
					if weight_decay != 0:
						d_p = d_p.add(p, alpha=weight_decay)
					if momentum != 0:
						param_state = self.state[p]
						if 'momentum_buffer' not in param_state:
							buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
						else:
							buf = param_state['momentum_buffer']
							buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
						if nesterov:
							d_p = d_p.add(buf, alpha=momentum)
						else:
							d_p = buf

					p.add_(d_p, alpha=-group['lr'])

		return loss
