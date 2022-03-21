import torch
from torch.optim import Optimizer
import numpy as np
import math
from .init import xavier_uniform_, kaiming_uniform_
from .stu.stu_function import SDtoFloat, stu_updater
from .optim_base import OptimLP_base
from ..quant.quant_function import quantizer

__all__ = ["OptimLP_STU", "Adam_STU", "SGD_STU"]


class OptimLP_STU(Optimizer):
	def __init__(self, optim,
				 grad_rounding='fp',
				 grad_structure=[5,2], 
				 grad_offset=24,
				 fp16_training=True, #use when fp16 acc
				 mode='floatsd4',
				 verbose=True
				 ):
		assert (isinstance(optim, Adam_STU) or isinstance(optim, SGD_STU))
		assert (mode in ['floatsd4', 'floatsd8']), "Mode should be floatsd4 or floatsd8."
		super(OptimLP_STU, self).__init__(optim.param_groups, optim.defaults) # place holder

		# python dictionary does not copy by default
		self.param_groups = optim.param_groups
		self.optim = optim
		self.grad_rounding=grad_rounding
		self.grad_structure =  grad_structure
		self.grad_offset = grad_offset
		self.fp16_training = fp16_training
		self.mode = mode
		self.verbose = verbose
		
		if (mode == 'floatsd4'):
			self.cg = 1
		else:
			self.cg = 2
		self.full_converter = SDtoFloat('floatsd4', 8)
		self.updater = stu_updater('floatsd4')
		self.compute_converter = SDtoFloat('floatsd4', self.cg)
		self._init_weight()
		self.is_adam = False
		if (isinstance(optim, Adam_STU)):
			self.is_adam = True

		if (self.grad_rounding != 'identity'):
			if (verbose):
				print("Creating gradient quantizer...")
			self.grad_quantizer = quantizer(forward_rounding=grad_rounding, forward_fp_structure=grad_structure,
											forward_fp_offset=grad_offset, verbose=verbose, adaptive_offset=True)
			if (fp16_training):
				self.grad_quantizer = patch_float_to_half(self.grad_quantizer)
		else:
			self.grad_quantizer = None

	def _init_weight(self):
		for group in self.param_groups: 		
			for i, (p, sd_group, sd_exp, quant) in enumerate(zip(group['params'], group['sd_group'], group['sd_exp'], group['quant'])):
				if (quant):
					#sd_group0, sd_exp0, offset = xavier_uniform_(p.data, gain=1., mode='FloatSD4', movement=-1)
					sd_group0, sd_exp0, offset = kaiming_uniform_(p.data, a=0, mode='fan_in', nonlinearity='relu', movement=0)
					p.data = self.compute_converter(sd_group0, sd_exp0, offset) #float
					if (self.fp16_training):
						p.data = p.data.half()
					sd_group.data = sd_group0.data
					sd_exp.data = sd_exp0.data
					group['offset'][i] = offset

	def step(self, closure=None):
		"""
		Performs one step of optimization with the underlying optimizer.
		Quantizes gradient and momentum before stepping. Quantizes gradient accumulator and weight after stepping.
		"""

		loss = self.optim.step(self.updater, self.full_converter, self.grad_quantizer, self.fp16_training)

		#get computing group
		for group in self.param_groups:
			for i, (p, sd_group, sd_exp, offset, quant) in enumerate(zip(group['params'], group['sd_group'], group['sd_exp'], group['offset'], group['quant'])):
				if (quant):
					p.data = self.compute_converter(sd_group, sd_exp, offset) #float
					if (self.fp16_training):
						p.data = p.data.half()

		return loss

	def __repr__(self):
		return "LP Optimizer: {}".format(self.optim.__repr__())

	def __str__(self):
		return "LP Optimizer: {}".format(self.optim.__str__())

#TODO: little complicated
class Adam_STU(OptimLP_base):
	def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
				 weight_decay=0, amsgrad=False, quant_bias=False):
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
		sd_defaults = dict(method='stu', quant_bias=quant_bias)
		super(Adam_STU, self).__init__(params, defaults, sd_defaults)

	def __setstate__(self, state):
		super(Adam_STU, self).__setstate__(state)
		for group in self.param_groups:
			group.setdefault('amsgrad', False)

	def step(self, updater, full_converter, grad_quantizer=None, fp16_training=True, closure=None):
		"""Performs a single optimization step.

		Arguments:
			closure (callable, optional): A closure that reevaluates the model
				and returns the loss.
			updater : STU updater
			full_converter: SDtoFloat for 8 cg (for weight decay)
		"""
		loss = None
		if closure is not None:
			loss = closure()

		for group in self.param_groups:
			for i, (p, sd_group, sd_exp, offset, quant) in enumerate(zip(group['params'], group['sd_group'], group['sd_exp'], group['offset'], group['quant'])):
				if (quant): #use stu
					if p.grad is None:
						continue
					#grad quant
					if (grad_quantizer is not None):
						grad = grad_quantizer(p.grad.data)
					else:
						grad = p.grad.data

					if grad.is_sparse:
						raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
					amsgrad = group['amsgrad']

					state = self.state[p]

					# State initialization
					if len(state) == 0:
						state['step'] = 0
						# Exponential moving average of gradient values
						state['exp_avg'] = torch.zeros_like(p.data)
						# Exponential moving average of squared gradient values
						state['exp_avg_sq'] = torch.zeros_like(p.data)
						if amsgrad:
							# Maintains max of all exp. moving avg. of sq. grad. values
							state['max_exp_avg_sq'] = torch.zeros_like(p.data)

					#begin update
					exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
					if amsgrad:
						max_exp_avg_sq = state['max_exp_avg_sq']
					beta1, beta2 = group['betas']

					state['step'] += 1

					if group['weight_decay'] != 0:
						p.data = full_converter(sd_group, sd_exp, offset).data
						grad.add_(group['weight_decay'], p.data)

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

					update_value = torch.zeros_like(p)
					update_value.data.addcdiv_(-step_size, exp_avg, denom)
					updater(sd_group, sd_exp, update_value, offset)

				else: #normal weight
					if p.grad is None:
						continue

					if (grad_quantizer is not None):
						grad = grad_quantizer(p.grad.data)
					else:
						grad = p.grad.data
					#print(grad.dtype)
					if grad.is_sparse:
						raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
					amsgrad = group['amsgrad']

					state = self.state[p]

					# State initialization
					if len(state) == 0:
						state['step'] = 0
						# Exponential moving average of gradient values
						state['exp_avg'] = torch.zeros_like(p.data)
						# Exponential moving average of squared gradient values
						state['exp_avg_sq'] = torch.zeros_like(p.data)
						if amsgrad:
							# Maintains max of all exp. moving avg. of sq. grad. values
							state['max_exp_avg_sq'] = torch.zeros_like(p.data)

					exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
					if amsgrad:
						max_exp_avg_sq = state['max_exp_avg_sq']
					beta1, beta2 = group['betas']

					state['step'] += 1

					if group['weight_decay'] != 0:
						grad.add_(group['weight_decay'], p.data)

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

					p.data.addcdiv_(-step_size, exp_avg, denom)

		return loss


class SGD_STU(OptimLP_base):
	def __init__(self, params, lr=1e-3, momentum=0, dampening=0,
				 weight_decay=0, nesterov=False, quant_bias=False):
		if lr < 0.0:
			raise ValueError("Invalid learning rate: {}".format(lr))
		if momentum < 0.0:
			raise ValueError("Invalid momentum value: {}".format(momentum))
		if weight_decay < 0.0:
			raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

		defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
						weight_decay=weight_decay, nesterov=nesterov)
		sd_defaults = dict(method='stu', quant_bias=quant_bias)
		if nesterov and (momentum <= 0 or dampening != 0):
			raise ValueError("Nesterov momentum requires a momentum and zero dampening")
		super(SGD_STU, self).__init__(params, defaults, sd_defaults)

	def __setstate__(self, state):
		super(SGD_STU, self).__setstate__(state)
		for group in self.param_groups:
			group.setdefault('nesterov', False)

	def step(self, updater, full_converter, grad_quantizer=None, fp16_training=True, closure=None):
		"""Performs a single optimization step.

		Arguments:
			closure (callable, optional): A closure that reevaluates the model
				and returns the loss.
		"""
		loss = None
		if closure is not None:
			loss = closure()

		for group in self.param_groups:
			weight_decay = group['weight_decay']
			momentum = group['momentum']
			dampening = group['dampening']
			nesterov = group['nesterov']

			for i, (p, sd_group, sd_exp, offset, quant) in enumerate(zip(group['params'], group['sd_group'], group['sd_exp'], group['offset'], group['quant'])):
				if (quant): #use stu
					if p.grad is None:
						continue
					#grad quant
					if (grad_quantizer is not None):
						d_p = grad_quantizer(p.grad.data)
					else:
						d_p = p.grad.data

					if weight_decay != 0:
						p.data = full_converter(sd_group, sd_exp, offset).data #float
						if (fp16_training):
							p.data = p.data.half()
						d_p.add_(weight_decay, p.data)

					if momentum != 0:
						param_state = self.state[p]
						if 'momentum_buffer' not in param_state:
							buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
							buf.mul_(momentum).add_(d_p)
						else:
							buf = param_state['momentum_buffer']
							buf.mul_(momentum).add_(1 - dampening, d_p)
						if nesterov:
							d_p = d_p.add(momentum, buf)
						else:
							d_p = buf

					if (fp16_training):
						p.data = p.data.float() #updater only support float update value
						d_p.data = d_p.data.float()
					update_value = torch.zeros_like(p)
					update_value.data.add_(-group['lr'], d_p)

					if (fp16_training):
						d_p.data = d_p.data.half()

					updater(sd_group, sd_exp, update_value, offset)

				else: #normal weight
					if p.grad is None:
						continue

					if (grad_quantizer is not None):
						d_p = grad_quantizer(p.grad.data)
					else:
						d_p = p.grad.data

					if weight_decay != 0:
						d_p.add_(weight_decay, p.data)
					if momentum != 0:
						param_state = self.state[p]
						if 'momentum_buffer' not in param_state:
							buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
							buf.mul_(momentum).add_(d_p)
						else:
							buf = param_state['momentum_buffer']
							buf.mul_(momentum).add_(1 - dampening, d_p)
						if nesterov:
							d_p = d_p.add(momentum, buf)
						else:
							d_p = buf

					p.data.add_(-group['lr'], d_p)

		return loss
