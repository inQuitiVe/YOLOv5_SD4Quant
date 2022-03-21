import torch
import numpy as np
import math
from ..quant.quant_function import quantizer
from .util import decide_struct_2, decide_struct_cwise
from ..utils import patch_float_to_half

class Post_training_quantizer():
	def __init__(self,
				 param,
				 weight_rounding='floatsd4_ex',
				 weight_structure=[5,2],
				 weight_offset=None,
				 channel_wise=False,
				 verbose=True
				 ):

		assert weight_rounding in ['floatsd4', 'floatsd4_ex', 'floatsd8', 'fp'], "Invalid rounding mode."
		if (channel_wise):
			assert weight_rounding in ['floatsd4_ex', 'fp'], "Channel-wise offset is only supported on floatsd4_ex / fp."

		self.param_groups = param
		self.weight_rounding = weight_rounding
		self.weight_offset = weight_offset
		self.weight_structure = weight_structure
		self.channel_wise = channel_wise
		self.verbose = verbose

		self._build_master_copy()

	def _build_master_copy(self):
		for group in self.param_groups:
			group['sd_master_copy'] = [None] * len(group['params'])
			for i, p in enumerate(group['params']):
				group['sd_master_copy'][i] = p.detach().clone().float()

	#quantize weight
	def _rebuild(self):
		for group in self.param_groups:
			for i, (p, master, offset, quant) in enumerate(zip(group['params'], group['sd_master_copy'], group['offset'], group['quant'])):
				if quant:  # use quant
					if (self.adaptive_structure):
						if (self.weight_rounding == 'fp'):
							ee, mm = group['weight_structure'][i]
							p.data.copy_(self.weight_quantizer(master.data, offset, ee, mm))
						else:
							p.data.copy_(self.weight_quantizer(master.data, offset))
					else:
						p.data.copy_(self.weight_quantizer(master.data, offset))

	def step(self, adaptive_structure=False, mode='anal'):
		assert mode in ['anal', 'exhaust']
		if (adaptive_structure):
			self.adaptive_structure = True
			if (mode == 'anal'):
				if (self.channel_wise):
					self._init_offset_and_structure_cwise()
				else:
					self._init_offset_and_structure()
			else:
				#create quantizer here since it would be used in seflf._init_offset_and_structure_2()
				self.weight_quantizer = quantizer(forward_rounding=self.weight_rounding, forward_fp_structure=self.weight_structure,
									   			  verbose=False, adaptive_offset=True, 
									   			  adaptive_structure=self.adaptive_structure, channel_wise=self.channel_wise)
				self._init_offset_and_structure_2()
		else:
			self.adaptive_structure = False
			if (self.channel_wise):
				self._init_offset_and_structure_cwise()
			else:
				self._init_offset_and_structure()
		#turn on adaptive structure since we doesn't support cwise quantization function for List[offset], exp, mantissa
		self.adaptive_structure = True 
		self.weight_quantizer = quantizer(forward_rounding=self.weight_rounding, forward_fp_structure=self.weight_structure,
										  verbose=False, adaptive_offset=True, 
										  adaptive_structure=self.adaptive_structure, channel_wise=self.channel_wise)
		self._rebuild()

	def reconfigure(self, weight_rounding=None, weight_structure=None,
				 	weight_offset=None, channel_wise=None):
		if (weight_rounding is not None):
			self.weight_rounding = weight_rounding
		if (weight_structure is not None):
			self.weight_structure = weight_structure
		if (channel_wise is not None):
			self.channel_wise = channel_wise
		self.weight_offset = weight_offset

	#helper function for decide struct & offset (anal mode)
	def _initial(self, idata):
		#structure
		if (self.adaptive_structure):
			total = self.weight_structure[0] + self.weight_structure[1] + 1
			ee, mm = decide_struct_2(idata, total=total)
		else:
			ee, mm = self.weight_structure
		#weight offset
		max_abs = torch.max(idata.abs()).item()
		bound = int(math.ceil(math.log2(max_abs)))
		d = 2**bound - max_abs
		if (d > 2**(bound-2-mm)):
			bound -= 1

		if (self.weight_rounding == 'floatsd8'):  #floatsd8
			offset = 8 - bound
		elif (self.weight_rounding == 'floatsd4_ex'):
			offset = 6 - bound
		elif (self.weight_rounding == 'floatsd4'):
			offset = 5 - bound
		else: #fp
			if (mm > 0):
				offset = 2**ee - 1 - bound
			else:
				offset = 2**ee - 2 - bound
		return ee, mm, offset

	#non channel-wise offset
	def _init_offset_and_structure(self):
		for group in self.param_groups:
			group['weight_structure'] = [None] * len(group['sd_master_copy'])
			group['offset'] = [None] * len(group['sd_master_copy'])
			for i, (p, quant) in enumerate(zip(group['sd_master_copy'], group['quant'])):
				if (quant):
					ee, mm, offset = self._initial(p)
					group['weight_structure'][i] = [ee, mm]
					if (self.weight_offset is not None):
						group['offset'][i] = self.weight_offset
					else:
						group['offset'][i] = offset

					if (self.verbose):
						if (self.weight_rounding == 'fp'):
							print("Weight index: {}, offset: {}, exp: {}, mantissa: {}".format(i, offset, ee, mm))
						else:
							print("Weight index: {}, offset: {}".format(i, offset))
	
	#channel-wise offset
	def _init_offset_and_structure_cwise(self):
		for group in self.param_groups:
			group['weight_structure'] = [None] * len(group['sd_master_copy'])
			group['offset'] = [None] * len(group['sd_master_copy'])
			for i, (p, quant) in enumerate(zip(group['sd_master_copy'], group['quant'])):
				if (quant):
					channel = p.shape[0]
					ee_list = []
					mm_list = []
					offset_list = []
					for c in range(channel):
						ee, mm , offset = self._initial(p[c])
						ee_list += [ee]
						mm_list += [mm]
						offset_list += [offset]

					group['weight_structure'][i] = [ee_list, mm_list]
					if (self.weight_offset is not None):
						group['offset'][i] = self.weight_offset
					else:
						group['offset'][i] = offset_list

					if (self.verbose):
						if (self.weight_rounding == 'fp'):
							print("Weight index: {}, offset: {}, exp: {}, mantissa: {}".format(i, offset_list[0], ee_list[0], mm_list[0]))
						else:
							print("Weight index: {}, offset: {}".format(i, offset_list[0]))

	#exhaust mode structure / non channel-wise offset
	def _init_offset_and_structure_2(self): #search for best structure
		for group in self.param_groups:
			group['weight_structure'] = [None] * len(group['sd_master_copy'])
			group['offset'] = [None] * len(group['sd_master_copy'])
			for i, (p, quant) in enumerate(zip(group['sd_master_copy'], group['quant'])):
				if (quant): 
					#bound
					max_abs = torch.max(p.abs()).item()
					bound = int(math.ceil(math.log2(max_abs)))
					d = 2**bound - max_abs
					# struct
					total = self.weight_structure[0] + self.weight_structure[1] + 1
					if (total == 8):
						search_list = [[2,5], [3,4], [4,3], [5,2]]
					elif (total == 6):
						search_list = [[2,3], [3,2], [4,1], [5,0]]
					else:
						search_list = [[2,1],[3,0]]

					criterion = torch.nn.MSELoss()
					e_min = float('inf')
					best_struct = 0
					for ii in range(len(search_list)):
						ee, mm = search_list[ii]
						if (d > 2**(bound-2-mm)):
							bound_2 = bound - 1
						else:
							bound_2 = bound
						if (mm > 0):
							offset = 2**ee - 1 - bound_2
						else:
							offset = 2**ee - 2 - bound_2
						tmp = torch.zeros_like(p)
						tmp.data = self.weight_quantizer(p.data, offset, ee, mm)
						e_now = criterion(p, tmp).item()
						if (e_now < e_min):
							e_min = e_now
							best_struct = ii

					ee, mm = search_list[best_struct]
					if (d > 2**(bound-2-mm)):
						bound -= 1
					#weight offset
					if (self.weight_rounding == 'floatsd8'):  #floatsd8
						offset = 8 - bound
						self.w_offset_min = -7
					elif (self.weight_rounding == 'floatsd4_ex'):
						offset = 6 - bound
						self.w_offset_min = -9
					elif (self.weight_rounding == 'floatsd4'):
						offset = 5 - bound
						self.w_offset_min = -10
					else: #float
						if (mm > 0):
							offset = 2**ee - 1 - bound
						else:
							offset = 2**ee - 2 - bound

					group['weight_structure'][i] = [ee, mm]
					if (self.weight_offset is not None):
						group['offset'][i] = self.weight_offset
					else:
						group['offset'][i] = offset

					if (self.verbose):
						if (self.weight_rounding == 'fp'):
							print("Weight index: {}, offset: {}, exp: {}, mantissa: {}".format(i, offset, ee, mm))
						else:
							print("Weight index: {}, offset: {}".format(i, offset))
