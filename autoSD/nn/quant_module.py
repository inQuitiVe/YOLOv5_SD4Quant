import torch
import torch.nn as nn
from ..quant.quant_function import quantizer
from ..utils import patch_float_to_half, calc_best_offset, calc_mse, calc_hist, calc_bd_from_hist, calc_minmax
from ..post_train.util import get_hist, decide_struct_and_offset_from_hist
import numpy as np
import math

__all__ = ['Quantizer']

class Quantizer(nn.Module):
    def __init__(self, forward_rounding="identity", backward_rounding="identity", forward_fp_structure=[5,2],
                 backward_fp_structure=[5,2], forward_fp_offset=24, backward_fp_offset=24,
                 verbose=True, half=False, adaptive=False):
        super(Quantizer, self).__init__()
        self.forward_rounding = forward_rounding
        self.backward_rounding = backward_rounding
        self.half = half
        self.forward_fp_offset = forward_fp_offset
        self.backward_fp_offset = backward_fp_offset
        self.forward_fp_structure = forward_fp_structure
        self.backward_fp_structure = backward_fp_structure
        self.verbose = verbose
        self.adaptive = adaptive #adaptive offset + structure

        self.quantize = quantizer(forward_rounding=forward_rounding, backward_rounding=backward_rounding, 
                                  forward_fp_structure=forward_fp_structure, backward_fp_structure=backward_fp_structure, 
                                  forward_fp_offset=forward_fp_offset, backward_fp_offset=backward_fp_offset, 
                                  verbose=verbose, adaptive_offset=adaptive, 
                                  adaptive_structure=adaptive)
        if self.half:
            self.quantize = patch_float_to_half(self.quantize)

        #for forward calibration
        self.collect_fwd = False
        self.forward_act_list = None
        self.forward_metric = "mse"
        self.forward_srange = 1
        self.iter_collect_fwd = 0
        #fwd adapter
        adpt_temp = quantizer(forward_rounding=forward_rounding, forward_fp_structure=forward_fp_structure, verbose=False, adaptive_offset=True)
        if self.half:
            adpt_temp = patch_float_to_half(adpt_temp)
        self.forward_adapter = adpt_temp

        #for post-training
        self.collect_hist = False
        self.fwd_hist = None

        #for c-wise forward offset
        self.collect_max = False
        self.fwd_max = None
        self.cwise = False

        #bias correction
        self.collect_iter_gt = 0
        self.collect_iter_q = 0
        self.collect_mean_gt = False
        self.collect_mean_q = False
        self.fwd_mean_gt = None
        self.fwd_mean_q = None
        self.dim = 0
        self.b_correct = False

        #for backward calibration
        self.handle = None
        self.backward_act_list = None
        self.backward_metric = "mse"
        self.backward_srange = 1
        self.iter_collect_bwd = 0
        #bwd adapter
        adpt_temp = quantizer(forward_rounding=backward_rounding, forward_fp_structure=backward_fp_structure, verbose=False, adaptive_offset=True)
        if self.half:
            adpt_temp = patch_float_to_half(adpt_temp)
        self.backward_adapter = adpt_temp

        #for post-training
        self.collect_bwd_hist = False
        self.bwd_hist = None

    #calculate minmum exp offset from fp struct
    def _calc_min_offset(self, struct):
        exp, mantissa = struct
        if (mantissa > 0):
            min_offset = 2**exp - 16
        else:
            min_offset = 2**exp - 16 - 1
        return min_offset

    def forward(self, x):
        if (self.collect_fwd):
            self._forward_collect_act(x)
            return x #identity
        elif (self.collect_hist):
            self._forward_collect_hist(x)
        elif (self.collect_max):
            self._forward_collect_max(x)

        #forward
        if (self.adaptive):
            if (self.cwise):
                channel = x.shape[1]
                ee = [self.forward_fp_structure[0]] * channel
                mm = [self.forward_fp_structure[1]] * channel
                x = x.transpose(1, 0) #since our cwise quantizer expect outermost dimension as channel
                out = self.quantize(x, self.forward_fp_cwise_offset, ee, mm)
                out = out.transpose(1,0)
            else:
                out = self.quantize(x, self.forward_fp_offset, self.forward_fp_structure[0], self.forward_fp_structure[1])
        else:
            out = self.quantize(x)

        #Mean collection should be performed after quantization
        if (self.collect_mean_gt):
            self._forward_collect_mean_gt(out)
        elif (self.collect_mean_q):
            self._forward_collect_mean_q(out)

        #bias correction
        if (self.b_correct):
            out = out - self.correction
        return out

    #=== Train
    #forward & collect activation for calibration
    def _forward_collect_act(self, x):
        tmpp = x.detach().clone()
        self.iter_collect_fwd += 1

        maxx = 24
        minn = self._calc_min_offset(self.forward_fp_structure)
        if self.forward_srange > 0:
            maxx = min(self.forward_fp_offset + self.forward_srange, maxx)
            minn = max(self.forward_fp_offset - self.forward_srange, minn)
        #create list of metrics for each offset
        if (self.forward_act_list is None):
            self.forward_act_list = [0 for i in range(minn, maxx+1)]
        #get metric of each offset
        for i, o in enumerate(range(minn, maxx+1)):
            if (self.forward_metric == "mse"):
                mse = calc_mse(tmpp, self.forward_adapter, o)
                self.forward_act_list[i] += mse
            elif (self.forward_metric == "minmax"):
                maxx = calc_minmax(tmpp, self.forward_adapter, o)
                if (maxx > self.forward_act_list[i]):
                    self.forward_act_list[i] = maxx
            elif (self.forward_metric == "bd"):
                p_hist, q_hist = calc_hist(tmpp, self.forward_adapter, o)
                if (p_hist is not None):
                    if (self.forward_act_list[i] == 0):
                        self.forward_act_list[i] = [p_hist, q_hist]
                    elif (self.forward_act_list[i][0] is not None):
                        self.forward_act_list[i][0] += p_hist
                        self.forward_act_list[i][1] += q_hist
                else:
                    self.forward_act_list[i] = [None, None]

    #find best fwd exp offset from collected data
    def forward_calibrate(self, name=None):
        dist_min = float('inf')
        offset_new = self.forward_fp_offset

        if (self.iter_collect_fwd > 0):
            maxx = 24
            minn = self._calc_min_offset(self.forward_fp_structure)
            if self.forward_srange > 0:
                maxx = min(self.forward_fp_offset + self.forward_srange, maxx)
                minn = max(self.forward_fp_offset - self.forward_srange, minn)
            #find best offset
            for i, offset_now in enumerate(range(minn, maxx+1)):
                if (self.forward_metric == "mse"):
                    mse_now = self.forward_act_list[i]
                    if (mse_now <= dist_min):
                        offset_new = offset_now
                        dist_min = mse_now

                elif (self.forward_metric == "minmax"):
                    max_now = self.forward_act_list[i]
                    if (max_now <= dist_min):
                        offset_new = offset_now
                        dist_min = max_now

                elif (self.forward_metric == "bd"):
                    if (self.forward_act_list[i] != 0):
                        p_hist, q_hist = self.forward_act_list[i]
                        if (p_hist is not None):
                            p_hist = p_hist / self.iter_collect_fwd
                            q_hist = q_hist / self.iter_collect_fwd
                            bd_now = calc_bd_from_hist(p_hist, q_hist)
                        else:
                            bd_now = float('inf')
                    else:
                        bd_now = float('inf')

                    if (bd_now <= dist_min):
                        offset_new = offset_now
                        dist_min = bd_now

            self.forward_act_list = None #reset fwd_act_list
            self.iter_collect_fwd = 0
        '''
        if (torch.distributed.is_initialized()):
            #print('before:',offset_new)
            offset_new_sync = torch.FloatTensor([offset_new]).cuda()
            torch.distributed.broadcast(offset_new_sync, 0)
            offset_new = int(offset_new_sync.item())
            #print('after:', offset_new)'''

        #only print on main rank
        rank_0_flag = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        self.verbose = self.verbose and rank_0_flag
        if self.verbose and (self.forward_fp_offset != offset_new):
            if (name is None):
                print('  Forward exp offset changes from {} to {}.'.format(self.forward_fp_offset, offset_new))
            else:
                print('  Forward exp offset ({}) changes from {} to {}.'.format(name, self.forward_fp_offset, offset_new))

        self.forward_fp_offset = offset_new
        #update quantizer
        self.rebuild()

    #register hook for collecting backward activation
    def register_calibration_hook_train(self):
        def backward_hook_fn(module, grad_in, grad_out):
            tmpp = grad_out[0].detach().clone()
            self.iter_collect_bwd += 1
            maxx = 24
            minn = self._calc_min_offset(self.backward_fp_structure)
            if self.backward_srange > 0:
                maxx = min(self.backward_fp_offset + self.backward_srange, maxx)
                minn = max(self.backward_fp_offset - self.backward_srange, minn)
            #create list of metrics for each offset
            if (self.backward_act_list is None):
                self.backward_act_list = [0 for i in range(minn, maxx+1)]
            #get metric of each offset
            for i, o in enumerate(range(minn, maxx+1)):
                if (self.backward_metric == "mse"):
                    mse = calc_mse(tmpp, self.backward_adapter, o)
                    self.backward_act_list[i] += mse
                elif (self.backward_metric == "minmax"):
                    maxx = calc_minmax(tmpp, self.backward_adapter, o)
                    if (maxx > self.backward_act_list[i]):
                        self.backward_act_list[i] = maxx
                elif (self.backward_metric == "bd"):
                    p_hist, q_hist = calc_hist(tmpp, self.backward_adapter, o)
                    if (p_hist is not None):
                        if (self.backward_act_list[i] == 0):
                            self.backward_act_list[i] = [p_hist, q_hist]
                        elif (self.backward_act_list[i][0] is not None):
                            self.backward_act_list[i][0] += p_hist
                            self.backward_act_list[i][1] += q_hist
                    else:
                        self.backward_act_list[strr][i] = [None, None]
            return None

        self.handle = self.register_backward_hook(backward_hook_fn)

    #find best bwd exp offset from collected data
    def backward_calibrate(self, name=None):
        dist_min = float('inf')
        offset_new = self.backward_fp_offset

        if (self.iter_collect_bwd > 0):
            maxx = 24
            minn = self._calc_min_offset(self.backward_fp_structure)
            if self.backward_srange > 0:
                maxx = min(self.backward_fp_offset + self.backward_srange, maxx)
                minn = max(self.backward_fp_offset - self.backward_srange, minn)
            #find best offset
            for i, offset_now in enumerate(range(minn, maxx+1)):
                if (self.backward_metric == "mse"):
                    mse_now = self.backward_act_list[i]
                    if (mse_now <= dist_min):
                        offset_new = offset_now
                        dist_min = mse_now

                elif (self.backward_metric == "minmax"):
                    max_now = self.backward_act_list[i]
                    if (max_now <= dist_min):
                        offset_new = offset_now
                        dist_min = max_now

                elif (self.backward_metric == "bd"):
                    if (self.backward_act_list[i] != 0):
                        p_hist, q_hist = self.backward_act_list[i]
                        if (p_hist is not None):
                            p_hist = p_hist / self.iter_collect_bwd
                            q_hist = q_hist / self.iter_collect_bwd
                            bd_now = calc_bd_from_hist(p_hist, q_hist)
                        else:
                            bd_now = float('inf')
                    else:
                        bd_now = float('inf')

                    if (bd_now <= dist_min):
                        offset_new = offset_now
                        dist_min = bd_now

            self.backward_act_list = None #reset bwd_act_list
            self.iter_collect_bwd = 0

        '''
        if (torch.distributed.is_initialized()): //all rank used same calibration offset
            #print('before:',offset_new)
            offset_new_sync = torch.FloatTensor([offset_new]).cuda()
            torch.distributed.broadcast(offset_new_sync, 0)
            offset_new = int(offset_new_sync.item())
            #print('after:', offset_new)'''

        #only print on main rank
        rank_0_flag = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        self.verbose = self.verbose and rank_0_flag
        if self.verbose and (self.backward_fp_offset != offset_new):
            if (name is None):
                print('  Backward exp offset changes from {} to {}.'.format(self.backward_fp_offset, offset_new))
            else:
                print('  Backward exp offset ({}) changes from {} to {}.'.format(name, self.backward_fp_offset, offset_new))

        self.backward_fp_offset = offset_new
        #update quantizer
        self.rebuild()

    #=== Post-train
    #forward & collect histogram for post training structure decision
    def _forward_collect_hist(self, x):
        tmpp = x.detach().clone()
        p_hist = get_hist(tmpp)
        if (self.fwd_hist is None):
            self.fwd_hist = p_hist
        else:
            self.fwd_hist += p_hist

    #forward & collect maximum number for each output channel 
    def _forward_collect_max(self, x):
        channel = x.shape[1]
        tmpp = x.detach().clone() #(batch, CO, W, H) or (batch, CO)
        tmpp = tmpp.transpose(1, 0) #(CO, batch, W, H) or (CO, batch)
        tmpp = tmpp.reshape(channel, -1).abs()
        maxx = torch.max(tmpp, dim=1)[0] #(CO)
        if (self.fwd_max is None):
            self.fwd_max = maxx
        else:
            self.fwd_max = torch.max(maxx, self.fwd_max)

    #forward & collect mean for each output channel (for ground truth in bias correction)
    def _forward_collect_mean_gt(self, x):
        self.collect_iter_gt += 1
        self.dim = x.dim()
        channel = x.shape[1]
        tmpp = x.detach().clone() #(batch, CO, W, H) or (batch, CO)
        tmpp = tmpp.transpose(1, 0) #(CO, batch, W, H) or (CO, batch)
        tmpp = tmpp.reshape(channel, -1)
        mean = torch.mean(tmpp, dim=1) #(CO)
        if (self.fwd_mean_gt is None):
            self.fwd_mean_gt = mean.clone()
        else:
            self.fwd_mean_gt.add_(mean)

    #forward & collect mean for each output channel (for quantized data in bias correction)
    def _forward_collect_mean_q(self, x):
        self.collect_iter_q += 1
        channel = x.shape[1]
        tmpp = x.detach().clone() #(batch, CO, W, H) or (batch, CO)
        tmpp = tmpp.transpose(1, 0) #(CO, batch, W, H) or (CO, batch)
        tmpp = tmpp.reshape(channel, -1)
        mean = torch.mean(tmpp, dim=1) #(CO)
        if (self.fwd_mean_q is None):
            self.fwd_mean_q = mean.clone()
        else:
            self.fwd_mean_q.add_(mean)

    #perform bias correction from collected gt_mean / q_mean
    def bias_correct(self):
        #normalize by collected iterations
        self.fwd_mean_q /= self.collect_iter_q
        self.fwd_mean_gt /= self.collect_iter_gt
        #calculate correction term
        self.correction = self.fwd_mean_q - self.fwd_mean_gt #correction term
        if (self.dim == 4):
            self.correction = self.correction.reshape(-1, 1, 1)
        #clear data
        self.fwd_mean_q = None
        self.fwd_mean_gt = None
        self.collect_iter_q = 0
        self.collect_iter_gt = 0
        self.collect_mean_gt = False
        self.collect_mean_q = False
        #turn on bias corrected mode
        self.b_correct = True

    #for post-training calibration => decide offset
    def post_forward_calibrate(self, name=None):
        (ee, mm), maxx = decide_struct_and_offset_from_hist(self.fwd_hist)
        self.fwd_hist = None
        ee, mm = self.forward_fp_structure #Not replace fp_structure since we only need maxx
        if (mm > 0):
            self.forward_fp_offset = 2**ee - 1 - maxx
        else:
            self.forward_fp_offset = 2**ee - 2 - maxx
        
        if (self.verbose):
            print("Layer: {}, offset:{}, exp: {}, mantissa: {}".format(name, self.forward_fp_offset, ee, mm))
        self.rebuild()

    #for post-training calibration => decide channel-wise offset
    def post_forward_calibrate_cwise(self, name=None):
        ee, mm = self.forward_fp_structure
        channel = self.fwd_max.shape[0]
        offsets = []
        for c in range(channel):
            maxx = self.fwd_max[c].item()
            if (maxx == 0):
                offsets += [self.forward_fp_offset]
                continue
            bound = int(math.ceil(math.log2(maxx)))
            d = 2**bound - maxx
            if (d > 2**(bound-2-mm)):
                bound -= 1

            if (self.forward_rounding == 'FloatSD8'):  # floatsd8
                offset = 8 - bound
            elif (self.forward_rounding == 'FloatSD4_ex'):
                offset = 6 - bound
            elif (self.forward_rounding == 'FloatSD4'):
                offset = 5 - bound
            else:
                if (mm > 0):
                    offset = 2**ee - 1 - bound
                else:
                    offset = 2**ee - 2 - bound
            offsets += [offset]
        self.fwd_max = None
        self.cwise = True
        self.forward_fp_cwise_offset = offsets
        print("Layer: {}, offset:{}, exp: {}, mantissa: {}".format(name, self.forward_fp_cwise_offset[0], ee, mm))
        self.rebuild()

    #register hook for collecting backward activation (post train)
    def register_calibration_hook_post_train(self):
        def backward_hook_fn(module, grad_in, grad_out):
            tmpp = grad_out[0].detach().clone()
            p_hist = get_hist(tmpp)
            if (self.bwd_list is None):
                self.bwd_list = p_hist
            else:
                self.bwd_list += p_hist
            return None

        self.handle = self.register_backward_hook(backward_hook_fn) 

    #for post-training calibration => adaptive structure
    def post_backward_calibrate(self, name=None):
        (ee, mm), maxx = decide_struct_and_offset_from_hist(self.bwd_hist)
        self.bwd_hist = None
        self.backward_structure = [ee, mm]
        self.backward_fp_offset = 2**ee - 1 - maxx
        print("Layer: {}, offset:{}, exp: {}, mantissa: {}".format(name, self.backward_fp_offset, ee, mm))
        self.rebuild()

    #reconstruct quantizer
    def rebuild(self):
        self.quantize = quantizer(self.forward_rounding, self.backward_rounding, forward_fp_structure=self.forward_fp_structure,
                                  backward_fp_structure=self.backward_fp_structure, forward_fp_offset=self.forward_fp_offset, 
                                  backward_fp_offset=self.backward_fp_offset, verbose=False,
                                  adaptive_offset=self.adaptive, adaptive_structure=self.adaptive, channel_wise=self.cwise)
        if self.half:
            self.quantize = patch_float_to_half(self.quantize)

    #reconfigure Quantizer setting
    def reconfigure(self, forward_rounding=None, backward_rounding=None, forward_fp_structure=None,
                    backward_fp_structure=None, forward_fp_offset=None, backward_fp_offset=None):
        if (forward_rounding is not None):
            self.forward_rounding = forward_rounding
        if (backward_rounding is not None):
            self.backward_rounding = backward_rounding
        if (forward_fp_structure is not None):
            self.forward_fp_structure = forward_fp_structure
        if (backward_fp_structure is not None):
            self.backward_fp_structure = backward_fp_structure
        if (forward_fp_offset is not None):
            self.forward_fp_offset = forward_fp_offset
        if (backward_fp_offset is not None):
            self.backward_fp_offset = backward_fp_offset
        self.rebuild()

    def __repr__(self):
        return "\n  Quantizer:\n    forward rounding: {} / forward structure: [{},{}]\n    backward rounding: {} / backward structure: [{},{}]".format(
                self.forward_rounding, self.forward_fp_structure[0], self.forward_fp_structure[1], 
                self.backward_rounding, self.backward_fp_structure[0], self.backward_fp_structure[1])

    def __str__(self):
        return "\n  Quantizer:\n    forward rounding: {} / forward structure: [{},{}]\n    backward rounding: {} / backward structure: [{},{}]".format(
                self.forward_rounding, self.forward_fp_structure[0], self.forward_fp_structure[1], 
                self.backward_rounding, self.backward_fp_structure[0], self.backward_fp_structure[1])
