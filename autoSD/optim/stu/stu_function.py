import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.cpp_extension import load
import os
current_path = os.path.dirname(os.path.realpath(__file__))

if torch.cuda.is_available():
     stu_cuda = load(
             name='stu_cuda',
             sources=[
                 os.path.join(current_path, "stu_cuda/stu_cuda.cpp"),
                 os.path.join(current_path, "stu_cuda/stu_helper.cu"),
                 os.path.join(current_path, "stu_cuda/stu_kernel.cu"),
                 os.path.join(current_path, "stu_cuda/stu.cu")
                 ]
             )

__all__ = ['stu_updater', 'SDtoFloat', 'random_init']


def stu_updater(mode='FloatSD4'):
    assert (mode not in ['FloatSD4, FloatSD8']), "Mode should be FloatSD4 or FloatSD8."
    if (mode == 'FloatSD4'):
        mode_num = 1
    else:
        mode_num = 0
    return lambda sd_group, sd_exp, update_value, offset: stu_cuda.fsd_update(sd_group, sd_exp, update_value, offset, mode_num)

def SDtoFloat(mode='FloatSD4', cg=1):
    assert (mode in ['FloatSD4', 'FloatSD8']), "Mode should be FloatSD4 or FloatSD8."
    assert ((cg > 0) and (cg < 9)), "Computing group should be 1~8."
    if (mode == 'FloatSD4'):
        mode_num = 1
    else:
        mode_num = 0
    return lambda sd_group, sd_exp, offset: stu_cuda.get_sd_value(sd_group, sd_exp, offset, mode_num, cg)

def random_init(mode='FloatSD4'):
    assert (mode not in ['FloatSD4, FloatSD8']), "Mode should be FloatSD4 or FloatSD8."
    if (mode == 'FloatSD4'):
        mode_num = 1
    else:
        mode_num = 0
    return lambda sd_group: stu_cuda.init_group(sd_group, mode_num)

