import torch

__all__ = ["split_parameters", "split_parameters_stu"]


def split_parameters(model, quant_bias=False, fp32_mastercopy=False, black_list=[], **optim_params):
	#split model parameters to different groups
	#Parameters:
	#	model: model which parameters would be splitted
	#   quant_bias: whether to quantize bias or batchnorm weight/bias
	#   black_list: List of param name, param in black list would not be quant
	#	optim_params: optimizer parameters, ex:lr, should be in type {param: [param of no quant, param of quant]}
    param_quant = []
    param_no_quant = []
    master_copy = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ((len(param.shape) == 1) and not quant_bias):
            param_no_quant.append(param)
        elif (name in black_list):
        	param_no_quant.append(param)
        else:
            if (fp32_mastercopy):
                sd_master_copy = param.detach().clone().float()
            else:
                sd_master_copy = param.detach().clone().half()
            offset = 0
            param_quant.append(param)
            master_copy.append([sd_master_copy, offset])

    group_no_quant = {'params': param_no_quant, 'quant': False, 'sd_info': []}
    group_quant = {'params': param_quant, 'quant': True, 'sd_info': master_copy}

    for key, value in optim_params.items():
    	group_no_quant[key] = value[0]
    	group_quant[key] = value[1]

    return [group_no_quant, group_quant]

def split_parameters_stu(model, quant_bias=False, black_list=[], **optim_params):
    #split model parameters to different groups
    #Parameters:
    #   model: model which parameters would be splitted
    #   quant_bias: whether to quantize bias or batchnorm weight/bias
    #   black_list: List of param name, param in black list would not be quant
    #   optim_params: optimizer parameters, ex:lr, should be in type {param: [param of no quant, param of quant]}
    param_quant = []
    param_no_quant = []
    stu_master_copy = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ((len(param.shape) == 1) and not quant_bias):
            param_no_quant.append(param)
        elif (name in black_list):
            param_no_quant.append(param)
        else:
            sd_group = 0 #would be change to tensor in optimizer
            sd_exp = 0 #would be change to tensor in optimizer
            offset = 0
            param_quant.append(param)
            stu_master_copy.append([sd_group, sd_exp, offset])

    group_no_quant = {'params': param_no_quant, 'quant': False, 'sd_info': []}
    group_quant = {'params': param_quant, 'quant': True, 'sd_info':stu_master_copy}

    for key, value in optim_params.items():
        group_no_quant[key] = value[0]
        group_quant[key] = value[1]

    return [group_no_quant, group_quant]