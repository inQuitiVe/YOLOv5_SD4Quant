import torch
import numpy as np

#calculate MSE between data and quantized data
def calc_mse(idata, quantizer, offset):
	tmp = torch.zeros_like(idata)
	tmp.data = quantizer(idata.data, offset)
	criterion = torch.nn.MSELoss()
	mse = criterion(tmp.data, idata.data)
	return mse.item()

#calculate MSE between data and quantized data (chaneel-wise offset)
def calc_mse_channel_wise(idata, quantizer, offset):
	channel = idata.shape[0]
	tmp = torch.zeros_like(idata)
	tmp.data = quantizer(idata.data, offset)
	criterion = torch.nn.MSELoss(reduction = 'None')
	mse = criterion(tmp.data.view(channel, -1), idata.data.view(channel, -1))
	return mse

#calculate maximum quantization error between data and quantized data
def calc_minmax(idata, quantizer, offset):
	tmp = torch.zeros_like(idata)
	tmp.data = quantizer(idata.data, offset)
	mmax = torch.max(torch.abs(tmp.data - idata.data))
	return mmax.item()

#calculate histogram of data and quantized data
def calc_hist(idata, quantizer, offset):
	tmp = torch.zeros_like(idata)
	tmp.data = quantizer(idata.data, offset)

	p = idata.data.view(-1).float()
	q = tmp.data.view(-1).float()

	idx_p0 = (p == 0)
	p1 = p[~idx_p0]  # inequal with zero

	idx_q0 = (q == 0)
	q1 = q[~idx_q0]

	if p1.shape[0] == 0 or q1.shape[0] == 0:
		return None, None

	p1 = p1.abs().log2()
	q1 = q1.abs().log2()

	max_p = p1.max().ceil()
	min_p = p1.min().floor()

	max_q = q1.max().ceil()
	min_q = q1.min().floor()

	max_v = max(max_p, max_q)
	min_v = min(min_p, min_q)
	if (float(max_v) == float('inf') or float(min_v) == -float('inf') or 
		float(max_v) == float('nan') or float(min_v) == float('nan')):
		return None, None

	max_h = 16
	min_h = -24

	P = torch.histc(p1, bins=int(max_h - min_h), max=int(max_h), min=int(min_h))
	#P = torch.cat((idx_p0.sum().float().reshape((1,)), P)) #zero bin
	P = P / p.shape[0]

	Q = torch.histc(q1, bins=int(max_h - min_h), max=int(max_h), min=int(min_h))
	#Q = torch.cat((idx_q0.sum().float().reshape((1,)), Q)) #zero bin
	Q = Q / q.shape[0]

	return P, Q

#calculate Bhattacharyya Distance from given histogram
def calc_bd_from_hist(p_hist, q_hist):
	#p_hist = p_hist / p_hist.shape[0]
	#q_hist = q_hist / q_hist.shape[0]
	BD = torch.mul(p_hist, q_hist)
	BD = -BD.sqrt().sum().log()
	return BD.item()

#calculate Bhattacharyya Distance between data and quantized data
def calc_bd(idata, quantizer, offset):
	tmp = torch.zeros_like(idata)
	tmp.data = quantizer(idata.data, offset)

	p = idata.data.view(-1).float()
	q = tmp.data.view(-1).float()

	idx_p0 = (p == 0)
	p1 = p[~idx_p0]  # inequal with zero

	idx_q0 = (q == 0)
	q1 = q[~idx_q0]

	if p1.shape[0] == 0 or q1.shape[0] == 0:
		return float('inf')

	p1 = p1.abs().log2()
	q1 = q1.abs().log2()

	max_p = p1.max().ceil()
	min_p = p1.min().floor()

	max_q = q1.max().ceil()
	min_q = q1.min().floor()

	max_v = max(max_p, max_q)
	min_v = min(min_p, min_q)
	if (float(max_v) == float('inf') or float(min_v) == -float('inf') or 
		float(max_v) == float('nan') or float(min_v) == float('nan') or 
		int(max_q - min_q) == 0):
		return float('inf')

	P = torch.histc(p1, bins=int(max_q - min_q), max=int(max_q), min=int(min_q))
	# P = torch.cat((idx_p0.sum().float().reshape((1,)), P))
	P = P / p.shape[0]

	Q = torch.histc(q1, bins=int(max_q - min_q), max=int(max_q), min=int(min_q))
	# Q = torch.cat((idx_q0.sum().float().reshape((1,)), Q))
	Q = Q / q.shape[0]

	BD = torch.mul(P, Q)
	BD = -BD.sqrt().sum().log()
	return BD.item()

#calculate cosine similarity between data and quantized data
def calc_cosine_sim(idata, quantizer, offset):
	tmp = torch.zeros_like(idata)
	tmp.data = quantizer(idata.data, offset)
	criterion = torch.nn.CosineSimilarity()
	similarity = criterion(tmp.data.view(1,-1), idata.data.view(1,-1))
	if (similarity != similarity): #nan
		return -1
	else:
		return similarity.item()

#fused mse+minmax+bd+cos
def calc_distance(idata, quantizer, offset, metric="mse"):
	assert (metric in ["mse", "minmax", "bd", "cos"]), print("Invalid metric.")
	if (metric == "mse"):
		return calc_mse(idata, quantizer, offset)
	elif (metric == "bd"):
		return calc_bd(idata, quantizer, offset)
	elif (metric == "minmax"):
		return calc_minmax(idata, quantizer, offset)
	else:
		return calc_cosine_sim(idata, quantizer, offset)

#calculate best offset from given quantizer / metric
def calc_best_offset(idata, quantizer, min_offset, max_offset, metric="mse"):
	dist_min = float("inf")
	offset_min = min_offset
	for o in range(min_offset, max_offset+1):
		dist_now = calc_distance(idata, quantizer, o, metric)
		if (dist_now <= dist_min):
			dist_min = dist_now
			offset_min = o
	return offset_min, dist_min

#for loop => slow
#calculate best offset (channel-wise) from given quantizer / metric
def calc_best_offset_channel_wise(idata, quantizer, min_offset, max_offset, metric="mse"):
	offset_min = []
	channel = idata.shape[0]
	for c in range(channel):
		o_min, _ = calc_best_offset(idata[c], quantizer, min_offset[c], max_offset[c], metric=metric)
		offset_min.append(o_min)
	return offset_min

#calculate best offset from given quatizer / cos similarity metric
def calc_best_offset_cos(idata, quantizer, min_offset, max_offset):
	sim_max = -1
	offset_max = min_offset
	for o in range(min_offset, max_offset+1):
		sim_now = calc_distance(idata, quantizer, o, "cos")
		if (sim_now >= sim_max):
			sim_max = sim_now
			offset_max = o
	return offset_max, sim_max