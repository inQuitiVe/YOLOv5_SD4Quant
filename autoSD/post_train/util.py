import numpy as np
import torch

def get_te(be=4, total=8):
	#c_value = {2: 1.0270572699885987, 3: 1.1880153276580563, 4: 1.4191346007268955, 5: 0.6539397209821505}
	#cos = c_value[be]
	bm = total - 1 - be
	threshold = 4  * (7 * 2**(2*bm+2)-1) / (3 * 2**(2**be))
	return threshold

def get_te_2(be=4, total=8):
	bm = total - 1 - be
	threshold = 4  / (3 * 2**(2**be))
	return threshold

#channel wise decide struct 2
def decide_struct_cwise(idata, total=8):
	channel = idata.shape[0]
	exp_list = []
	m_list = []
	for c in range(channel):
		ee, mm = decide_struct_2(idata[c], total=total)
		exp_list += [ee]
		m_list += [mm]
	return exp_list, m_list

#consider saturation
def decide_struct_2(idata, total=8):
	tmp = idata.data.view(-1).float()
	idx0 = (tmp == 0)
	tmp1 = tmp[~idx0]  # inequal with 0
	tmp1 = tmp1.abs().log2()

	max_p = tmp1.max().ceil()
	min_p = tmp1.min().floor()

	if (max_p == min_p):
		return total-1, 0

	hist = torch.histc(tmp1, bins=int(max_p - min_p), max=int(max_p), min=int(min_p))
	hist = hist.cpu().numpy()
	hist = hist[::-1]  # big to small
	if (len(hist) < 33):
		diff = 33 - len(hist)
		hist = np.append(hist, [0] * diff)
	else:
		hist = hist[0:33]

	if (total >= 6):
		exp_list = [2, 3, 4]  # exp can be 2,3,4,5
	else:
		exp_list = [1, 2]  # exp can be 1,2,3

	threshold = []
	idx_t = []
	for exp in exp_list:
		threshold += [get_te_2(exp + 1, total)]
		idx_t += [2 ** exp]

	case = 0
	for idx, e in zip(idx_t, exp_list):
		hist_q = hist[0:idx]
		hist_z = hist[idx:2*idx+1]
		bm = total - 2 - e
		e_list_q = [(0.25) ** i for i in range(idx)]
		e_list_z = []
		for i in range(idx+1):
			if (i == 0):
				aa = (1 / 2**bm + 1)**3 - (1 / 2**bm)**3
				e_list_z += [(aa * 2 ** (2 * bm + 2) - 1) * ((0.25) ** i)]
			elif (i == idx):
				aa = (2 / 2**bm + 1)**3 - (2 / 2**bm)**3
				e_list_z += [(7 - aa) / 3 * ((0.25) ** i) * (2 ** (2 * bm + 2))]
			else:
				e_list_z += [(7 * 2 ** (2 * bm + 2) - 1) * ((0.25) ** i)]
		inner_q = np.inner(hist_q, e_list_q)
		inner_z = np.inner(hist_z, e_list_z)
		if (inner_z == 0 or inner_q / inner_z > threshold[case]):
			exp = exp_list[case]
			return exp, total - 1 - exp
		else:
			if (case == len(exp_list) - 1):
				exp = exp_list[-1] + 1
				return exp, total - 1 - exp
			else:
				case += 1

def decide_struct(idata, total=8):
	tmp = idata.data.view(-1).float()
	idx0 = (tmp == 0)
	tmp1 = tmp[~idx0] #inequal with 0
	tmp1 = tmp1.abs().log2()

	max_p = tmp1.max().ceil()
	min_p = tmp1.min().floor()

	hist = torch.histc(tmp1, bins=int(max_p-min_p), max=int(max_p), min=int(min_p))
	hist = hist.cpu().numpy()
	hist = hist[::-1] #big to small
	if (len(hist) < 32):
		diff = 32 - len(hist)
		hist = np.append(hist, [0] * diff)
	else:
		hist = hist[0:32]

	if (total >= 6):
		exp_list = [2, 3, 4] #exp can be 2,3,4,5
	else:
		exp_list = [1, 2] #exp can be 1,2,3

	threshold = []
	idx_t = []
	for exp in exp_list:
		threshold += [get_te(exp+1, total)]
		idx_t += [2**exp]
	
	case = 0
	for idx in idx_t:
		hist_q = hist[0:idx]
		hist_z = hist[idx:2*idx]
		e_list = np.array([(0.25)**i for i in range(idx)])
		inner_q = np.inner(hist_q, e_list)
		inner_z = np.inner(hist_z, e_list)
		if (inner_z == 0 or inner_q/inner_z > threshold[case]):
			exp = exp_list[case]
			return exp, total-1-exp
		else:
			if (case == len(exp_list)-1):
				exp = exp_list[-1] + 1
				return exp, total-1-exp
			else:
				case += 1

def decide_struct_and_offset_from_hist(hist, total=8):
	if (total >= 6):
		exp_list = [2, 3, 4] #exp can be 2,3,4,5
	else:
		exp_list = [1, 2] #exp can be 1,2,3

	threshold = []
	idx_t = []
	for exp in exp_list:
		threshold += [get_te(exp+1, total)]
		idx_t += [2**exp]

	case = 0
	count = 0
	max_exp = 0
	start_flag = False
	q_list = []
	z_list = []
	e_list = np.array([(0.25)**i for i in range(idx_t[case])])
	for i in range(39, -1, -1): #from big to small
		bin_count = hist[i].item()
		if (bin_count > 0 and not start_flag):
			start_flag = True
			max_exp = i-24

		if (start_flag):
			if (count < idx_t[case]): #quatized bin
				q_list += [bin_count]
			else:
				z_list += [bin_count]
			if (count == idx_t[case]*2-1 or i == 0):
				count_q = np.inner(q_list, e_list)
				len_z = len(z_list)
				count_z = np.inner(z_list, e_list[:len_z])
				if (count_z == 0 or count_q/count_z > threshold[case]):
					exp = exp_list[case]
					return (exp, total-1-exp), max_exp
				else:
					if (case == len(exp_list)-1):
						exp = exp_list[-1] + 1
						return (exp, total-1-exp), max_exp
					else:
						case += 1
						q_list += z_list
						z_list = []
						e_list = np.array([(0.25)**i for i in range(idx_t[case])])
			count += 1

#for post-training quantization
def get_hist(idata):
	p = idata.data.view(-1).float()

	idx_p0 = (p == 0)
	p1 = p[~idx_p0]  # inequal with zero

	p1 = p1.abs().log2()

	max_h = 16
	min_h = -24

	P = torch.histc(p1, bins=int(max_h - min_h), max=int(max_h), min=int(min_h))
	return P