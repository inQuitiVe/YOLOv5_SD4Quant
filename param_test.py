from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import numpy as np

pytorch_pt_file = "./MNIST/mnist_test.pt"
state = torch.load(pytorch_pt_file)

possible_list = [-64,-32,-16,-8,-4,-2,-1,0,1,2,4,8,16,32,64]

for name, param in state.items():
    print(name)
    if name.split(".")[-1] == "weight":
        weight       = param.int_repr().detach().numpy()
        wight_orig   = param.detach()
        all_possible = np.unique(weight)
        for i in range(len(all_possible)):
            assert all_possible[i] in possible_list, "error: there is a number {} do not in possible list".format(all_possible[i])
        # print("These are all possible weights in CONV = ", all_possible)
    elif name.split(".")[-1] == "_packed_params":
        weight       = param[0].int_repr().detach().numpy()
        all_possible = np.unique(weight)
        for i in range(len(all_possible)):
            assert all_possible[i] in possible_list, "error: there is a number {} do not in possible list".format(all_possible[i])
        # print("These are all possible weights in FC = ", all_possible)
