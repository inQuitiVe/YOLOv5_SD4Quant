'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

print(torch.__version__)

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import os
import argparse
import numpy as np

from utils import progress_bar
from torch.optim.lr_scheduler import StepLR
from modules import *

import sys
sys.path.insert(0,'..')
from int_quantization import observer, mappings, fake_quantize

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.02, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', default=False,
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

## need to change data path here
trainset = torchvision.datasets.CIFAR10(
    root="./data/CIFAR10/", train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(
    root="./data/CIFAR10/", train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = MobilenetV2_with_quant()
net = net.to(device)


int_act_fake_quant = fake_quantize.default_fake_quant
log_weight_fake_quant_per_channel = fake_quantize.default_per_channel_log_weight_fake_quant
log_weight_fake_quant_per_tensor  = fake_quantize.default_log_weight_fake_quant

# qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')

log4_per_channel_config = torch.quantization.QConfig(activation=int_act_fake_quant, weight=log_weight_fake_quant_per_channel)
log4_per_tensor_config  = torch.quantization.QConfig(activation=int_act_fake_quant, weight=log_weight_fake_quant_per_tensor)

net.qconfig = log4_per_channel_config

for mod in net.modules():
    if type(mod) == ConvBNReLU and mod.groups == mod.in_channels:
        mod.qconfig = log4_per_tensor_config

    if type(mod) == torch.nn.Linear:
        mod.qconfig = log4_per_tensor_config

net.fuse_model()

torch.quantization.prepare_qat(net, inplace=True)
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = args.lr,
                    momentum=0.9, weight_decay=5e-4)
#optimizer = optim.Adam(net.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# Training
train_loss_np = []
valid_loss_np = []
train_acc_np = []
valid_acc_np = []

def train(epoch):

    print('\nEpoch: ', epoch,'LR: ', scheduler.get_lr())
    print('\nEpoch: ', epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    scheduler.step()
    train_loss_np.append(train_loss)
    train_acc_np.append(100.*correct/total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        valid_loss_np.append(test_loss)
        valid_acc_np.append(100.*correct/total)



    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+100):
    train(epoch)
    test(epoch)

x = torch.ones(1,3,32,32).to(device)
output = net(x).cpu().detach().numpy()
# output = model.conv1(x)
print("final output = ", output)
np.save("final_output.npy", output)


net.cpu()
net.eval()
torch.quantization.convert(net, mapping= mappings.LOG_MODULE_MAPPING, inplace=True)
torch.save(net.state_dict(), 'mytraining_mobilenet.pth')
