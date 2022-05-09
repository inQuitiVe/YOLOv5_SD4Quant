from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm
import sys
import numpy as np
from torch.quantization import QuantStub, DeQuantStub
# from utils import progress_bar
from torch.autograd import Variable

import time

import warnings

import sys
sys.path.insert(0,'..')
from int_quantization import observer, mappings, fake_quantize

print(torch.__version__)

int_act_fake_quant = fake_quantize.default_fake_quant
log_weight_fake_quant_per_channel = fake_quantize.default_per_channel_log_weight_fake_quant
log_weight_fake_quant_per_tensor  = fake_quantize.default_log_weight_fake_quant


log4_per_channel_config = torch.quantization.QConfig(activation=int_act_fake_quant, weight=log_weight_fake_quant_per_channel)
log4_per_tensor_config  = torch.quantization.QConfig(activation=int_act_fake_quant, weight=log_weight_fake_quant_per_tensor)

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, groups=1,bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv =  nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn   =  nn.BatchNorm2d(out_planes, momentum=0.1)
        self.relu =  nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1)

        self.conv2_depthwise = ConvBNReLU(8, 8,kernel_size=3, padding=0,groups=8,bias=False)
        self.conv2_pointwise = ConvBNReLU(8,16,kernel_size=1, padding=0,bias=False)
        self.conv3 = nn.Conv2d(16, 32, 5, 1)
        self.fc1 = nn.Linear(32, 64, bias=True)
        self.fc2 = nn.Linear(64, 10, bias=True)

        self.max_pooling = nn.MaxPool2d(kernel_size = 2 , stride = 2 )
        self.soft_max = nn.Softmax(dim=1)
        self.flatten =  nn.Flatten(1)
        self.relu     = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pooling(x)
        x = self.relu(x)

        x = self.conv2_depthwise(x)
        x = self.conv2_pointwise(x)

        x = self.max_pooling(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.soft_max(x)
        return output

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                torch.quantization.fuse_modules(m, [['conv', 'bn', 'relu']], inplace=True) #conv + bn + relu

class Net_quant(nn.Module):
    def __init__(self):
        super(Net_quant, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.body = Net()

    def forward(self, x):
        x = self.quant(x)
        x = self.body(x)
        output = self.dequant(x)
        return output

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                torch.quantization.fuse_modules(m, [['conv', 'bn', 'relu']], inplace=True) #conv + bn + relu

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # print(data.max())
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, half=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.half()) if half else model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return correct


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--retrain-epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # torch.use_deterministic_algorithms(True)


    torch.manual_seed(args.seed)
    np.random.seed(1)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor()
        ])

    ## need to modify data path here
    dataset1 = datasets.MNIST("/home/user2/data/MNIST/", train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST("/home/user2/data/MNIST/", train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    ### train FP32
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_accuracy = test(model, device, test_loader)

        scheduler.step()

    torch.save(model.state_dict(), "mnist_test_FP32.pt")

    ### post-quant
    model = Net().to(device)
    model.load_state_dict(torch.load("mnist_test_FP32.pt"))
    model.fuse_model()

    test_accuracy = test(model, device, test_loader)
    print("Final FP32 accuracy = ", test_accuracy/100)

    model.qconfig = log4_per_channel_config
    torch.quantization.prepare(model, inplace=True)
    model.eval()
    test_accuracy = test(model.half(), device, test_loader, half=True)
    # torch.quantization.convert(model, mapping= mappings.LOG_MODULE_MAPPING, inplace=True)

    print("FloatSD4 accuracy after post quant = ", test_accuracy/100)


    ### retrain model
    torch.quantization.prepare_qat(model, inplace=True)

    for epoch in range(1, args.retrain_epochs + 1):
        train(args, model.half(), device, train_loader, optimizer, epoch)

        model.apply(fake_quantize.disable_observer)
        test(model, device, test_loader)
        model.apply(fake_quantize.enable_observer)

        scheduler.step()

    model.cpu()
    model.eval()

    torch.quantization.convert(model, mapping= mappings.LOG_MODULE_MAPPING, inplace=True)
    torch.save(model.state_dict(), "mnist_test.pt")

if __name__ == '__main__':
    main()