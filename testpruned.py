import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import time
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import *

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=16,
                    help='depth of the vgg')
parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def test(model):
    time_start = time.time()
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    time_end=time.time()#time.time()为1970.1.1到当前时间的毫秒数
    
    print('\nTest set: Accuracy: {}/{} ({:.1f}%) Consuming Time: {}\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset), time_end-time_start))
    return correct / float(len(test_loader.dataset))

if not os.path.exists(args.save):
    os.makedirs(args.save)

if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        model = vgg(dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
        if args.cuda:
            model.cuda()
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'"
              .format(args.model))
    else:
        print("=> no checkpoint found")

print(model)
test(model)

