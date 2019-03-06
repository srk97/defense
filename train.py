'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from src.models.resnet import ResNet18 
from src.utils.model_utils import progress_bar
from src.utils.load_data import get_data


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--num_epochs', default=100, type=int, help='Number of training epochs')
parser.add_argument('--steps', default =0, type=int, help='No of steps in an epoch') 
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
batch_size = 128
# get the data 
trainloader, testloader = get_data(batch_size)

"""Fucntion to instantiate the resnet model and return it"""
def create_model():
    net = ResNet18()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
    # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    return net, criterion, optimizer

# Training
def train(steps, trainloader, net, criterion, optimizer):
    print('\nStep: %d' % steps)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    n_epochs = 1
    batch_idx = 0
    iterator = iter(trainloader)
    for batch_idx in range(steps):
        if batch_idx == n_epochs * len(trainloader):
            n_epochs = n_epochs + 1
            iterator = iter(trainloader)
        inputs, targets = iterator.next()
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

        progress_bar(batch_idx, steps, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(steps ,testloader, net, criterion, optimizer):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    n_epochs = 1
    with torch.no_grad():
        iterator = iter(trainloader)
        for batch_idx in range(steps):
            if batch_idx == n_epochs * len(trainloader):
                n_epochs = n_epochs + 1
                iterator = iter(trainloader)
            inputs, targets = iterator.next()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, steps, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'step': steps,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc


if __name__ == "__main__":
    net, criterion, optimizer = create_model()
    trainloader, testloader = get_data()
    if args.steps !=0 :
        train(args.steps, trainloader, net, criterion, optimizer)
        test(args.steps, testloader, net, criterion, optimizer)
    else:
        steps = (int)((args.num_epochs * 50000) / batch_size)        
        train(steps, trainloader, net, criterion, optimizer)
        test(steps, testloader, net, criterion, optimizer)
