'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms

import os
import argparse

from src.models.resnet import ResNet18
from src.utils.misc_utils import progress_bar
from src.utils.load_data import get_data
from src.hparams.registry import get_hparams
from src.models.registry import get_model

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument(
    '--hparams', type=str, required=True, help='Hyperparameters string')
parser.add_argument(
    '--steps', default=0, type=int, help='No of steps in an epoch')
parser.add_argument(
    '--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument(
    '--output_dir',
    type=str,
    help='Output directory for storing ckpts. Default is in runs/hparams')
parser.add_argument(
    '--use_colab', type=bool, default=False, help='Use Google colaboratory')
args = parser.parse_args()

hparams = get_hparams(args.hparams)

if not args.use_colab:
  OUTPUT_DIR = 'runs/' + args.hparams if args.output_dir is None else args.output_dir
  if args.output_dir is None and not os.path.isdir('runs'):
    os.mkdir('runs')
else:
  from google.colab import drive
  drive.mount('/content/gdrive')
  OUTPUT_DIR = '/content/gdrive/My Drive/runs'
  if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)
  OUTPUT_DIR = OUTPUT_DIR + '/' + args.hparams
  if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_step = 0  # start from epoch 0 or last checkpoint epoch
# get the data
trainloader, testloader = get_data(hparams.batch_size)
"""Fucntion to instantiate the resnet model and return it"""


def create_model():
  net = get_model(hparams.model)
  net = net.to(device)
  if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

  if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(OUTPUT_DIR), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(OUTPUT_DIR + '/' +
                            sorted(os.listdir(OUTPUT_DIR))[-1])
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    global start_step
    start_step = checkpoint['step']

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(
      net.parameters(),
      lr=hparams.learning_rate,
      momentum=hparams.momentum,
      weight_decay=hparams.weight_decay)

  return net, criterion, optimizer


# Training
def train(steps, trainloader, net, criterion, optimizer, test_loader=None):
  net.train()
  train_loss = 0
  correct = 0
  total = 0
  n_epochs = 1
  batch_idx = 0
  iterator = iter(trainloader)
  for batch_idx in tqdm(range(start_step, steps, 1)):
    if batch_idx == n_epochs * len(trainloader):
      n_epochs = n_epochs + 1
      iterator = iter(trainloader)
    inputs, targets = iterator.next()
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    outputs = net(inputs, hparams)
    loss = criterion(outputs, targets)
    loss.backward()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()

    if batch_idx % hparams.eval_and_save_every == 0:
      print("Train Accuracy: {}".format(correct / total))
      test(hparams.eval_steps, testloader, net, criterion, int(batch_idx))

    optimizer.step()
    train_loss += loss.item()


def test(steps, testloader, net, criterion, curr_step):
  net.eval()
  test_loss = 0
  correct = 0
  total = 0
  n_epochs = 1
  with torch.no_grad():
    iterator = iter(testloader)
    for batch_idx in range(steps):
      if batch_idx == (n_epochs * len(testloader)):
        n_epochs = n_epochs + 1
        iterator = iter(testloader)
      inputs, targets = iterator.next()
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = net(inputs, hparams)
      loss = criterion(outputs, targets)

      test_loss += loss.item()
      _, predicted = outputs.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()

  # Save checkpoint.
  acc = 100. * correct / total
  print("Test Accuracy: ", acc)
  print('Saving..')
  state = {
      'net': net.state_dict(),
      'acc': acc,
      'step': curr_step,
  }
  if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

  torch.save(state, OUTPUT_DIR + '/ckpt-{}.t7'.format(str(curr_step)))


if __name__ == "__main__":
  net, criterion, optimizer = create_model()
  trainloader, testloader = get_data()
  if args.steps != 0:
    train(
        args.steps,
        trainloader,
        net,
        criterion,
        optimizer,
        test_loader=testloader)
    test(args.steps, testloader, net, criterion, args.steps + 1)
  else:
    steps = (int)((hparams.num_epochs * 50000) / hparams.batch_size)
    train(steps, trainloader, net, criterion, optimizer, test_loader=testloader)
    test(hparams.eval_steps, testloader, net, criterion, steps + 1)
