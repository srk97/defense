'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import re
import os
import argparse
import re
import logging
from advertorch.attacks import LinfPGDAttack

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
    '--attack', '-a', action='store_true', help='Attack Instead')
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
  output_eval_file = 'runs/results_' + args.hparams + '.txt'
  if args.output_dir is None and not os.path.isdir('runs'):
    os.mkdir('runs')
else:
  from google.colab import drive
  drive.mount('/content/gdrive')
  OUTPUT_DIR = '/content/gdrive/My Drive/runs'
  output_eval_file = OUTPUT_DIR + '/results_' + args.hparams + '.txt'
  if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)
  OUTPUT_DIR = OUTPUT_DIR + '/' + args.hparams
  if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)

if args.resume:
  filemode = 'a'
else:
  filemode = 'w'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_step = 0  # start from epoch 0 or last checkpoint epoch
# get the data
trainloader, testloader = get_data(hparams.batch_size)
"""Fucntion to instantiate the resnet model and return it"""

logging.basicConfig(
    filename=output_eval_file,
    format='%(asctime)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
    filemode=filemode)
logger = logging.getLogger(__name__)
"""Function to print current Learning rate"""


def get_lr(optimizer):
  for param_group in optimizer.param_groups:
    return param_group['lr']


def create_model():
  net = get_model(hparams.model)
  net = net.to(device)
  optimizer = optim.SGD(
      net.parameters(),
      lr=hparams.learning_rate,
      momentum=hparams.momentum,
      weight_decay=hparams.weight_decay)
  scheduler = optim.lr_scheduler.MultiStepLR(
      optimizer, milestones=[40000, 60000, 80000], gamma=0.1)
  if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

  if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(OUTPUT_DIR), 'Error: no checkpoint directory found!'
    model_dir = (OUTPUT_DIR + '/' + 'ckpt-' + str(
        max([
            int(re.findall("[0-9]+\.", x)[0][:-1])
            for x in os.listdir(OUTPUT_DIR)
        ])) + '.t7')
    print(model_dir)
    checkpoint = torch.load(model_dir)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    global start_step
    start_step = checkpoint['step']
    hparams.gs = start_step
    scheduler.load_state_dict(checkpoint['scheduler'])

  criterion = nn.CrossEntropyLoss()

  return net, criterion, optimizer, scheduler


# Training
def train(steps,
          trainloader,
          net,
          criterion,
          optimizer,
          scheduler,
          test_loader=None):

  net.train()
  train_loss = 0
  correct = 0
  total = 0
  n_epochs = (start_step // len(trainloader)) + 1
  batch_idx = 0
  iterator = iter(trainloader)
  for batch_idx in tqdm(range(start_step, steps, 1)):
    hparams.gs = batch_idx
    if batch_idx == n_epochs * len(trainloader):
      n_epochs = n_epochs + 1
      iterator = iter(trainloader)
    inputs, targets = iterator.next()
    inputs, targets = inputs.to(device), targets.to(device)
    inputs_mean = torch.mean(inputs, 0)
    inputs_std = torch.clamp(
        torch.std(inputs, 0), min=1. / np.sqrt(inputs[0].numel()))
    inputs = (inputs - inputs_mean) / inputs_std
    optimizer.zero_grad()
    outputs = net(inputs, hparams)
    loss = criterion(outputs, targets)
    loss.backward()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()

    if batch_idx % hparams.eval_and_save_every == 0:
      print("\nTrain Accuracy: {}, Loss: {}".format((correct / total), loss))
      logger.info("Steps {}".format(batch_idx))
      logger.info("Train Accuracy: {}, Loss: {}".format((correct / total),
                                                        loss))
      test(hparams.eval_steps, testloader, net, criterion, scheduler,
           int(batch_idx))

    optimizer.step()
    scheduler.step()
    train_loss += loss.item()


def test(steps, testloader, net, criterion, scheduler, curr_step):
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
      inputs_mean = torch.mean(inputs, 0)
      inputs_std = torch.clamp(
          torch.std(inputs, 0), min=1. / np.sqrt(inputs[0].numel()))
      inputs = (inputs - inputs_mean) / inputs_std
      outputs = net(inputs, hparams)
      loss = criterion(outputs, targets)

      test_loss += loss.item()
      _, predicted = outputs.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()

  # Save checkpoint.
  acc = 100. * correct / total
  print("Test Accuracy: ", acc)
  logger.info("Test Accuracy: {} \n".format(acc))
  print('Saving..')
  print("Filename " + OUTPUT_DIR + '/ckpt-{}.t7'.format(str(curr_step)))
  state = {
      'net': net.state_dict(),
      'scheduler': scheduler.state_dict(),
      'acc': acc,
      'step': curr_step,
  }
  if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

  torch.save(state, OUTPUT_DIR + '/ckpt-{}.t7'.format(str(curr_step)))
  net.train()


def attack(net, testloader, steps):
  iterator = iter(testloader)
  adversary = LinfPGDAttack(
      net,
      loss_fn=nn.CrossEntropyLoss(reduction="sum"),
      eps=0.031,
      nb_iter=40,
      eps_iter=0.01,
      rand_init=True,
      clip_min=0.0,
      clip_max=1.0,
      targeted=False,
      hparams=hparams)
  correct = 0
  total = 0
  for batch_idx in range(steps):
    inputs, targets = iterator.next()
    inputs, targets = inputs.to(device), targets.to(device)

    inputs_mean = torch.mean(inputs, 0)
    inputs_std = torch.clamp(
        torch.std(inputs, 0), min=1. / np.sqrt(inputs[0].numel()))
    inputs = (inputs - inputs_mean) / inputs_std

    adv_untargeted = adversary.perturb(inputs, targets)

    adv_outputs = net(adv_untargeted, hparams)
    _, predicted = adv_outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()

  print("Accuracy after attack: ", 100. * float(correct / total))


if __name__ == "__main__":
  net, criterion, optimizer, scheduler = create_model()
  trainloader, testloader = get_data()

  if args.attack:
    net.eval()
    attack(net, testloader, hparams.eval_steps)
  else:
    print(vars(hparams))
    if args.steps != 0:
      train(
          args.steps,
          trainloader,
          net,
          criterion,
          optimizer,
          scheduler,
          test_loader=testloader)
      test(args.steps, testloader, net, criterion, scheduler, args.steps + 1)
    else:
      steps = (int)((hparams.num_epochs * 50000) / hparams.batch_size)
      #train(steps, trainloader, net, criterion, optimizer, scheduler, test_loader=testloader)
      test(hparams.eval_steps, testloader, net, criterion, scheduler, steps + 1)
