from .registry import get_attack
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import argparse
from ..utils.load_data import get_data
from ..models.registry import get_model
from .aparams.registry import get_attack_params
from ..hparams.registry import get_hparams
import os
import re
parser = argparse.ArgumentParser(description='PyTorch Adversarial Attacks')

parser.add_argument(
    '--attk_hparams', type=str, help='Attack Hyperparameters string')
parser.add_argument(
    '--model_hparams', type=str, help='Model Hyperparameters string')
parser.add_argument(
    '--model_dir',
    type=str,
    default=None,
    help='directory where model is present.')
parser.add_argument(
    '--use_colab', type=bool, default=False, help='Use Google colaboratory')
parser.add_argument(
    '--eval_steps', type=int, default=None, help='No of Eval Steps')
args = parser.parse_args()
if args.model_dir == None:
  OUTPUT_DIR = 'runs/' + args.model_hparams
  args.model_dir = OUTPUT_DIR + '/' + 'ckpt-' + str(
      max([
          int(re.findall("[0-9]+\.", x)[0][:-1]) for x in os.listdir(OUTPUT_DIR)
      ])) + '.t7'

if args.use_colab:
  from google.colab import drive
  drive.mount('/content/gdrive')
  args.model_dir = '/content/gdrive/My Drive/' + args.model_dir
hparams = get_attack_params(args.attk_hparams)
model_hparams = get_hparams(args.model_hparams)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainloader, testloader = get_data(hparams.batch_size)
if args.eval_steps != None:
  hparams.eval_steps = args.eval_steps


def create_model():
  net = get_model(model_hparams.model)
  net = net.to(device)

  if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
  print(args.model_dir)
  checkpoint = torch.load(args.model_dir)
  net.load_state_dict(checkpoint['net'])
  net.eval()

  criterion = nn.CrossEntropyLoss()
  return net, criterion


class Test_Attack:

  def __init__(self, attack, test_data, device, epsilons, eval_steps):
    self.attack = attack
    self.test_data = test_data
    self.device = device
    self.epsilons = epsilons
    self.batch_size = test_data.batch_size
    self.eval_steps = eval_steps

  def test(self):

    accuracies = []
    examples = []  #Run test for each epsilon
    for eps in self.epsilons:

      acc, ex = self.evaluate(eps, self.eval_steps)
      accuracies.append(acc)
      examples.append(ex)
    return accuracies, examples

  def evaluate(self, epsilon, eval_steps=None):
    total_examples = len(
        self.test_data
    ) * self.batch_size if eval_steps is None else self.batch_size * eval_steps

    eval_step_no = 0
    correct = 0
    total = 0
    adv_examples = []
    n_epochs = 1
    iterator = iter(self.test_data)
    for batch_idx in range(eval_steps):

      if batch_idx == (n_epochs * len(testloader)):
        n_epochs = n_epochs + 1
        iterator = iter(testloader)
      inputs, targets = iterator.next()
      inputs, targets = inputs.to(device), targets.to(device)

      init_pred, perturbed_data, final_pred = self.attack.generate(
          inputs, epsilon, y=targets)
      total += targets.size(0)
      correct += final_pred.eq(targets).sum().item()
    final_acc = correct / float(total)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(
        epsilon, correct, total, final_acc))

    return final_acc, adv_examples


if __name__ == '__main__':

  model, criterion = create_model()
  attack = get_attack(model, device, criterion, model_hparams, hparams)
  testd = Test_Attack(attack, testloader, device, hparams.epsilons,
                      hparams.eval_steps)
  testd.test()
