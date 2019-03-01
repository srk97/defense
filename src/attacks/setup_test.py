from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


class Net(nn.Module):

  def __init__(self):

    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = x.view(-1, 320)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)


def create_model(batch_size, pretrained_model_path="lenet_mnist_model.pth"):
  use_cuda = True

  test_loader = torch.utils.data.DataLoader(
      datasets.MNIST(
          '../data',
          train=False,
          download=True,
          transform=transforms.Compose([
              transforms.ToTensor(),
          ])),
      batch_size=batch_size,
      shuffle=True)
  device = torch.device("cuda" if (
      use_cuda and torch.cuda.is_available()) else "cpu")
  model = Net().to(device)
  model.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'))
  model.eval()
  return model, test_loader, device
