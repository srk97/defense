"""Code based on official pytorch implementation of resnet models """
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair as pair
from torch.nn import init
import torch.nn.functional as F
from src.utils import model_utils
from src.models.registry import register


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, in_planes, planes, stride=1):
    super(BasicBlock, self).__init__()
    self.stride = stride
    self.floatTensor = (torch.FloatTensor if not torch.cuda.is_available() else
                        torch.cuda.FloatTensor)
    self.conv1_weight = nn.Parameter(
        self.floatTensor(planes, in_planes, *pair(3)))
    self.conv2_weight = nn.Parameter(self.floatTensor(planes, planes, *pair(3)))
    self.conv_short_weight = nn.Parameter(
        self.floatTensor(self.expansion * planes, in_planes, *pair(1)))
    self.bn1 = nn.BatchNorm2d(planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.use_shortcut = False
    self.weights = [
        self.conv1_weight, self.conv2_weight, self.conv_short_weight
    ]
    if stride != 1 or in_planes != self.expansion * planes:
      self.use_shortcut = True
      self.short_batch_norm = nn.BatchNorm2d(self.expansion * planes)

    self.reset_parameters()

  def reset_parameters(self):
    for weight in self.weights:
      init.kaiming_normal_(weight, mode="fan_in")

  def forward(self, x, hparams):
    out = model_utils.conv2d(x, self.conv1_weight, self.stride, 1, hparams,
                             self.training)
    out = F.relu(self.bn1(out))
    out = model_utils.conv2d(out, self.conv2_weight, 1, 1, hparams,
                             self.training)
    out = self.bn2(out)
    out_short = 0
    if self.use_shortcut:
      out_short = model_utils.conv2d(x, self.conv_short_weight, self.stride, 0,
                                     hparams, self.training)
      out_short = self.short_batch_norm(out_short)
    out += out_short
    out = F.relu(out)
    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, in_planes, planes, stride=1):
    super(Bottleneck, self).__init__()
    self.stride = stride
    self.floatTensor = (torch.FloatTensor if not torch.cuda.is_available() else
                        torch.cuda.FloatTensor)
    self.conv1_weight = nn.Parameter(
        self.floatTensor(planes, in_planes, *pair(1)))
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2_weight = nn.Parameter(self.floatTensor(planes, planes, *pair(3)))
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3_weight = nn.Parameter(
        self.floatTensor(self.expansion * planes, planes, *pair(1)))
    self.bn3 = nn.BatchNorm2d(self.expansion * planes)
    self.use_shortcut = False
    self.conv_short_weight = nn.Parameter(
        self.floatTensor(self.expansion * planes, in_planes, *pair(1)))
    self.weights = [
        self.conv1_weight, self.conv2_weight, self.conv3_weight,
        self.conv_short_weight
    ]
    if stride != 1 or in_planes != self.expansion * planes:
      self.use_shortcut = True
      self.short_batch_norm = nn.BatchNorm2d(self.expansion * planes)

    self.reset_parameters()

  def reset_parameters(self):
    for weight in self.weights:
      init.kaiming_normal_(weight, mode="fan_in")

  def forward(self, x, hparams):
    out = model_utils.conv2d(x, self.conv1_weight, 1, 0, hparams, self.training)
    out = F.relu(self.bn1(out))
    out = model_utils.conv2d(out, self.conv2_weight, self.stride, 1, hparams,
                             self.training)
    out = F.relu(self.bn2(out))
    out = model_utils.conv2d(out, self.conv3_weight, 1, 0, hparams,
                             self.training)
    out = self.bn3(out)

    out_short = 0
    if self.use_shortcut:
      out_short = model_utils.conv2d(x, self.conv_short_weight, self.stride, 0,
                                     hparams, self.training)
      out_short = self.short_batch_norm(out_short)

    out += out_short
    out = F.relu(out)
    return out


class ResNet(nn.Module):

  def __init__(self, block, num_blocks, num_classes=10):
    super(ResNet, self).__init__()
    self.in_planes = 64
    self.floatTensor = (torch.FloatTensor if not torch.cuda.is_available() else
                        torch.cuda.FloatTensor)
    self.conv1_weight = nn.Parameter(self.floatTensor(64, 3, *pair(3)))
    self.weights = [self.conv1_weight]
    self.bn1 = nn.BatchNorm2d(64)
    self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
    self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
    self.linear = nn.Linear(512 * block.expansion, num_classes)

    self.reset_parameters()

  def reset_parameters(self):
    for weight in self.weights:
      init.kaiming_normal_(weight, mode="fan_in")

  def _make_layer(self, block, planes, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, stride))
      self.in_planes = planes * block.expansion
    return nn.Sequential(*layers)

  def forward(self, x, hparams):
    out = model_utils.conv2d(x, self.conv1_weight, 1, 1, hparams, self.training)
    out = F.relu(self.bn1(out))
    for i in range(len(self.layer1)):
      out = self.layer1[i](out, hparams)
    for i in range(len(self.layer2)):
      out = self.layer2[i](out, hparams)
    for i in range(len(self.layer3)):
      out = self.layer3[i](out, hparams)
    for i in range(len(self.layer4)):
      out = self.layer4[i](out, hparams)
    out = F.avg_pool2d(out, 4)
    out = out.view(out.size(0), -1)
    out = self.linear(out)
    return out


@register
def ResNet18():
  return ResNet(BasicBlock, [2, 2, 2, 2])


@register
def ResNet34():
  return ResNet(BasicBlock, [3, 4, 6, 3])


@register
def ResNet50():
  return ResNet(Bottleneck, [3, 4, 6, 3])


@register
def ResNet101():
  return ResNet(Bottleneck, [3, 4, 23, 3])


@register
def ResNet152():
  return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
  net = ResNet18()
  net.eval()
  y = net(torch.randn(1, 3, 32, 32), 1, 1.0, 0.8)
  print(y.size())
