"""Code based on official pytorch implementation of resnet models """
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair as pair
from torch.nn import init
import torch.nn.functional as F
from src.utils.model_utils import TDConv
from src.models.registry import register


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, in_planes, planes, stride=1):
    super(BasicBlock, self).__init__()
    self.conv1 = TDConv(
        in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = TDConv(
        planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)

    self.use_short = False
    if stride != 1 or in_planes != self.expansion * planes:
      self.use_short = True
      self.short_conv = TDConv(
          in_planes,
          self.expansion * planes,
          kernel_size=1,
          stride=stride,
          bias=False)
      self.short_bn = nn.BatchNorm2d(self.expansion * planes)

  def forward(self, x, hparams):

    if hparams.linearize:
      out = hparams.linearize_coeff * self.bn1(self.conv1(x, hparams))
    else:
      out = F.relu(self.bn1(self.conv1(x, hparams)))
    out = self.bn2(self.conv2(out, hparams))

    if self.use_short:
      out_short = self.short_bn(self.short_conv(x, hparams))
      out += out_short

    if hparams.linearize:
      return hparams.linearize_coeff * out

    out = F.relu(out)
    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, in_planes, planes, stride=1):
    super(Bottleneck, self).__init__()
    self.conv1 = TDConv(in_planes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = TDConv(
        planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = TDConv(
        planes, self.expansion * planes, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(self.expansion * planes)

    self.use_short = False
    if stride != 1 or in_planes != self.expansion * planes:
      self.use_short = True
      self.short_conv = TDConv(
          in_planes,
          self.expansion * planes,
          kernel_size=1,
          stride=stride,
          bias=False)
      self.short_bn = nn.BatchNorm2d(self.expansion * planes)

  def forward(self, x, hparams):

    if hparams.linearize:
      out = hparams.linearize_coeff * self.bn1(self.conv1(x, hparams))
      out = hparams.linearize_coeff * self.bn2(self.conv2(out, hparams))
    else:
      out = F.relu(self.bn1(self.conv1(x, hparams)))
      out = F.relu(self.bn2(self.conv2(out, hparams)))
    out = self.bn3(self.conv3(out, hparams))
    if self.use_short:
      out_short = self.short_bn(self.short_conv(x, hparams))
      out += out_short

    if hparams.linearize:
      return hparams.linearize_coeff * out

    out = F.relu(out)
    return out


class ResNet(nn.Module):

  def __init__(self, block, num_blocks, num_classes=10):
    super(ResNet, self).__init__()
    self.in_planes = 64

    self.conv1 = TDConv(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
    self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
    self.linear = nn.Linear(512 * block.expansion, num_classes)

  def _make_layer(self, block, planes, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, stride))
      self.in_planes = planes * block.expansion
    return nn.Sequential(*layers)

  def forward(self, x, hparams):
    if hparams.linearize:
      out = hparams.linearize_coeff * self.bn1(self.conv1(x, hparams))
    else:
      out = F.relu(self.bn1(self.conv1(x, hparams)))
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
