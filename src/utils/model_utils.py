import torch
import torch.nn as nn
from torch.nn.modules import Module
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from torch.nn import init
from src.utils.dropout import *


class TDConv(Module):

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               stride=1,
               padding=0,
               dilation=1,
               groups=1,
               bias=False):
    super(TDConv, self).__init__()
    self.kernel_size = _pair(kernel_size)
    self.stride = _pair(stride)
    self.padding = _pair(padding)
    self.dilation = _pair(dilation)
    self.groups = groups
    self.floatTensor = (torch.FloatTensor if not torch.cuda.is_available() else
                        torch.cuda.FloatTensor)
    self.weight = nn.Parameter(
        self.floatTensor(out_channels, in_channels, *self.kernel_size))

    self.reset_parameters()

  def reset_parameters(self):
    init.kaiming_normal_(self.weight, mode="fan_in")

  def forward(self, x, hparams):

    if hparams.targeted_weight:
      dropped_weights = targeted_weight_dropout(self.weight, hparams,
                                                self.training)
    elif hparams.targeted_unit:
      dropped_weights = targeted_unit_dropout(self.weight, hparams,
                                              self.training)
    elif hparams.ramping_targeted_weight:
      dropped_weights = ramping_targeted_weight_dropout(self.weight, hparams,
                                                        self.training)
    elif hparams.ramping_targeted_unit:
      dropped_weights = ramping_targeted_unit_dropout(self.weight, hparams,
                                                      self.training)
    else:
      dropped_weights = self.weight

    return F.conv2d(x, dropped_weights, None, self.stride, self.padding,
                    self.dilation, self.groups)
