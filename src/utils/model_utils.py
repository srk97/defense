import torch
import torch.nn.functional as F
from src.utils.dropout import targeted_weight_dropout, targeted_unit_dropout


def conv2d(x, weight, stride, padding, params, is_training):
  dropped_weights = weight

  if params.targeted_weight:
    dropped_weights = targeted_weight_dropout(weight, params, is_training)
  elif params.targeted_unit:
    dropped_weights = targeted_weight_dropout(weight, params, is_training)

  return F.conv2d(x, dropped_weights, stride=stride, padding=padding)
