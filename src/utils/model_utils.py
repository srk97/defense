import torch
from src.utils.dropout import targeted_weight_dropout, targeted_unit_dropout

def conv(weight, params, is_training):

  #TODO - actually pass the is_training mode here. Not just True/False
  if params.targeted_weight:
    weight = torch.nn.Parameter(targeted_weight_dropout(weight, params, is_training))
  elif params.targeted_unit:
    weight = torch.nn.Parameter(targeted_weight_dropout(weight, params, is_training))

  return weight
