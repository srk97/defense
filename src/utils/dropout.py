import torch
"""
References:
Targeted Dropout by Aidan N. Gomez, Ivan Zhang, Kevin Swersky, Yarin Gal, and Geoffrey E. Hinton
The code release for the same.
"""
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def targeted_weight_dropout(w, params, is_training):
  drop_rate = params.drop_rate
  targ_perc = params.targ_perc
  w_shape = list(w.size())
  w = w.view(w_shape[0], -1).transpose(0, 1)
  norm = torch.abs(w)
  idx = int(targ_perc * w.shape[0])
  threshold = (torch.sort(norm, dim=0)[0])[idx]
  mask = norm < threshold

  if not is_training:
    #Inference
    w = (1. - mask.float()) * w
    w = w.transpose(0, 1).view(w_shape)
    return w

  m = (torch.empty(list(w.size())).uniform_(0, 1) < drop_rate).to(device)
  # m = m.to(device)
  mask = m & mask

  mask = mask.float()
  w = (1. - mask) * w

  w = w.transpose(0, 1).view(w_shape)

  return w


def targeted_unit_dropout(w, params, is_training):
  drop_rate = params.drop_rate
  targ_perc = params.targ_perc

  w_shape = list(w.size())
  w = w.view(w_shape[0], -1).transpose(0, 1)

  norm = torch.norm(w, dim=0)
  idx = int(targ_perc * int(w.shape[1]))
  sorted_norms = torch.sort(norm)
  threshold = (sorted_norms[0])[idx]
  mask = (norm < threshold)[None, :]
  mask = mask.repeat(w.shape[0], 1)

  m = ((1. - drop_rate) < torch.empty(list(w.size())).uniform_(0, 1)).to(device)
  mask = torch.where(m & mask,
                     torch.ones(w.shape, dtype=torch.float32).to(device),
                     torch.zeros(w.shape,
                                 dtype=torch.float32).to(device)).to(device)

  x = (1 - mask) * w
  x = x.transpose(0, 1).view(w_shape)
  return (x)


def ramping_targeted_weight_dropout(w, params, is_training):
  drop_rate = params.drop_rate * min(1.0, float(params.gs) / 40000.)

  if params.extreme_pruning:
    num_weights = w.numel() / w.size()[0]
    targ_perc_target = (num_weights - params.xtreme_keep) / num_weights
    targ_perc = 0.95 * targ_perc_target * min(1.0, float(params.gs) / 20000.)
    targ_perc = targ_perc + 0.05 * targ_perc_target * max(
        0.0, min(1.0, (float(params.gs) - 20000.) / 20000.))

    if targ_perc == 1:
      raise ValueError
  else:
    targ_perc = 0.95 * params.targ_perc * min(1.0, float(params.gs) / 20000.)
    targ_perc = targ_perc + 0.05 * params.targ_perc * max(
        0.0, min(1.0, (float(params.gs) - 20000.) / 20000.))

  w_shape = list(w.size())
  w = w.view(w_shape[0], -1).transpose(0, 1)
  norm = torch.abs(w)
  idx = int(targ_perc * w.shape[0])
  threshold = (torch.sort(norm, dim=0)[0])[idx]
  mask = norm < threshold

  if not is_training:
    #Inference
    w = (1. - mask.float()) * w
    w = w.transpose(0, 1).view(w_shape)
    return w

  m = (torch.empty(list(w.size())).uniform_(0, 1) < drop_rate).to(device)
  # m = m.to(device)
  mask = m & mask

  mask = mask.float()
  w = (1. - mask) * w

  w = w.transpose(0, 1).view(w_shape)

  return w


def ramping_targeted_unit_dropout(w, params, is_training):
  drop_rate = params.drop_rate * min(1.0, float(params.gs) / 40000.)

  targ_perc = 0.95 * params.targ_perc * min(1.0, float(params.gs) / 20000.)
  targ_perc = targ_perc + 0.05 * params.targ_perc * max(
      0.0, min(1.0, (float(params.gs) - 20000.) / 20000.))

  w_shape = list(w.size())
  w = w.view(w_shape[0], -1).transpose(0, 1)

  norm = torch.norm(w, dim=0)
  idx = int(targ_perc * int(w.shape[1]))
  sorted_norms = torch.sort(norm)
  threshold = (sorted_norms[0])[idx]
  mask = (norm < threshold)[None, :]
  mask = mask.repeat(w.shape[0], 1)

  m = ((1. - drop_rate) < torch.empty(list(w.size())).uniform_(0, 1)).to(device)
  mask = torch.where(m & mask,
                     torch.ones(w.shape, dtype=torch.float32).to(device),
                     torch.zeros(w.shape,
                                 dtype=torch.float32).to(device)).to(device)

  x = (1 - mask) * w
  x = x.transpose(0, 1).view(w_shape)
  return (x)
