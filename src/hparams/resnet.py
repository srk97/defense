from .registry import register


class HParams():

  def __init__(self):
    self.model = "ResNet18"
    self.batch_size = 128
    self.targeted_weight = False
    self.targeted_unit = False
    self.ramping_targeted_weight = False
    self.ramping_targeted_unit = False
    self.targ_perc = 0.0
    self.drop_rate = 0.0
    self.learning_rate = 0.1
    self.momentum = 0.9
    self.weight_decay = 5e-4
    self.num_epochs = 256
    self.eval_and_save_every = 1000
    self.eval_steps = 100
    self.linearize = False
    self.linearize_coeff = 1.0
    self.gs = -1
    self.extreme_pruning = False
    #TODO enforce the below flags
    self.image_aug = False
    self.per_image_standardization = True


@register
def resnet18_default():
  hps = HParams()
  return hps


#=======================================
# Targeted Weight
#========================================


@register
def resnet18_default_linearize():
  hps = HParams()
  hps.linearize = True
  return hps


#=======================================
# Targeted Weight
#========================================


@register
def resnet18_targ_weight_075_drop_033():
  hps = resnet18_default()
  hps.targeted_weight = True
  hps.targ_perc = 0.75
  hps.drop_rate = 0.33
  hps.eval_steps = 10

  return hps


@register
def resnet18_targ_weight_075_drop_033_linearize():
  hps = resnet18_default()
  hps.targeted_weight = True
  hps.targ_perc = 0.75
  hps.drop_rate = 0.33
  hps.linearize = True
  hps.linearize_coeff = 1.0

  return hps


@register
def resnet18_targ_weight_075_drop_050():
  hps = resnet18_targ_weight_075_drop_033()
  hps.drop_rate = 0.5

  return hps


@register
def resnet18_targ_weight_075_drop_066():
  hps = resnet18_targ_weight_075_drop_050()
  hps.drop_rate = 0.66

  return hps


@register
def resnet18_targ_weight_050_drop_050():
  hps = resnet18_targ_weight_075_drop_050()
  hps.targ_perc = 0.50

  return hps


@register
def resnet18_targ_weight_099_drop_099_ramping():
  hps = resnet18_default()
  hps.targ_perc = 0.99
  hps.drop_rate = 0.99
  hps.ramping_targeted_weight = True

  return hps


@register
def resnet18_targ_weight_ramping_xtreme_2():
  hps = resnet18_default()
  hps.targ_perc = 0.99
  hps.drop_rate = 0.99
  hps.ramping_targeted_weight = True
  hps.extreme_pruning = True
  hps.xtreme_keep = 2

  return hps


@register
def resnet18_targ_weight_ramping_xtreme_3():
  hps = resnet18_default()
  hps.targ_perc = 0.99
  hps.drop_rate = 0.99
  hps.ramping_targeted_weight = True
  hps.extreme_pruning = True
  hps.xtreme_keep = 3

  return hps


@register
def resnet18_targ_weight_ramping_xtreme_4():
  hps = resnet18_default()
  hps.targ_perc = 0.99
  hps.drop_rate = 0.99
  hps.ramping_targeted_weight = True
  hps.extreme_pruning = True
  hps.xtreme_keep = 4

  return hps


@register
def resnet18_targ_weight_099_drop_099_ramping_linearize():
  hps = resnet18_default()
  hps.targ_perc = 0.99
  hps.drop_rate = 0.99
  hps.ramping_targeted_weight = True
  hps.linearize = True

  return hps


#==============================================
# Targeted Unit
#==============================================
@register
def resnet18_targ_unit_075_drop_033():
  hps = resnet18_default()
  hps.targeted_unit = True
  hps.targ_perc = 0.75
  hps.drop_rate = 0.33

  return hps


@register
def resnet18_targ_unit_075_drop_050():
  hps = resnet18_targ_unit_075_drop_033()
  hps.drop_rate = 0.5

  return hps


@register
def resnet18_targ_unit_075_drop_066():
  hps = resnet18_targ_unit_075_drop_050()
  hps.drop_rate = 0.66

  return hps


@register
def resnet18_targ_unit_050_drop_050():
  hps = resnet18_targ_unit_075_drop_050()
  hps.targ_perc = 0.50

  return hps


@register
def resnet18_targ_unit_099_drop_090_ramping():
  hps = resnet18_default()
  hps.targ_perc = 0.99
  hps.drop_rate = 0.90
  hps.ramping_targeted_unit = True

  return hps


#==============================================


@register
def resnet34_default():
  hps = HParams()
  hps.model = "ResNet34"
  hps.learning_rate = 0.4
  return hps


#=======================================
# Targeted Weight
#========================================


@register
def resnet34_targ_weight_075_drop_033():
  hps = resnet34_default()
  hps.targeted_weight = True
  hps.targ_perc = 0.75
  hps.drop_rate = 0.33

  return hps


@register
def resnet34_targ_weight_075_drop_050():
  hps = resnet34_targ_weight_075_drop_033()
  hps.drop_rate = 0.5

  return hps


@register
def resnet34_targ_weight_075_drop_066():
  hps = resnet34_targ_weight_075_drop_050()
  hps.drop_rate = 0.66

  return hps


@register
def resnet34_targ_weight_050_drop_050():
  hps = resnet34_targ_weight_075_drop_050()
  hps.targ_perc = 0.50

  return hps


@register
def resnet34_targ_weight_099_drop_099_ramping():
  hps = resnet34_default()
  hps.targ_perc = 0.99
  hps.drop_rate = 0.99
  hps.ramping_targeted_weight = True

  return hps


#==============================================
# Targeted Unit
#==============================================


@register
def resnet34_targ_unit_075_drop_033():
  hps = resnet34_default()
  hps.targeted_unit = True
  hps.targ_perc = 0.75
  hps.drop_rate = 0.33

  return hps


@register
def resnet34_targ_unit_075_drop_050():
  hps = resnet34_targ_unit_075_drop_033()
  hps.drop_rate = 0.5

  return hps


@register
def resnet34_targ_unit_075_drop_066():
  hps = resnet34_targ_unit_075_drop_050()
  hps.drop_rate = 0.66

  return hps


@register
def resnet34_targ_unit_050_drop_050():
  hps = resnet34_targ_unit_075_drop_050()
  hps.targ_perc = 0.50

  return hps


@register
def resnet34_targ_unit_099_drop_090_ramping():
  hps = resnet34_default()
  hps.targ_perc = 0.99
  hps.drop_rate = 0.90
  hps.ramping_targeted_unit = True

  return hps
