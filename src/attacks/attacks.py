from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from abc import ABC, abstractmethod
from torch.autograd import Variable


class Attack(ABC):

  def __init__(self, model, device, min_value=0, max_value=1):
    self.model = model
    self.device = device
    self.min_value = min_value
    self.max_value = max_value

  @abstractmethod
  def generate(self, data, epsilon, y=None, y_target=None):
    raise NotImplementedError


class FGSM(Attack):

  def __init__(self, model, device, min_value=0, max_value=1):
    """
        parameters:-
        model :-The model under attack
        device :- CPU or GPU acc to usage
        data :- input image
        epsilon :- value of the perturbation
        y :- target /output labels
        targeted :- targeted version of attack

        4 Cases are possible according to the combination of targeted and y variables
        Case 1 :-y is specified and targeted is False .. then y is treated as the real output labels
        Case 2 :-y is specified and targeted is True ... then the targeted version of the attack takes place and y is the target label
        Case 3 :-y is None and targeted is False ... then the predicted outputs of the model are treated as the real outputs and the attack takes place
        Case 4 :-y is None and targeted is True .. Invalid Input"""

    super().__init__(model, device, min_value, max_value)

  def perturb(self, data, epsilon, output, target, y_target):
    """
    performs perturbation on the input in accordance with fgsm attack
    inputs :- 
          data :- the input image
          epsilon :- the epsilon values with which to perturb
          output :- output values of the network
          target :-target values 
          y_target :- None if Untargeted attack


    returns :-the perturbed matrix
    """
    loss = F.nll_loss(output, target)
    if y_target is not None:
      loss = -loss
    self.model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    sign_data_grad = data_grad.sign()
    perturbed_matrix = epsilon * sign_data_grad
    return perturbed_matrix

  def generate(self, data, epsilon, y=None, y_target=None):
    data = data.to(self.device)
    data.requires_grad = True
    output = self.model(data)
    init_pred = output.max(1, keepdim=True)[1]
    if y_target is not None:  # if no y is specified use predictions as the label for the attack
      target = y_target
    elif y is None:
      target = init_pred
    else:
      target = y  # use y itself as the target
    target = target.to(self.device)
    perturbed_matrix = self.perturb(data, epsilon, output, target, y_target)
    perturbed_data = data + perturbed_matrix
    perturbed_data = torch.clamp(perturbed_data, self.min_value, self.max_value)
    output = self.model(perturbed_data)
    final_pred = output.max(1, keepdim=True)[1]
    return init_pred, perturbed_data, final_pred


class PGD(FGSM):

  def __init__(self, model, device, iters=40, step_size=0.01,
               random_start=True):

    self.iters = iters
    self.step_size = step_size
    self.rand = random_start
    self.loss_fn = nn.CrossEntropyLoss()

    super().__init__(model, device)

  def generate(self, X_img, epsilon, y=None, y_target=None):
    if self.rand:
      X = X_img.cpu().numpy() + np.random.uniform(
          -epsilon, epsilon,
          X_img.cpu().numpy().shape).astype('float32')
    else:
      X = np.copy(X)

    data = torch.from_numpy(X)
    data = data.to(self.device)
    data.requires_grad = True
    output = self.model(data)
    init_pred = output.max(1, keepdim=True)[1]
    if y_target is not None:  # if no y is specified use predictions as the label for the attack
      target = y_target
    elif y is None:
      target = init_pred
    else:
      target = y  # use y itself as the target
    for i in range(self.iters):

      y_var = Variable(torch.LongTensor(target)).to(self.device)
      X_var = torch.from_numpy(X).to(self.device)
      X_var.requires_grad = True
      output = self.model(X_var)
      perturbed_matrix = self.perturb(X_var, self.step_size, output, y_var,
                                      y_target)
      X += perturbed_matrix.cpu().numpy()
      X = np.clip(X,
                  X_img.cpu().numpy() - epsilon,
                  X_img.cpu().numpy() + epsilon)
      X = np.clip(X, self.min_value, self.max_value)  # ensure valid pixel range
    perturbed_image = torch.from_numpy(X)
    perturbed_image = perturbed_image.to(self.device)
    output = self.model(perturbed_image)
    final_pred = output.max(1, keepdim=True)[1]
    return init_pred, perturbed_image, final_pred