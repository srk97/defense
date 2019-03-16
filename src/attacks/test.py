import torch

from .attacks import PGD, FGSM
from .setup_test import create_model

from ..hparams.registry import get_hparams


class Test_Attack:

  def __init__(self, attack, test_data, device, epsilons):
    self.attack = attack
    self.test_data = test_data
    self.device = device
    self.epsilons = epsilons
    self.batch_size = test_data.batch_size

  def test(self):

    accuracies = []
    examples = []  #Run test for each epsilon
    for eps in self.epsilons:

      acc, ex = self.evaluate(eps)
      accuracies.append(acc)
      examples.append(ex)
    return accuracies, examples

  def evaluate(self, epsilon, eval_steps=None):
    total_examples = len(
        self.test_data
    ) * self.batch_size if eval_steps is None else self.batch_size * eval_steps

    eval_step_no = 0
    correct = 0
    adv_examples = []
    for data, target in self.test_data:
      data, target = data.to(self.device), target.to(self.device)

      init_pred, perturbed_data, final_pred = self.attack.generate(
          data, epsilon, y=target)
      target = target.view((20, 1))
      correct += torch.mm((init_pred == target).t(),
                          (init_pred == final_pred)).item()
      eval_step_no += 1
      if eval_steps is not None and eval_steps_no == eval_steps:
        break

    final_acc = correct / float(total_examples)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(
        epsilon, correct, total_examples, final_acc))

    return final_acc, adv_examples


if __name__ == '__main__':

  epsilons = [0, .05, .2, .25, .3]
  model, test_loader, device = create_model(20)
  attack = attacks.FGSM(model, device)
  testd = Test_Attack(attack, test_loader, device, epsilons)
  testd.test()
  attack = attacks.PGD(model, device)
  testd = Test_Attack(attack, test_loader, device, epsilons)
  testd.test()
