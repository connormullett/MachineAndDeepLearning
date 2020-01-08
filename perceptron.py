
import random

from operator import mul


class Perceptron:
  def __init__(self, inputs, targets, weight=0, bias=1, lr=0.1):
    self.inputs = inputs
    self.targets

    self.num_inputs = num_inputs

    self.weights = [0 for weight in range(num_inputs)]
    self.bias = bias
    self.lr = lr

  def forward(self, x):
    dot = sum(map(mul, x, self.weights))
    if dot + self.bias > 0:
      return 0
    else:
      return 1

  def __call__(self, x):
    return self.forward(x)

  def update_weights(self, last_input, target):
    # keep going till full
    # iteration with no errors
    for i in range(len(self.weights)):
      # get new weight
      new_weight = self.weights[i] + self.lr
      # calc loss?? (target - last_output) * x^j,i
      loss = target - last_input[i]
      loss *= last_input[i]
      self.weights[i] = new_weight * loss
    self.weight = new_weight

