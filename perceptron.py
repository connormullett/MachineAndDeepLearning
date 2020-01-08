
import random

from operator import mul


class Perceptron:
  def __init__(self, num_inputs, weight=0, bias=0, lr=0.1):
    # num of weights == num_inputs
    self.num_inputs = num_inputs
    # initialize weights based on number of inputs
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
    # sum every weight with learning rate
    for i in range(len(self.weights)):
      # get new weight
      new_weight = self.weights[i] + self.lr
      # calc loss?? (target - last_output) * x^j,i
      loss = target - last_input[i]
      loss *= last_input[i]
      self.weights[i] = new_weight * loss
    self.weight = new_weight

if __name__ == '__main__':
  
  inp = [[random.randint(1, 10) for _ in range(3)] for i in range(10)]
  print(inp)
  trg = [1 if sum(i) >= 15 else 0 for i in inp]
  print('trg: ', trg)

  perceptron = Perceptron(len(inp[0]))

  for i in range(len(inp)):
    out = perceptron(inp[i])
    if trg[i] != out:
      perceptron.update_weights(inp[i], trg[i])
    print('%d' % (i + 1))
    print('input  :', inp[i])
    print('out    : ', out)
    print('target : ', trg[i])
    print('weight : ', perceptron.weights)

  eval_inp = [5, 5, 5]
  eval_trg = 1
  eval_out = perceptron(eval_inp)
  print(eval_trg)
  print(eval_out)

