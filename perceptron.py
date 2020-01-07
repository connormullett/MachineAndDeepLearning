
from operator import mul


class Perceptron:
  def __init__(self, weight=0, bias=0, lr=0.1):
    # add input size list where w^0 is the bias
    self.weight = weight
    self.bias = bias
    self.lr = lr

  def forward(self, x):
    # dot = sum(map(mul, x, self.weight))
    dot = x * self.weight
    if dot + self.bias > 0:
      return 0
    else:
      return 1

  def __call__(self, x):
    return self.forward(x)

  def update_weights(self, last_input, target):
    new_weight = (self.weight + self.lr) * (last_input - target)
    self.weight = new_weight

if __name__ == '__main__':
  perceptron = Perceptron()
  i, t = 5, 0
  print('input: ', i)
  print('target: ', t)
  print('starting weight: ', perceptron.weight)
  for iter in range(2):
    out = perceptron(i)
    if t != out:
      perceptron.update_weights(i, t)
    print('iteration %d' % (iter))
    print('out: ', out)
    print('weight: ', perceptron.weight)

