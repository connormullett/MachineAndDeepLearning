
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
  inp = (3, 4, 5, 10, 12)
  trg = 0
  input_dimension = len(inp)
  perceptron = Perceptron(input_dimension)
  print('input  : ', inp)
  print('target : ', trg)
  print('weight : ', perceptron.weights)
  print('')
  for iter in range(1, 5):
    # pass to p
    out = perceptron(inp)
    if trg != out:
      perceptron.update_weights(inp, trg)
    print('%d' % (iter))
    print('out   : ', out)
    print('weight: ', perceptron.weights)
    print('')

