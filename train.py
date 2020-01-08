
from perceptron import Perceptron

perceptron = Perceptron(2)

inp = [[2,4], [2,1], [3,3], [3,2], [1,2], [4,1]]
trg = [1, 0, 1, 0, 1, 0]

for i in range(len(inp)):
  out = perceptron(inp[i])
  if trg[i] != out:
    perceptron.update_weights(inp[i], trg[i])
  print('%d' % (i + 1))
  print('input  :', inp[i])
  print('out    : ', out)
  print('target : ', trg[i])
  print('weight : ', perceptron.weights)

