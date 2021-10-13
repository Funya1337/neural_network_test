import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])

training_outputs = np.array([[0, 1, 0, 1]]).T

bias = -0.000001

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3, 1)) - 1

print(synaptic_weights)

for i in range(20000):
  input_layer = training_inputs
  outputs = sigmoid(np.dot(input_layer, synaptic_weights))

  err = training_outputs - outputs

  adjustments = np.dot(input_layer.T, err * (outputs * (1 - outputs)))

  synaptic_weights += adjustments
  
#new_inputs = np.array([0, 1, 0])
#output = sigmoid(np.dot(new_inputs, synaptic_weights) + bias)

#print(output)

test_data = [[0, 0, 1],
             [0, 1, 0],
             [0, 1, 1],
             [1, 0, 0],
             [1, 0, 1],
             [1, 1, 0],
             [1, 1, 1],
             [0, 0, 0]]

for arr in test_data:
  new_inputs = np.array(arr)
  output = sigmoid(np.dot(new_inputs, synaptic_weights) + bias)
  if output < 0.5:
    print(arr, output, "result = 0")
  else:
    print(arr, output, "result = 1")
