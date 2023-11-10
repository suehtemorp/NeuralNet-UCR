# OS and System utilities
import os
import sys

# Numpy data science
import numpy as np

# Inject dependencies
def Install_Dependencies() -> None:
    current_working_directory = os.path.dirname( __file__ )
    library_directory = os.path.join(current_working_directory, '..', 'Library')
    sys.path.append( library_directory )
    import NeuralNetwork as NN

# Install dependencies
Install_Dependencies()

# Activation function
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# Create a MLP for XOR operations
xor_network = NN.Neural_Network

# Add the input layer
input_layer : NN.Neural_Layer(2, 2, sigmoid)
xor_network.Add_Layer(input_layer)

# Add the hidden layer
hidden_layer : NN.Neural_Layer(2, 2, sigmoid)
xor_network.Add_Layer(hidden_layer)

# Add the output layer
output_layer : NN.Neural_Layer(2, 1, sigmoid)
xor_network.Add_Layer(hidden_layer)

# Train the network
final_error = xor_network.Train([[0,0], [0,1], [1,0], [1,1]], [[0],[1],[1],[0]], 1, 1000, 10)