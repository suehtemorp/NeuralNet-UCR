# # OS and System utilities
# import os
# import sys

# # Numpy data science
# import numpy as np

# # Inject dependencies
# current_working_directory = os.path.dirname( __file__ )
# library_directory = os.path.join(current_working_directory, '..', 'Library')
# sys.path.append( library_directory )
# import NeuralNetwork as NN

# # Activation function
# def sigmoid(x, derivative=False):
#   return 1 / (1 + np.exp(-x)) if derivative == False else x * (1 - x)
import numpy as np

from Library import Sequential, Dense_Layer, sigmoid
# Create a MLP for XOR operations
xor_network = Sequential()

# Add the input layer and hiden layer
input_layer = Dense_Layer(output_dim=2, input_dim=2, activation_function=sigmoid)
xor_network.Add_Layer(input_layer)

# Add the output layer
output_layer = Dense_Layer(output_dim=1, input_dim=2, activation_function=sigmoid)
xor_network.Add_Layer(output_layer)


# Example usage:
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

print(f"Shape of X {X.shape}")


# Train the network
final_error = xor_network.Train(X, y, learning_rate=0.01, epochs=10000, epoch_report_rate=10)

predictions = xor_network.Predict(X)
binary_predictions = np.round(predictions)
print("Predictions after training:")
print(predictions)