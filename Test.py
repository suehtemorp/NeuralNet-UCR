# Numpy data science
import numpy as np

# Dependencies
from Library import Sequential, Max_Pooling_Layer, Dense_Layer, sigmoid

# Create a MLP for XOR operations
xor_network = Sequential()

# Add the input layer and hiden layer
input_layer = Dense_Layer(output_dim=16, input_dim=16, activation_function=sigmoid)
xor_network.Add_Layer(input_layer)

hidden_layer = Max_Pooling_Layer(input_shape=(4,4), output_shape=(2,2))
xor_network.Add_Layer(hidden_layer)

output_layer = Dense_Layer(output_dim=4, input_dim=4, activation_function=sigmoid)
xor_network.Add_Layer(output_layer)

# Example usage:
X = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]).T
y = np.array([[1, 2, 3, 4]]).T

print("Inputs with shape", X.shape)
print(X)

print("Expected outputs with shape", y.shape)
print(y)

# Train the network
print("Training time!")
errors_per_epoch : np.ndarray = xor_network.Train(inputs=X, outputs=y, learning_rate=0.25, epochs=4000, epoch_report_rate=1000, loss_floor=0.01)

# Make predictions on the data set
print("Prediction time!")
predictions = xor_network.Predict(X)

print("Predictions after training:")
print(predictions)

print("Rounded-up predictions after training:")
print(np.rint(predictions))