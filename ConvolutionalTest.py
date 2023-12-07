import cv2

# Numpy data science
import numpy as np

# import cupy as np


# Data viz
import matplotlib.pyplot as plt

# Dependencies
from Library import Sequential, Dense_Layer, Convolutional_Layer, Flatten_Layer, sigmoid, relu

# Assuming you have a Sequential class and Convolutional_Layer defined

# Define the Sequential model
model = Sequential()

dummy_input = np.random.rand(2, 16, 1022, 1022)
# Define the Convolutional Layer
convolutional_layer = Convolutional_Layer(
    input_shape=(1024, 1024, 1),  # Grayscale image with one channel
    num_filters=16,
    filter_size=3,
    activation_function=np.tanh,  # You can choose an activation function
    output_shape=(16, 1022, 1022)
)

# Add the Convolutional Layer to the model
model.Add_Layer(convolutional_layer)

# Predict on a dummy input to initialize last_output
# dummy_input = np.random.rand(1024, 1024, 1)
# convolutional_layer.Predict(dummy_input)

# Flatten layer to connect to the Dense layer
flatten_layer = Flatten_Layer()
flatten_layer.Predict(dummy_input)
model.Add_Layer(flatten_layer)

# Now you can safely connect the layers
# model.layers[-2].Connect_Layer(model.layers[-1])
# model.layers[-3].Connect_Layer(model.layers[-2])

# print ("Shape:", np.prod(convolutional_layer.last_output.shape))
# Shape: 16711744

# Define the Dense Layer with one output
dense_layer = Dense_Layer(
    output_dim=1,
    input_dim=flatten_layer.last_output.shape[1],
    activation_function=sigmoid  # You can choose an activation function
)
model.Add_Layer(dense_layer)
# Define the Dense Layer with one output

# # Read and preprocess the image
image_path = "./input/input.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (1024, 1024))  # Resize the image to match the model's input size
image = image.reshape((1024, 1024, 1))  # Add batch dimension and channel dimension

# Normalize pixel values to be between 0 and 1
image = image.astype(np.float64) / 255.0

# # Read and preprocess the image
image_path = "./input/input2.png"
image2 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image2 = cv2.resize(image, (1024, 1024))  # Resize the image to match the model's input size
image2 = image2.reshape((1024, 1024, 1))  # Add batch dimension and channel dimension

# Normalize pixel values to be between 0 and 1
image2 = image2.astype(np.float64) / 255.0


# print(image.shape)
# dummy_output = convolutional_layer.Predict(image)

# print("dummy_output_shape:", dummy_output.shape)
# print("input_dim:", np.prod(dummy_output.shape))
# Predict using the model
output = model.Predict(np.array([image, image2]))

# Print the predicted output
print("Predicted Output:", output.flatten())
