import cv2
import numpy as np

# Numpy data science
import numpy as np

# Data viz
import matplotlib.pyplot as plt

# Dependencies
from Library import Sequential, Dense_Layer, Convolutional_Layer, Flatten_Layer, sigmoid, relu

# Assuming you have a Sequential class and Convolutional_Layer defined

# Define the Sequential model
model = Sequential()

# Define the Convolutional Layer
convolutional_layer = Convolutional_Layer(
    input_shape=(1024, 1024, 1),  # Grayscale image with one channel
    num_filters=16,
    filter_size=3,
    activation_function=np.tanh  # You can choose an activation function
)

# Add the Convolutional Layer to the model
model.Add_Layer(convolutional_layer)

# Flatten layer to connect to the Dense layer
flatten_layer = Flatten_Layer()
model.Add_Layer(flatten_layer)

# Define the Dense Layer with one output
dense_layer = Dense_Layer(
    output_dim=1,
    input_dim=np.prod(convolutional_layer.last_output.shape),
    activation_function=np.sigmoid  # You can choose an activation function
)

# Add the Dense Layer to the model
model.Add_Layer(dense_layer)

# # Read and preprocess the image
# image_path = "path/to/your/image.jpg"
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# image = cv2.resize(image, (1024, 1024))  # Resize the image to match the model's input size
# image = image.reshape((1, 1024, 1024, 1))  # Add batch dimension and channel dimension

# # Normalize pixel values to be between 0 and 1
# image = image.astype(np.float64) / 255.0

# # Predict using the model
# output = model.Predict(image)

# # Print the predicted output
# print("Predicted Output:", output)

print("read data")
