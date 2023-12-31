import cv2

# Numpy data science
import numpy as np

# Data viz
import matplotlib.pyplot as plt

# Dependencies
from Library import Sequential, Dense_Layer, Convolutional_Layer, Flatten_Layer, sigmoid, relu

# Define the Sequential model
model = Sequential()

# Define the Convolutional Layer
convolutional_layer = Convolutional_Layer(
    input_shape=(128, 128, 1),  # Grayscale image with one channel
    num_filters=16,
    filter_size=3,
    activation_function=np.tanh,  # You can choose an activation function
)

# Add the Convolutional Layer to the model
model.Add_Layer(convolutional_layer)

# Flatten layer to connect to the Dense layer
flatten_layer = Flatten_Layer(input_shape=convolutional_layer.output_shape)

model.Add_Layer(flatten_layer)

# Define the Dense Layer with one output
dense_layer = Dense_Layer(
    output_dim=1,
    input_dim=flatten_layer.output_dim,
    activation_function=sigmoid  # You can choose an activation function
)
model.Add_Layer(dense_layer)

# # Read and preprocess the image
image_path = "./input/input.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (128, 128))  # Resize the image to match the model's input size
image = image.reshape((128, 128, 1))  # Add batch dimension and channel dimension

# Normalize pixel values to be between 0 and 1
image = image.astype(np.float64) / 255.0

# # Read and preprocess the image
image_path = "./input/input2.png"
image2 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image2 = cv2.resize(image, (128, 128))  # Resize the image to match the model's input size
image2 = image2.reshape((128, 128, 1))  # Add batch dimension and channel dimension

# # Read and preprocess the image
image_path = "./input/input3.png"
image3 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image3 = cv2.resize(image, (128, 128))  # Resize the image to match the model's input size
image3 = image3.reshape((128, 128, 1))  # Add batch dimension and channel dimension


# Normalize pixel values to be between 0 and 1
image3 = image3.astype(np.float64) / 255.0

# X = np.array([image, image2])
X = np.array([image, image2, image3])
Y = np.array([[1], [0], [0.5]]).T

# Train the network
print("Training time!")
errors_per_epoch = model.Train(inputs=X, outputs=Y, learning_rate=0.25, epochs=3, epoch_report_rate=10, loss_floor=0.01)

# Make predictions on the data set
print("Prediction time!")
predictions = model.Predict(X)

print("Predictions after training:")
print(predictions)

print("Rounded-up predictions after training:")
print(np.rint(predictions))

# Visualize the loss over time
plt.plot(errors_per_epoch)
plt.xlabel("Epochs")
plt.ylabel("Loss with averaged MSE")
plt.title("Evolution of average loss on the entire dataset with respect to epoch")
plt.show()
