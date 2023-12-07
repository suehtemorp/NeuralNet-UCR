# Numpy numericals
import numpy as np

# Type hints
import typing

# Neural Network interfaces
from .NeuralNetwork import Neural_Layer

# Error funtion utilities
from .ErrorFunctions import Mean

class Convolutional_Layer(Neural_Layer):
    """Convolutional layer for a neural network"""

    def __init__(
        self,
        input_shape: typing.Tuple[int, int, int],  # Shape of the input data (height, width, channels)
        num_filters: int,  # Number of filters in the layer
        filter_size: int,  # Size of each filter
        activation_function: typing.Callable[[float], float],  # Activation function
    ):
        super().__init__(output_dim=num_filters, input_dim=np.prod(input_shape), activation_function=activation_function)
        
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.filter_size = filter_size

        # Initialize filters and biases
        self.filters = np.random.uniform(low=-1, high=1, size=(num_filters, filter_size, filter_size, input_shape[2]))
        self.biases = np.random.uniform(low=-1, high=1, size=(num_filters, 1))

    def _apply_filter(self, input_slice, filter):
        return np.sum(input_slice * filter)

    def _convolve(self, input_data):
        height, width, _ = self.input_shape
        output_height = height - self.filter_size + 1
        output_width = width - self.filter_size + 1

        convolved_output = np.zeros((self.num_filters, output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):
                input_slice = input_data[i:i + self.filter_size, j:j + self.filter_size, :]
                for k in range(self.num_filters):
                    convolved_output[k, i, j] = self._apply_filter(input_slice, self.filters[k])

        return convolved_output

    def Predict(self, layer_input: np.ndarray) -> np.ndarray:
        if layer_input.shape != self.input_shape:
            raise ValueError("Input shape does not match the expected input shape for the convolutional layer")

        # Perform the convolution operation
        convolved_output = self._convolve(layer_input)

        # Broadcast biases across the last two dimensions
        biases_broadcasted = self.biases[:, :, np.newaxis]

        # Add biases to the convolved output
        self.last_preactivations = convolved_output + biases_broadcasted

        # Apply activation function
        self.last_output = self.activation_function(self.last_preactivations)

        return self.last_output

    def Connect_To_Next_Layer(
        self,
        next_layer: "Neural_Layer"  # Layer to connect to.
    ) -> None:
		# Check if dimension hint for the next layer is correct
        # If not, abort connection
		# Check if the number of filters matches the next layer's input channels
        if next_layer.input_dim is not None and self.output_dim != next_layer.input_dim:
            raise Exception("Number of filters in the convolutional layer must match the input channels of the next layer")
		
    def Connect_To_Previous_Layer(
		self,
        previous_layer: "Neural_Layer"  # Layer to connect to.
    ) -> None:
        # Check if dimension hint for previous layer is correct
        # If not, abort connection
		# Check if the number of filters matches the previous layer's input channels
        if self.input_dim is not None and self.input_dim != previous_layer.output_dim:
            raise Exception("Number of filters in the convolutional layer must match the input channels of the next layer")
    
    def Produce_Backwards_Layer_Error(
        self, 
        layer_input: np.ndarray  # Matrix, with each column representing an error correction vector
    ) -> np.ndarray: # Matrix, with each column representing an error correction vector
        """Produce a layer error based on the fed layer error from the next layer"""
        # Assuming a simple gradient passing for now

        # Compute the gradients for the pre-activations of this layer with respect to the activations of the previous layer 
        preactivation_jacobian_at_layer = np.zeros_like(self.last_preactivations)
        for i in range(self.num_filters):
            for j in range(self.input_shape[0] - self.filter_size + 1):
                for k in range(self.input_shape[1] - self.filter_size + 1):
                    input_slice = self.last_input[j:j + self.filter_size, k:k + self.filter_size, :]
                    preactivation_jacobian_at_layer[i, j:j + self.filter_size, k:k + self.filter_size, :] += layer_input[i, j, k] * input_slice

        # Compute the gradients for the activations of this layer with respect to the pre-activations of this layer
        activation_jacobians = np.apply_along_axis(
            func1d=self.activation_function,
            axis=(1, 2),
            arr=self.last_preactivations[:, :, :, 0],
            derivative=True
        )

        # Compute the Jacobian for the activated outputs of this layer with respect to the previous layer 
        activation_jacobians_at_layer = np.multiply(preactivation_jacobian_at_layer, activation_jacobians[:, :, np.newaxis, np.newaxis])

        return activation_jacobians_at_layer
    
    def Update_From_Layer_Error(
        self, 
        layer_input: np.ndarray  # Matrix, with each column representing an gradient vector
		, learning_rate : np.float64 # Learning rate. Bigger values mean higher rate
    ) -> None:
        """Update any hidden state within the layer based on the layer error"""

        weights_delta : np.ndarray = Mean(
				x = np.matmul(self.last_input, layer_input.T)
				, axis=1
				, get_derivative_instead=False
				, keepdims=True
			).T

        biases_delta : np.ndarray = Mean(
				x = layer_input
				, axis=1
				, get_derivative_instead=False
				, keepdims=True
			)

		# Normalize the deltas and then scale them by the learning rate. 
		# We desire to keep the direction of these deltas, but scale them by the learning rate 
        if (np.linalg.norm(weights_delta, ord='fro') > 0):
            weights_delta =  weights_delta / np.linalg.norm(weights_delta, ord='fro')
		
        if (np.linalg.norm(biases_delta, ord='fro') > 0):
            biases_delta = biases_delta / np.linalg.norm(biases_delta, ord='fro')

        weights_delta *= -learning_rate
        biases_delta *= -learning_rate
		
		# Apply the deltas respectively
        self.weights += weights_delta
        self.biases += biases_delta
