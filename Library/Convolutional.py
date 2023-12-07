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

        self.output_shape = (self.num_filters, input_shape[0] - filter_size + 1, input_shape[1] - filter_size + 1)

        # Initialize filters and biases
        self.filters = np.random.uniform(low=-1, high=1, size=(num_filters, filter_size, filter_size, input_shape[2]))
        self.biases = np.random.uniform(low=-1, high=1, size=(num_filters, 1))

    def _apply_filter(self, input_slice, filter):
        return np.sum(input_slice * filter)

    def _convolve(self, input_data, *args):
        if input_data.ndim == 1:
            # If input is 1D, reshape it to 3D
            input_data = input_data.reshape((self.input_shape[0], self.input_shape[1], self.input_shape[2]))

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

    def Predict(self, input: np.ndarray) -> np.ndarray:
        self.last_output = np.empty((0,) + self.output_shape)
        self.last_preactivations = np.empty((0,) + self.output_shape)
        for layer_input in input:
            if layer_input.shape != self.input_shape:
                raise ValueError("Input shape does not match the expected input shape for the convolutional layer")

            # Perform the convolution operation
            convolved_output = self._convolve(layer_input)

            # Broadcast biases across the last two dimensions
            biases_broadcasted = self.biases[:, :, np.newaxis]

            # Add biases to the convolved output
            last_preactivations = convolved_output + biases_broadcasted

            # Apply activation function
            last_output = self.activation_function(last_preactivations)
            
            # Append values to last_preactivations and last output.
            self.last_preactivations = np.append(self.last_preactivations, last_preactivations[np.newaxis, :], axis=0)
            self.last_output = np.append(self.last_output, last_output[np.newaxis, :], axis=0);
        # print(f"Output shape: {self.last_output.shape}")
        # print(f"Last Preactivations dim: {self.last_preactivations.shape}")
        self.last_input = input;

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
        # Assuming a simple gradient passing for now

        # Compute the gradients for the pre-activations of this layer with respect to the activations of the next layer 
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
        layer_input: np.ndarray,  # Matrix, with each column representing a gradient vector
        learning_rate: np.float64  # Learning rate. Bigger values mean a higher rate
    ) -> None:
        """Update any hidden state within the layer based on the layer error"""

        # Compute the mean of the biases correction
        mean_biases_correction = np.mean(layer_input, axis=(0, 2), keepdims=True)

        # Compute the weights correction for filters
        weights_delta_filters = np.zeros_like(self.filters)

        for i in range(layer_input.shape[0]):  # Loop over the batch dimension
            for j in range(self.num_filters):
                temp_layer_input = layer_input[i, j][:, :, np.newaxis]
                temp_last_input = self.last_input[i]
                
                # Define the amount of padding for each dimension
                pad_width = ((1, 1), (1, 1), (0, 0))  # (before, after) for each dimension

                # Pad the matrix with zeros
                padded_matrix = np.pad(temp_layer_input, pad_width, mode='constant', constant_values=0)
                weights_delta_filters[j] += np.sum(temp_last_input * padded_matrix, axis=(0, 1))

        weights_delta_filters /= layer_input.shape[0]

        # Normalize the deltas and then scale them by the learning rate.
        # We desire to keep the direction of these deltas but scale them by the learning rate
        norm_factor = np.linalg.norm(weights_delta_filters)

        if norm_factor > 0:
            weights_delta_filters /= norm_factor

        mean_biases_correction *= -learning_rate
        weights_delta_filters *= -learning_rate

        # Apply the deltas respectively
        self.biases += np.mean(mean_biases_correction.squeeze())
        self.filters += weights_delta_filters