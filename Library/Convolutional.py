import numpy as np
import typing

from .NeuralNetwork import Neural_Layer

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

        convolved_output = self._convolve(layer_input) + self.biases
        self.last_input = layer_input
        self.last_preactivations = convolved_output
        self.last_output = self.activation_function(convolved_output)
        return self.last_output

    def Connect_Layer(self, next_layer: "Neural_Layer") -> None:
        # Check if the number of filters matches the next layer's input channels
        if self.num_filters != next_layer.input_dim[2]:
            raise Exception("Number of filters in the convolutional layer must match the input channels of the next layer")
    
    def Gradients_From_Activation(
      self,
      layer_error_correction: np.ndarray,
      previous_layer: Neural_Layer
    ) -> np.ndarray:
      # Assuming a simple gradient passing for now

      # Compute the gradients for the pre-activations of this layer with respect to the activations of the previous layer 
      preactivation_jacobian_at_layer = np.zeros_like(self.last_preactivations)
      for i in range(self.num_filters):
          for j in range(self.input_shape[0] - self.filter_size + 1):
              for k in range(self.input_shape[1] - self.filter_size + 1):
                  input_slice = self.last_input[j:j + self.filter_size, k:k + self.filter_size, :]
                  preactivation_jacobian_at_layer[i, j:j + self.filter_size, k:k + self.filter_size, :] += layer_error_correction[i, j, k] * input_slice

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
