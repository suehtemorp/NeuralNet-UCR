import numpy as np
from .NeuralNetwork import Neural_Layer

class Flatten_Layer(Neural_Layer):
    """Flatten layer for a neural network"""

    def __init__(self):
        super().__init__(output_dim=None, input_dim=None, activation_function=None)
        self.last_input_shape = None

    def Predict(self, layer_input: np.ndarray) -> np.ndarray:
        # Save the input shape for later use (during backpropagation)
        self.last_input_shape = layer_input.shape

        # Flatten the input
        self.last_output = layer_input.reshape((layer_input.shape[0], -1))
        print(f"Last output: {self.last_output.shape}")
        # print(f"Last transposed output: {self.last_output.T.shape}")

        return self.last_output.T

    def Connect_Layer(self, next_layer):
        # No connection needed for Flatten layer
        pass

    def Gradients_From_Activation(self, layer_error_correction, previous_layer):
        # Reshape the error correction to match the input shape before flattening
        return layer_error_correction.reshape(self.last_input_shape)
