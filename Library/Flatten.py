import numpy as np
from .NeuralNetwork import Neural_Layer

class Flatten_Layer(Neural_Layer):
    """Flatten layer for a neural network"""

    def __init__(self, input_shape):
        super().__init__(output_dim=None, input_dim=None, activation_function=None)
        self.input_shape = input_shape
        self.output_dim = np.prod(input_shape)

    def Predict(self, layer_input: np.ndarray) -> np.ndarray:
        # Save the input shape for later use (during backpropagation)
        self.input_shape = layer_input.shape

        # Flatten the input
        self.last_output = layer_input.reshape((layer_input.shape[0], -1))
        print(f"Last output: {self.last_output.shape}")
        # print(f"Last transposed output: {self.last_output.T.shape}")

        return self.last_output.T

    def Connect_To_Next_Layer(
        self,
        next_layer: "Neural_Layer"  # Layer to connect to.
    ) -> None:
		# Check if dimension hint was for previous layer is correct
        # If not, abort connection
        if self.output_dim != next_layer.input_dim:
            raise Exception("Next layer input dimension doesn't match this layer's output dimension")
		
    def Connect_To_Previous_Layer(
        self,
        previous_layer: "Neural_Layer"  # Layer to connect to.
    ) -> None:
        """Connects the current layer to its previous layer"""
        if self.input_shape != previous_layer.output_shape:
            raise Exception("Previous layer output dimension doesn't match this layer's input dimension")

    def Produce_Backwards_Layer_Error(self,
        layer_error_correction: np.ndarray # Matrix, with each column representing an error correction vector
        ) -> np.ndarray: # Matrix, with each column representing an error correction vector
        # Reshape the error correction to match the input shape before flattening
        return layer_error_correction.reshape(self.input_shape)
    
    def Update_From_Layer_Error(
        self, 
        layer_input: np.ndarray  # Matrix, with each column representing an gradient vector
		, learning_rate : np.float64 # Learning rate. Bigger values mean higher rate
    ) -> None:
        """Update any hidden state within the layer based on the layer error"""
        pass
