# Numpy numericals
import numpy as np

# Type hints
import typing

# Neural Network interfaces
from .NeuralNetwork import Neural_Layer

class Dense_Layer(Neural_Layer):
	"""A densely connected, Feed-Forward Neural Network layer"""

	def __init__(
		self
        , output_dim : np.uint # Length of output vector
        , input_dim: np.uint # Length of input vector (merely a hint)
        , activation_function :  typing.Callable[[float], float] # Activation function
	):
		super().__init__(output_dim=output_dim, input_dim=input_dim, activation_function=activation_function)
		# Weights for this layer's activation
		self.weights : np.ndarray = np.random.uniform(low=-1, high=1, size=(output_dim, input_dim))
		# Biases for this layer's activation
		self.biases : np.ndarray = np.zeros(shape=(output_dim, 1), dtype=np.float64)

	def Predict(
		self
		, layer_input : np.ndarray # Vector values aligned with this layer's input dimension
    ) -> np.ndarray: # Predicted output vector
		# Compute the weighted sum of the previous layer, and after applying the
        # activation function, set it as this layer's activation 
		print(f"Shape of weights {self.weights.shape}, Shape of layer input {layer_input.T.shape}")
		weighted_sum = np.dot(self.weights, layer_input.T) + self.biases
		self.output = self.activation_function(weighted_sum)
		
		print(f"Predict: Layer output shape {self.output.T.shape}, {self.output}")
		return self.output.T
	
	def Connect_Layer(
		self
		, next_layer: "Neural_Layer" # Layer to connect to.
	) -> None:
		# Check if dimension hint was for previous layer is correct
        # If not, abort connection
		if self.output_dim != next_layer.input_dim:
			raise Exception("Previous layer output dimension doesn't match this layer's input dimension")
