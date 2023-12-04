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
		self.biases : np.ndarray = np.random.uniform(low=-1, high=1, size=(output_dim, 1))

	def Predict(
		self
		, layer_input : np.ndarray # Matrix, with each column representing an input vector
    ) -> np.ndarray: # Matrix, with each column representing an output vector
		# Compute the weighted sum of the previous layer, and after applying the
        # activation function, set it as this layer's activation 
		# print(">>>>> Layer prediction time!")
		
		self.last_input = layer_input

		self.last_preactivations = np.dot(self.weights, layer_input) + np.outer(a=self.biases, b=np.ones(shape=(layer_input.shape[1], 1)))

		self.last_output = self.activation_function(self.last_preactivations)

		return self.last_output
	
	def Connect_Layer(
		self
		, next_layer: "Neural_Layer" # Layer to connect to.
	) -> None:
		# Check if dimension hint was for previous layer is correct
        # If not, abort connection
		if self.output_dim != next_layer.input_dim:
			raise Exception("Previous layer output dimension doesn't match this layer's input dimension")

	def Gradients_From_Activation(
		self,
		layer_error_correction : np.ndarray,
		previous_layer : Neural_Layer
	) -> np.ndarray: # Matrix, with each column representing the gradient vector for each output of this layer
		# print(">>>>> Gradient time!")
		
		# print(f"Layer error correction with shape {layer_error_correction.shape}")
		# print(layer_error_correction)

		# Compute the gradients for the pre-activations of this layer with respect to the activations of the previous layer 
		preactivation_jacobian_at_layer : np.ndarray = np.dot(self.weights.T, layer_error_correction)

		# print(f"Preactivation jacobian at layer with shape: {preactivation_jacobian_at_layer.shape}")
		# print(preactivation_jacobian_at_layer)

		# print("Previous layer pre-activation")
		# print(previous_layer.last_preactivations)

		# Compute the gradients for the activations of this layer with respect to the pre-activations this layer
		activation_jacobians : np.ndarray = np.apply_along_axis(
			func1d=previous_layer.activation_function
			, axis=0, arr=previous_layer.last_preactivations
			, derivative=True)
		activation_jacobians = activation_jacobians.diagonal().T

		# print(f"Activation jacobian with shape: {activation_jacobians.shape}")
		# print(activation_jacobians)

		# Compute the Jacobian for the activated outputs of this layer with respect to the previous layer 
		activation_jacobians_at_layer = np.multiply(preactivation_jacobian_at_layer, activation_jacobians)
		
		# print(f"Activation jacobian at layer with shape: {activation_jacobians_at_layer.shape}")
		# print(activation_jacobians_at_layer)

		return activation_jacobians_at_layer
