# Numpy numericals
import numpy as np

# Type hints
import typing

# Neural Network interfaces
from .NeuralNetwork import Neural_Layer

# Error funtion utilities
from .ErrorFunctions import Mean

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
		
		self.last_input = layer_input

		self.last_preactivations = np.dot(self.weights, layer_input) + np.outer(a=self.biases, b=np.ones(shape=(layer_input.shape[1], 1)))

		self.last_output = self.activation_function(self.last_preactivations)

		return self.last_output
	
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
		if self.input_dim != previous_layer.output_dim:
			raise Exception("Previous layer output dimension doesn't match this layer's input dimension")
		
		# Cache the previous layer for backpropagation calculus
		self.previous_layer = previous_layer

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

	def Produce_Backwards_Layer_Error(
        self, 
        layer_input: np.ndarray  # Matrix, with each column representing an error correction vector
    ) -> np.ndarray: # Matrix, with each column representing an error correction vector
		"""Produce a layer error based on the fed layer error from the next layer"""

		# Compute the gradients for the pre-activations of this layer with respect to the activations of the previous layer 
		preactivation_jacobian_at_layer : np.ndarray = np.dot(self.weights.T, layer_input)

		# Compute the gradients for the activations of this layer with respect to the pre-activations this layer
		activation_jacobians : np.ndarray = np.apply_along_axis(
			func1d=self.previous_layer.activation_function
			, axis=0, arr=self.previous_layer.last_output
			, derivative=True)
		activation_jacobians = activation_jacobians.diagonal().T

		# Compute the Jacobian for the activated outputs of this layer with respect to the previous layer 
		activation_jacobians_at_layer = np.multiply(preactivation_jacobian_at_layer, activation_jacobians)

		# All is done
		return activation_jacobians_at_layer
