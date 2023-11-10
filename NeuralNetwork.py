# import libaryName.Models
import numpy as np

# Type hints
import typing

# Abstract meta-class behavior
from abc import ABCMeta as AbstractClass, abstractmethod as AbstractMethod

# Base definition for a neural network layer
class Neural_Layer (metaclass=AbstractClass):
	"""Neural network layer interface"""

	def __init__(
        self
        , output_dim : np.uint
        , input_dim: np.uint
        , activation_function :  typing.Callable[[float], float]
        , weights : typing.List[typing.List[float]]
        , biases : typing.List[float]):
		"""Construct a neural network layer"""

		# Node count of nodes inside layer
		self.output_dim : np.uint = output_dim
        # Node count of nodes on input layer
		self.input_dim : np.uint = input_dim
		# Activation function
		self.activation_function : np.uint = activation_function
		# Weight matrix for weighted sums
		self.weights = np.zeros(shape=(output_dim, input_dim), dtype=np.float64)
		# Bias vector for weighted sums
		self.biases : typing.List[float] = biases
		# Output values stored inside layer
		self.output = np.zeros(shape=(output_dim, 1), dtype=np.float64)

	@AbstractMethod
	def Predict(
		self
		, layer_input) -> np.ndarray:
		"""Compute the output based on an input layer and the ouput size"""
		pass

# Base definition for a neural network
class Neural_Network (metaclass=AbstractClass):
	"""A sequential neural network interface"""

	def __init__(
        self
		, input_dim : np.uint
		, output_dim : np.uint):
		"""Construct a neural network"""

        # Node count of input layer
		self.input_dim = input_dim
		# Node count of output layer
		self.output_size = output_dim
		# Neural network layer sequence
		self.layers = typing.List[Neural_Layer]

	# Base class
	@AbstractMethod	
	def Add_Layer(
		self
		, next_layer : Neural_Layer
		, activation_function)  -> bool:
		"""Add a neuron layer to the sequence of layers already stored on the network"""
		pass


	@AbstractMethod
	def Feed_Forward(
		self
		, inputs : np.ndarray) -> np.ndarray:
		"""Feed a set of inputs through the network, and get a set of predictions based on those"""
		pass

	@AbstractMethod
	def Backpropagate(
		self
		, inputs : np.ndarray
		, outputs : np.ndarray
		, learning_rate : np.float64) -> None:
		"""Readjust weights based on a subset of inputs, and the corresponding outputs """
		pass

	@AbstractMethod
	def Compute_Error(
		computed_outputs : np.ndarray
		, expected_outputs : np.ndarray) -> np.float64:
		"""Compute the cost function based on the set of outputs and the expected outputs"""
		pass

	def Train(
		self
		, inputs : np.ndarray
		, outputs : np.ndarray
		, learning_rate : np.float64
		, epochs : np.uint64
		, epoch_report_rate : np.uint = 1) -> np.float64:
		"""Train the Neural Network, and get the final error after the last epoch"""
		for epoch in range(epochs):

			computed_outputs : np.ndarray = self.Feed_Forward(inputs=input)
			self.Backpropagate(inputs=inputs, outputs=outputs, learning_rate=learning_rate)

			if epoch % epoch_report_rate == 0:
				error = Neural_Network.Compute_Error(outputs=outputs, computed_outputs=computed_outputs)
				print(f'Error after {epoch} epochs: {error}')
		
		computed_outputs : np.ndarray = self.Feed_Forward(inputs=input)
		return Neural_Network.Compute_Error(outputs=outputs, computed_outputs=computed_outputs)