# Numpy numericals
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
        , output_dim : np.uint # Length of output vector
        , input_dim: np.uint # Length of input vector
        , activation_function :  typing.Callable[[float], float] # Activation function
		):
		"""Construct a neural network layer"""
		# Node count of nodes inside layer
		self.output_dim : np.uint = output_dim
        # Node count of nodes on input layer
		self.input_dim : np.uint = input_dim
		# Activation function
		self.activation_function : np.uint = activation_function
		# Output values stored inside layer
		self.output = np.zeros(shape=(output_dim, 1), dtype=np.float64)

	@AbstractMethod
	def Predict(
		self
		, layer_input : "Neural_Layer" # Other layer with length of this layer's input dimension
		) -> np.ndarray: # Predicted output vector
		"""Compute an individual output based on an individual input"""
		pass

# Base definition for a neural network
class Neural_Network (metaclass=AbstractClass):
	"""A sequential neural network interface"""

	def __init__(
        self
		):
		"""Construct a neural network"""
		self.layers = typing.List[Neural_Layer]

	@AbstractMethod	
	def Add_Layer(
		self
		, next_layer : Neural_Layer # Neural layer to add
		)  -> bool: # Whether or not the layer was added succesfully
		"""Add a neuron layer to the sequence of layers already stored on the network"""
		pass

	@AbstractMethod
	def Feed_Forward(
		self
		, inputs : np.ndarray # Matrix, with each row representing an input vector
		) -> np.ndarray: # Matrix, with each row representing an output vector
		"""Feed a set of inputs through the network, and get a set of predictions based on those"""
		pass

	@AbstractMethod
	def __Backpropagate(
		self
		, inputs : np.ndarray # Matrix, with each row representing an input vector
		, outputs : np.ndarray # Matrix, with each row representing an output vector
		, learning_rate : np.float64 # Learning rate. Bigger values mean higher rate
		) -> None:
		"""Readjust weights based on a subset of inputs, and the corresponding outputs """
		pass

	@AbstractMethod
	def __Compute_Error(
		computed_outputs : np.ndarray # Matrix, with each row representing an input vector
		, expected_outputs : np.ndarray # Matrix, with each row representing an output vector
		) -> np.float64: # Computed cost over all input-output pairs
		"""Compute the cost function based on the set of outputs and the expected outputs"""
		pass

	def Train(
		self
		, inputs : np.ndarray # Matrix, with each row representing an input vector
		, outputs : np.ndarray # Matrix, with each row representing an output vector
		, learning_rate : np.float64 # Learning rate. Bigger values mean higher rate
		, epochs : np.uint64 # Iteration count for training backpropagation cycles
		, epoch_report_rate : np.uint = 1 # How many epochs until next error report
		) -> np.float64: # Computed cost over all input-output pairs, after training
		"""Train the Neural Network, and get the final error after the last epoch"""

		# This naive implementation trains the network on all input-output pairs at once
		# We should consider estochastic training instead for perfomance
		for epoch in range(epochs):

			computed_outputs : np.ndarray = self.Feed_Forward(inputs=input)
			self.Backpropagate(inputs=inputs, outputs=outputs, learning_rate=learning_rate)

			if epoch % epoch_report_rate == 0:
				error = Neural_Network.Compute_Error(outputs=outputs, computed_outputs=computed_outputs)
				print(f'Error after {epoch} epochs: {error}')
		
		computed_outputs : np.ndarray = self.Feed_Forward(inputs=input)
		return Neural_Network.Compute_Error(outputs=outputs, computed_outputs=computed_outputs)