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
        , input_dim: np.uint # Length of input vector (merely a hint)
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
		self.last_output = np.zeros(shape=(output_dim, 1), dtype=np.float64)
		self.last_input : np.ndarray

	@AbstractMethod
	def Predict(
		self
		, layer_input : np.ndarray # Matrix, with each row representing an input vector
	) -> np.ndarray: # Matrix, with each row representing an output vector
		"""Compute an individual output based on an individual input"""
		pass

	@AbstractMethod
	def Connect_Layer(
		self
		, next_layer: "Neural_Layer" # Layer to connect to.
	) -> None:
		"""Connects Next layer with current layer"""
		pass

# Base definition for a neural network
class Neural_Network (metaclass=AbstractClass):
	"""Neural Network interface"""

	def __init__(
		self
	):
		"""Construct a neural network"""

	@AbstractMethod	
	def Add_Layer(
		self
		, next_layer : Neural_Layer # Neural layer to add
	):
		"""Add a neuron layer to the sequence of layers already stored on the network"""
		pass

	@AbstractMethod
	def Cost_Overall(
		computed_outputs : np.ndarray # Matrix, with each row representing an output vector
		, expected_outputs : np.ndarray # Matrix, with each row representing an output vector
	) -> np.float64: # Computed cost over all input-output pairs
		"""Compute the cost function based on the set of outputs and the expected outputs"""
		pass

	@AbstractMethod
	def Train(
		self
		, inputs : np.ndarray # Matrix, with each row representing an input vector
		, outputs : np.ndarray # Matrix, with each row representing an output vector
		, learning_rate : np.float64 # Learning rate. Bigger values mean higher rate
		, epochs : np.uint64 # Iteration count for training backpropagation cycles
		, epoch_report_rate : np.uint = 1 # How many epochs until next error report
	) -> np.float64: # Computed cost over all input-output pairs, after training
		"""Train the Neural Network, and get the final error after the last epoch"""
		pass

	@AbstractMethod
	def Predict(
		self
		, inputs : np.ndarray # Matrix, with each row representing an input vector
	) -> np.ndarray: # Matrix, with each row representing an output vector
		"""Compute a set of outputs based on a set of inputs"""
		pass
