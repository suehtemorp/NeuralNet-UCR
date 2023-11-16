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
        , activation_function :  typing.Callable[[float, bool], float] # Activation function, with optional derivative usage
	):
		"""Construct a neural network layer"""
		# Node count of nodes inside layer
		self.output_dim : np.uint = output_dim
        # Node count of nodes on input layer
		self.input_dim : np.uint = input_dim
		# Activation function
		self.activation_function : np.uint = activation_function
		# Output values stored inside layer
		self.last_output : np.ndarray = np.zeros(shape=(output_dim, 1), dtype=np.longdouble)
		self.last_preactivations : np.ndarray = np.zeros(shape=(output_dim, 1), dtype=np.longdouble)
		self.last_input : np.ndarray # Shape decided by implementation

	@AbstractMethod
	def Predict(
		self
		, layer_input : np.ndarray # Matrix, with each column representing an input vector
	) -> np.ndarray: # Matrix, with each column representing an output vector
		"""Compute an individual output based on an individual input"""
		pass

	@AbstractMethod
	def Connect_Layer(
		self
		, next_layer: "Neural_Layer" # Layer to connect to.
	) -> None:
		"""Connects Next layer with current layer"""
		pass

	@AbstractMethod
	def Gradients_From_Activation(
		self
	) -> np.ndarray: # Matrix, with each column representing the gradient vector for each output of this layer
		"""Compute a matrix of Gradients, one on each column, from the previous input of this layer"""
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
		computed_outputs : np.ndarray # Matrix, with each column representing an output vector
		, expected_outputs : np.ndarray # Matrix, with each column representing an output vector
	) -> np.float64: # Computed cost over all input-output pairs
		"""Compute the cost function based on the set of outputs and the expected outputs"""
		pass

	@AbstractMethod
	def Train(
		self
		, inputs : np.ndarray # Matrix, with each column representing an input vector
		, outputs : np.ndarray # Matrix, with each column representing an output vector
		, learning_rate : np.float64 # Learning rate. Bigger values mean higher rate
		, epochs : np.uint64 # Iteration count for training backpropagation cycles
		, epoch_report_rate : np.uint64 = 1 # How many epochs until next error report
		, loss_floor : np.float64 = None # Loss to match or go under to stop iterations
	) -> np.ndarray: # Matrix, with each row representing a cost on an epoch. Higher index means later epoch
		"""Train the Neural Network, and get the final error after the last epoch"""
		pass

	@AbstractMethod
	def Predict(
		self
		, inputs : np.ndarray # Matrix, with each column representing an input vector
	) -> np.ndarray: # Matrix, with each column representing an output vector
		"""Compute a set of outputs based on a set of inputs"""
		pass
