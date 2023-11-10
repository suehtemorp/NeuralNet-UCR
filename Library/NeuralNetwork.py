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
		, num_nodes : np.uint # Number of nodes in layer
		, activation_function :  typing.Callable[[float], float] # Activation function
		, input_dim: np.uint = None# Number of inputs that the layer takes
		):
		"""Construct a neural network layer"""
		# Number of nodes in the layer
		self.num_nodes = num_nodes
		# Number of nodes in the layer
		self.output_dim : np.uint = self.num_nodes
		# Node count of nodes on input layer
		self.input_dim : np.uint = input_dim if input_dim != None else 0
		# Activation function
		self.activation_function : np.uint = activation_function
		# Output values stored inside layer
		self.output: np.ndarray
		# Values to store in layer.
		self.weights: np.ndarray = np.random.uniform(-1, 1, (self.input_dim, self.num_nodes)) if input_dim != 0 else None
		self.biases: np.ndarray = np.zeros((self.num_nodes, 1), dtype=np.float64)
		self.input: np.ndarray

	@AbstractMethod
	def ConnectLayer(
		self
		, next_layer: "Neural_Layer" # Layer to connect to.
	):
		"""Connects Next layer with current layer"""
		pass

	@AbstractMethod
	def Predict(
		self
		, layer_input : "Neural_Layer" # Other layer with length of this layer's input dimension
		) -> np.ndarray: # Predicted output vector
		"""Compute an individual output based on an individual input"""
		pass


class Dense_Layer(Neural_Layer):
	"""A densly connected Neural Layer"""
	def __init__(
		self
		, num_nodes : np.uint # Length of output vector
		, activation_function :  typing.Callable[[float], float] # Activation function
		, input_dim: np.uint = None# Length of input vector
	):
		super().__init__(num_nodes, activation_function, input_dim)

	def ConnectLayer(
		self
		, next_layer: "Neural_Layer" # Layer to connect to.
	):
		self.output_dim = self.num_nodes * next_layer.num_nodes
		next_layer.input_dim = self.output_dim
		next_layer.weights = np.random.uniform(-1, 1, (next_layer.input_dim, next_layer.num_nodes))
		

	def Predict(
		self
		, layer_input: np.ndarray
		) -> np.ndarray:
		self.input = layer_input
		# print(f"Input: {self.input.shape}", self.input)
		# print(f"Biases: {self.biases.shape}")
		# print(f"Weights: {self.weights.shape}", self.weights)
		# print(f"Dot Product: {np.dot(self.input, self.weights).shape}")
		weighted_sum = np.dot(self.input, self.weights) + self.biases
		# print(f"Weighted Sum: {weighted_sum.shape}", weighted_sum)
		self.output = self.activation_function(weighted_sum)
		# print(f"Output: {self.output.shape}", self.output,  "\n\n")
		return self.output


# Base definition for a neural network
class Neural_Network (metaclass=AbstractClass):
	"""Neural Network interface"""

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
	def Compute_Error(
		computed_outputs : np.ndarray # Matrix, with each row representing an input vector
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
	def Evaluate(
		self
		, inputs
	):
		"""Get output for current input"""
		pass

class Sequential(Neural_Network):
	"""A sequential neural network interface"""

	def __init__(self):
		"""Construct a neural network"""
		super().__init__()
		self.layers = []

	def Add_Layer(self, next_layer: "Neural_Layer"):
		"""Add a neuron layer to the sequence of layers already stored on the network"""
		if isinstance(next_layer, Neural_Layer):
			if len(self.layers) > 0 : self.layers[-1].ConnectLayer(next_layer)
			self.layers.append(next_layer)
			return True
		else:
			print("Invalid layer type. Only Neural_Layer instances can be added.")
			return False

	def Feed_Forward(self, inputs):
		"""Feed a set of inputs through the network and get a set of predictions based on those"""
		layer_output = inputs
		for layer in self.layers:
			layer_output = layer.Predict(layer_output)
		return layer_output

	def Backpropagate(self, outputs, learning_rate):
		"""Readjust weights based on a subset of inputs and the corresponding outputs"""
		# Initialize gradients for backpropagation
		gradients = np.ones_like(outputs)

		# Iterate through layers in reverse order
		for layer in reversed(self.layers):
			gradients = gradients * layer.activation_function(layer.output, derivative=True)
			layer.weights += learning_rate * np.dot(layer.input, gradients)
			adjusted_gradients = learning_rate * np.sum(gradients, axis=0, keepdims=True)
			adjusted_gradients = adjusted_gradients.reshape(layer.biases.shape)
			layer.biases += adjusted_gradients
			# print(learning_rate * np.sum(gradients, axis=0, keepdims=True))
			gradients = np.dot(gradients, layer.weights.T)

	def Compute_Error(self, computed_outputs, expected_outputs):
		"""Compute the cost function based on the set of outputs and the expected outputs"""
		return np.mean(np.square(computed_outputs - expected_outputs)) / 2

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
			total_error = 0.0
			for input_data, output_data in zip(inputs, outputs):
				computed_output = self.Feed_Forward(inputs=input_data[np.newaxis])
				self.Backpropagate(outputs=output_data[np.newaxis], learning_rate=learning_rate)
				error = self.Compute_Error(computed_outputs=computed_output, expected_outputs=output_data[np.newaxis])
				total_error += error

			if epoch % epoch_report_rate == 0:
				print(f'Error after {epoch} epochs: {total_error / len(inputs)}')
					
		return total_error
	
	def Evaluate(self, x):
		# Apply the method to each sub-array using a list comprehension
		result_array = np.array([self.Feed_Forward(arr) for arr in x])
		return result_array