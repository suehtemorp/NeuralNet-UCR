# Numpy numericals
import numpy as np

# Type hints
import typing

# Neural Network interfaces
from .NeuralNetwork import Neural_Layer, Neural_Network

class Sequential(Neural_Network):
	"""A sequential, Feed-Forward Neural Network"""

	def __init__(self):
		super().__init__()
		# Store neural network layers
		self.layers : typing.List[Neural_Layer] = []

	def Add_Layer(
		self, 
		next_layer: "Neural_Layer"
	):
		# Perform a hard-coded check to make sure the layer is a valid data type
		if not isinstance(next_layer, Neural_Layer):
			raise Exception("Invalid layer type. Only Neural_Layer instances can be added.")
		
		# If there are previous layers, attempt to connect to them
		if len(self.layers) > 0: 
			self.layers[-1].Connect_Layer(next_layer)

		# Append the layer to the list
		self.layers.append(next_layer)

	def Compute_Error(
		computed_outputs : np.ndarray # Matrix, with each row representing an output vector
		, expected_outputs : np.ndarray # Matrix, with each row representing an output vector
	) -> np.float64: # Computed cost over all input-output pairs
		# Compute the MSE per each expected-computed pairwise output
		mse_per_output : np.ndarray = Sequential.MSE(computed_outputs, expected_outputs)

		# Compute the mean of the MSE as the overall error
		mse_across_outputs : np.float64 = np.mean(a=mse_per_output)
		return mse_across_outputs 

	def Train(
		self
		, inputs : np.ndarray # Matrix, with each row representing an input vector
		, outputs : np.ndarray # Matrix, with each row representing an output vector
		, learning_rate : np.float64 # Learning rate. Bigger values mean higher rate
		, epochs : np.uint64 # Iteration count for training backpropagation cycles
		, epoch_report_rate : np.uint64 = 1 # How many epochs until next error report
		) -> np.float64: # Computed cost over all input-output pairs, after training

		# This naive implementation trains the network on all input-output pairs at once
		# We should consider estochastic training instead for perfomance
		
		computed_outputs : np.ndarray = []
		for epoch in range(epochs):
			# Keep track of computed outputs
			computed_outputs = self.Predict(inputs=inputs)
			
			# Backpropagate the error through the network
			self.Backpropagate(outputs=outputs, learning_rate=learning_rate)

			if epoch % epoch_report_rate == 0:
				error = Sequential.Compute_Error(expected_outputs=outputs, computed_outputs=computed_outputs)
				print(f'Error after {epoch} epochs: {error}')

		return Sequential.Compute_Error(expected_outputs=outputs, computed_outputs=computed_outputs)

	def Predict(
		self
		, inputs : np.ndarray # Matrix, with each row representing an input vector
	) -> np.ndarray: # Matrix, with each row representing an output vector
		# Take the input activation and pass it to through every layer
		# Each layer's output is the following's input 
		layer_output = inputs
		for layer in self.layers:
			layer_output = layer.Predict(layer_output)

		# The predicted output is taken from the last activation
		return layer_output

	def Backpropagate(self, inputs, outputs, learning_rate):
		"""Readjust weights based on a subset of inputs and the corresponding outputs"""
		pass
		# # Initialize gradients for backpropagation
		# gradients = np.ones_like(outputs)

		# # Iterate through layers in reverse order
		# for layer in reversed(self.layers):
		# 	gradients = gradients * layer.activation_function(layer.output, derivative=True)
		# 	layer.weights += learning_rate * np.dot(layer.input, gradients)
		# 	adjusted_gradients = learning_rate * np.sum(gradients, axis=0, keepdims=True)
		# 	adjusted_gradients = adjusted_gradients.reshape(layer.biases.shape)
		# 	layer.biases += adjusted_gradients
		# 	# print(learning_rate * np.sum(gradients, axis=0, keepdims=True))
		# 	gradients = np.dot(gradients, layer.weights.T)

	def MSE(
		computed_outputs : np.ndarray # Matrix, with each row representing an output vector
		, expected_outputs : np.ndarray # Matrix, with each row representing an output vector
	) -> np.ndarray: # Computed cost over all input-output pairs
		# Compute the MSE per each expected-computed pairwise output
		squared_error : np.ndarray = np.power((computed_outputs - expected_outputs), 2)
		mse_per_output : np.ndarray = np.mean(a=squared_error, axis=1)
		return mse_per_output
