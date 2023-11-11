# Numpy numericals
import numpy as np

# Error funtion utilities
from .ErrorFunctions import Mean, Squared_Error

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
			# for out, expected in zip(computed_outputs, outputs):
			self.Backpropagate(computed_outputs=computed_outputs, expected_outputs=outputs, learning_rate=learning_rate)

			if epoch % epoch_report_rate == 0:
				error = Sequential.Cost_Overall(expected_outputs=outputs, computed_outputs=computed_outputs)
				print(f'Error after {epoch} epochs: {error}')

		return Sequential.Cost_Overall(expected_outputs=outputs, computed_outputs=computed_outputs)

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

	# def Backpropagate(self, inputs, outputs, learning_rate):
	# 	"""Readjust weights based on a subset of inputs and the corresponding outputs"""
	# 	pass

	def Backpropagate(self, computed_outputs, expected_outputs, learning_rate):
		"""Readjust weights based on a subset of inputs and the corresponding outputs"""
		# Keep track of the error
		error = Sequential.Cost_Overall(expected_outputs=expected_outputs, computed_outputs=computed_outputs)

		# Propagate the error backward through the network
		for i in reversed(range(len(self.layers))):
			layer = self.layers[i]
			if i == len(self.layers) - 1:
				# For the output layer, use the error computed above
				layer_error = error
			else:
				# For hidden layers, use the error from the previous layer
				l = np.dot(layer.weights, layer.activation_function(layer.last_output, derivative=True))
				layer_error = np.mean(np.dot(l, layer_error))
				# print(f"Error: {layer_error.shape}", layer_error)
				# exit()

			layer.weights -= np.dot(layer_error, layer.weights) + learning_rate * np.random.uniform(-1, 1, layer.weights.shape)
			layer.biases -= np.dot(layer_error, layer.biases) + learning_rate * np.random.uniform(-1, 1, layer.biases.shape)

	def Cost_Overall(
		computed_outputs : np.ndarray # Matrix, with each row representing an output vector
		, expected_outputs : np.ndarray # Matrix, with each row representing an output vector
	) -> np.float64: # Computed cost over all input-output pairs
		"""Compute the average of the cost function for each computed-to-expected output"""

		# Compute the MSE per each expected-computed pairwise output
		mse_per_output : np.ndarray = Sequential.Cost_Per_Output(computed_outputs, expected_outputs)

		# Compute the mean of the MSE as the overall error
		mse_across_outputs : np.float64 = np.mean(a=mse_per_output)
		return mse_across_outputs 

	def Cost_Per_Output(
		computed_outputs : np.ndarray # Matrix, with each row representing an output vector
		, expected_outputs : np.ndarray # Matrix, with each row representing an output vector
	) -> np.ndarray: # Matrix, with each row representing the cost per output row
		"""Compute the cost function for each computed output to its corresponding expected output"""
		# Compute the MSE per each expected-computed pairwise output
		return Mean(Squared_Error(computed_outputs, expected_outputs), axis=1)
