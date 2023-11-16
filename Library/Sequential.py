# Numpy numericals
import numpy as np

# Error funtion utilities
from .ErrorFunctions import Mean, Squared_Error, Mean_Squared_Error

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
		, inputs : np.ndarray # Matrix, with each column representing an input vector
		, outputs : np.ndarray # Matrix, with each column representing an output vector
		, learning_rate : np.float64 # Learning rate. Bigger values mean higher rate
		, epochs : np.uint64 # Iteration count for training backpropagation cycles
		, epoch_report_rate : np.uint64 = 1 # How many epochs until next error report
		, loss_floor : np.float64 = None # Loss to match or go under to stop iterations
	) -> np.float64: # Computed cost over all input-output pairs, after training

		# Keep track of the last error cost
		error : np.float64 = 0

		# This naive implementation trains the network on all input-output pairs at once
		# We should consider estochastic training instead for perfomance 
		for epoch in range(epochs):
			# Keep track of computed outputs
			computed_outputs : np.ndarray = self.Predict(inputs=inputs)
			
			# Backpropagate the error through the network
			# for out, expected in zip(computed_outputs, outputs):
			self.Backpropagate(computed_outputs=computed_outputs, expected_outputs=outputs, learning_rate=learning_rate)
			error = Sequential.Cost_Overall(expected_outputs=outputs, computed_outputs=computed_outputs)
			
			if epoch % epoch_report_rate == 0:
				print(f'>>>> Error after {epoch + 1} epochs: {error}')
			
			if (loss_floor != None and error <= loss_floor):
				print(f'>>>> Loss floor reached or surpassed at {epoch + 1} epochs. Loss is: {error}')
				break
		
		return error

	def Predict(
		self
		, inputs : np.ndarray # Matrix, with each column representing an input vector
	) -> np.ndarray: # Matrix, with each column representing an output vector
		# Take the input activation and pass it to through every layer
		# Each layer's output is the following's input 
		layer_output = inputs
		for layer in self.layers:
			layer_output = layer.Predict(layer_output)

		# print("Network predicted activations")
		# print(layer_output)

		# The predicted output is taken from the last activation
		return layer_output

	def Backpropagate(self, computed_outputs, expected_outputs, learning_rate):
		"""Readjust weights based on a subset of inputs and the corresponding outputs"""

		# Keep track of the error correction per layer
		layer_error_correction : np.ndarray = np.ndarray

		# And of the deltas to apply
		weights_deltas : list = list()
		biases_deltas : list = list()

		# Propagate the error correction backward through the network and compute
		# the deltas respectively
		for i in reversed(range(len(self.layers))):
			# Keep track of the current layer
			layer : Neural_Layer = self.layers[i]

			# The error correction of the last layer is well known
			# We will be considering MSE for these calculations
			if i == len(self.layers) - 1:

				# Compute the MSE's gradients for each expected-to-computed output
				# pair, with respect to the expected output
				activations_to_cost_gradients = Mean_Squared_Error(
					computed_outputs=computed_outputs
					, expected_outputs=expected_outputs
					, get_derivative_instead = True)
				
				# print("Activation to cost gradients")
				# print(activations_to_cost_gradients)

				# As a neat trick, we'll compute the activations gradients with respect to the preactivations 
				# directly instead of computing the jacobians for each activation in a tensor
				# and only then multiplying such tensor with the preactivations
				preactivations_to_activations_gradients : np.ndarray = np.apply_along_axis(
					func1d=layer.activation_function
					, axis=0, arr=layer.last_preactivations
					, derivative=True)
				preactivations_to_activations_gradients = preactivations_to_activations_gradients.diagonal().T

				# print("Preactivation to cost gradients")
				# print(preactivations_to_activations_gradients)

				# The last layer deltas is exactly the same as the gradients from the preactivations to the cost function, 
				# computed as the product element-wise between these other set of gradients
				layer_error_correction = np.multiply(preactivations_to_activations_gradients, activations_to_cost_gradients)

			# The error correction of previous layers is influenced by those of the next layers
			else:
				# For hidden layers, use the error correction from the next layer
				next_layer : Neural_Layer = self.layers[i+1]
				layer_error_correction = next_layer.Gradients_From_Activation(layer_error_correction, layer)

			weights_delta : np.ndarray = Mean(
				x = np.matmul(layer.last_input, layer_error_correction.T)
				, axis=1
				, get_derivative_instead=False
				, keepdims=True
			).T

			biases_delta : np.ndarray = Mean(
				x = layer_error_correction
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

			# print("Weights delta")
			# print(weights_delta)

			# print("Bias delta")
			# print(biases_delta)

			# Store but don't apply just yet the adjustment for each weight and bias
			# This will ensure application of deltas don't muddy the calculations for other deltas
			weights_deltas.insert(0, weights_delta)
			biases_deltas.insert(0, biases_delta)

		# Apply the deltas for each layer
		for (layer, weights_delta, biases_delta) in zip(self.layers, weights_deltas, biases_deltas):
			layer.weights += weights_delta
			layer.biases += biases_delta

			# print("Weights")
			# print(layer.weights)
			# print("Biases")
			# print(layer.biases)

	def Cost_Overall(
		computed_outputs : np.ndarray # Matrix, with each column representing an output vector
		, expected_outputs : np.ndarray # Matrix, with each column representing an output vector
	) -> np.float64: # Computed cost over all input-output pairs
		"""Compute the average of the cost function for each computed-to-expected output"""

		# Compute the MSE per each expected-computed pairwise output
		mse_per_output : np.ndarray = Sequential.Cost_Per_Output(computed_outputs, expected_outputs)

		# Compute the mean of the MSE as the overall error
		mse_across_outputs : np.float64 = np.mean(a=mse_per_output)
		return mse_across_outputs 

	def Cost_Per_Output(
		computed_outputs : np.ndarray # Matrix, with each column representing an output vector
		, expected_outputs : np.ndarray # Matrix, with each column representing an output vector
	) -> np.ndarray: # Matrix, with each column representing the cost per output row
		"""Compute the cost function for each computed output to its corresponding expected output"""
		# Compute the MSE per each expected-computed pairwise output
		return Mean_Squared_Error(computed_outputs, expected_outputs)
