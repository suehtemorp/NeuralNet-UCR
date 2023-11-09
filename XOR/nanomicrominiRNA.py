# import libaryName.Models
import numpy as np
from abc import ABC, abstractmethod

def sigmoid(x, derivative=False):
	if derivative is False:
		return 1 / (1 + np.exp(-x))
	else:
		return x * (1 - x)

def relu(x, derivative=False):
	return max(0, x)

# layer = {
# 	'num_nodes': num_nodes,
# 	'activation_function': activation_function,
# 	'weights': np.random.uniform(-1, 1, (layer_input_size, num_nodes)),
# 	'biases': np.zeros((1, num_nodes))
# }

# def feedforward(self, X):
# 	layer_outputs = []
# 	for layer in self.layers:
# 		layer_input = X if not layer_outputs else layer_outputs[-1]
# 		weighted_sum = np.dot(layer_input, layer['weights']) + layer['biases']
# 		layer_output = layer['activation_function'](weighted_sum)
# 		layer_outputs.append(layer_output)
# 	return layer_outputs

class Layer(ABC):
	def __init__(self, num_nodes, activation_function, input_dim, weights, biases):
		self.num_nodes = num_nodes
		self.activation_function = activation_function
		self.input_dim = input_dim
		self.weights = weights
		self.biases = biases
		self.output = np.zeros((1, num_nodes))
  
	@abstractmethod
	def predict(self, layer_input):
		pass

# Les	
class Perceptron (Layer):
	def __init__(self, num_nodes, activation_function, input_dim=None, weights=np.empty((0,0)), biases=np.empty(0)):
		weights = weights if weights.count() == 0 else np.random.uniform(-1, 1, (self.input_dim, self.num_nodes))
		biases = biases if biases.count() == 0 else np.zeros((1, num_nodes))
		input_dim = input_dim if input_dim != None else num_nodes
		super().__init__(self, num_nodes, activation_function, input_dim, weights, biases)
	
	def predict(self, layer_input):
		weighted_sum = np.dot(layer_input, self.weights) + self.biases
		return self.activation_function(weighted_sum)
	


class NeuralNetwork :
	def __init__(self, input_size, output_size):
		self.input_size = input_size
		self.output_size = output_size
		self.layers = []

	# Base class	
	def add_layer(self, num_nodes, activation_function):
		if not self.layers:
			# If it's the first layer, set the input size accordingly
			layer_input_size = self.input_size
		else:
			# Otherwise, set the input size to the previous layer's output size
			layer_input_size = self.layers[-1]['num_nodes']
		
		layer = {
			'num_nodes': num_nodes,
			'activation_function': activation_function,
			'weights': np.random.uniform(-1, 1, (layer_input_size, num_nodes)),
			'biases': np.zeros((1, num_nodes))
		}
		self.layers.append(layer)


	# Layer class specialization
	def feedforward(self, X):
		layer_outputs = []
		for layer in self.layers:
			layer_input = X if not layer_outputs else layer_outputs[-1]
			weighted_sum = np.dot(layer_input, layer['weights']) + layer['biases']
			layer_output = layer['activation_function'](weighted_sum)
			layer_outputs.append(layer_output)
		return layer_outputs

	# Layer class specialization
	def backward(self, X, y, outputs, learning_rate):
		errors = [y - outputs[-1]]
		for i in range(len(self.layers) - 1, 0, -1):
			layer = self.layers[i]
			prev_output = outputs[i - 1]
			delta = errors[-1] * layer['activation_function'](outputs[i], derivative=True)
			layer['weights'] += np.dot(prev_output.T, delta) * learning_rate
			layer['biases'] += np.sum(delta, axis=0) * learning_rate
			errors.append(delta.dot(layer['weights'].T))
	
	# Layer class specialization
	def train(self, X, y, epochs, learning_rate):
		for epoch in range(epochs):
			outputs = self.feedforward(X)
			self.backward(X, y, outputs, learning_rate)
			if epoch % 1000 == 0:
				error = np.mean(np.abs(y - outputs[-1]))
				print(f'Error after {epoch} epochs: {error}')
	
	# Layer class specialization.
	def evaluate(self, X):
		outputs = self.feedforward(X)
		return outputs[-1]

# Example usage:
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create a neural network with one hidden layer and 2 nodes, using the sigmoid activation function
nn = NeuralNetwork(input_size=2, output_size=1)
nn.add_layer(num_nodes=2, activation_function=sigmoid)
nn.add_layer(num_nodes=1, activation_function=sigmoid)

# Train the network
nn.train(X, y, epochs=10000, learning_rate=0.1)

# Test the network
predictions = nn.evaluate(X)
binary_predictions = np.round(predictions)
print("Predictions after training:")
print(predictions)