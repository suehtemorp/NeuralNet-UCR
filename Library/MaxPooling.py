# Numpy numericals
import numpy as np

# Type hints
import typing

# Neural Network interfaces
from .NeuralNetwork import Neural_Layer

# Error funtion utilities
from .ActivationFunctions import identity

# Blocking utilities
from skimage.util import view_as_blocks
from skimage.measure import block_reduce

class Max_Pooling_Layer(Neural_Layer):
	"""A max-pooling, non-overlapping layer"""

	def __init__(
		self
        , output_shape : np.shape # Ouput tensor dimensions
        , input_shape: np.shape # Input tensor dimensions
	):
		super().__init__(output_dim=np.prod(output_shape), input_dim=np.prod(input_shape), activation_function=identity)
		# Keep track of the input and output shapes for reshaping the batches
		self.input_shape: tuple = input_shape
		self.output_shape: tuple = output_shape

		# The masks for each pooling
		self.batch_output_masks : typing.List[np.ndarray] = list()

		# The axis to compute the max on
		self.pooling_axes = tuple([-x for x in range(1, len(self.input_shape) + 1)])

		# The axis to expand the backpropagated layer error on
		self.error_axes = [x for x in range(len(self.input_shape))]

		# Make sure the output shape divides the input shape
		if np.sum(np.mod(input_shape, output_shape)) != 0:
			raise ValueError("Input shape does not subdivide output shape in the pooling layer")
        
		# Compute the block size
		self.block_size : tuple =  tuple(np.floor_divide(self.input_shape, self.output_shape))
		# print("Block size", self.block_size)

        # Weights and biases for this layer are nil
		self.weights : np.ndarray = None
		self.biases : np.ndarray = None

	def Predict(
		self
		, layer_input : np.ndarray # Matrix, with each column representing an input vector
    ) -> np.ndarray: # Matrix, with each column representing an output vector
		# Cache the previous input
		self.last_input = layer_input
		
		# Separate the input batch
		batch_inputs : typing.List[np.ndarray] = layer_input.T.tolist()

		# Compute the max pooling for each input in the batch
		# Clean the cached masks and store the activations and preactivations
		self.batch_output_masks.clear()
		preactivations : typing.List[np.ndarray] = list()
		activations : typing.List[np.ndarray] = list() 

		for input in batch_inputs:
			# Reshape the batch input
			current_input : np.ndarray = np.array(input).reshape(self.input_shape)
			# print("Current input reshaped", current_input)

			# Subdivide the input into blocks
			input_as_blocks = view_as_blocks(arr_in=current_input, block_shape=self.output_shape)
			# print("Input as block", input_as_blocks)

			# Compute the output mask and save it
			output_mask = np.max(input_as_blocks, axis=self.pooling_axes, keepdims=True) == input_as_blocks
			self.batch_output_masks.append(output_mask)
			# print("Output mask", output_mask)

			# Compute the pre-activations as a blockwise, max-pooling reduction
			preactivation : np.ndarray = block_reduce(current_input, block_size=self.block_size, func=np.max)
			# print("Block preactivation", preactivation)

			# Flatten it, compute the activation, and save those
			preactivation = preactivation.flatten()
			preactivations.append(preactivation)
			# print("Flatenned preactivation", preactivation)
			
			activation = self.activation_function(preactivation, False)
			activations.append(activation)
			# print("Flatenned activation", activation)

		# Join the outputs into a proper matrix
		self.last_preactivations = np.stack(preactivations).T
		self.last_output = np.stack(activations).T

		# print("Final preactivations", self.last_preactivations, "with shape", self.last_preactivations.shape)
		# print("Final outputs", self.last_output, "with shape", self.last_output.shape)

		# All is done
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
		
		# Cache the previous layer for gradient calculus purposes


	def Update_From_Layer_Error(
        self, 
        layer_input: np.ndarray  # Matrix, with each column representing an gradient vector
		, learning_rate : np.float64 # Learning rate. Bigger values mean higher rate
    ) -> None:
		"""Update any hidden state within the layer based on the layer error"""
		pass

	def Produce_Backwards_Layer_Error(
        self, 
        layer_input: np.ndarray  # Matrix, with each column representing an error correction vector
    ) -> np.ndarray: # Matrix, with each column representing an error correction vector
		"""Produce a layer error based on the fed layer error from the next layer"""

		# Compute the layer error per batch item
		output_errors : typing.List[np.ndarray] = list() 
		for (input_error, input_mask) in zip(layer_input.T, self.batch_output_masks):
			# Turn the layer mask into a binary matrix
			binary_mask : np.ndarray = input_mask.astype(np.int32).reshape(self.input_shape)
			# print("Binary mask", binary_mask, "with shape", binary_mask.shape)

			# Tile the error components onto the same shape of the binary mask
			tiled_error = input_error.reshape(self.output_shape)
			tiled_error = np.kron(tiled_error, np.ones(self.block_size)) 
			# print("Matching components in error", tiled_error, "with shape", tiled_error.shape)
			
			# Multiply them both element-wise, then flatten. This is the new layer error
			new_error = tiled_error * binary_mask
			# print("New error before flatenning", new_error, "with shape", new_error.shape)
			new_error : np.ndarray = new_error.flatten()
			# print("New error after flatenning", new_error, "with shape", new_error.shape)

			# Save it
			output_errors.append(new_error)
		
		# Merge the layer errors onto a matrix
		output_error : np.ndarray = np.stack(output_errors).T
		# print("Output error", output_error, "with shape", output_error.shape)

		# All is done
		return output_error
