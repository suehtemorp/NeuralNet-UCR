# Numpy math
import numpy as np

def Mean(
    x : np.ndarray # Matrix to compute mean on
    , axis: None | int | tuple = None # Axises to apply mean on
    , get_derivative_instead : bool = False # Whether to get or not the derivative instead
    , keepdims : bool = True # Whether to keep or not to the dimensions along the axis of the mean
) -> np.ndarray: # Matrix, with either means computed along the axises, or a scalar if no axis is specified
    """Compute the mean either as a scalar or along some axis, or get its derivative instead"""
    mean_along_axis : np.ndarray = np.mean(a=x, axis=axis, keepdims=keepdims)

    if get_derivative_instead:
        element_count : np.uint64 = np.prod(a=x.shape) / np.prod(a=mean_along_axis.shape)
        return np.dot(a = (1 / element_count), b = np.ones(shape=x.shape))
    else:
        return mean_along_axis
    
def Squared_Error(
    computed_outputs : np.ndarray # Matrix, with each column representing an output vector
    , expected_outputs : np.ndarray # Matrix, with each column representing an output vector
    , get_derivative_instead : bool = False # Whether to get or not the derivative instead
) -> np.ndarray: # Matrix
    """Compute the element-wise square of the error, or get its derivative instead"""
    if get_derivative_instead:
        return np.diagflat(
            np.multiply(
                x1 = computed_outputs
                , x2 = np.dot(
                    a = 2
                    , b = np.absolute(computed_outputs-expected_outputs)
                )
            )
        )
    else:
        return np.power((expected_outputs - computed_outputs), 2)

def Mean_Squared_Error(
    computed_outputs : np.ndarray # Matrix, with each column representing an output vector
    , expected_outputs : np.ndarray # Matrix, with each column representing an output vector
    , get_derivative_instead : bool = False # Whether to get or not the derivative instead

) -> np.ndarray: # Matrix, with each column representing the MSE, or its gradient, for the computed-expected output pair
    """Compute the MSE per each expected-to-computed output pair"""
    if get_derivative_instead:
        # To avoid using hardcore math for higher dimensions (e.g: tensors if the output dimension is greater than 1) 
        # let's take a shortcut
        component_count : np.uint64 = computed_outputs.shape[0]

        return np.dot(
            2 / component_count
            , computed_outputs - expected_outputs
        )

    else:
        return Mean(Squared_Error(computed_outputs, expected_outputs), axis=1)