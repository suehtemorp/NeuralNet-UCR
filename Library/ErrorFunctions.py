# Numpy math
import numpy as np

def Mean(
    x : np.ndarray # Matrix to compute mean on
    , axis: None | int | tuple = None # Axises to apply mean on
    , get_derivative_instead : bool = False # Whether to get or not the derivative instead
) -> np.ndarray: # Matrix, with either means computed along the axises, or a scalar if no axis is specified
    """Compute the mean either as a scalar or along some axis, or get its derivative instead"""
    mean_along_axis : np.ndarray = np.mean(a=x, axis=axis)

    if get_derivative_instead:
        element_count : np.uint64 = np.prod(a=mean_along_axis.shape)
        return np.dot(a = (1 / element_count), b = np.ones(shape=mean_along_axis.shape))
    else:
        return mean_along_axis
    
def Squared_Error(
    x : np.ndarray # Matrix
    , y : np.ndarray # Matrix
    , get_derivative_instead : bool = False # Whether to get or not the derivative instead
) -> np.ndarray: # Matrix
    """Compute the element-wise square of the error, or get its derivative instead"""
    if get_derivative_instead:
        return np.dot(a=2, b=np.absolute(x-y))
    else:
        return np.power((y - x), 2)
