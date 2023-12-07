import numpy as np

def identity(x, derivative=False):
  if (derivative):
    return np.eye(N=np.prod(x.shape))
  else:
    return x

def sigmoid(x, derivative=False):
  if derivative:
    return np.diagflat(np.multiply(sigmoid(x), 1-sigmoid(x)))
  else: 
    return 1 / (1 + np.exp(-x)) 
  
def relu(x, derivative=False):
  if derivative:
    # Leaky ReLu, to bypass dead ReLus
    return np.diagflat(np.where(x > 0, 1, 0.01))
  else:
    return np.where( x > 0, x, 0)
  
def identity(x, derivative=False):
  if derivative:
    return np.ones_like(x)
  else:
    return x.reshape((x.shape[0], -1))