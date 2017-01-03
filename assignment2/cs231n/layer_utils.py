from cs231n.layers import *
from cs231n.fast_layers import *

###########################################################################################
# The following two functions stitched affine - bn - ReLU(W * X + B) forward and backward #
###########################################################################################

def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
  """
  Convenience layer that performs an affine transform followed by a batch normalization followed by a ReLU activation
  Inputs:
  - x: Input to the affine layer
  - w, b: weights and bias for the affine layer
  - gamma, beta, bn_params: parameters for translation, rescaling and training/testing running

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, af_cache = affine_forward(x, w, b) # def affine_forward(x, w, b): return out, cache;

  na, bn_cache = batchnorm_forward(a, gamma, beta, bn_param) # def batchnorm_forward(x, gamma, beta, bn_param): return out, cache;

  out, relu_cache = relu_forward(na) # def relu_forward(x): return out, cache;

  cache = (af_cache, bn_cache, relu_cache)

  return out, cache

def affine_bn_relu_backward(dout, cache):
  """
  the backward process of what is defined above
  """
  af_cache, bn_cache, relu_cache = cache

  dna = relu_backward(dout, relu_cache) # def relu_backward(dout, cache): return dx;

  da, dgamma, dbeta = batchnorm_backward(dna, bn_cache) # def batchnorm_backward(dout, cache): return dx, dgamma, dbeta;

  dx, dw, db = affine_backward(da, af_cache) # def affine_backward(dout, cache): return dx, dw, db;

  return dx, dw, db, dgamma, dbeta

###########################################################################################
# The following two functions stitched ReLU(W * X + B) together both forward and backward #
###########################################################################################

def affine_relu_forward(x, w, b):
  """
  Convenience layer that performs an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, af_cache = affine_forward(x, w, b) # fc_cache = (x, w, b) cached for backprop

  out, relu_cache = relu_forward(a) # relu_cache = (x)

  cache = (af_cache, relu_cache)

  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  af_cache, relu_cache = cache

  da = relu_backward(dout, relu_cache) # relu_backward(dout from lower layer, relu_cache = (x))
  
  dx, dw, db = affine_backward(da, af_cache) # affine_backward(da from relu_backward, fc_cache = (x, w, b))
  
  return dx, dw, db




def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db

###########################################################################################
# The following two functions stitched conv - bn - ReLU(W * X + B) forward and backward   #
###########################################################################################

def conv_sbn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
  """
  Convenience layer that performs a conv transform followed by a spatial batch normalization followed by a ReLU activation
  Inputs:
  - x: Input to the conv layer
  - w, b: weights and bias for the affine layer
  - conv_params: parameters for performing convolving translation.
  - gamma, beta, bn_params: parameters for translation, rescaling and training/testing running

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """

  a, conv_cache = conv_forward_fast(x, w, b, conv_param)

  # def spatial_batchnorm_forward(x, gamma, beta, bn_param): return out, cache;
  na, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)

  s, relu_cache = relu_forward(na)

  cache = (conv_cache, bn_cache, relu_cache)

  return s, cache

def conv_sbn_relu_backward(dout, cache):
  """
  the backward process of the above process
  """
  conv_cache, bn_cache, relu_cache = cache

  dna = relu_backward(dout, relu_cache) # def relu_backward(dout, cache): return dx;

  da, dgamma, dbeta = spatial_batchnorm_backward(dna, bn_cache) # def spatial_batchnorm_backward(dout, cache): return dx, dgamma, dbeta;

  dx, dw, db = conv_backward_fast(da, conv_cache)

  return dx, dw, db, dgamma, dbeta

























