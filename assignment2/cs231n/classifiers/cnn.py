import numpy as np

from cs231n import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer, filter is always a square
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    ############################################################################
    
    # Store weights and biases for the convolutional layer using the keys 'W1' and 'b1'; 
    C, H, W = input_dim
    filter_sizes = (filter_size, filter_size)
    self.params['W1'] = np.random.normal(0, weight_scale, [num_filters, C, filter_sizes[0], filter_sizes[1]])
    self.params['b1'] = np.zeros((num_filters, ))

    # use keys 'W2' and 'b2' for the weights and biases of the hidden affine layer;
    # In this case, ConvLayer doesn't reduce the spatial size of the input, (N, C, H, W) -> Conv -> (N, F, H, W)
    # To satisfy this constraint, (W + 2 * pad - filter_size) / stride + 1 = W need to hold, which led to pad = (F - S) / 2 where S == 1
    # (N, C, H, W) -> Conv -> (N, F, H, W) -> Pooling -> (N, F, H/2, W/2)
    # In a FC_NN, FCL weights (input_dim, hidden_dim) where every img is flatten into a 1D array of length D = F * H/2 * W/2.
    self.params['W2'] = np.random.normal(0, weight_scale, [num_filters * (H / 2) * (W / 2), hidden_dim])
    self.params['b2'] = np.zeros((hidden_dim, ))

    # And the keys 'W3' and 'b3' for the weights and biases of the output affine layer. 
    self.params['W3'] = np.random.normal(0, weight_scale, [hidden_dim, num_classes])
    self.params['b3'] = np.zeros((num_classes, ))

    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    
    scores = None    
    cache = {}
    # def conv_relu_pool_forward(x, w, b, conv_param, pool_param): return out, cache;
    out, cache['layer1'] = layer_utils.conv_relu_pool_forward(X, W1, b1, conv_param, pool_param) 
    # def affine_relu_forward(x, w, b): return out, cache;
    out, cache['layer2'] = layer_utils.affine_relu_forward(out, W2, b2)
    # def affine_forward(x, w, b): return out, cache;
    scores, cache['layer3'] = layers.affine_forward(out, W3, b3)


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    
    loss, grads = 0, {}

    # def softmax_loss(x, y): return loss, dscore;
    loss, dscores = layers.softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))

    # def affine_backward(dout, cache): return dx, dw, db;
    dout, dW3, db3 = layers.affine_backward(dscores, cache['layer3']) 
    # def affine_relu_backward(dout, cache): return dx, dw, db;
    dout, dW2, db2 = layer_utils.affine_relu_backward(dout, cache['layer2'])
    # def conv_relu_pool_backward(dout, cache): return dx, dw, db;
    dout, dW1, db1 = layer_utils.conv_relu_pool_backward(dout, cache['layer1'])

    # reg
    grads['W3'], grads['b3'] = dW3 + self.reg * W3, db3
    grads['W2'], grads['b2'] = dW2 + self.reg * W2, db2
    grads['W1'], grads['b1'] = dW1 + self.reg * W1, db1

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads

#########################################################################
# To build a fully modulized convNets constructor, quite a few changes  #
# need to be made on Solver class, such as updating process. Therefore  #
# I will build a convNets with fixed architecture including batchnorm   #
# for now. 
#########################################################################



class FourLayerConvNets(object):
  """
  A four-layer convolutional network with the following architecture:
  
  conv - sbn - relu - conv - sbn - relu - 2x2 max pool - affine - bn - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, 
                input_dim=(3, 32, 32), 
                num_filters = (32, 64), filter_sizes = (7, 7), conv_param = {"stride": 1, "pad": 3},
                hidden_dim= 100, num_classes=10, weight_scale=1e-3, reg=0.0,
                dtype=np.float32
                ):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer, filter is always a square
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.conv_param = conv_param
    self.filter_sizes = filter_sizes
    self.num_layers = 4
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    ############################################################################
    
    C, H, W = input_dim
    filter_size1, filter_size2 = filter_sizes
    num_filters1, num_filters2 = num_filters

    # conv layer 1: (N, C, H, W) -> (N, num_filters1, H, W)
    self.params['W1'] = np.random.normal(0, weight_scale, [num_filters1, C, filter_size1, filter_size1]) # square filter
    self.params['b1'] = np.zeros((num_filters1, ))
    self.params["sbnGamma1"] = np.ones((num_filters1, )) # scale parameter one for each color channel during spatial batch norm
    self.params["sbnBeta1"] = np.zeros((num_filters1, )) # shift parameter one for each color channel during spatial batch norm

    # conv layer 2: (N, num_filters1, H, W) -> (N, num_filters2, H, W)
    self.params['W2'] = np.random.normal(0, weight_scale, [num_filters2, num_filters1, filter_size2, filter_size2]) # square filter
    self.params['b2'] = np.zeros((num_filters2, ))
    self.params["sbnGamma2"] = np.ones((num_filters2, ))
    self.params["sbnBeta2"] = np.zeros((num_filters2, ))

    # (2, 2, 2) maxpool: (N, num_filters2, H, W) -> (N, num_filters2, H/2. W/2)
    # maxpool layer contributes nothing to self.params that need to be updated.
    self.maxpool_params = {"pool_height": 2, "pool_width": 2, "stride": 2}

    # affine layer 3: (N, num_filters2, H/2. W/2) -> (N, hidden_dim)
    self.params['W3'] = np.random.normal(0, weight_scale, [num_filters2 * (H / 2) * (W / 2), hidden_dim])
    self.params['b3'] = np.zeros((hidden_dim, ))
    self.params["bnGamma3"] = np.ones((hidden_dim, ))
    self.params["bnBeta3"] = np.zeros((hidden_dim, ))

    # output affine - sfmx layer 4: (N, hidden_dim) -> (N, num_classes)
    self.params['W4'] = np.random.normal(0, weight_scale, [hidden_dim, num_classes])
    self.params['b4'] = np.zeros((num_classes, ))

    self.bn_params = [{"mode": "train"} for _ in range(self.num_layers)]

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """

    # In dev testing, the loss fnc stops at "scores" , unfollowed by "softmax" probability prediction.
    # In real testing, "self.predict()" needs to be implemented in Solver() class.
    
    if y is None:
        for bn_param in self.bn_params:
            bn_param["mode"] = "test"


    W1, b1 = self.params['W1'], self.params['b1']
    gamma1, beta1 = self.params["sbnGamma1"], self.params["sbnBeta1"]
    bn_param1 = self.bn_params[0]

    W2, b2 = self.params['W2'], self.params['b2']
    gamma2, beta2 = self.params["sbnGamma2"], self.params["sbnBeta2"]
    bn_param2 = self.bn_params[1]

    W3, b3 = self.params['W3'], self.params['b3']
    gamma3, beta3 = self.params["bnGamma3"], self.params["bnBeta3"]
    bn_param3 = self.bn_params[2]

    W4, b4 = self.params['W4'], self.params['b4']
    
    # pass conv_param to the forward pass for the convolutional layer
    conv_param = self.conv_param

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = self.maxpool_params

    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    
    scores = None    
    cache = {}
    # def conv_sbn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param): return out, cache;
    out, cache["layer1"] = layer_utils.conv_sbn_relu_forward(X, W1, b1, gamma1, beta1, conv_param, bn_param1) 
    out, cache["layer2"] = layer_utils.conv_sbn_relu_forward(out, W2, b2, gamma2, beta2, conv_param, bn_param2)

    # def max_pool_forward_fast(x, pool_param): return out, cache;
    out, cache["maxpool"] = fast_layers.max_pool_forward_fast(out, pool_param)

    # def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param): return out, cache;
    
    out, cache["layer3"] = layer_utils.affine_bn_relu_forward(out, W3, b3, gamma3, beta3, bn_param3)

    # def affine_forward(x, w, b): return out, cache;
    scores, cache["layer4"] = layers.affine_forward(out, W4, b4)


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    
    loss, grads = 0, {}

    # def softmax_loss(x, y): return loss, dscore;
    loss, dscores = layers.softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3) + np.sum(W4 * W4))

    # def affine_backward(dout, cache): return dx, dw, db;
    dout, dW4, db4 = layers.affine_backward(dscores, cache["layer4"]) 

    # def affine_bn_relu_backward(dout, cache): return dx, dw, db, dgamma, dbeta;
    dout, dW3, db3, dgamma3, dbeta3 = layer_utils.affine_bn_relu_backward(dout, cache["layer3"])

    # print cache["layer3"]

    # def max_pool_backward_fast(dout, cache): return max_pool_backward_im2col(dout, real_cache);
    # def max_pool_backward_im2col(dout, cache): return dx;
    dout = fast_layers.max_pool_backward_fast(dout, cache["maxpool"])

    # def conv_sbn_relu_backward(dout, cache): return dx, dw, db, dgamma, dbeta;
    dout, dW2, db2, dgamma2, dbeta2 = layer_utils.conv_sbn_relu_backward(dout, cache["layer2"])
    _, dW1, db1, dgamma1, dbeta1 = layer_utils.conv_sbn_relu_backward(dout, cache["layer1"])

    # reg
    grads['W4'], grads['b4'] = dW4 + self.reg * W4, db4
    
    grads['W3'], grads['b3'] = dW3 + self.reg * W3, db3
    grads["bnGamma3"], grads["bnBeta3"] = dgamma3, dbeta3

    grads['W2'], grads['b2'] = dW2 + self.reg * W2, db2
    grads["sbnGamma2"], grads["sbnBeta2"] = dgamma2, dbeta2

    grads['W1'], grads['b1'] = dW1 + self.reg * W1, db1
    grads["sbnGamma1"], grads["sbnBeta1"] = dgamma1, dbeta1

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads



















