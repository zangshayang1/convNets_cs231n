import numpy as np

from cs231n import layer_utils
from cs231n import layers

####################################################################################
# Different from previous all-in-one TwoLayerNet(), the classes in this module     #
# implement only the NN model that holds model-level parameters.                   #
# The training process is encapsulated in a Solver() object, where the model object#
# will be used and all the params for training such as learning rate will be hold  #
# within.                                                                          #
####################################################################################

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    self.params["W1"] = np.random.randn(input_dim, hidden_dim) * weight_scale
    self.params["W2"] = np.random.randn(hidden_dim, num_classes) * weight_scale
    self.params["b1"] = np.zeros(hidden_dim)
    self.params["b2"] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    W1, b1 = self.params["W1"], self.params["b1"]
    W2, b2 = self.params["W2"], self.params["b2"]

    scores, affine_relu_cache = layer_utils.affine_relu_forward(X, W1, b1)
    scores, fc_cache = layers.affine_forward(scores, W2, b2)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss, dscores = layers.softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2)) # Don't forget to add L2 regularization to loss!

    dx, dW2, db2 = layers.affine_backward(dscores, fc_cache)
    dx, dW1, db1 = layer_utils.affine_relu_backward(dx, affine_relu_cache)

    # don't forget to add L2 regularization to gradients!
    dW1 += self.reg * W1
    dW2 += self.reg * W2
    # make sure that grads[k] holds the gradients for self.params[k]
    grads["W1"] = dW1
    grads["W2"] = dW2
    grads["b1"] = db1
    grads["b2"] = db2
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    last_hd = None
    
    for i, hd in enumerate(hidden_dims):
      wi = 'W' + str(i + 1)
      bi = 'b' + str(i + 1)
      if i == 0:
        self.params[wi] = np.random.randn(input_dim, hd) * weight_scale
        self.params[bi] = np.zeros(hd)
        last_hd = hd

      else:
        assert not last_hd is None, "last_hidden_layer number of nodes initialization failed."

        self.params[wi] = np.random.randn(last_hd, hd) * weight_scale
        self.params[bi] = np.zeros(hd)
        last_hd = hd
      
    # {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    # now this is for the last affine
    last_w = 'W' + str(self.num_layers)
    last_b = 'b' + str(self.num_layers)
    self.params[last_w] = np.random.randn(last_hd, num_classes) * weight_scale
    self.params[last_b] = np.zeros(num_classes)

    ############################################################################
    #               initialize the dropout parameters                          #
    ############################################################################

    self.dropout_params = [{'mode': 'train', 'p': dropout, 'seed': seed} for i in xrange(self.num_layers - 1)]
    # Self.dropout_params will be set regardless of self.use_dropout. In case it is False, dropout = 0, dropout layer doens't change anything.
    # Initially set mode = train, which will be switched to "test" when y is not detected in evaluating loss.
    # This makes it a different design schema than self.bn_params, which will only be set upon self.use_batchnorm.
    # because there's no way inputs/gradients can go through batchnorm layer untouch as it is a learnable layer.

    # a few questions about dropout implementation:
    # 1. in __init__() dropout is set to be a single value - global dropout probability p - what if I want to modify p layer by layer?
    # 2. at first, self.dropout_params = {} is a single dict - globally use case, which determines dropout strategy once for the whole net.
    #    then, I thought I need to create affine_bn_relu_dropout_forward() and affine_bn_relu_dropout_backward() to implement it layer by layer.
    #    however, it is unnecessary. affine_bn_relu are chained together because gradients flow through them and each of them has a few params to be learned during training.
    #    but there's no such learnable params in a dropout layer.
    # 3. ToDO: self.dropout_params = [{}, ..., {}] is now initialized to hold multiple dict as number of layers grow.
    #          in future, self.dropout_params can hold a list of dropout probabilities provided by user.

    ############################################################################
    #                 initialize the batch norm parameters                     #
    ############################################################################
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)] 
      # running_mean & running_var will be created within each {} when training begins as well as updated when training progresses.
      for i, hd in enumerate(hidden_dims):
        # init gamma, beta for batch normalization as "layer-level parameters" instead of leaving them in self.bn_params
        # benefit1: they can be updated through solver.update_rule() the same way as w, b got updated.
        bnGammai = "bnGamma" + str(i + 1)
        bnBetai = "bnBeta" + str(i + 1)
        self.params[bnGammai] = np.ones(hd)
        self.params[bnBetai] = np.zeros(hd)

    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)

    # it is not only for testing our loss function but also for solver.check_accuracy() to calculate validation accuracy.
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    for dropout_param in self.dropout_params:
      dropout_param['mode'] = mode

    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param['mode'] = mode

    # intermediate_outputs = {} Turns out no need to store intermediate outputs.
    cache = {}
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    
    oj = X
    # oj tracks the output of the current layer and thus the input of the next layer.
    for i in range(self.num_layers - 1):
      # print "round: ", i
      wi, bi = self.params['W' + str(i + 1)], self.params['b' + str(i + 1)]

      if self.use_batchnorm:
        bnGammai, bnBetai = self.params["bnGamma" + str(i + 1)], self.params["bnBeta" + str(i + 1)]
        oi, ci = layer_utils.affine_bn_relu_forward(oj, wi, bi, bnGammai, bnBetai, self.bn_params[i]) # self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)] 
      
      else:
        oi, ci = layer_utils.affine_relu_forward(oj, wi, bi)
      
      # now both cache['backprop' + str(i)] and variable ci point to the tuple object that was hold only by ci
      # if you point ci to other tuple object, cache['backprop' + str(i)] doesn't change
      cache['backprop' + str(i + 1)] = ci

      # add dropout layer anyway and cache['dropouti'] always exists as well.
      oi, di = layers.dropout_forward(oi, self.dropout_params[i]) # def dropout_forward(x, dropout_param): return out, cache;
      cache['dropout' + str(i + 1)] = di

      # update oj for next forward pass  
      oj = oi
    
    # {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    # now this is for the last affine
    last_w, last_b = self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)]
    scores, cache['backprop' + str(self.num_layers)] = layers.affine_forward(oj, last_w, last_b)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If it is test mode, no need to go through backprop, simply just return early. 
    # In the case batch normalization, in test mode, the forward pass will be calculated using running_mean/running_var, updated gamma/beta.
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################

    loss, dscores = layers.softmax_loss(scores, y) # dscores are the gradients poping out from the softmax gate

    # add regularization to loss
    for i in range(1, self.num_layers + 1):
      wi = self.params['W' + str(i)]
      loss += 0.5 * self.reg * np.sum(wi * wi)

    # backprop last affine - store last_dw and last_db to future update weights on this layer
    last_backprop_cache = cache['backprop' + str(self.num_layers)]
    last_dx, last_dw, last_db = layers.affine_backward(dscores, last_backprop_cache) 

    # last_dw - gradient wrt weights, is computed by local_grad of this gate * grad from lower layers.
    # last_dx - gradient wrt x, is also computed the same way but used differently (involved in higher level grad backprop)
    grads['W' + str(self.num_layers)] = last_dw + self.reg * self.params['W' + str(self.num_layers)]
    grads['b' + str(self.num_layers)] = last_db

    # backprop affine - bn - ReLU - dropout
    for i in range(self.num_layers - 1, 0, -1):
      
      backprop_cache = cache["backprop" + str(i)]
      dropout_cache = cache["dropout" + str(i)]
      
      last_dx = layers.dropout_backward(last_dx, dropout_cache) # def dropout_backward(dout, cache): return dx;

      if self.use_batchnorm:
        last_dx, dwi, dbi, dbnGammai, dbnBetai = layer_utils.affine_bn_relu_backward(last_dx, backprop_cache)
        grads["bnGamma" + str(i)] = dbnGammai
        grads["bnBeta" + str(i)] = dbnBetai

      else:
        last_dx, dwi, dbi = layer_utils.affine_relu_backward(last_dx, backprop_cache)
      
      grads['W' + str(i)] = dwi + self.reg * self.params['W' + str(i)]
      grads['b' + str(i)] = dbi

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads












