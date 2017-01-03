import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  x_shape = x.shape
  N = x_shape[0]
  x = x.reshape(N, -1)

  out = np.dot(x, w) + b # (N, M) + (M, ) broadcast == (N, M) + (1, M) broadcast

  x = x.reshape(x_shape)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  x_shape = x.shape
  N = x_shape[0]
  x = x.reshape(N, -1) # for computing dw

  M = dout.shape[1]
  
  dx = np.dot(dout, w.T)
  dx = dx.reshape(x_shape)

  dw = np.dot(x.T, dout)

  db = np.sum(dout, axis = 0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  # initially I didn't use copy() which lead to the changes occurred in x, wasted 2 hours on this.
  out = np.copy(x)
  out[out <= 0] = 0
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dx = np.copy(dout)
  dx[x <= 0] = 0
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var: Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    
    batch_mean = np.mean(x, axis = 0)
    batch_var = np.std(x, axis = 0)
    running_mean = momentum * running_mean + (1 - momentum) * batch_mean
    running_var = momentum * running_var + (1 - momentum) * batch_var

    x_mu = x - batch_mean

    var = np.mean(np.square(x_mu), axis = 0)

    sqrtvar = np.sqrt(var + eps)

    i_sqrtvar = 1.0 / sqrtvar

    x_hat = x_mu * i_sqrtvar

    out = np.multiply(x_hat, gamma) + beta

    cache = (x_hat, i_sqrtvar, sqrtvar, var, x_mu, gamma, eps)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    x = (x - running_mean) / running_var
    out = gamma * x + beta
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  x_hat, i_sqrtvar, sqrtvar, var, x_mu, gamma, eps = cache
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  
  # gate: f = x + beta - broadcast
  dbeta = np.sum(1 * dout, axis = 0) # with 1 being the local gradient df/dbeta
  backprop_x = 1 * dout # with 1 being the local gradient df/dx

  # gate: f = x * gamma
  dgamma = np.sum(np.multiply(x_hat, backprop_x), axis = 0) # with x_hat being the local gradient
  backprop_x = np.multiply(backprop_x, gamma) # with gamma being the local gradient

  # gate: f = x_mu * i_sqrtvar ((N, D) * (D,))
  dx_mu1 = np.multiply(backprop_x, i_sqrtvar) # with i_sqrtvar being the local grad
  di_sqrtvar = np.sum(np.multiply(backprop_x, x_mu), axis = 0) # with x_mu being the local grad and itself broadcasted

  # gate: i_sqrtvar = 1 / sqrtvar
  dsqrtvar = np.multiply((-1.0 / sqrtvar ** 2), di_sqrtvar) # with (-1.0 / sqrtvar ** 2) being the local grad

  # gate: f = (var + eps) ^ (1/2)
  dvar = 0.5 * (var + eps) ** (-0.5) * dsqrtvar # with (-0.5 * (var + eps) ** (-1.5)) being the local grad

  # gate: var = np.mean(np.square(x_mu), axis = 0)
  dsq = np.multiply(np.ones(x_mu.shape), dvar) / float(x_mu.shape[0])

  # gate: sq = (x_mu) ^ 2
  dx_mu2 = np.multiply(2 * x_mu, dsq) # with 2 * x_mu being the local grad

  # gate: x_mu = x - mu
  dx1 = 1 * (dx_mu1 + dx_mu2) # with 1 being the local grad
  dmu = -1 * (np.sum(dx_mu1, axis = 0) + np.sum(dx_mu2, axis = 0))

  # gate: mu = np.mean(x, axis = 0)
  dx2 = np.multiply(np.ones(x_mu.shape), dmu) / float(x_mu.shape[0])

  # When multiple gradients come to the same node, add them together.
  dx = dx1 + dx2

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  
  pass
  # refer to http://cthorey.github.io./backpropagation/

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). 
    dropout mask in training mode, None mask in testing mode.
  """
  out, mask = None, None
  p, mode = dropout_param['p'], dropout_param['mode']

  ################################################################
  # In the case of p == 0, this layer acts as if it doesn't exist.
  if p == 0:
    return x, (dropout_param, mask)
  ################################################################

  if not dropout_param['seed'] is None:
    np.random.seed(dropout_param['seed'])

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################

    mask = np.random.rand(*x.shape) < p 
    # Here's sth new: call fnc(*(1, 2, 3)) "*" is a splate operator unpacking list or tuple to feed into the fnc call
    # np.random.rand(m, n) is valid while np.random.rand((m, n)) is not.
    # Now mask is a bool array of the same size as x

    out = x * mask / p # if p == 0, this fnc shouldn't invoked at all.
    # The vanilla dropout is simply out = x * mask. Its expected out became p times of that of the original out without dropout.
    # Accordingly, in testing mode, forward pass should output p times testing out so that it matches the training magnitude.
    # The above calculation however, slows down the forward pass during testing time.
    # Therefore the inverted dropout is implemented here to compensate the expected magnitude of out.

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  p, mode = dropout_param['p'], dropout_param['mode']
  
  ############################################################
  # In the case of p == 0, this layer acts as if it doesn't exist.
  if p == 0:
    return dout
  ############################################################

  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    
    # gate: x * mask / p with (mask / p) being the local gradient
    dx = dout * mask / p

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx

def images_padded_with(x, pad):
  """
  Input:
  - x: Input image data of shape (N, C, H, W)
  - pad: number of pixels to be paded
  Output:
  - padded image data
  """
  
  N, C, H, W = x.shape
  paddedX = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
  for i in range(N):
    for j in range(C):
      paddedX[i, j, :, :] = np.pad(x[i, j], pad, mode = "constant", constant_values = 0) # it pads x[i, j] all around with 0 for pad pixels
      # np.pad(x[i, j], (2, 3), mode = "constant", constant_values = (0, 9)) # it pads x[i, j] with 2 leading 0s and 3 trailing 9s on each of its axises.
  return paddedX

def images_cropped_with(paddedX, pad):
  """
  This fnc undo what the images_padded_with() fnc does

  Input: 
  - paddedX: Input image data of shape (N, C, H, W)
  - pad: pixels to be cropped
  Output:
  - out: cropped image data
  """
  _, _, H, W = paddedX.shape
  croppedX = paddedX[:, :, pad: H - pad, pad: W - pad]
  return croppedX


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  stride, pad = conv_param["stride"], conv_param["pad"]
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  paddedX = images_padded_with(x, pad)

  N, C, H, W = x.shape
  F, _, HH, WW = w.shape
  OH = 1 + (H + 2 * pad - HH) / stride
  OW = 1 + (W + 2 * pad - WW) / stride
  out = np.zeros((N, F, OH, OW))
  for n in range(N):
    for i in range(OH):
      for j in range(OW):
        start_i, start_j = i * stride, j * stride
        cube = np.zeros((C, HH, WW)) # exactly like a filter
        for c in range(C):
          window = np.zeros((HH, WW))  # a slice of the cube
          for k in range(HH):
            for t in range(WW): # this is really tricky to code. I still can't believe I got it right. 
              window[k, t] = paddedX[n, c, start_i + k, start_j + t]
          cube[c, :, :] = np.copy(window)
        for f in range(F):
          out[n, f, i, j] = np.dot(cube.reshape(1, C * HH * WW), w[f, :, :, :].reshape(C * HH * WW, 1)) + b[f]
          # ReLU activation will follow.

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache

def imgs2col_matrice(x, filter_size, stride, pooling = False):
  """
  Input:
  - x: Input data of shape (N, C, H, W)
  - window_size: a tuple consisting of (window_height, window_width) - essentially like HH, WW in w.shape
  - stride: pace
  - pooling: the fnc was originally designed for conv layer, now got modified a bit so it can be used in pooling layer as well
  
  Return:
  - out: x being reshaped into a 2D matrice - (number_local_windows, window_size)

  For a single (most likely padded) img(3, 227, 227)
    convolving with 96 filters of (3, 11, 11) at stride 4
    it forms output(96, 55, 55).
  Now we:
    1. cut off a chunk of img of the same size with filter at each stride, there will be 55 * 55 = 3025 such chunks.
       pull each chunk into a 1D array(1, 3 * 11 * 11) and lay each of them on top of another, img -> (3025, 3 * 11 * 11)
    2. reshape filters w into (96, 3 * 11 * 11), each filter became a 1D array(1, 3 * 11 * 11)
    3. apply np.dot(img, filters.T) -> output(3025, 96) 

  This fnc implements the 1st of the above operations for N imgs.
  """
  out = None

  N, D, H, W = x.shape
  C, HH, WW = filter_size
  num_vertical_local_windows = (H - HH) / stride + 1
  num_horizontal_local_windows = (W - WW) / stride + 1
  
  # I could combine this two situations since there are only a little change in the middle but it'd require checking pooling bool for a lot more times. 
  if pooling:
    out_shape = (num_horizontal_local_windows * num_vertical_local_windows * N * D, HH * WW)
    # completely comb it into a vertical line of imgbars(N * OH * OW, C * HH * WW) -> (number_local_windows, window_size)
    out = np.zeros(out_shape)

    counter = 0
    for i in range(N):
      for d in range(D):
        rb = (HH, WW)
        while rb[0] <= H and rb[1] <= W:
          lt = (rb[0] - HH, rb[1] - WW)
          out[counter, :] = x[i, d, lt[0]:rb[0], lt[1]:rb[1]].reshape(-1)
          counter += 1
          if rb[1] == W:
            rb = (rb[0] + stride, WW)
          else:
            rb = (rb[0], rb[1] + stride)

  else:
    out_shape = (num_horizontal_local_windows * num_vertical_local_windows * N, C * HH * WW)
    out = np.zeros(out_shape)
    
    counter = 0
    for i in range(N): # outer loop -> N images
      rb = (HH, WW) # right bottom marks where we are # immutable tuple
      while rb[0] <= H and rb[1] <= W: # inner loop -> each image makes (num_horizontal_local_windows * num_vertical_local_windows) windows
        lt = (rb[0] - HH, rb[1] - WW)
        # During passing a forward conv layer, the depth of filter always match up with the input depth, so we safely leave : on the axis = 1
        out[counter, :] = x[i, :, lt[0]:rb[0], lt[1]:rb[1]].reshape(-1) 
        counter += 1
        if rb[1] == W:
          rb = (rb[0] + stride, WW)
        else:
          rb = (rb[0], rb[1] + stride)
  
  return out

def col2imgs_matrice(x_unfolded, x_shape, filter_size, stride, accumulative = True, pooling = False):
  """
  this fnc implements the reverse transformation of what imgs2col_matrice() does

  Input:
  - x_unfolded: the output of imgs2col_matrice(), shape: (N * OH * OW, C * HH * WW)
  - x_shape: specify the output shape
  - window_size: specify the window scale
  - stride: pace
  - accumulative: If it is called to restore the image, accumulative = False; If it is called to accumulate gradients dx, accumulative = True.
  - pooling: bool default is False, again, this fnc helps to reconstruct the pooling gradients into the shape of the input x when they flow back

  Output:
  - x: image data of shape (N, C, H, W)
  """
  N, D, H, W = x_shape
  C, HH, WW = filter_size

  x = np.zeros(x_shape)

  if pooling:
    counter = 0
    for i in range(N):
      for d in range(D):
        rb = (HH, WW)
        while rb[0] <= H and rb[1] <= W:
          lt = (rb[0] - HH, rb[1] - WW)
          x[i, d, lt[0]: rb[0], lt[1]: rb[1]] = x_unfolded[counter].reshape(HH, WW) 
          # NOTE the difference here - the depth of pooling is 1 while the depth of convolved x is C == depth of filter == depth of x
          counter += 1
          if rb[1] == W:
            rb = (rb[0] + stride, WW)
          else:
            rb = (rb[0], rb[1] + stride)    
  
  else:
    counter = 0
    for i in range(N):
      rb = (HH, WW)
      while rb[0] <= H and rb[1] <= W:
        lt = (rb[0] - HH, rb[1] - WW)
        if accumulative:
          x[i, :, lt[0]: rb[0], lt[1]: rb[1]] += x_unfolded[counter].reshape(C, HH, WW)
        else:
          x[i, :, lt[0]: rb[0], lt[1]: rb[1]] = x_unfolded[counter].reshape(C, HH, WW)
        counter += 1
        if rb[1] == W:
          rb = (rb[0] + stride, WW)
        else:
          rb = (rb[0], rb[1] + stride)
  return x

def conv_forward_efficient(x, w, b, conv_param):
  """
  It implements the conv layer forward pass based on efficient matrices dot product operation.

  The above imgs2col_matrice() implemented the 1st step
  Now we complete the following
    2. reshape filters w into (96, 3 * 11 * 11), each filter became a 1D array(1, 3 * 11 * 11)
    3. apply np.dot(img, filters.T) -> output(3025, 96) -> reshape into (96, 55, 55)

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  stride, pad = conv_param["stride"], conv_param["pad"]

  F, C, HH, WW = w.shape
  N, _, H, W = x.shape
  OH = 1 + (H + 2 * pad - HH) / stride
  OW = 1 + (W + 2 * pad - WW) / stride
  
  paddedX = images_padded_with(x, pad) # x(N, C, H, W) -> paddedX(N, C, H + 2pad, W + 2pad)

  imgbars = imgs2col_matrice(paddedX, (C, HH, WW), stride)
  # completely comb it into a vertical line of imgbars(N * OH * OW, C * HH * WW) -> (number_local_windows, window_size)
  
  w_strips = w.reshape(F, C * HH * WW).T # transpose it (filter_size, filter_numbers) ready for dot product.
  
  product = np.dot(imgbars, w_strips) + b # (N * OH * OW, F) + (, F) -> (number_local_windows, number_filters)

  reshaped_product = product.T.reshape(F, N, OH, OW) # .T so that it can be correctly reshaped (((OH, OW) for N times) for F times)

  out = np.transpose(reshaped_product, axes = (1, 0, 2, 3)) # switch the first and the second axes to match out format

  cache = (x, w, b, conv_param, imgbars, w_strips)

  return out, cache

def conv_backward_efficient(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x - (N, C, H, W)
  - dw: Gradient with respect to w - (F, C, HH, WW)
  - db: Gradient with respect to b - (F, )
  
  """
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  
  x, w, b, conv_param, imgbars, w_strips = cache
  stride, pad = conv_param["stride"], conv_param["pad"]
  # imgbars: (N * OH * OW, C * HH * WW)
  # w_strips: (C * HH * WW, F)
  # product: (N * OH * OW, F)

  N, _, H, W = x.shape
  F, C, HH, WW = w.shape  
  
  # the following is the reverse transformation of "product -> out" in forward pass
  # this transformation is critical and becomes the basis for all the matrices computation that follow.
  dout = np.transpose(dout, axes = (1, 0, 2, 3))
  dout = dout.reshape(F, -1)
  # gate: out = x * w + b; So: dw = x * dout
  dw = np.dot(dout, imgbars)
  dw = dw.reshape(F, C, HH, WW)

  # gate: out = x * w + b; So: dx = w * dout
  dx_bars = np.dot(w_strips, dout).T # .T so that dx_bars.shape matches imgbars_like.shape to be correctly processed by col2imgs_matrice() as follows
  # the following is the reverse transformation of what the imgs2col_matrice() does in forward pass
  padded_x_shape = (N, C, H + 2 * pad, W + 2 * pad)
  dx = col2imgs_matrice(dx_bars, padded_x_shape, (C, HH, WW), stride, accumulative = True)
  dx = images_cropped_with(dx, pad)

  # gate: out = x * w + b; So: db = np.sum(dout, axis = 1)
  db = np.sum(dout, axis = 1)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_efficient(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """

  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  out = None
  PH, PW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
  
  imgbars = imgs2col_matrice(x, (1, PH, PW), stride, pooling = True) 
  # def imgs2col_matrice(x, filter_size, stride, pooling = False): return out;

  maxbars = np.amax(imgbars, axis = 1, keepdims = True) # 2D

  N, D, H, W = x.shape
  OH = (H - PH) / stride + 1
  OW = (W - PW) / stride + 1
  # need to figure out the dimensions so that I can reshape maxbars into output
  out = maxbars.reshape(N, D, OH, OW)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x.shape, imgbars, maxbars, pool_param)

  return out, cache


def max_pool_backward_efficient(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  
  x_shape_recovered, imgbars, maxbars, pool_param = cache
  PH, PW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # gate: out_pixel = max(x_window) -> pass dout through where imgbars == maxbars is True (indices of the max value chosen) and 0 through other indices else
  
  # expand dout -> into doutbars of the same shape as imgbars, so the correct value can be calved out of doutbars when "imgbars == maxbar" works as a mask
  doutbars = np.dot(dout.reshape(-1, 1), np.ones((1, imgbars.shape[1])))

  # np.where(condition(bool table), value_table where True, value_table where False) # these three tables have the same shape
  dx = np.where(imgbars == maxbars, doutbars, np.zeros_like(doutbars))

  # def col2imgs_matrice(x_unfolded, x_shape, filter_size, stride, accumulative = True, pooling = False): return x;
  dx = col2imgs_matrice(dx, x_shape_recovered, (1, PH, PW), stride, pooling = True) 
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Normally batch-normalization accepts inputs of shape (N, D) and produces outputs of shape (N, D), where we normalize across the minibatch dimension N. 
    For data coming from convolutional layers, batch normalization needs to accept inputs of shape (N, C, H, W) and produce outputs of shape (N, C, H, W) 
    where the N dimension gives the minibatch size and the (H, W) dimensions give the spatial size of the feature map.

  If the feature map was produced using convolutions, then we expect the statistics of each feature channel to be relatively consistent both between different images and different locations within the same image. 
    Therefore spatial batch normalization computes a mean and variance for each of the C feature channels by computing statistics over both the minibatch dimension N and the spatial dimensions H and W.
  
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  
  out, cache = None, None

  N, C, H, W = x.shape
  
  # imagine you are looking at the color channels of an image (C, H, W), then you take each channel as a feature and roll it out into (-1, C)
  # now the following does that on a mini-batch of N images.
  rollout = np.transpose(x, (0, 2, 3, 1)).reshape(-1, C)

  # def batchnorm_forward(x, gamma, beta, bn_param): return out, cache;
  out, cache = batchnorm_forward(rollout, gamma, beta, bn_param)

  # restore the shape of the output
  out = np.transpose(out.reshape(N, H, W, C), (0, 3, 1, 2))

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  
  dx, dgamma, dbeta = None, None, None

  N, C, H, W = dout.shape

  rollout = np.transpose(dout, (0, 2, 3, 1)).reshape(-1, C)

  # def batchnorm_backward(dout, cache): return dx, dgamma, dbeta
  dx, dgamma, dbeta = batchnorm_backward(rollout, cache)

  dx = np.transpose(dx.reshape(N, H, W, C), (0, 3, 1, 2))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / float(N)
  dx = probs.copy()
  dx[np.arange(N), y] -= 1 # only appears at the softmax layer
  dx /= float(N)
  return loss, dx
