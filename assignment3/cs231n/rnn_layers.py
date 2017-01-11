import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for the backward pass.
  """
  next_h, cache = None, None
  ##############################################################################
  # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
  # hidden state and any values you need for the backward pass in the next_h   #
  # and cache variables respectively.                                          #
  ##############################################################################

  # draw a vannila RNN graph and understand the following:

  # the output at timestep t = the dot product of the weights from hidden state
  # to output and the values of hidden variables -> y_t = np.dot(W_hy, h_t) + b
  # but y_t is not what's fed into the next hidden layer, next_h is.
  # y_label = sfmx(y_t)

  # the values of hidden variables h_t = activation(dot product of the values
  # of the last hidden variables and corresponding weights + the dot product of
  # the current input and corresponding weights), which
  # -> h_t = activation(np.dot(W_hh, h_t-1) + np.dot(W_xh, X_t) + b)

  a = np.dot(prev_h, Wh) + np.dot(x, Wx) + b
  next_h = np.tanh(a)

  cache = (x, prev_h, Wx, Wh, a)

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return next_h, cache


def rnn_step_backward(dnext_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN.

  Inputs:
  - dnext_h: Gradient of loss with respect to next hidden state
  - cache: Cache object from the forward pass

  Returns a tuple of:
  - dx: Gradients of input data, of shape (N, D)
  - dprev_h: Gradients of previous hidden state, of shape (N, H)
  - dWx: Gradients of input-to-hidden weights, of shape (N, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """
  dx, dprev_h, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
  #                                                                            #
  # HINT: For the tanh function, you can compute the local derivative in terms #
  # of the output value from tanh.                                             #
  ##############################################################################

  x, prev_h, Wx, Wh, a = cache

  # computational graph:
  # ax = Wx * X
  # ah = Wh * H(t)
  # a = ax + ah + b
  # H(t+1) = tanh(a)

  # dH(t+1)/da: local - (1 - tanh^2(a)); upstream - dnext_h flow through tanh gate
  da = np.multiply((np.ones_like(a) - np.square(np.tanh(a))), dnext_h)
  # da/dax = da/dah = da/db: local - 1; upstream - da flow through [+] gate
  dax = da
  dah = da
  db = np.sum(da, axis = 0)
  # dax/dX: local - Wx; upstream - dax flow through [*] gate
  dx = np.dot(dax, Wx.T)
  # dax/dWx: local - X; upstream - dax flow through [*] gate
  dWx = np.dot(x.T, dax)
  # dah/dH(t): local - Wh; upstream - dah flow through [*] gate
  dprev_h = np.dot(dah, Wh.T)
  # dah/dWh: local - H(t); upstream - dah flow through [*] gate
  dWh = np.dot(prev_h.T, dah)

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.

  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  h, cache = None, None
  ##############################################################################
  # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
  # input data. You should use the rnn_step_forward function that you defined  #
  # above.                                                                     #
  ##############################################################################

  N, T, D = x.shape
  _, H = h0.shape

  h = np.zeros((T, N, H))
  cache_list = [None for _ in range(T)]

  # NOTE: At every hidden state, we advance input along T axis, going through N
  # sequences at the same time.
  x = x.transpose(1, 0, 2)

  # def rnn_step_forward(x, prev_h, Wx, Wh, b): return next_h, cache
  # cache = (x, prev_h, Wx, Wh, a)
  next_h, cache = rnn_step_forward(x[0], h0, Wx, Wh, b)
  h[0] = next_h
  cache_list[0] = cache

  for i in range(1, T):
      next_h, cache = rnn_step_forward(x[i], next_h, Wx, Wh, b)
      h[i] = next_h
      cache_list[i] = cache

  h = h.transpose(1, 0, 2)

  dimensions = N, T, D, H
  forward_cache = (dimensions, cache_list)

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return h, forward_cache


def rnn_backward(dh, forward_cache):
  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data.

  Inputs:
  - dh: Upstream gradients of all hidden states, of shape (N, T, H), coming down
        from output layer y_label

  Returns a tuple of:
  - dx: Gradient of inputs, of shape (N, T, D)
  - dh0: Gradient of initial hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - db: Gradient of biases, of shape (H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a vanilla RNN running an entire      #
  # sequence of data. You should use the rnn_step_backward function that you   #
  # defined above.                                                             #
  ##############################################################################

  dimensions, cache_list = forward_cache
  N, T, D, H = dimensions

  # prepare output
  dx = np.zeros((T, N, D))
  dh0 = np.zeros((N, H))
  dWx = np.zeros((D, H))
  dWh = np.zeros((H, H))
  db = np.zeros((H, ))

  # prepare input, the following two gradients flow through the same hidden unit
  # - gradients from output layers for all hidden layers
  dh = dh.transpose(1, 0, 2)
  # - gradients from the last hidden layer
  dprev_h = np.zeros((N, H))

  for i in range(T - 1, -1, -1):
      # def rnn_step_backward(dnext_h, cache): return dx, dprev_h, dWx, dWh, db
      ddx, dprev_h, ddWx, ddWh, ddb = rnn_step_backward(dh[i] + dprev_h, cache_list[i])
      dx[i] = ddx
      dWx += ddWx
      dWh += ddWh
      db += ddb

  # don't forget the format output accordingly
  dh0 = dprev_h
  dx = dx.transpose(1, 0, 2)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
  """
  Forward pass for word embeddings. We operate on minibatches of size N where
  each sequence has length T. We assume a vocabulary of V words, assigning each
  to a vector of dimension D.

  Inputs:
  - x: Integer array of shape (N, T) giving indices of words. Each element idx
    of x muxt be in the range 0 <= idx < V.
  - W: Weight matrix of shape (V, D) giving word vectors for all words.

  Returns a tuple of:
  - out: Array of shape (N, T, D) giving word vectors for all input words.
  - cache: Values needed for the backward pass
  """
  out, cache = None, None
  ##############################################################################
  # TODO: Implement the forward pass for word embeddings.                      #
  ##############################################################################

  N, T = x.shape
  V, D = W.shape

  # out = np.zeros((N, T, D))
  #
  # for i in range(N):
  #     for j in range(T):
  #         out[i, j, :] = W[x[i, j], :]

  out = W[x] # quite amazing this line of code can do so much!

  cache = W.shape, x

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return out, cache


def word_embedding_backward(dout, cache):
  """
  Backward pass for word embeddings. We cannot back-propagate into the words
  since they are integers, so we only return gradient for the word embedding
  matrix.

  HINT: Look up the function np.add.at

  Inputs:
  - dout: Upstream gradients of shape (N, T, D)
  - cache: Values from the forward pass

  Returns:
  - dW: Gradient of word embedding matrix, of shape (V, D).
  """
  dW = None
  ##############################################################################
  # TODO: Implement the backward pass for word embeddings.                     #
  ##############################################################################

  W_shape, x = cache

  dW = np.zeros((W_shape))

  # np.add.at(a, indices, b) performs addition on operand a indiced by "indices"
  # and operand b. b could be constant, broadcastable array, OR in this case -
  # an array of the same length as the indices array, so that, elements in b will
  # be added into a[indices] in order.

  # paste the following code in a notebook to see how it works
  ##############################################################################
    # N, T, V, D = 3, 2, 9, 4
    # x = np.random.randint(0, 9, size = (N, T))
    # print "array of integers representing words"
    # print x
    # w = np.random.randn(V, D)
    # print "word vectors for each word in vocabulary"
    # print w
    # print "word vectors for each word of input"
    # print w[x]
    # print np.prod(x.shape)
    #
    # out = w[x]
    # dout = np.arange(N * T * D).reshape(N, T, D) # same shape as out
    # dw = np.zeros_like(w)
    # np.add.at(dw, x.reshape(-1), dout.reshape(N*T, D))
    # print "indices:"
    # print x.reshape(-1)
    # print "the other operand:"
    # print dout.reshape(N * T, D)
    # print "result:"
    # print dw
  ##############################################################################

  # len(x.reshape(-1)) == N * T
  N, T = x.shape
  np.add.at(dW, x.reshape(-1), dout.reshape(N * T, -1))

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dW


def sigmoid(x):
  """
  A numerically stable version of the logistic sigmoid function.
  """
  pos_mask = (x >= 0)
  neg_mask = (x < 0)
  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])
  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  """
  Forward pass for a single timestep of an LSTM.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  next_h, next_c, cache = None, None, None
  #############################################################################
  # TODO: Implement the forward pass for a single timestep of an LSTM.        #
  # You may want to use the numerically stable sigmoid implementation above.  #
  #############################################################################

  # RNN hidden state (as opposed to y emission state):
  # H(t+1) = tanh(X * Wx + H(t) * Wh + b)

  # LSTM hidden cell state: c(t+1) = f * c(t) + g * i
  # LSTM hidden state: H(t+1) = o * tanh(c(t+1))

  # i: input state based on current input and current hidden state
  #     i = sigmoid(X(t) * Wx_i + H(t) * Wh_i) + b_i
  # g: how much the input state negatively/positively impact the hidden cell state
  #     g = tanh(X(t) * Wx_g + H(t) * Wh_g) + b_g
  # f: how much the hidden cell state count on the next hidden cell state
  #     f = sigmoid(X(t) * Wx_f + H(t) * Wh_f) + b_f
  # o: how much the next hidden cell state count on the next hidden state
  #     o = sigmoid(X(t) * Wx_o + H(t) * Wh_o) + b_o

  _, H = prev_h.shape

  ii, ff, oo, gg = H, 2 * H, 3 * H, 4 * H

  # big dot product (N, 4H)
  dp = np.dot(x, Wx) + np.dot(prev_h, Wh) + b

  i = sigmoid(dp[:, :ii])
  f = sigmoid(dp[:, ii:ff])
  o = sigmoid(dp[:, ff:oo])
  g = np.tanh(dp[:, oo:gg])

  next_c = np.multiply(f, prev_c) + np.multiply(g, i)
  next_h = np.multiply(o, np.tanh(next_c))

  cache = x, Wx, Wh, b, dp, i, f, o, g, next_c, prev_c, prev_h
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
  """
  Backward pass for a single timestep of an LSTM.

  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass

  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  # dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for a single timestep of an LSTM.       #
  #                                                                           #
  # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
  # the output value from the nonlinearity.                                   #
  #############################################################################

  x, Wx, Wh, b, dp, i, f, o, g, next_c, prev_c, prev_h = cache

  N, H = dnext_h.shape

  dWx = np.zeros_like(Wx)
  dWh = np.zeros_like(Wh)
  db = np.zeros_like(b)

  ii, ff, oo, gg = H, 2 * H, 3 * H, 4 * H

  # computational graph:
  # -------------------
  # i:
  # axi = x * Wx_i
  # ahi = prev_h * Wh_i
  # ai = axi + ahi + bi
  # i = sig(ai)
  # -------------------
  # f:
  # axf = x * Wx_f
  # ahf = prev_h * Wh_f
  # af = axf + ahf + bf
  # f = sig(af)
  # -------------------
  # o:
  # axo = x * Wx_o
  # aho = prev_h * Wh_o
  # ao = axo + aho + bf
  # o = sig(ao)
  # -------------------
  # g:
  # axg = x * Wx_g
  # ahg = prev_h * Wh_g
  # ag = axg + ahg + bg
  # g = tanh(ag)
  # -------------------
  # intermediate results for future use
  ai = dp[:, :ii]
  af = dp[:, ii:ff]
  ao = dp[:, ff:oo]
  ag = dp[:, oo:gg]
  # -------------------
  # 1. gi = g * i
  # 2. f_c_prev = f * prev_c
  # 3. next_c = f_c_prev + gi
  # 4. tanh_c_next = tanh(next_c)
  # 5. h_next = o * tanh_c_next
  # -------------------

  # 5:
  # dh_next/dtanh_c_next: local - o; upstream - dnext_h flow through [*] gate
  # dh_next/do: local - tanh(next_c); upstream - dnext_h flow through [*] gate
  dtanh_c_next = o * dnext_h # (N, H)
  do = np.tanh(next_c) * dnext_h # (N, H)
  # 4: dtanh_c_next/dc_next_from_h: local - (1 - tanh^2(c_next)); upstream - dtanh_c_next flow through [tanh] gate
  dc_next_from_h = (np.ones_like(next_c) - np.square(np.tanh(next_c))) * dtanh_c_next # (N, H)

  # NOTE: (this is also the gist of LSTM)
  # one gradients stream from the next hidden cell state directly;
  # another gradients stream from the hidden state, h
  dc_next_from_c = dnext_c
  dc_next = dc_next_from_h + dc_next_from_c

  # 3:
  # dnext_c/dgi: local - 1; upstream - dc_next flow through [+] gate;
  # dnext_c/df_c_prev: local - 1; upstream - dc_next flow through [+] gate;
  dgi = dc_next # (N, H)
  df_c_prev = dc_next # (N, H)
  # 2:
  # df_c_prev/df: local - prev_c; upstream - df_c_prev flow through [*] gate;
  # df_c_prev/dprev_c: local - f; upstream - df_c_prev flow through [*] gate;
  df = prev_c * df_c_prev # (N, H)
  dprev_c = f * df_c_prev # (N, H)
  # 1:
  # dgi/dg: local - i; upstream - dgi flowthrough [*] gate;
  # dgi/di: local -g; upstream - dgi flowthrough [*] gate;
  dg = i * dgi # (N, H)
  di = g * dgi # (N, H)
  # ----------------------------------------------------------------------------
  # g:
  # dg/dag: local - tanh - (1 - tanh^2(ag)); upstream - dg flowthrough [tanh] gate;
  dag = (np.ones_like(ag) - np.square(np.tanh(ag))) * dg # (N, H)
  # dag/daxg = dag/dahg = dag/dbg: local - 1; upstream - dag flowthrough [+] gate;
  daxg, dahg, dbg = dag, dag, dag

  # dahg/dprev_h: local - Wh_g; upstream - dahg == dag flowthrough [dot *] gate;
  dprev_h = np.dot(dahg, Wh[:, oo:gg].T) # (N, H) = np.dot((N, H), (H, H).T)
  # dahg/dWh_g: local - prev_h; upstream - dahg == dag flowthrough [dot *] gate;
  dWh[:, oo:gg] = np.dot(prev_h.T, dahg) # (H, H) = np.dot((N, H).T, (N, H))
  # daxg/dx: local - Wx_g; upstream - daxg == dag flowthrough [*] gate;
  dx = np.dot(daxg, Wx[:, oo:gg].T) # (N, D) = np.dot((N, H), (D, H).T)
  # daxg/dWx_g: local - x; upstream - daxg == dag flowthrough [*] gate;
  dWx[:, oo:gg] = np.dot(x.T, daxg) # (D, H) = np.dot((N, D).T, (N, H))

  db[oo:gg] = np.sum(dbg, axis = 0) # (H, ) = np.sum((N, H), axis = 0)

  # likewise:
  # o:
  # do/dao: local - sig - (1 - sigmoid(ao))*sigmoid(ao); upstream - do flowthrough [sigmoid] gate
  dao = (np.ones_like(ao) - sigmoid(ao)) * sigmoid(ao) * do # (N, H)
  daf = (np.ones_like(af) - sigmoid(af)) * sigmoid(af) * df # (N, H)
  dai = (np.ones_like(ai) - sigmoid(ai)) * sigmoid(ai) * di # (N, H)

  daxo, daho, dbo = dao, dao, dao
  daxf, dahf, dbf = daf, daf, daf
  daxi, dahi, dbi = dai, dai, dai

  dprev_h += np.dot(daho, Wh[:, ff:oo].T)
  dprev_h += np.dot(dahf, Wh[:, ii:ff].T)
  dprev_h += np.dot(dahi, Wh[:, :ii].T)

  dWh[:, ff:oo] = np.dot(prev_h.T, daho)
  dWh[:, ii:ff] = np.dot(prev_h.T, dahf)
  dWh[:, :ii] = np.dot(prev_h.T, dahi)

  dx += np.dot(daxo, Wx[:, ff:oo].T)
  dx += np.dot(daxf, Wx[:, ii:ff].T)
  dx += np.dot(daxi, Wx[:, :ii].T)

  dWx[:, ff:oo] = np.dot(x.T, daxo)
  dWx[:, ii:ff] = np.dot(x.T, daxf)
  dWx[:, :ii] = np.dot(x.T, daxi)

  db[ff:oo] = np.sum(dbo, axis = 0)
  db[ii:ff] = np.sum(dbf, axis = 0)
  db[:ii] = np.sum(dbi, axis = 0)

  return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.

  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.

  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)

  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  h, cache = None, None
  #############################################################################
  # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
  # You should use the lstm_step_forward function that you just defined.      #
  #############################################################################

  # def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b): return next_h, next_c, cache;
  N, T, D = x.shape
  _, H = h0.shape
  h = np.zeros((N, T, H))
  cache_list = [None for _ in range(T)]

  prev_h = h0
  prev_c = np.zeros_like(prev_h) # init prev_c
  for i in range(T):
      x_i = x[:, i, :].reshape(N, D)
      prev_h, prev_c, cache_list[i] = lstm_step_forward(x_i, prev_h, prev_c, Wx, Wh, b)
      h[:, i, :] = prev_h

  cache = (x.shape, cache_list)

  return h, cache


def lstm_backward(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.

  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass

  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
  # You should use the lstm_step_backward function that you just defined.     #
  #############################################################################

  x_shape, cache_list = cache
  _, _, D = x_shape
  N, T, H = dh.shape

  dx = np.zeros((N, T, D))
  dWx = np.zeros((D, 4 * H))
  dWh = np.zeros((H, 4 * H))
  db = np.zeros((4 * H, ))

  dlast_c = np.zeros((N, H)) # To init the direct flow from dnext_c
  dlast_h = np.zeros((N, H))
  for i in range(T - 1, -1, -1):
      dupstream_h = dh[:, i, :].reshape(N, H)
      # def lstm_step_backward(dnext_h, dnext_c, cache): return dx, dprev_h, dprev_c, dWx, dWh, db
      dx[:, i, :], dlast_h, dlast_c, dWx_i, dWh_i, db_i = lstm_step_backward(dupstream_h + dlast_h, dlast_c, cache_list[i])
      # ----------------------------------------------------
      # the gist of LSTM is there're two timeseries highway:
      # 1. hidden state highway that helps to roll back the timeseries info
      # 2. hidden cell state super highway that creates a more direct gradients flow between hidden cells
      # ----------------------------------------------------
      dWx += dWx_i
      dWh += dWh_i
      db += db_i

  dh0 = dlast_h

  return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
  """
  Forward pass for a temporal affine layer. The input is a set of D-dimensional
  vectors arranged into a minibatch of N timeseries, each of length T. We use
  an affine function to transform each of those vectors into a new vector of
  dimension V.

  Inputs:
  - x: Input data of shape (N, T, D)
  - w: Weights of shape (D, V)
  - b: Biases of shape (V,)

  Returns a tuple of:
  - out: Output data of shape (N, T, V)
  - cache: Values needed for the backward pass
  """
  N, T, D = x.shape
  V = b.shape[0]
  out = x.reshape(N * T, D).dot(w).reshape(N, T, V) + b
  cache = x, w, b, out
  return out, cache


def temporal_affine_backward(dout, cache):
  """
  Backward pass for temporal affine layer.

  Input:
  - dout: Upstream gradients of shape (N, T, M)
  - cache: Values from forward pass

  Returns a tuple of:
  - dx: Gradient of input, of shape (N, T, D)
  - dw: Gradient of weights, of shape (D, M)
  - db: Gradient of biases, of shape (M,)
  """
  x, w, b, out = cache
  N, T, D = x.shape
  M = b.shape[0]

  dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
  dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
  db = dout.sum(axis=(0, 1))

  return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
  """
  A temporal version of softmax loss for use in RNNs. We assume that we are
  making predictions over a vocabulary of size V for each timestep of a
  timeseries of length T, over a minibatch of size N. The input x gives scores
  for all vocabulary elements at all timesteps, and y gives the indices of the
  ground-truth element at each timestep. We use a cross-entropy loss at each
  timestep, summing the loss over all timesteps and averaging across the
  minibatch.

  As an additional complication, we may want to ignore the model output at some
  timesteps, since sequences of different length may have been combined into a
  minibatch and padded with NULL tokens. The optional mask argument tells us
  which elements should contribute to the loss.

  Inputs:
  - x: Input scores, of shape (N, T, V)
  - y: Ground-truth indices, of shape (N, T) where each element is in the range
       0 <= y[i, t] < V
  - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss.

  Returns a tuple of:
  - loss: Scalar giving loss
  - dx: Gradient of loss with respect to scores x.
  """

  N, T, V = x.shape

  x_flat = x.reshape(N * T, V)
  y_flat = y.reshape(N * T)

  # each word corresponds to one boolean mask in a minibatch
  mask_flat = mask.reshape(N * T)

  probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)

  # sum loss over each word from input, according to the ground-truth
  # with small loss for high possibility and large lss for low possibility
  loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N

  # gradients of lss wrt scores x
  dx_flat = probs.copy()
  dx_flat[np.arange(N * T), y_flat] -= 1
  dx_flat /= N
  dx_flat *= mask_flat[:, None]

  if verbose: print 'dx_flat: ', dx_flat.shape

  dx = dx_flat.reshape(N, T, V)

  return loss, dx
