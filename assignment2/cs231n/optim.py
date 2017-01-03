import numpy as np

"""
This file implements various first-order update rules that are commonly used for
training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning rate,
    momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not perform
well; however the default values of the other hyperparameters should work well
for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""


def sgd(w, dw, config=None):
  """
  Performs vanilla stochastic gradient descent.

  config format:
  - learning_rate: Scalar learning rate.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2) #

  w -= config['learning_rate'] * dw
  return w, config


def sgd_momentum(w, dw, config=None):
  """
  Performs stochastic gradient descent with momentum.

  config format:
  - learning_rate: Scalar learning rate.
  - momentum: Scalar between 0 and 1 giving the momentum value.
    Setting momentum = 0 reduces to sgd.
  - velocity: A numpy array of the same shape as w and dw used to store a moving
    average of the gradients.
  """
  if config is None: config = {}
  
  # dict.setdefault(k, v): return dict[k] if k in dict else return v and set dict[k] = v
  config.setdefault('learning_rate', 1e-2)
  config.setdefault('momentum', 0.9)
  
  # dict.get(k, v): return dict[k] if k in dict else return v but k will still not be in dict.
  v = config.get('velocity', np.zeros_like(w))

  m = config['momentum']
  lr = config['learning_rate']

  #############################################################################
  # TODO: Implement the momentum update formula. Store the updated value in   #
  # the next_w variable. You should also use and update the velocity v.       #
  #############################################################################
  
  v = - lr * dw + m * v # initially it makes no changes as compared with w -= learning_rate * dw

  w += v
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  config['velocity'] = v           # but it accumulates "dw" into velocity and thus builds up the speed to descent 
                                   # if dw stays positive/negative for quite a few updates
  return w, config



def rmsprop(w, dw, config=None):
  """
  Uses the RMSProp update rule, which uses a moving average of squared gradient
  values to set adaptive per-parameter learning rates.

  config format:
  - learning_rate: Scalar learning rate.
  - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
    gradient cache.
  - epsilon: Small scalar used for smoothing to avoid dividing by zero.
  - cache: Moving average of second moments of gradients.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)
  config.setdefault('decay_rate', 0.99)
  config.setdefault('epsilon', 1e-8)
  config.setdefault('cache', np.zeros_like(w))

  lr = config['learning_rate']
  dr = config['decay_rate']
  eps = config['epsilon']
  cache = config['cache']
  #############################################################################
  # TODO: Implement the RMSprop update formula, storing the next value of x   #
  # in the next_x variable. Don't forget to update cache value stored in      #  
  # config['cache'].                                                          #
  #############################################################################

  # rmsprop is built on adagrad which follows
  # ----------------------------------------
  # cache += dw ** 2 -> tracks of per-parameter sum of squared historical gradients
  # w -= dw * learning_rate / (sqrt(cache) + eps) -> benefit: equalizing learning rate for big/small gradients
  #                                                -> disadvantage: monotonically decreasing dw, too aggresive
  # ----------------------------------------
  
  # rmsprop, instead of using sum of histroical effects, uses a moving average (leaky) strategy to adaptively determine the learning_rate
  cache = dr * cache + (1 - dr) * dw ** 2
  w -= dw * lr / (np.sqrt(cache) + eps)
  config['cache'] = cache
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return w, config


def adam(w, dw, config=None):
  """
  Uses the Adam update rule, which incorporates moving averages of both the
  gradient and its square and a bias correction term.

  config format:
  - learning_rate: Scalar learning rate.
  - beta1: Decay rate for moving average of first moment of gradient.
  - beta2: Decay rate for moving average of second moment of gradient.
  - epsilon: Small scalar used for smoothing to avoid dividing by zero.
  - velocity: Moving average of gradient.
  - cache: Moving average of squared gradient.
  - t: Iteration number.
  """
  if config is None: config = {}
  lr = config.setdefault('learning_rate', 1e-3)
  beta1 = config.setdefault('beta1', 0.9)
  beta2 = config.setdefault('beta2', 0.999)
  eps = config.setdefault('epsilon', 1e-8)
  momentum = config.setdefault('momentum', np.zeros_like(w))
  cache = config.setdefault('cache', np.zeros_like(w))
  t = config.setdefault('t', 0)
  
  #############################################################################
  # TODO: Implement the Adam update formula, storing the next value of x in   #
  # the next_x variable. Don't forget to update the m, v, and t variables     #
  # stored in config.                                                         #
  #############################################################################

  # adam is the combination of rmsprop (cache part) and sgd_momentum (only difference: raw dw -> (1 - beta1) * dw for a smoothier update)
  momentum = beta1 * momentum + (1 - beta1) * dw
  cache = beta2 * cache + (1 - beta2) * dw ** 2
  w -= lr * momentum / (np.sqrt(cache) + eps)
  
  t += 1

  config['momentum'] = momentum
  config['cache'] = cache
  config['t'] = t
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return w, config

  
  
  

