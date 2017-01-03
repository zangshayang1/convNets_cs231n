import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N, D = X.shape
  D, C = W.shape
  
  for i in range(N):
    scores = np.dot(X[i, :], W)
    # sum_exp_scores = np.sum(np.exp(scores)) this line could easily go exploding if scores are big numbers
    # therefore we do the following to ensure numerical stability
    scores -= np.max(scores)
    sum_exp_scores = np.sum(np.exp(scores))

    loss += - np.log(np.exp(scores[y[i]]) / sum_exp_scores)
    for j in range(C):
      # NOTE: for each class, for each feature, you need to update dW!
      #       L[i] = -score[y[i]] + log(sum_exp_scores)
      #       dL_j/dW_j = 0 + 1/sum_exp_scores * d(sum_exp_scores)/dW_j when j != i else -X[i, :] + /sum_exp_scores * d(sum_exp_scores)/dW_j
      #       d(sum_exp_scores)/dW_j = d(sum(exp(np.dot(X[i, :], W))))/dW_j = exp(score_j) * dScore_j/dW_j (score_j = np.dot(X[i, :], W[:, j]))
      #       dScore_j/dW_j = X[i, j]

      # HERE: I vecterized updates for each feature! The weight on each feature got updates for N times!
      dW[:, j] += 1/sum_exp_scores * np.exp(scores[j]) * X[i, :]
      if j == y[i]:
        dW[:, j] += - X[i, :].T

  loss += 0.5 * reg * np.sum(W * W)

  loss /= N
  dW /= N
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  N, D = X.shape
  D, C = W.shape
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X, W)

  # avoid numerical unstability due to large number exponential
  scores = scores - np.max(scores, axis = 1).reshape(N, 1)

  sum_exp_scores = np.sum(np.exp(scores), axis = 1).reshape(N, 1)

  possibility_scores = np.exp(scores) / sum_exp_scores # broadcast

  loss = np.sum(- np.log(possibility_scores[np.arange(N), y]))
  loss += 0.5 * reg * np.sum(W * W)

  # NOTE: Rather that for each class, update weight on each features for N times
  # HERE: np.dot(X.T, scores) -> np.dot(the input of featureX for all instances, the scores of classY for all instances)
  # NEAT: possibility_scores = np.exp(scores) * (1 / sum_exp_scores) -> according to gradient equation
  dW = np.dot(X.T, possibility_scores)

  # for i in range(N):
  #   dW[:, y[i]] -= X[i, :].T
  y_mask = np.zeros((N, C))
  y_mask[np.arange(N), y] = 1
  dW -= np.dot(X.T, y_mask)


  loss /= float(N)
  dW /= float(N)

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW





















