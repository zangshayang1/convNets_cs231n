import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W) 
    # NOTE: W is initialized differently than that in KNN and this dot product can be broadcasted 1 x N dot_with N x c = 1 x c
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      # NOTE: loss += 0 only when scores[j] < correct_class_score - 1
      if margin > 0:
        loss += margin
        # NOTE: to each datapoint, loss function L = sum_over_j(max(0, np.dot(W_j.T, X[i]) - np.dot(W_yi.T, X[i]) + 1)
        #       gradescent on W_j = 0 if margin <= 0 else X[i, :].T, they should increment over each point
        #       gradescent on W_yi = 0 if margin <= 0 else -X[i, :].T, increment over each point.
        dW[:, j] += X[i, :].T
        dW[:, y[i]] -= X[i, :].T
        
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train ## NOTE: don't forget to average gradescent

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  num_train = X.shape[0]
  scores = np.dot(X, W) 
  
  correct_scores = scores[np.arange(num_train), y] # NOTE: mask scores matrice with [np.arange(num_train) in axis=0, y in axis = 1]
                                                   #       its shape got reduced to (num_train, ) -> 1D array while (num_train, 1) -> 2D array
  delta = np.ones((scores.shape[0], 1))

  L = scores - correct_scores.reshape((num_train, 1)) + delta # +/- can be broadcasted through these three 2D arrays

  L[L <= 0] = 0.0
  L[np.arange(scores.shape[0]), y] = 0.0
  loss = np.sum(L) / float(X.shape[0]) + 0.5 * reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  L = scores - correct_scores.reshape((num_train, 1)) + delta

  L[L > 0] = 1
  L[L <= 0] = 0
  
  L[np.arange(num_train), y] = 0 # NOTE: this step is necessary otherwise the following np.sum() will count.
  L[np.arange(num_train), y] = - np.sum(L, axis = 1)

  dW = np.dot(X.T, L) # NOTE: quite neat!
  dW = dW / float(num_train)


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
