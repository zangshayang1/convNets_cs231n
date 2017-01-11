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
  loss += 0.5 * reg * np.sum(W * W)

  dW /= num_train ## NOTE: don't forget to average gradescent
  dW += reg * W
  # Add regularization to the loss.


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
  loss = 0.0
  dW = np.zeros_like(W) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  N, _ = X.shape
  _, C = W.shape

  scores = np.dot(X, W) # (N, C)

  correct_scores = scores[np.arange(N), y]
  # NOTE: mask scores matrice with [np.arange(num_train) in axis=0, y in axis = 1]
  #       its shape got reduced to (num_train, ) -> 1D array while (num_train, 1) -> 2D array
  correct_scores = np.dot(correct_scores.reshape(N, 1), np.ones((1, C)))

  delta = np.ones((N, C))

  L = np.fmax(0, scores - correct_scores + delta)
  # everything between +/- is in (N, C)
  # np.fmax gives element-wise comparison

  L[np.arange(N), y] = 0

  loss = np.sum(L) / N + 0.5 * reg * np.sum(W * W)

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

  L[L > 0] = 1 # these contribute to the loss

  # for each instance i in range(N):
  #     for each class in range(C):
  #         if L[i, j] == 1 contributing to the loss:
  #             L[i, y_i] -= 1, the right class score got boosted.

  L[np.arange(N), y] = - np.sum(L, axis = 1)


  dW = np.dot(X.T, L) # NOTE: quite neat!

  dW = dW / N + reg * W


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
