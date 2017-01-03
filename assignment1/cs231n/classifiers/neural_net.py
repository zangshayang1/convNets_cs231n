import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################

    hidden_scores = np.maximum(0, np.dot(X, W1) + b1) # use ReLu activation here

    # number of deactivated ReLUs
    # print hidden_scores[hidden_scores == 0].shape
    
    scores = np.dot(hidden_scores, W2) + b2 # no activation applied on output layer
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    # Now scores should be a N x C matrice

    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################

    # scores -= np.max(scores, axis = 1).reshape(N, 1)
    # it is not necessary in this assignment since a good initialization will prevent numerical overflow.
      
    scores_exp = np.exp(scores)

    sum_scores_exp = np.sum(scores_exp, axis = 1, keepdims = True)

    probability_scores = scores_exp / sum_scores_exp

    ll_loss = np.sum(-np.log(probability_scores[np.arange(N), y]))

    avg_data_loss = ll_loss / float(N)

    reg_loss = 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

    loss = avg_data_loss + reg_loss
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################

    ### notes from softmax.py in terms of how calc grad with input X and scores

    # # NOTE: Rather that for each class, update weight on each features for N times
    # # HERE: np.dot(X.T, scores) -> np.dot(the input of featureX for all instances, the scores of classY for all instances)
    # # NEAT: possibility_scores = np.exp(scores) * (1 / sum_exp_scores) -> according to gradient equation
    # dW = np.dot(X.T, possibility_scores)

    # # for i in range(N):
    # #   dW[:, y[i]] -= X[i, :].T
    # y_mask = np.zeros((N, C))
    # y_mask[np.arange(N), y] = 1
    # dW -= np.dot(X.T, y_mask)

    ### first backprop dW2
    # hidden_scores = np.maximum(0, np.dot(X, W1) + b1) # use ReLu activation here
    # scores = np.dot(hidden_scores, W2) + b2 # no activation applied on output layer

    # the gradient backflow starts from the probability because: 
    # the last gate is softmax gate where Loss = - log (e^(score_y[i]) / sum(e^score_j))
    # when i != j, its gradient == probability_j * X
    # when i == j, its graident == probability_j * X - 1
    back_prob = probability_scores
    back_prob[np.arange(N), y] -= 1 # take care of grad when i == j -y_i in loss equation, namely replacing "dW -= np.dot(X.T, y_mask)" section
    # back_prob /= float(N)


    grads["W2"] = np.dot(hidden_scores.T, back_prob) / float(N)
    grads["b2"] = np.sum(back_prob, axis = 0) / float(N)

    ### for next backprop dHidden some equations involved.
    # H1 = np.dot(X, W1); P = np.dot(H1, W2) -> P.T = np.dot(W2.T, H1.T)
    # dW1 = np.dot(X.T, H1);                    dH1.T = np.dot(W2, P.T) -> dH1 = np.dot(P, W2.T)
    dHidden = np.dot(back_prob, W2.T)
    dHidden[hidden_scores <= 0] = 0 # backprop on ReLu

    # linear, no exp involved
    grads["W1"] = np.dot(X.T, dHidden) / float(N)
    grads["b1"] = np.sum(dHidden, axis = 0) / float(N)

    # reg_loss = 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
    grads["W2"] += reg * W2
    grads["W1"] += reg * W1

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]

    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################

      # random_indices = np.floor(np.random.rand(batch_size) * num_train).astype(int)
      # np.random.rand -> uniformly distributed random numbers [0, 1)
      # np.random.randn -> standard normally distributed random numbers centered at 0;
      
      random_indices = np.random.choice(num_train, batch_size, replace = False)
      # why it makes the loss curve very smooth?

      X_batch = X[random_indices]
      y_batch = y[random_indices]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      for param in self.params:
        self.params[param] -= learning_rate * grads[param]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    results_prob = np.dot(X, self.params["W1"]) + self.params["b1"]
    results_prob[results_prob <= 0] = 0
    results_prob = np.dot(results_prob, self.params["W2"]) + self.params["b2"]
    y_pred = np.argmax(results_prob, axis = 1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


