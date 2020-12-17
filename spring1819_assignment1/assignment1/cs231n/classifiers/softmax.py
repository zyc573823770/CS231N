from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    class_num = W.shape[1]
    sample_num = X.shape[0]

    for i in range(sample_num):
      score_all = np.exp(X[i].dot(W))
      correct_score = score_all[y[i]]
      loss += -np.log(correct_score / np.sum(score_all))
      for j in range(class_num):
        dW[:,j] += X[i].T*np.exp(X[i].dot(W[:,j]))/np.sum(score_all)
        if j==y[i]:
          dW[:,j] += -X[i].T
    dW /= sample_num
    dW += 2*reg*W
    
    loss /= sample_num
    loss += reg * np.sum(W*W)



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    class_num = W.shape[1]
    sample_num = X.shape[0]

    score = X.dot(W)
    correct_score = np.exp(score[range(sample_num),y].T)
    denominator = np.sum(np.exp(score), axis=1)
    loss = -np.log(correct_score / denominator)
    loss = np.sum(loss)
    loss /= sample_num
    loss += reg * np.sum(W*W)

    row_sum = np.sum(np.exp(score), axis=1)
    temp = np.ones_like(score)
    temp = temp * row_sum[:,np.newaxis]
    temp = np.exp(score) / temp
    dW += X.T.dot(temp)
    xx,_ = np.meshgrid(range(class_num), range(sample_num))
    temp = np.ones_like(score)
    temp = y[:,np.newaxis] * temp
    dW += -X.T.dot((temp==xx))
    dW /= sample_num
    dW += 2*reg*W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
