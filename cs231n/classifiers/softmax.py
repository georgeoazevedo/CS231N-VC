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
    
    num_train = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_train):
        #compute the loss
        scores = X[i].dot(W)
        
        #subtrai de scores o maior valor de socres (todos os valores deles são negativos)
        scores -=np.max(scores)
        #a função softmax é a razão entre o valor da exponencial do score dividido pelo somatório das exponenciais de cada score
        exp_scores = np.sum(np.exp(scores))
        softmax = np.exp(scores[y[i]])/exp_scores
        loss -= np.log(softmax)
        
        #calculando o gradiente de coluna a coluna
        scores_p = np.exp(scores)/exp_scores
        for j in range(num_classes):
            if j==y[i]:
                dscore = scores_p[j] - 1
            else:
                dscore = scores_p[j]
            dW[:,j] += dscore*X[i]
            
    loss /= num_train
    #regularization
    loss += reg*np.sum(W**2)
    dW /= num_train
    dW += reg*W
    
    
    pass

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
    
    
    scores = np.dot(X,W)
    
    
    scores -= np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores)/np.sum(np.exp(scores), axis=1, keepdims=True)

    #Fator de suavização para evitar divisão por zero no log
    fator_suav = 1e-14
    N,K=np.shape(X)
    
    loss = -np.sum(np.log(probs[np.arange(N), y]+fator_suav))/N
    
    dscores=probs.copy()
    dscores[np.arange(N), y] -=1
    dscores /= N
    dW = np.dot(X.T, dscores)
    

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
