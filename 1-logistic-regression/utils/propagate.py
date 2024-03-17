# GRADED FUNCTION: propagate
import numpy as np
import sigmoid
def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    grads -- dictionary containing the gradients of the weights and bias
            (dw -- gradient of the loss with respect to w, thus same shape as w)
            (db -- gradient of the loss with respect to b, thus same shape as b)
    cost -- negative log-likelihood cost for logistic regression
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    # number of samples equivalent to number of columns of X matrix (n,m).
    m = X.shape[1]
    # FORWARD PROPAGATION (FROM X TO COST)
    # compute activation
    A=sigmoid(np.dot(w.T,X)+b)
    # compute cost by using np.dot to perform multiplication
    cost=-(Y*np.log(A)+(1-Y)*np.log(1-A))
    cost=np.sum(cost)/m

    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw=np.dot(X,(A-Y).T)/m
    db=np.sum(A-Y)/m

    cost = np.squeeze(np.array(cost))

    grads = {"dw": dw,
             "db": db}
    
    return grads, cost