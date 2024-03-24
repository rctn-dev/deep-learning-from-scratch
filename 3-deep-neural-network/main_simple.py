import numpy as np
import matplotlib.pyplot as plt
import h5py


def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A

def relu(Z):
    A = np.maximum(0,Z)
    return A


def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z<=0] = 0
    assert (dZ.shape == Z.shape)
    return dZ

def sigmoid_backward(dA, Z):
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)
    return dZ

def predict(X, params):
    A = X
    L = len(params) // 2   
    for l in range(1, L):
        A_prev = A 
        W=params['W' + str(l)]
        b=params['b' + str(l)]
        Z = W.dot(A_prev) + b
        A= relu(Z)
    A_prev = A 
    W=params['W' + str(L)]
    b=params['b' + str(L)]
    Z = W.dot(A_prev) + b
    AL= sigmoid(Z)
    return (AL>0.5)


if __name__=='__main__':

    # load_dataset
    f_train=h5py.File('./catvnoncat/train_catvnoncat.h5', 'r')
    f_test=h5py.File('./catvnoncat/test_catvnoncat.h5', 'r')

    # Add [:] to the end, in order to convert from h5 into nd.array
    train_set_x=f_train['train_set_x'][:]
    train_set_y=f_train['train_set_y'][:]
    test_set_x=f_test['test_set_x'][:]
    test_set_y=f_test['test_set_y'][:]
    classes=f_train['list_classes'][:]

    # Reshape and normalize the dataset
    X_train=train_set_x.reshape(train_set_x.shape[0],-1).T/255
    X_test=test_set_x.reshape(test_set_x.shape[0],-1).T/255
    Y_train=train_set_y.reshape(1,train_set_y.shape[0])
    Y_test=test_set_y.reshape(1,test_set_y.shape[0])
    m_train=X_train.shape[1]
    m_test=X_test.shape[1]

    # initialize parameters
    np.random.seed(1)
    params = {}
    layers_dims = [12288, 20, 7, 5, 1]
    learning_rate=0.01
    num_iterations=2500
    L = len(layers_dims) 
    costs=[]
    for l in range(1, L):
        params['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) / np.sqrt(layers_dims[l-1]) #*0.01
        params['b' + str(l)] = np.zeros((layers_dims[l], 1))
        
    # iterations, epochs
    for iter in range(num_iterations):

        # forward propagation
        caches = []
        A = X_train
        L = len(params) // 2   
        for l in range(1, L):
            A_prev = A 
            W=params['W' + str(l)]
            b=params['b' + str(l)]
            Z = W.dot(A_prev) + b
            A= relu(Z)
            cache = (A_prev,Z,W,b)
            caches.append(cache)
        A_prev = A 
        W=params['W' + str(L)]
        b=params['b' + str(L)]
        Z = W.dot(A_prev) + b
        AL= sigmoid(Z)
        cache = (A_prev,Z,W,b)
        caches.append(cache)

        #compute cost
        m = Y_train.shape[1]
        cost = (1./m) * (-np.dot(Y_train,np.log(AL).T) - np.dot(1-Y_train, np.log(1-AL).T))
        cost = np.squeeze(cost)  
        costs.append(cost)
        if iter%100==0:
            print("cost at iter ",str(iter)," : ",str(cost))

        # backward_propagation
        grads = {}
        L = len(caches) 
        m = AL.shape[1]
        Y = Y_train.reshape(AL.shape)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        
        current_cache = caches[L-1]
        A_prev,Z,W,b=current_cache
        dZ = sigmoid_backward(dAL,Z)
        m = A_prev.shape[1]
        dW = 1./m * np.dot(dZ,A_prev.T)
        db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(W.T,dZ)
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] =  dA_prev, dW, db 
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            A_prev,Z,W,b=current_cache
            dZ = relu_backward(dA_prev,Z)
            m = A_prev.shape[1]
            dW = 1./m * np.dot(dZ,A_prev.T)
            db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
            dA_prev = np.dot(W.T,dZ)
            grads["dA" + str(l+1)], grads["dW" + str(l+1)], grads["db" + str(l+1)] =  dA_prev, dW, db 

        # update parameters
        L = len(params) // 2 
        for l in range(1,L+1):
            params["W" + str(l)] = params["W" + str(l)] - learning_rate * grads["dW" + str(l)]
            params["b" + str(l)] = params["b" + str(l)] - learning_rate * grads["db" + str(l)]
  
    Y_train_predict =predict(X_train,params)
    Y_test_predict =predict(X_test,params)
    train_accuracy=100 - np.mean(np.abs(Y_train_predict - Y_train)) * 100
    test_accuracy=100 - np.mean(np.abs(Y_test_predict - Y_test)) * 100

    # Print train/test Errors
    print("train accuracy: {} %".format(train_accuracy))
    print("test accuracy: {} %".format(test_accuracy))
            