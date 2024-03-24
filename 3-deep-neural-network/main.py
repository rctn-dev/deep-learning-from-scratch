import numpy as np
import matplotlib.pyplot as plt
import h5py
from utils import *

if __name__=='__main__':

    # load data
    X_train,Y_train,X_test,Y_test=load_data()
    m_train=X_train.shape[1]
    m_test=X_test.shape[1]

    # initialize parameters

    layers_dims = [12288, 20, 7, 5, 1]
    learning_rate=0.01
    num_iterations=2500
    costs=[]
    params=initialize_parameters(layers_dims)
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
            A,Z=forward_prop_one(A_prev,W,b,activation="relu")
            cache = (A_prev,Z,W,b)
            caches.append(cache)
        A_prev = A 
        W=params['W' + str(L)]
        b=params['b' + str(L)]
        AL,Z=forward_prop_one(A_prev,W,b,activation="sigmoid")
        cache = (A_prev,Z,W,b)
        caches.append(cache)

        #compute cost 
        cost=compute_cost(AL,Y_train)
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
        dA_prev, dW, db=back_prop_one(dAL,current_cache,"sigmoid")
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = dA_prev, dW, db
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev, dW, db=back_prop_one(dA_prev,current_cache,"relu")
            grads["dA" + str(l+1)], grads["dW" + str(l+1)], grads["db" + str(l+1)] =  dA_prev, dW, db 

        # update parameters
        params=update_parameters(params,grads,learning_rate)

  
    Y_train_predict =predict(X_train,params)
    Y_test_predict =predict(X_test,params)
    train_accuracy=100 - np.mean(np.abs(Y_train_predict - Y_train)) * 100
    test_accuracy=100 - np.mean(np.abs(Y_test_predict - Y_test)) * 100

    # Print train/test Errors
    print("train accuracy: {} %".format(train_accuracy))
    print("test accuracy: {} %".format(test_accuracy))
            