#  2-Layer NN from scratch
import numpy as np
import matplotlib.pyplot as plt
import h5py
from utils import *

if __name__=='__main__':

    # load_dataset
    X_train,Y_train,X_test,Y_test=load_data()

    # intialize W,b with zeros
    m_train=X_train.shape[1]
    m_test=X_test.shape[1]
    n_x=X_train.shape[0]
    n_h=4
    n_y=Y_train.shape[0]
 
    params=initialize_parameters(n_x,n_h,n_y) 


    # Some hyper-parameters and cost
    max_iterations=2500
    learning_rate=0.005
    costs=[]

    # train and model fit: multiple forward-backward propagations
    for iter in range(max_iterations):
        AL,cache=forward_prop(X_train,params)
        cost=compute_cost(AL,Y_train)
        costs.append(cost)
        if iter%100==0:
            print(f"cost after iteration {iter} is: {costs[iter]}")

        #backward propagation
        grads=back_prop(AL,cache,X_train,Y_train,params)

        #update
        params["W2"]-=learning_rate*grads["dW2"]
        params["b2"]-=learning_rate*grads["db2"]
        params["W1"]-=learning_rate*grads["dW1"]
        params["b1"]-=learning_rate*grads["db1"]
    
    # predict
    Y_train_predict=predict(X_train,params)
    Y_test_predict=predict(X_test,params)

    # Accuracy, L1
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_train_predict - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_test_predict - Y_test)) * 100))
