import numpy as np
import matplotlib.pyplot as plt
import h5py
from utils import *

if __name__=='__main__':

    # load data
    X_train,Y_train,X_test,Y_test=load_moon_dataset()

    # initialize parameters
    layers_dims = [2, 5, 3, 1]
    learning_rate=0.05
    num_iterations=5000
    activation="tanh"
    costs=[]
    params=initialize_parameters(layers_dims)

    # iterations, epochs
    for iter in range(num_iterations):
        # forward propagation
        AL,caches=forward_prop_L(X_train,params,activation)

        #compute cost 
        cost=compute_cost(AL,Y_train)
        costs.append(cost)
        if iter%100==0:
            print("cost at iter ",str(iter)," : ",str(cost))

        # backward_propagation
        grads=back_prop_L(AL,Y_train,caches,activation)

        # update parameters
        params=update_parameters(params,grads,learning_rate)

    Y_train_predict =predict(X_train,params,activation)
    Y_test_predict =predict(X_test,params,activation)
    train_accuracy=100 - np.mean(np.abs(Y_train_predict - Y_train)) * 100
    test_accuracy=100 - np.mean(np.abs(Y_test_predict - Y_test)) * 100

    # Print train/test Errors
    print("train accuracy: {} %".format(train_accuracy))
    print("test accuracy: {} %".format(test_accuracy))

    plt.figure(figsize=(8, 6))
    plot_decision_boundary(X_train,Y_train, params,activation)