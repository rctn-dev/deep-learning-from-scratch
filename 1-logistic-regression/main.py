import numpy as np
import h5py
from utils import *

if __name__=='__main__':

    # load dataset
    X_train,Y_train,X_test,Y_test,classes=load_data('./catvnoncat/train_catvnoncat.h5','./catvnoncat/test_catvnoncat.h5')

    # initialize parameters
    n=X_train.shape[0]
    params=initialize_parameters(n)

    num_iterations=2000
    learning_rate=0.005
    costs=[]
    for iter in range(num_iterations):

        # forward propagation
        AL=forward_prop(X_train,params)
        
        # compute cost
        cost=compute_cost(AL,Y_train) 
        if iter%100==0:
            costs.append(cost)
            print ("cost at iter "+str(iter)+" : "+str(cost))

        # backward propagation
        grads=backward_prop (AL,X_train,Y_train)

        # update parameters
        params=update_parameters (params,grads, learning_rate)
    
    # predict and analysis
    Y_train_predict=predict(X_train,params)
    Y_test_predict=predict(X_test,params)
     # Accuracy, L1
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_train_predict - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_test_predict - Y_test)) * 100))