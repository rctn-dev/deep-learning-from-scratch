#  logistic regression from scratch
import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py

def sigmoid(z):

    s=1/(1+np.exp(-z))
    return s

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

    # intialize W,b with zeros
    m_train=X_train.shape[1]
    m_test=X_test.shape[1]
    n=X_train.shape[0]
    W=np.zeros((n,1))
    b=0.0
    dW=np.zeros((n,1))
    db=0.0

    # Some hyper-parameters and cost
    max_iterations=2000
    learning_rate=0.005
    costs=[]
    
    # train and model fit: multiple forward-backward propagations
    for iter in range(0,max_iterations):

        #forward propagation
        Z=np.dot(W.T,X_train)+b
        A=sigmoid(Z)
        dZ=A-Y_train 
        costs.append(-np.sum(Y_train*np.log(A)+(1-Y_train)*np.log(1-A))/m_train)

        #backward propagation
        dW=np.dot(X_train,dZ.T)/m_train
        db=np.sum(dZ)/m_train

        #update
        W-=learning_rate*dW
        b-=learning_rate*db
    
    #cost
    plt.plot(costs[0:max_iterations:100])
    plt.show()

    # predict
    A_train=sigmoid(np.dot(W.T,X_train)+b)
    A_test=sigmoid(np.dot(W.T,X_test)+b)
    Y_train_predict=np.array([ A_train[0,i]>0.5 for i in range(A_train.shape[1])])
    Y_test_predict=np.array([ A_test[0,i]>0.5 for i in range(A_test.shape[1])])

    # Accuracy, L1
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_train_predict - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_test_predict - Y_test)) * 100))
