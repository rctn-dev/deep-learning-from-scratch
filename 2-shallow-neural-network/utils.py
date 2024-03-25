import numpy as np
import h5py


def sigmoid(z):

    s=1/(1+np.exp(-z))
    return s

def predict(X,params):
    AL,cache=forward_prop(X,params)
    return AL>=0.5

def load_data():
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

    return X_train,Y_train,X_test,Y_test

def initialize_parameters(n_x,n_h,n_y):
    params={}
    np.random.seed(1)
    params["W1"]=np.random.randn(n_h,n_x) * 0.001
    params["b1"]=np.zeros((n_h,1))
    params["W2"]=np.random.randn(n_y,n_h) * 0.001
    params["b2"]=np.zeros((n_y,1))

    return params

def compute_cost (A,Y):
    m=Y.shape[1]
    loss=Y*np.log(A)+(1-Y)*np.log(1-A)
    cost=-np.sum(loss)/m
    return cost

def forward_prop(X_train,params):
    W1=params["W1"]
    b1=params["b1"]
    W2=params["W2"]
    b2=params["b2"]
    #forward propagation
    Z1=np.dot(W1,X_train)+b1
    A1=np.tanh(Z1)
    Z2=np.dot(W2,A1)+b2
    AL=sigmoid(Z2)
    return AL,A1
def back_prop(AL,cache,X_train,Y_train,params):
    grads={}
    A1=cache
    W2=params["W2"]
    m=Y_train.shape[1]
    dZ2=AL-Y_train 
    dZ1=np.dot(W2.T,dZ2)*(1-np.power(A1,2))
    grads["dW2"]=np.dot(dZ2,A1.T)/m
    grads["db2"]=np.sum(dZ2,axis=1,keepdims=True)/m
    grads["dW1"]=np.dot(dZ1,X_train.T)/m
    grads["db1"]=np.sum(dZ1,axis=1,keepdims=True)/m
    return grads