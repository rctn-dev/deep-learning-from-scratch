import numpy as np
import h5py


def sigmoid(z):
    """
    Compute the sigmoid of z
    Arguments:
    z: A scalar or numpy array of any size.
    Return:
    s: sigmoid(z)
    """
    s=1/(1+np.exp(-z))

    return s

# Loading the data (cat/non-cat)
def load_data(file_train,file_test):
    """
    Convert .h5 data into numpy.ndarray
    Arguments:
    file_name: file name and extension as string
    Return:
    train_set_x: numpy.ndarray of shape (m_train,64,64,3) 
    train_set_y: numpy.ndarray of shape (m_train,) 
    test_set_x: numpy.ndarray of shape (m_test,64,64,3) 
    test_set_y: numpy.ndarray of shape (m_test,) 
    classes: numpy.ndarray of shape (2,), cat or non-cat in this dataset 
    """
    f_train = h5py.File(file_train, 'r')
    f_test = h5py.File(file_test, 'r')
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

    return X_train,Y_train,X_test,Y_test, classes

def initialize_parameters(n):
    params={}
    W=np.zeros((n,1))
    b=0.0
    params["W"]=W
    params["b"]=b
    return params

def forward_prop(X,params):
    W=params["W"]
    b=params["b"]
    Z=np.dot(W.T,X)+b
    A=sigmoid(Z)
    return A

def predict(X,params):
    A=forward_prop(X,params)
    return A>=0.5

def compute_cost(A,Y):
    m_train=Y.shape[1]
    cost=-np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))/m_train
    return cost

def  backward_prop(A,X,Y):
    grads={}
    dZ=A-Y 
    m=Y.shape[1]
    dW=np.dot(X,dZ.T)/m
    db=np.sum(dZ)/m
    grads["dW"]=dW
    grads["db"]=db
    return grads

def update_parameters (params,grads,learning_rate):
    params["W"]-=learning_rate*grads["dW"]
    params["b"]-=learning_rate*grads["db"]
    return params