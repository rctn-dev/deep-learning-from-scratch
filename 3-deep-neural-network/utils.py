import numpy as np
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
    return X_train,Y_train, X_test,Y_test
def initialize_parameters(layers_dims):
    np.random.seed(1)
    params = {}
    L = len(layers_dims) 
    for l in range(1, L):
        params['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) / np.sqrt(layers_dims[l-1]) #*0.01
        params['b' + str(l)] = np.zeros((layers_dims[l], 1))
    return params

def forward_prop_one(A_prev,W,b,activation):
    Z = W.dot(A_prev) + b
    if activation=="relu":
        A= relu(Z)
    if activation=="sigmoid":
        A= sigmoid(Z)
    return A,Z

def  compute_cost(A,Y):
    m = Y.shape[1]
    cost = (1./m) * (-np.dot(Y,np.log(A).T) - np.dot(1-Y, np.log(1-A).T))
    cost = np.squeeze(cost) 
    return cost

def back_prop_one(dA,cache,activation):
    A_prev,Z,W,b=cache
    if activation=="sigmoid":
        dZ = sigmoid_backward(dA,Z)
    if activation=="relu":
        dZ = relu_backward(dA,Z)
    m = A_prev.shape[1]
    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    return dA_prev, dW, db 

def update_parameters(params,grads,learning_rate):
        L = len(params) // 2 
        for l in range(1,L+1):
            params["W" + str(l)] = params["W" + str(l)] - learning_rate * grads["dW" + str(l)]
            params["b" + str(l)] = params["b" + str(l)] - learning_rate * grads["db" + str(l)]
        return params

    
    




        