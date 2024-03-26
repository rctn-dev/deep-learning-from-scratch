import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A

def tanh(Z):
    A = np.tanh(Z)
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

def tanh_backward(dA, Z):
    s = tanh(Z)
    dZ = dA * (1-np.power(s,2))
    assert (dZ.shape == Z.shape)
    return dZ

def predict(X, params,activation):
    AL, caches=forward_prop_L(X,params,activation)
    return (AL>0.5)

def load_moon_dataset():
    X, y = make_moons(n_samples=1000, noise = 0.1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=25)
    colors = ['maroon', 'forestgreen']
    vectorizer = np.vectorize(lambda x: colors[x % len(colors)])
    fig, ax= plt.subplots(1,2, sharey=True,figsize=(10,4))
    fig.suptitle('Moon Dataset')
    ax[0].scatter(X_train[:,0], X_train[:,1], c=vectorizer(Y_train))
    ax[0].set_title('train data')
    ax[1].scatter(X_test[:,0], X_test[:,1], c=vectorizer(Y_test))
    ax[1].set_title('test data')
    plt.show()

    X_train=np.transpose(X_train)
    X_test=np.transpose(X_test)
    Y_train=Y_train.reshape(1,Y_train.shape[0])
    Y_test=Y_test.reshape(1,Y_test.shape[0])
   
   
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)
 

    return X_train, Y_train, X_test, Y_test

def initialize_parameters(layers_dims):
    np.random.seed(1)
    params = {}
    L = len(layers_dims) 
    for l in range(1, L):
        # params['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) / np.sqrt(layers_dims[l-1]) #*0.01
        params['W' + str(l)] = np.random.rand(layers_dims[l], layers_dims[l-1])
        # params['W' + str(l)] = np.random.rand(layers_dims[l], layers_dims[l-1])*0.01
        params['b' + str(l)] = np.zeros((layers_dims[l], 1))
    return params

def forward_prop_one(A_prev,W,b,activation):
    Z = W.dot(A_prev) + b
    if activation=="relu":
        A= relu(Z)
    if activation=="sigmoid":
        A= sigmoid(Z)
    if activation=="tanh":
        A= tanh(Z)
    return A,Z

def forward_prop_L(X_train,params,activation):
    caches = []
    A = X_train
    L = len(params) // 2   
    for l in range(1, L):
        A_prev = A 
        W=params['W' + str(l)]
        b=params['b' + str(l)]
        A,Z=forward_prop_one(A_prev,W,b,activation)
        cache = (A_prev,Z,W,b)
        caches.append(cache)
    A_prev = A 
    W=params['W' + str(L)]
    b=params['b' + str(L)]
    AL,Z=forward_prop_one(A_prev,W,b,activation="sigmoid")
    cache = (A_prev,Z,W,b)
    caches.append(cache)
    return AL, caches

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
    if activation=="tanh":
        dZ = tanh_backward(dA,Z)
    m = A_prev.shape[1]
    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    return dA_prev, dW, db 

def back_prop_L(A,Y,caches,activation):
    grads = {}
    L = len(caches) 
    m = A.shape[1]
    Y = Y.reshape(A.shape)
    dAL = - (np.divide(Y, A) - np.divide(1 - Y, 1 - A)) 
    current_cache = caches[L-1]
    dA_prev, dW, db=back_prop_one(dAL,current_cache,"sigmoid")
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = dA_prev, dW, db
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev, dW, db=back_prop_one(dA_prev,current_cache,activation)
        grads["dA" + str(l+1)], grads["dW" + str(l+1)], grads["db" + str(l+1)] =  dA_prev, dW, db 
    return grads

def update_parameters(params,grads,learning_rate):
        L = len(params) // 2 
        for l in range(1,L+1):
            params["W" + str(l)] = params["W" + str(l)] - learning_rate * grads["dW" + str(l)]
            params["b" + str(l)] = params["b" + str(l)] - learning_rate * grads["db" + str(l)]
        return params

def plot_decision_boundary(X, y,params,activation):

    # find the minimum and maximum values for the first
    # and second feature of the dataset

    x1_min, x1_max = X[0, :].min() - 1, X[0,:].max() + 1
    x2_min, x2_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.02

    # generate a grid of data points between maximum and minimum feature values
    x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))

    # make prediction on all points in the grid by flattening with ravel() and concatenating with vstack()
    X_mesh=np.vstack((x1.ravel(), x2.ravel()))
    AL, caches = forward_prop_L(X_mesh,params,activation)
    A_prev,Z,W,b=caches[-1]
    AL = AL.reshape(x1.shape)

    # convert sigmoid outputs to binary
    AL = np.where(AL > 0.5, 1, 0)

    # plot countourf plot to fill the grid with data points
    # the colour of the data points correspond to prediction (0 or 1)
    plt.contourf(x1, x2, AL, cmap=plt.cm.Spectral)

    # plot the original scatter plot to see where the data points fall
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y)
    plt.show()





        