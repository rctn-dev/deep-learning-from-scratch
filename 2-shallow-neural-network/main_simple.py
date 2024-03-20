#  2-Layer NN from scratch
import numpy as np
import matplotlib.pyplot as plt
import h5py

def sigmoid(z):

    s=1/(1+np.exp(-z))
    return s

def predict(W1,W2,b1,b2,X):
    Z1=np.dot(W1,X)+b1
    A1=np.tanh(Z1)
    Z2=np.dot(W2,A1)+b2
    A2=sigmoid(Z2)
    return A2

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
    n_x=X_train.shape[0]
    n_h=4
    n_y=Y_train.shape[0]
    W1=np.random.randn(n_h,n_x) * 0.001
    b1=np.zeros((n_h,1))
    W2=np.random.randn(n_y,n_h) * 0.001
    b2=np.zeros((n_y,1))  


    # Some hyper-parameters and cost
    max_iterations=2000
    learning_rate=0.005
    costs=[]

    # train and model fit: multiple forward-backward propagations
    for iter in range(0,max_iterations):

        #forward propagation
        Z1=np.dot(W1,X_train)+b1
        A1=np.tanh(Z1)
        Z2=np.dot(W2,A1)+b2
        A2=sigmoid(Z2)
        loss=Y_train*np.log(A2)+(1-Y_train)*np.log(1-A2)
        costs.append(-np.sum(loss)/m_train)

        #backward propagation
        dZ2=A2-Y_train 
        dZ1=np.dot(W2.T,dZ2)*(1-np.power(A1,2))
        dW2=np.dot(dZ2,A1.T)/m_train
        db2=np.sum(dZ2,axis=1,keepdims=True)/m_train
        dW1=np.dot(dZ1,X_train.T)/m_train
        db1=np.sum(dZ1,axis=1,keepdims=True)/m_train

        #update
        W2-=learning_rate*dW2
        b2-=learning_rate*db2
        W1-=learning_rate*dW1
        b1-=learning_rate*db1
    
    #cost
    plt.plot(costs[0:max_iterations:100])
    plt.show()
    for iter in range(max_iterations):
        if iter%100==0:
            print(f"cost after iteration {iter} is: {costs[iter]}")

    # predict
    A2_train=predict(W1,W2,b1,b2,X_train)
    A2_test=predict(W1,W2,b1,b2,X_test)
    Y_train_predict=A2_train>=0.5
    Y_test_predict=A2_test>=0.5

    # Accuracy, L1
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_train_predict - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_test_predict - Y_test)) * 100))
