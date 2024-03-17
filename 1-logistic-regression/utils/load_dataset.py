import h5py
import numpy as np
# Loading the data (cat/non-cat)
def load_dataset(file_train,file_test):
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
    # print(list(f.keys()))
    # print(f['list_classes'])
    # print(type(f['train_set_x']))
    # print(type(f['train_set_x'][:]))
    # print(f['train_set_x'].shape)
    # print(f['train_set_y'].shape)
 
    train_set_x=f_train['train_set_x'][:]
    train_set_y=f_train['train_set_y'][:]
    test_set_x=f_test['test_set_x'][:]
    test_set_y=f_test['test_set_y'][:]
    classes=f_train['list_classes'][:]

    return train_set_x, train_set_y, test_set_x, test_set_y,classes


