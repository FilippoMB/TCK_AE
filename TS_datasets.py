import numpy as np
from scipy.integrate import odeint
import scipy.io
from sklearn import preprocessing
import sys
from utils import ideal_kernel
import pandas as pd

"""
Data manager for different time series datasets.
"""


def getBlood(kernel='TCK', inp='zero'):
    blood_data = scipy.io.loadmat('Data/TCK_data.mat')
    
    # ------ train -------
    train_data = blood_data['X']
    train_data = np.transpose(train_data,axes=[1,0,2]) # time_major=True
    train_len = [train_data.shape[0] for _ in range(train_data.shape[1])]
    
    if inp=='zero': # substitute NaN with 0
        train_data[np.isnan(train_data)] = 0 
    
    elif inp == 'last': # replace NaN with the last seen value
       train_data0 = train_data[0,:,:]
       train_data0[np.isnan(train_data0)] = 0
       train_data[0,:,:] = train_data0
       for i in range(train_data.shape[1]):
           train_data_i = pd.DataFrame(train_data[:,i,:])
           train_data_i.fillna(method='ffill',inplace=True)  
           train_data[:,i,:] = train_data_i.values
           
    elif inp=='mean':
        imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
        for i in range(train_data.shape[2]):
            print(train_data[:,:,i])
            train_data[:,:,i] = imp.fit_transform(train_data[:,:,i])
            print(train_data[:,:,i])
           
    train_labels = np.asarray(blood_data['Y'])
    
    # ----- test -------
    test_data = blood_data['Xte'] 
    test_data = np.transpose(test_data,axes=[1,0,2]) # time_major=True
    test_len = [test_data.shape[0] for _ in range(test_data.shape[1])]
    
    if inp == 'zero': # substitute NaN with 0
        test_data[np.isnan(test_data)] = 0 
    
    elif inp == 'last': # replace NaN with the last seen value
       test_data0 = test_data[0,:,:]
       test_data0[np.isnan(test_data0)] = 0
       test_data[0,:,:] = test_data0
       for i in range(test_data.shape[1]):
           test_data_i = pd.DataFrame(test_data[:,i,:])
           test_data_i.fillna(method='ffill',inplace=True)
           test_data[:,i,:] = test_data_i.values
    
    elif inp=='mean':
        imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
        for i in range(test_data.shape[2]):
            test_data[:,:,i] = imp.fit_transform(test_data[:,:,i])
        
    test_labels = np.asarray(blood_data['Yte'])
                
    # valid == train   
    valid_data = train_data
    valid_labels = train_labels
    valid_len = train_len
    
    # target outputs
    train_targets = train_data
    valid_targets = valid_data
    test_targets = test_data    
    
    if kernel=='TCK':
        K_tr = blood_data['Ktrtr']
        K_vs = K_tr
        K_ts = blood_data['Ktete']
    else:
        K_tr = ideal_kernel(train_labels)
        K_vs = ideal_kernel(valid_labels)
        K_ts = ideal_kernel(test_labels)
    
    return (train_data, train_labels, train_len, train_targets, K_tr,
        valid_data, valid_labels, valid_len, valid_targets, K_vs,
        test_data, test_labels, test_len, test_targets, K_ts)