import numpy as np
import scipy.io
from utils import ideal_kernel
import pandas as pd

"""
Data manager for loading blood data and (precomputed) TCK kernel
"""


def getBlood(kernel='TCK', inp='zero'):
    blood_data = scipy.io.loadmat('Data/TCK_data.mat')
    
    # ------ train -------
    train_data = blood_data['X']
    train_data = np.transpose(train_data,axes=[1,0,2]) # time_major=True
    train_len = [train_data.shape[0] for _ in range(train_data.shape[1])]
   
    train_data0 = train_data[0,:,:]
    train_data0[np.isnan(train_data0)] = 0
    train_data[0,:,:] = train_data0
    for i in range(train_data.shape[1]):
        train_data_i = pd.DataFrame(train_data[:,i,:])
        if inp == 'last': 
            train_data_i.fillna(method='ffill',inplace=True)  
        elif inp =='zero':
            train_data_i.fillna(0,inplace=True)
        elif inp=='mean':
            train_data_i.fillna(train_data_i.mean(),inplace=True)
        train_data[:,i,:] = train_data_i.values
                    
    train_labels = np.asarray(blood_data['Y'])
    
    # ----- test -------
    test_data = blood_data['Xte'] 
    test_data = np.transpose(test_data,axes=[1,0,2]) # time_major=True
    test_len = [test_data.shape[0] for _ in range(test_data.shape[1])]
        
    test_data0 = test_data[0,:,:]
    test_data0[np.isnan(test_data0)] = 0
    test_data[0,:,:] = test_data0
    for i in range(test_data.shape[1]):
        test_data_i = pd.DataFrame(test_data[:,i,:])
        if inp == 'last':
            test_data_i.fillna(method='ffill',inplace=True)
        elif inp =='zero':
            test_data_i.fillna(0,inplace=True)
        elif inp=='mean':
            test_data_i.fillna(test_data_i.mean(),inplace=True)
        test_data[:,i,:] = test_data_i.values
            
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