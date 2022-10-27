# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 15:31:47 2022

@author: HD
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from scipy import interpolate
import scipy.io as sio
from numpy import *

min_max_scaler = preprocessing.MinMaxScaler()

#Import dataset
RUL_F001 = np.loadtxt('./cmapss/RUL_FD001.txt')
train_F001 = np.loadtxt('/cmapss/train_FD001.txt')
test_F001 = np.loadtxt('/cmapss/test_FD001.txt')
train_F001[:, 2:] = min_max_scaler.fit_transform(train_F001[:, 2:])
test_F001[:, 2:] = min_max_scaler.transform(test_F001[:, 2:])
train_01_nor = train_F001
test_01_nor = test_F001


#Delete worthless sensors
train_01_nor = np.delete(train_01_nor, [5, 9, 10, 14, 20, 22, 23], axis=1) 
test_01_nor = np.delete(test_01_nor, [5, 9, 10, 14, 20, 22, 23], axis=1)  
train_01_nor = np.delete(train_01_nor, [2,3,4,5, 9, 10, 14, 20, 22, 23], axis=1) 
test_01_nor = np.delete(test_01_nor, [2,3,4,5, 9, 10, 14, 20, 22, 23], axis=1) 

#parameters of data process
RUL_max = 125.0  
window_Size = 40 

trainX = []
trainY = []
trainY_bu = []
testX = []
testY = []
testY_bu = []
testInd = []
testLen = []
testX_all = []
testY_all = []
test_len = []

#Training set sliding time window processing
for i in range(1, int(np.max(train_01_nor[:, 0])) + 1):  
    ind = np.where(train_01_nor[:, 0] == i)  
    ind = ind[0]
    data_temp = train_01_nor[ind, :] 
    for j in range(len(data_temp) - window_Size + 1): 
        trainX.append(data_temp[j:j + window_Size, 2:].tolist()) 
        train_RUL = len(data_temp) - window_Size - j  
        train_bu = RUL_max - train_RUL
        if train_RUL > RUL_max:
            train_RUL = RUL_max
            train_bu = 0.0
        trainY.append(train_RUL)
        trainY_bu.append(train_bu)
        
        
#Test set sliding time window processing
for i in range(1, int(np.max(test_01_nor[:, 0])) + 1): 
    ind = np.where(test_01_nor[:, 0] == i)
    ind = ind[0]
    testLen.append(float(len(ind)))
    data_temp = test_01_nor[ind, :] 
    testY_bu.append(data_temp[-1, 1])
    if len(data_temp) < window_Size:  
        data_temp_a = []
        for myi in range(data_temp.shape[1]):
            x1 = np.linspace(0, window_Size - 1, len(data_temp))
            x_new = np.linspace(0, window_Size - 1, window_Size)
            tck = interpolate.splrep(x1, data_temp[:, myi])
            a = interpolate.splev(x_new, tck)
            data_temp_a.append(a.tolist())
        data_temp_a = np.array(data_temp_a)
        data_temp = data_temp_a.T
        data_temp = data_temp[:, 2:]
    else:
        data_temp = data_temp[-window_Size:, 2:]  

    data_temp = np.reshape(data_temp, (1, data_temp.shape[0], data_temp.shape[1])) 
    
    if i == 1:
        testX = data_temp
    else:
        testX = np.concatenate((testX, data_temp), axis=0)
    if RUL_F001[i - 1] > RUL_max:
        testY.append(RUL_max)
        #testY_bu.append(0.0)
    else:
        testY.append(RUL_F001[i - 1])    
        
        
#All data processing of test set
for i in range(1, int(np.max(test_01_nor[:, 0])) + 1):
    ind = np.where(test_01_nor[:, 0] == i)
    ind = ind[0]
    data_temp = test_01_nor[ind, :]
    data_RUL = RUL_F001[i - 1] 
    test_len.append(len(data_temp) - window_Size + 1) 
    for j in range(len(data_temp) - window_Size + 1):
        testX_all.append(data_temp[j:j + window_Size, 2:].tolist())
        test_RUL = len(data_temp) + data_RUL - window_Size - j 
        if test_RUL > RUL_max:
            test_RUL = RUL_max
        testY_all.append(test_RUL)
        
                
trainX = np.array(trainX)
testX = np.array(testX)
trainY = np.array(trainY)/RUL_max 
trainY_bu = np.array(trainY_bu)/RUL_max
testY = np.array(testY)/RUL_max
testY_bu = np.array(testY_bu)/RUL_max


testX_all = np.array(testX_all)
testY_all = np.array(testY_all)


sio.savemat('F001_window_size_trainX.mat', {"train1X": trainX})
sio.savemat('F001_window_size_trainY.mat', {"train1Y": trainY})
sio.savemat('F001_window_size_testX.mat', {"test1X": testX})
sio.savemat('F001_window_size_testY.mat', {"test1Y": testY})

     
        
# Statistical features process 
        
regr = linear_model.LinearRegression()  # feature of linear coefficient

def fea_extract1(data):  # feature 1
    fea = []
    x = np.array(range(data.shape[0]))
    for i in range(data.shape[1]):
        regr.fit(x.reshape(-1, 1), np.ravel(data[:, i]))
        fea = fea + list(regr.coef_)
    return fea

def fea_extract2(data):  # feature 2
    fea = []
    for i in range(data.shape[1]):
        fea.append(np.mean(data[:, i]))
    return fea


trainX = sio.loadmat('.../slide_window_processed/')  #upload sliding time window processed data
testX =  sio.loadmat('.../slide_window_processed/')  #upload sliding time window processed data
trainX_fea1 = []
testX_fea1 = []
trainX_fea2 = []
testX_fea2 = []
window_size = 40 
Feasize = 14  # the number of choosed sensors

trainX = np.reshape(trainX, [trainX.shape[0], window_size, Feasize, 1])
testX = np.reshape(testX, [testX.shape[0], window_size, Feasize, 1])

for i in range(len(trainX)): 
    data_temp = trainX[i]
    trainX_fea1.append(fea_extract1(data_temp))
    trainX_fea2.append(fea_extract2(data_temp))

for i in range(len(testX)):
    data_temp = testX[i]
    testX_fea1.append(fea_extract1(data_temp))
    testX_fea2.append(fea_extract2(data_temp))

scale1 = preprocessing.MinMaxScaler().fit(trainX_fea1)#归一化
trainX_fea1 = scale1.transform(trainX_fea1)
testX_fea1 = scale1.transform(testX_fea1)

scale2 = preprocessing.MinMaxScaler().fit(trainX_fea2)
trainX_fea2 = scale2.transform(trainX_fea2)
testX_fea2 = scale2.transform(testX_fea2)


trainX_new = []
testX_new = []


for i in range(len(trainX)):
    data_temp0 = trainX[i]
    data_temp1 = np.reshape(trainX_fea1[i], [1, Feasize, 1])  # regr.coef_
    data_temp2 = np.reshape(trainX_fea2[i], [1, Feasize, 1])  # mean_value
    data_temp = np.vstack((data_temp0, data_temp1, data_temp2))
    trainX_new.append(data_temp)
trainX_new = np.array(trainX_new)    

for i in range(len(testX)):
    data_temp0 = testX[i]
    data_temp1 = np.reshape(testX_fea1[i], [1, Feasize, 1])  # regr.coef_
    data_temp2 = np.reshape(testX_fea2[i], [1, Feasize, 1])  # mean_value
    data_temp = np.vstack((data_temp0, data_temp1, data_temp2))
    testX_new.append(data_temp)
testX_new = np.array(testX_new)
        
sio.savemat('F001_window_size_trainX_new.mat', {"train1X_new": trainX_new})
sio.savemat('F001_window_size_testX_new.mat', {"test1X_new": testX_new})   
        
        
        
        
        




