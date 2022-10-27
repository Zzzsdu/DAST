# Statistical features process 

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
Feasize = window_size

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
