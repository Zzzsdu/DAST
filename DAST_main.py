# -*- coding: utf-8 -*-
"""
@author: HD

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import torch.utils.data.dataloader as Data
import os
import time
from torch.autograd import Variable
from DAST_utils import *
from DAST_Network import *
from torch.utils.data import TensorDataset,DataLoader


#Myscore function
def myScore(Target, Pred):
    tmp1 = 0
    tmp2 = 0
    for i in range(len(Target)):
        if Target[i] > Pred[i]:
            tmp1 = tmp1 + math.exp((-Pred[i] + Target[i]) / 13.0) - 1
        else:
            tmp2 = tmp2 + math.exp((Pred[i] - Target[i]) / 10.0) - 1
    tmp = tmp1 + tmp2
    return tmp

if __name__ == '__main__':
    
    # Load preprocessed data
    X_train = sio.loadmat('../trainX_new_40.mat')  # load sliding window preprocessed and feature extracted (mean value and regression coefficient estimates feature) data

    X_train = X_train['trainX_new']
    X_train = X_train.reshape(len(X_train),42,14)
    Y_train = sio.loadmat('../trainY_40.mat')
    Y_train = Y_train['trainY']
    Y_train = Y_train.transpose()
    
    X_test = sio.loadmat('../testX_new_40.mat')
    X_test = X_test['testX_new']
    X_test = X_test.reshape(len(X_test),42,14)
    Y_test = sio.loadmat('../testY_40.mat')
    Y_test = Y_test['testY']
    Y_test = Y_test.transpose()
    
    X_train = Variable(torch.Tensor(X_train).float())
    Y_train = Variable(torch.Tensor(Y_train).float())
    X_test = Variable(torch.Tensor(X_test).float())
    Y_test = Variable(torch.Tensor(Y_test).float())
    
    
    #Hyperparameters
    batch_size = 256
    dim_val = 64
    dim_attn = 64
    dim_val_t = 64 
    dim_attn_t = 64 
    dim_val_s = 64
    dim_attn_s = 64
    n_heads = 4  
    n_decoder_layers = 1
    n_encoder_layers = 2
    max_rul = 125
    lr = 0.001 
    epochs = 150
    time_step = 62  
    dec_seq_len = 4
    output_sequence_length = 1  
    input_size = 14 
    
    
    #Dataloader 
    train_dataset = TensorDataset(X_train,Y_train)
    train_loader = Data.DataLoader(dataset=train_dataset,batch_size = batch_size,shuffle=True)
    test_dataset = TensorDataset(X_test,Y_test)
    test_loader = Data.DataLoader(dataset=test_dataset,batch_size = batch_size,shuffle=False)
    
    
    # Initialize model parameters
    model = DAST(dim_val_s,dim_attn_s,dim_val_t,dim_attn_t,dim_val, dim_attn,time_step,input_size,dec_seq_len,output_sequence_length, n_decoder_layers, n_encoder_layers, n_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    #Training  and testing 
    loss_list = []
    train_loss_list = []
    test_loss_list = []
    train_time = []
    test_time = []
    model_loss = 1000
    
    for epoch in range(epochs):
        #training
        model.train()
        start1 = time.time()
        for i,(X, Y) in enumerate(train_loader):
            out = model(X)
            loss = torch.sqrt(criterion(out*max_rul, Y*max_rul))
            optimizer.zero_grad()   
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        end1 = time.time()
        train_time.append(end1 - start1)
        loss_eopch = np.mean(np.array(loss_list)) 
        train_loss_list.append(loss_eopch)
        print('epoch = ',epoch,
              'train_loss = ',loss_eopch.item())
        
        #testing
        model.eval()
        prediction_list = []
        for j ,(batch_x,batch_y) in enumerate(test_loader):
            start2= time.time()
            prediction = model(batch_x)
            end2 = time.time()
            test_time.append(end2 - start2)
            prediction[prediction<0] = 0        
            prediction_list.append(prediction)
            
        out_batch_pre = torch.cat(prediction_list).detach().numpy()
        prediction_tensor = torch.from_numpy(out_batch_pre)                
        test_loss = torch.sqrt(criterion(prediction_tensor*125, Y_test*125))
        test_loss_list.append(test_loss)    
        Y_test_numpy = Y_test.detach().numpy()
        test_score = myScore(Y_test_numpy*125, out_batch_pre*125)
        print('test_loss = ', test_loss.item(),
              'test_score = ', test_score)

        #Model save
        if epoch > 1:
            if test_loss.item() < model_loss:    
                model_loss = test_loss.item()
                File_Path = '..' + '\\' + 'DAST' + '\\' + 'F004' + '\\' 
                if not os.path.exists(File_Path):
                    os.makedirs(File_Path)
                torch.save(model, File_Path + '/' + 'F004_DAST_prediciton_model')
        

    test_time_mean = np.mean(test_time)
    train_time_sum = np.sum(train_time)
    train_time_mean = np.mean(train_time)
    print('Test_time:', test_time_mean)
    print('Train_time:', train_time_sum)



'''
# load test
model = torch.load('../F001_DAST_prediciton_model_11.46')
Y_test_numpy = Y_test.detach().numpy()
test_list = []

for k ,(batch_x,batch_y) in enumerate(test_loader):
    prediction = model(batch_x)
    prediction[prediction<0] = 0
    test_list.append(prediction)
    #test_loss = torch.sqrt(criterion(prediction*125, batch_y*125))
    #prediction_numpy = prediction.detach().numpy()   
    #batch_y_numpy = batch_y.detach().numpy()
    #test_score = myScore(batch_y_numpy*125, prediction_numpy*125)

test_all =  torch.cat(test_list).detach().numpy()
test_all_tensor = torch.from_numpy(test_all)
test_loss = torch.sqrt(criterion(test_all_tensor*125, Y_test*125))
test_score = myScore(Y_test_numpy*125, test_all*125)
print('test_loss = ', test_loss.item(),
          'test_score = ', test_score)


'''



