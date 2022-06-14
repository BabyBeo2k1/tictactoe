import numpy as np
import pandas as pd
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size=400
hidden_size=100
num_epochs=8
batch_size=80
learningrate=0.01
f_log=open("log.txt","r")
f_res=open("res.txt","r")
x_input=np.array([])
x_input=np.reshape(x_input,(0,20,20))
y_output=np.array([])
y_output=np.reshape(y_output,(0,1))
for i in range(1000):
    x=""
    for i in  range(20):
        x=x+f_log.readline()+"\n"
    cur_input=np.fromstring(x,sep=' ')
    cur_input=np.reshape(cur_input,(1,20,20))
    y=f_res.readline()
    cur_output=np.fromstring(y,sep=" ")
    cur_output=np.reshape(cur_output,(1,1))
    y_output=np.append(y_output,cur_output,axis=0)
    x_input=np.append(x_input,cur_input,axis=0)
print (x_input.shape)

print(y_output.shape)
x_train=torch.from_numpy(x_input[:800])
x_train.requires_grad()
y_train=torch.from_numpy(y_output[800:])
print(x_train.shape)
"""class tictactoe_train(Dataset):
    def __init__(self,transform=None):
        self.x=torch.from_numpy(x_input[:800])
        self.y=torch.from_numpy(y_output[:800])
        self.n_samples=x_input.shape[0]
        self.transform=transform
    def __getitem__(self, item):
        return self.x[item],self.y[item]
        return sample
    def __len__(self):
        return self.n_samples
class tictactoe_test(Dataset):
    def __init__(self,transform=None):
        self.x=torch.from_numpy(x_input[800:])
        self.y=torch.from_numpy(y_output[800:])
        self.n_samples=x_input.shape[0]
        self.transform=transform
    def __getitem__(self, item):
        return self.x[item],self.y[item]
        return sample
    def __len__(self):
        return self.n_samples
dataset_train=tictactoe_train()
dataset_test=tictactoe_test()
print(dataset_test.x.shape)
train_data=DataLoader(dataset=dataset_train,
                      batch_size=batch_size,
                      shuffle=True)
test_data=DataLoader(dataset=dataset_test,
                     batch_size=batch_size,
                     shuffle=False)
class CNN(nn.module):
    def __init__(self,input,layer1,layer2,):"""