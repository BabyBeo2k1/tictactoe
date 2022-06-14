import numpy as np
import pandas as pd
import os
import torch
import torchvision
from torch.utils.data import DataLoader,Dataset

f=open("log.txt","r")
x_input=np.zeros((0,20,20))
for i in range(2):
    x=""
    for i in  range(20):
        x=x+f.readline()
    print(x)
    cur_input=np.fromstring(x,sep=' ')
    print(cur_input)
    cur_input.reshape((20,20))
    print(cur_input.shape)
    x=np.append(x,cur_input,axis=0)
