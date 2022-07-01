import numpy as np
import pandas as pd
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
device=torch.device('cpu')
input_size=400
hidden_size=100
num_epochs=8
batch_size=100
learning_rate=0.01
momentum=0.5
f_log=open("log.txt","r")
f_res=open("res.txt","r")
x_input=np.array([])
x_input=np.reshape(x_input,(0,1,20,20))
y_output=np.array([])
y_output=np.reshape(y_output,(0,1))
for i in range(10000):
    x=""
    for i in  range(20):
        x=x+f_log.readline()+"\n"
    cur_input=np.fromstring(x ,dtype=float,sep=' ')
    cur_input=np.reshape(cur_input,(1,1,20,20))
    y=f_res.readline()
    cur_output=np.fromstring(y,dtype=int,sep=" ")
    if cur_output==-1:
        cur_output=0
    cur_output=np.reshape(cur_output,(1,1))
    y_output=np.append(y_output,cur_output,axis=0)
    x_input=np.append(x_input,cur_input,axis=0)
print (x_input.shape)
print(y_output.shape)

class tictactoe_train(Dataset):
    def __init__(self,transform=None):
        self.x=torch.from_numpy(x_input)
        self.y=torch.from_numpy(y_output)
        self.n_samples=x_input.shape[0]
        self.transform=transform
    def __getitem__(self, item):
        return self.x[item],self.y[item]
    def __len__(self):
        return self.n_samples

class tictactoe_test(Dataset):
    def __init__(self,transform=None):
        self.x=torch.from_numpy(x_input[8000:])
        self.y=torch.from_numpy(y_output[8000:])
        self.n_samples=2000
        self.transform=transform
    def __getitem__(self, item):
        return self.x[item],self.y[item]

    def __len__(self):
        return self.n_samples

dataset_train=tictactoe_train()
dataset_test=tictactoe_test()
print(dataset_test.x.shape)
train_loader=DataLoader(dataset=dataset_train,
                      batch_size=batch_size,
                      shuffle=True)
test_loader=DataLoader(dataset=dataset_test,
                     batch_size=batch_size,
                     shuffle=False)
                     
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1,10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()  #Dropout
        self.fc1 = nn.Linear(80, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        #Convolutional Layer/Pooling Layer/Activation
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #Convolutional Layer/Dropout/Pooling Layer/Activation
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 80)
        #Fully Connected Layer/Activation
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        #Fully Connected Layer/Activation
        x = self.fc2(x)

        #Softmax gets probabilities. 
        return F.sigmoid(x)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #target=target.type(torch.LongTensor)
        #Variables in Pytorch are differenciable.

        data, target = Variable(data), Variable(target)
        #This will zero out the gradients for this batch. 
        optimizer.zero_grad()
        output = model(data)
        # Calculate the loss The negative log likelihood loss. It is useful to train a classification problem with C classes.
        loss = nn.BCELoss()(output, target)
        #dloss/dx for every Variable 
        loss.backward()
        #to do a one-step update on our parameter.
        optimizer.step()
        #Print out the loss periodically. 
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += nn.BCELoss()(output, target).data # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

model = CNN()
model=model.double()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

for epoch in range(1,11):
    train(epoch)
    test()
torch.save(model.state_dict(),"model.pth")