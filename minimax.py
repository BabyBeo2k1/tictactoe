import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
def ai_brain(_input,mode):
    if mode==3:
        res_row,res_col=if_else(_input)
    else:
        depth=2
        minimax_res=algo(_input,depth,True,mode,-pow(200,6),pow(200,6))
        res,res_row,res_col=minimax_res

    return res_row,res_col
def heuristic_nn(_input):
    _input=np.reshape(_input,(1,1,20,20))
    _input=torch.from_numpy(_input)

    return model(_input)
def heuristic(_input):
    res=0
    # hàm heuristic chấm điểm cho từng thế cờ bằng cách tính hàng dọc ngang chéo
    for row in range(20):
        x=[]
        for column in range(20):
            x.append(_input[row][column])
        res+=check_row(x,20)

    for column in range(20):
        x = []
        for row in range(20):
            x.append(_input[row][column])
        res+=check_row(x,20)

    for column in range(20):
        x=[]
        for row in range(20-column):
            x.append(_input[row][column+row])
        res+=check_row(x,20-column)
        y=[]
        for row in range(19-column,20):
            y.append(_input[row][row-19+column])
        res+=check_row(y,column+1)
    for row in range(20):
        x=[]
        for column in range(row+1):
            x.append(_input[row-column][column])
        res+=check_row(x,row)
        y=[]
        for column in range(row,20):
            y.append(_input[19+row-column][column])
        res+=check_row(y,20-row)
    return res
def algo(_input, depth, is_ai,mode,alpha,beta):
    # thuật minimax
    res_row=-1
    res_col=-1
    if depth==0:
        if mode==1:
            return  heuristic(_input),0,0
        elif mode==2:
            return heuristic_nn(_input),0,0
    if is_ai:
        maxEval=-pow(200,6)
        for i in range(20):
            for j in range(20):
                if _input[i][j]==0:
                    _input[i][j]=1
                    eval,m,n=algo(_input,depth-1,False,mode,alpha,beta)
                    if eval>maxEval:
                        res_col=j
                        res_row=i
                        maxEval=eval
                    alpha=max(alpha,eval)
                    if beta<=alpha:
                        break
                    _input[i][j] = 0

        return maxEval,res_row,res_col
    else:
        minEval=pow(200,6)
        for i in range(20):
            for j in range(20):
                if _input[i][j] == 0:
                    _input[i][j] = -1
                    eval,m,n = algo(_input, depth - 1, True,mode,alpha,beta)
                    if eval<minEval:
                        minEval=eval
                        res_col=j
                        res_row=i
                    _input[i][j] = 0
                    beta =min(beta,eval)
                    if beta <=alpha:
                        break
        return minEval,res_row,res_col

def check_row(_input,size):
    _input.append(0)
    # check điểm của từng hàng, cột, đường chéo
    res=0
    near_ai=-1
    near_pl=-1
    for i in range(size+1):
        local =-1
        if _input[i]==1 or i==size:
            if i-near_ai>5:
                for j in range(near_ai+1,i):
                    if _input[j]==-1:
                        local*=200
                    else:
                        local=-1
                        res+=local
            near_ai=i
            local=-1
            res+=local
    for i in range(size+1):
        local=1
        if _input[i]==-1 or i==size:
            if i-near_pl>5:
                for j in range(near_pl+1,i):
                    if _input[j]==-1:
                        local*=200
                    else:
                        local=1
                        res+=local
            near_pl=i
            local=1
            res+=local
        """if _input[i]!=_input[i-1]:
            cur=cur*200
        else:
            if _input[i - 1] == 1:
                near_ai = i - 1
            else:
                near_pl = i - 1
            if _input[i]==0:
                local+=cur
                cur=0

            elif _input[i]==1:
                if (i-near_ai)>5:
                    res+=local
                    local=0
                else:
                    local=0
            else:
                if(i-near_pl)>5:
                    res+=local
                    local=0
                else:
                    local=0"""
    return res

x=[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],]
#test thuật toán trong hàm
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
model = CNN()
model.double()
model.load_state_dict(torch.load("model.pth"))
#code ở hàm if_else
def if_else(_input):
    #_input là mảng 20x20 biểu diễn thế cờ
    #muốn dùng hàm heuristic ở trên phải đưa vào đúng định dạng là mảng 20x20
    #tức là nếu bạn muốn đánh nước i j mà muốn tính hàm heuristic của nước này thì gán _input[i][j]=1
    #rồi dùng res=heuristic(_input)
    row,col=0,0
    return row,col #return hàng trước cột sau
"""x=np.array(x,dtype=float)
i,y,z=ai_brain(x,2)
print (i,y,z,x[i][y] )"""
print(heuristic(x))