import numpy as np

def ai_brain(_input):
    res_row=0
    res_col=0
    depth=2
    res=-pow(200,5)
    for i in range(20):
        for j in range(20):
            if _input[i][j]==0:
                _input[i][j] = 1
                cur=algo(_input,depth,True)
                if cur>res:
                    res_row=i
                    res_col=j
                    res=cur
                _input[i][j]=0
    return res_row,res_col,res

def heuristic(_input):
    res=0

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
def algo(_input, depth, is_ai):
    if depth==0:
        return heuristic(_input)
    if is_ai:
        maxEval=-pow(200,5)
        for i in range(20):
            for j in range(20):
                if _input[i][j]!=0:
                    _input[i][j]=1
                    eval=algo(_input,depth-1,False)
                    maxEval=max(eval,maxEval)
                    _input[i][j] = 0

        return maxEval
    else:
        minEval=pow(200,5)
        for i in range(20):
            for j in range(20):
                if _input[i][j] != 0:
                    _input[i][j] = -1
                    eval = algo(_input, depth - 1, True)
                    minEval =min(eval, minEval)
                    _input[i][j] = 0

        return minEval
def check_row(_input,size):
    res=0
    local=0
    near_ai=0
    near_pl=0
    cur=0
    first=20
    for i in range(size):
        if _input[i]!=0:
            first=i
            if _input[i]==1:
                near_ai=1
                cur=1
            if _input[i]==-1:
                near_pl=1
                cur=-1
            break
    for i in range(first,size):
        if _input[i]==0:
            local+=cur
            cur=0
            near_pl+=1
            near_ai+=1
            
        elif _input[i] == 1:
            if near_ai==0:
                if i>4:
                    cur=1
                near_ai=1
                near_pl+=1
            elif near_ai==1:
                cur*=200
                if near_pl!=0:
                    near_pl+=1
            else:
                if near_ai>near_pl:
                    if near_ai>5:
                        local+=cur
                        res+=local
                    local=0
                cur=1
                near_ai=1
                if near_pl!=0:
                    near_pl+=1
        else:
            if near_pl==0:
                if i>4:
                    cur=-1
                near_pl=1
                near_ai+=1
            elif near_pl==1:
                cur*=200
                if near_ai!=0:
                    near_ai+=1
            else:
                if near_ai<near_pl:
                    if near_pl>5:
                        local+=cur
                        res+=local
                    local=0

                if near_ai!=0:
                    near_ai+=1
                cur=-1
                near_pl=1
    return res


def update_pre_elim(_input):
    max_i=20
    min_i=0
    max_j=20
    min_j=0
    for i in range(20):
        for j in range(20):
            if _input[i][j]!=0:
                if i>3:
                    min_i=i-4

    for i in range(20):
        for j in range(20):
            if _input[j][i]!=0:
                if i > 3:
                    min_j = i - 4

    for i in range(20):
        for j in range(20):
            if _input[19-i][19-j]!=0:
                if i>3:
                    max_i=24-i
    for i in range(20):
        for j in range(20):
            if _input[19-j][19-i]!=0:
                if i>3:
                    max_j=24-i
    return min_i,max_i,min_j,max_j
x=[1,1,1,0,0,0,0,-1,-1]
print (check_row(x,9))