import numpy as np
def ai_brain(_input):
    depth=1
    res_col=0
    res_row=0
    res=-pow(200,5)
    for i in range(20):
        for j in range(20):
            if pre_elim(_input,i,j):
                _input[i][j] = 1
                cur=algo(_input,depth,True)
                if cur>res:
                    res_row=i
                    res_col=j
                _input[i][j]=0
    return res_row,res_col

def heuristic(_input):
    res=0
    return res
def algo(_input, depth, is_ai):
    if depth==0:
        return heuristic(_input)
    if is_ai:
        maxEval=-pow(200,5)
        for i in range(20):
            for j in range(20):
                if pre_elim(_input,i,j):
                    _input[i][j]=1
                    eval=algo(_input,depth-1,False)
                    maxEval=max(eval,maxEval)
                    _input[i][j]=0
        return maxEval
    else:
        minEval=pow(200,5)
        for i in range(20):
            for j in range(20):
                if pre_elim(_input,i,j):
                    _input[i][j] = -1
                    eval = algo(_input, depth - 1, True)
                    minEval =min(eval, minEval)
                    _input[i][j] = 0
        return minEval
def pre_elim(_input, row, column):
    if _input[row][column]==0:
        return True
    else:
        return False
