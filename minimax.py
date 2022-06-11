import numpy as np

def ai_brain(_input):
    # xử lý bước cuối để ra quyết định
    res_row=0
    res_col=0
    depth=2
    minimax_res=algo(_input,depth,True)
    res,res_log=minimax_res
    for i in range(20):
        for j in range(20):
            if res_log[i][j]!=_input[i][j]:
                res_row=i
                res_col=j
    return res_row,res_col,res

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

def algo(_input, depth, is_ai):
    # thuật minimax
    res_log=_input
    if depth==0:
        return heuristic(_input),_input
    if is_ai:
        maxEval=-pow(200,5)
        for i in range(20):
            for j in range(20):
                if _input[i][j]!=0:
                    _input[i][j]=1
                    eval,cur_log=algo(_input,depth-1,False)
                    if eval>maxEval:
                        res_log=cur_log
                        maxEval=eval

                    _input[i][j] = 0

        return maxEval
    else:
        minEval=pow(200,5)
        for i in range(20):
            for j in range(20):
                if _input[i][j] != 0:
                    _input[i][j] = -1
                    eval,cur_log = algo(_input, depth - 1, True)
                    if eval<minEval:
                        minEval=eval
                        res_log=cur_log
                    _input[i][j] = 0

        return minEval,res_log

def check_row(_input,size):
    # check điểm của từng hàng, cột, đường chéo
    res=0
    local=0
    near_ai=-1
    near_pl=-1
    cur=0
    for i in range(size):
        if _input[i]==1:
            if i-near_ai>5 :
                res+=local
                local=0
            near_ai=i
            while _input[i]==1:
                i+=1
                if i==size:
                    break
            if i == size:
                if near_pl < size-5:
                    local += pow(200, i - near_ai)

                    res += local
            else:
                local+=pow(200, i - near_ai)


            near_ai=i
        elif _input[i]==-1:
            if i-near_pl>5 :
                res+=local
                local=0

            near_pl=i
            while _input[i]==-1 :
                i+=1
                if i==size:
                    break

            if i == size:
                if near_ai < size-5:
                    local += -pow(200, i - near_ai)
                    res += local
            else:
                local+=-pow(200, i - near_ai)
            near_ai=i
        else:
            if i==size-1:
                res+=local

    return res


def update_pre_elim(_input):
    # lọc các đầu vào không cần thiết
    max_i=20
    min_i=0
    max_j=20
    min_j=0
    #mục tiêu thu nhỏ phạm vi của nghiệm, đầu vào phải nằm trong hình chữ nhật với chỉ số hàng/cột nhỏ nhất/lớn nhất
    #không cách quá các ô đã được đánh 5 đơn vị


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
x=[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0],
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
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],]
#test thuật toán trong hàm

i,y,z=ai_brain(x)
print (i,y,z,x[i][y] )