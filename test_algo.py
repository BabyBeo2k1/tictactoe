def test(_input):

    min_i = 0

    min_j = 0
    for i in range(20):
        for j in range(20):
            if _input[i][j] != 0:
                if i > 3:
                    min_i = i - 4

    for i in range(20):
        for j in range(20):
            if _input[j][i] != 0:
                if i > 3:
                    min_j = i - 4
    for i in range(min_i,20):
        for j in range(min_j,20):
            if _input[i][j]!=0:
                min_i=i
                min_j=j
            else:
                return min_i,min_j

    return min_i,min_j