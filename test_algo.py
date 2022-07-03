def if_else(x):
    for i in range(17):
        for j in range(17):
            if x[i][j-1] == 0 and x[i][j] == 1 and x[i][j+1] == 1 & & x[i][j+2] == 1 & & x[i][j+3] == 0){
        if (heuristic(i, j-1) > heuristic(i, j+3))
        return i, j - 1;
        else return i, j + 3;

    }
    if (x[i - 1][j] == 0 & & x[i][j] == 1 & & x[i + 1][j] == 1 & & x[i + 2][j] == 1 & & x[i + 3][j] == 0){
    if (heuristic(i - 1, j) > heuristic(i + 3, j)) return i-1, j;
    else return i+3, j;
    }
    if (x[i - 1][j - 1] == 0 & & x[i][j] == 1 & & x[i + 1][j + 1] == 1 & & x[i + 2][j + 2] == 1 & & x[i + 3][j + 3] == 0){
    if (heuristic(i - 1, j - 1) > heuristic(i + 3, j + 3)) return i-1, j-1;
    else return i+3, j+3;
    }
    if (x[i + 3][j - 1] == 0 & & x[i + 2][j] == 1 & & x[i + 1][j + 1] == 1 & & x[i][j + 2] == 1 & & x[i - 1][j + 3] == 0){
    if (heuristic(i + 3, j - 1) > heuristic(i - 1, j + 3)) return i+3, j-1;
    else return i-1, j+3;
    }
    if (x[i][j - 1] == 1 & & x[i][j] == 1 & & x[i][j + 1] == 1 & & x[i][j + 2] == 1 & & x[i][j + 3] == 0){
    return i, j + 3;
    }
    if (x[i - 1][j] == 1 & & x[i][j] == 1 & & x[i + 1][j] == 1 & & x[i + 2][j] == 1 & & x[i + 3][j] == 0){
    return i + 3, j;
    }
    if (x[i - 1][j - 1] == 1 & & x[i][j] == 1 & & x[i + 1][j + 1] == 1 & & x[i + 2][j + 2] == 1 & & x[i + 3][j + 3] == 0){
    return i + 3, j + 3;
    }
    if (x[i + 3][j - 1] == 1 & & x[i + 2][j] == 1 & & x[i + 1][j + 1] == 1 & & x[i][j + 2] == 1 & & x[i - 1][j + 3] == 0){
    return i - 1, j + 3;
    }
    if (x[i][j - 1] == 0 & & x[i][j] == 1 & & x[i][j + 1] == 1 & & x[i][j + 2] == 1 & & x[i][j + 3] == 1){
    return i, j - 1;
    }
    if (x[i - 1][j] == 0 & & x[i][j] == 1 & & x[i + 1][j] == 1 & & x[i + 2][j] == 1 & & x[i + 3][j] == 1){
    return i - 1, j;
    }
    if (x[i - 1][j - 1] == 0 & & x[i][j] == 1 & & x[i + 1][j + 1] == 1 & & x[i + 2][j + 2] == 1 & & x[i + 3][j + 3] == 1){
    return i - 1, j - 1;
    }
    if (x[i + 3][j - 1] == 0 & & x[i + 2][j] == 1 & & x[i + 1][j + 1] == 1 & & x[i][j + 2] == 1 & & x[i - 1][j + 3] == 1){
    return i + 3, j - 1;
    }
    }
    }