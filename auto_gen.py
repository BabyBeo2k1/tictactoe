import random
import numpy as np
def gen_step():
    global player, log_ai, winner

    row = random.randint(0, 19)
    column = random.randint(0, 19)
    while check_winner() is False:
        while log_ai[row][column]!=0:
            row = random.randint(0, 19)
            column = random.randint(0, 19)
        if player == "x":
            log_ai[row][column] = -1
            if check_winner() is False:
                player = 'o'
            elif check_winner() is True:
                winner = -1
                new_game()
                return
            elif check_winner() == 'Tie':
                new_game()
                return
        elif player == "o":
            log_ai[row][column] = 1
            if check_winner() is False:
                player = "x"
            elif check_winner() is True:
                winner = 1
                new_game()
                return
            elif check_winner() == 'Tie':
                new_game()
                return
def check_winner():
    # check điều kiện thắng
    counter = 0
    for row in range(20):
        for column in range(19):
            if log_ai[row][column] == log_ai[row][column + 1] !=0:
                counter += 1
                if counter == 4:
                    return True
            else:
                counter = 0
        counter = 0
    for column in range(20):
        for row in range(19):
            if log_ai[row][column] == log_ai[row + 1][column] !=0:
                counter += 1
                if counter == 4:
                    return True
            else:
                counter = 0
        counter = 0
    for column in range(20):
        for row in range(19 - column):
            if log_ai[row][column + row] == log_ai[row + 1][column + row + 1] !=0:
                counter += 1
                if counter == 4:
                    return True
            else:
                counter = 0
        for row in range(19 - column, 19):
            if log_ai[row][row - 19 + column] == log_ai[row + 1][row - 19 + column + 1] !=0:
                counter += 1
                if counter == 4:
                    return True
            else:
                counter = 0
    for row in range(20):
        for column in range(row):
            if log_ai[row - column][column] == log_ai[row - column - 1][column + 1] !=0:
                counter += 1
                if counter == 4:
                    return True
            else:
                counter = 0
        for column in range(row, 19):
            if log_ai[19 + row - column][column] == log_ai[19 + row - column - 1][column + 1] !=0:
                counter += 1
                if counter == 4:
                    return True
            else:
                counter = 0
    if empty_space() is False:
        return "Tie"
    else:
        return False
def empty_space():
    def empty_space():
        # check xem còn quân nào có thể đi trên bàn cờ không
        spaces = 400
        for i in range(20):
            for j in range(20):
                if  log_ai[i][j] != 0:
                    spaces -= 1
        if spaces == 0:
            return False
        else:
            return True
def new_game():
    global  log_ai, player
    # tat ca cac log file duoc luu vao history va dan nhan
    i = 0
    path_log = "log.txt"
    path_res = "res.txt"
    f_res = open(path_res, "a")
    f_log = open(path_log, "a")
    for i in range(20):
        for j in range(20):
            f_log.write(str( log_ai[i][j]))
            f_log.write(" ")
        f_log.write("\n")
    f_res.write(str(winner))
    f_res.write("\n")
    # tao game moi
    player = random.choice(players)
    log_ai = np.zeros((20, 20), dtype=int)
log_ai=np.zeros((20,20),dtype=int)
players=["x","o"]
player=random.choice(players)
winner =0
for i in range(10000):
    gen_step()