from tkinter import *
import random
import os
import numpy as np
def next_turn(row, column):
    #verify which player is playing
    global player,log_ai,winner
    if buttons[row][column]['text']=="" and check_winner() is False:
        if player=="x":
            buttons[row][column]['text'] ="x"
            log_ai[row][column] = -1
            if check_winner() is False:
                player='o'
                label.config(text='o turn')
            elif check_winner() is True:
                label.config(text='x win')
                winner=-1
            elif check_winner() == 'Tie':
                label.config(text='Tie!')
        elif player=="o":
            buttons[row][column]['text'] ="o"
            log_ai[row][column] = 1
            if check_winner() is False:
                player="x"
                label.config(text='x turn')
            elif check_winner() is True:
                label.config(text='o win')
                winner=1
            elif check_winner() == 'Tie':
                label.config(text='Tie!')
def check_winner():
    # check điều kiện thắng
    counter = 0
    for row in range(20):
        for column in range(19):
            if buttons[row][column]['text'] == buttons[row][column + 1]['text'] != "":
                counter += 1
                if counter == 4:
                    return True
            else:
                counter = 0
        counter = 0
    for column in range(20):
        for row in range(19):
            if buttons[row][column]['text'] == buttons[row + 1][column]['text'] != "":
                counter += 1
                if counter == 4:
                    return True
            else:
                counter = 0
        counter = 0
    for column in range(20):
        for row in range(19 - column):
            if buttons[row][column + row]['text'] == buttons[row + 1][column + row + 1]['text'] != "":
                counter += 1
                if counter == 4:
                    return True
            else:
                counter = 0
        for row in range(19 - column, 19):
            if buttons[row][row - 19 + column]['text'] == buttons[row + 1][row - 19 + column + 1]['text'] != "":
                counter += 1
                if counter == 4:
                    return True
            else:
                counter = 0
    for row in range(20):
        for column in range(row):
            if buttons[row - column][column]['text'] == buttons[row - column - 1][column + 1]['text'] != "":
                counter += 1
                if counter == 4:
                    return True
            else:
                counter = 0
        for column in range(row, 19):
            if buttons[19 + row - column][column]['text'] == buttons[19 + row - column - 1][column + 1]['text'] != "":
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
    #check xem còn quân nào có thể đi trên bàn cờ không
    spaces =400
    for i in range(20):
        for j in range(20):
            if buttons[i][j]['text']!="":
                spaces -=1
    if spaces ==0:
        return False
    else:
        return True
def new_game():
    global log_ai, player
    #tat ca cac log file duoc luu vao history va dan nhan
    i=0
    path_log= "log.txt"
    path_res= "res.txt"
    f_res=open(path_res,"a")
    f_log= open(path_log,"a")
    for i in range (20):
        for j in range(20):
            f_log.write(str(log_ai[i][j]))
            f_log.write(" ")
        f_log.write("\n")
    f_res.write(str(winner))
    f_res.write("\n")
    # tao game moi
    player=random.choice(players)
    log_ai = np.zeros((20, 20))
    label.config(text=player+ 'turn')
    for i in range(20):
        for j in range(20):
            buttons[i][j].config(text="" )

window=Tk()
window.title("gomoku")
players=["x","o"]
winner=0
log_ai= np.zeros((20,20),dtype=int)
buttons= []
for i in range(20):
    x=[]
    for j in range(20):
        x.append(0)
    buttons.append(x)
player=random.choice(players)
label= Label(text=player+" turn",font=('consolas',40))
label.pack(side="top")
reset_button=Button(text="restart",font=('consolas',20),command=new_game)
reset_button.pack(side="top")
frame=Frame(window)
frame.pack()
pixelVirtual = PhotoImage(width=1, height=1)
for row in range(20):
    for column in range(20):
        buttons[row][column]=Button(frame,
                                    text="",
                                    font=('consolas',16),
                                    width=3, height=1,
                                    command= lambda row=row,column=column:next_turn(row,column))
        buttons[row][column].grid(row=row,column=column)
window.mainloop()