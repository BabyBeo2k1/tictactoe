from tkinter import *
import random
import ann as ann
import numpy as np
import minimax as mnm
#define next turn
def next_turn(row, column):
    global player,mode
    #player turn
    while mode == 0:
        return
    if buttons[row][column]['text']=="" and check_winner() is False:
        buttons[row][column]['text']="x"
        log_ai[row][column]=-1
        if check_winner() is False:
            label.config(text='ai turn')
        elif check_winner() is True:
            label.config(text='player win')

        elif check_winner() =='Tie':
            label.config(text='Tie!')


        #choose mode for ai
        if mode==1:
            x,y=mnm.ai_brain(log_ai)
        elif mode==2:
            x,y=ann.ai_brain(log_ai)
        if buttons[x][y]['text']==""and check_winner() is False :
            log_ai[x][y]=1
            buttons[x][y]['text'] = 'o'
            if check_winner() is False:
                label.config(text=player +' turn')
            elif check_winner() is True:
                label.config(text='ai win')
            elif check_winner() == 'Tie':
                label.config(text='Tie!')


def check_winner():
    #check winning condition
    counter=0
    for row in range(20):
        for column in range(19):
            if buttons[row][column]['text']==buttons[row][column+1]['text']!="":
                counter+=1
                if counter==4:
                    return True
            else:
                counter=0
        counter=0
    for column in range(20):
        for row in range(19):
            if buttons[row][column]['text']==buttons[row+1][column]['text']!="":
                counter+=1
                if counter==4:
                    return True
            else:
                counter=0
        counter=0
    for column in range(20):
        for row in range(19-column):
            if buttons[row][column+row]['text']==buttons[row+1][column+row+1]['text']!="":
                counter+=1
                if counter==4:
                    return True
            else:
                counter=0
        for row in range(19-column,19):
            if buttons[row][row-19+column]['text']==buttons[row+1][row-19+column+1]['text']!="":
                counter+=1
                if counter==4:
                    return True
            else:
                counter=0
    for row in range(20):
        for column in range(row):
            if buttons[row-column][column]['text']==buttons[row-column-1][column+1]['text']!="":
                counter+=1
                if counter==4:
                    return True
            else:
                counter=0
        for column in range(row,19):
            if buttons[19+row-column][column]['text']==buttons[19+row-column-1][column+1]['text']!="":
                counter+=1
                if counter==4:
                    return True
            else:
                counter=0
    if empty_space() is False:
        return "Tie"
    else:
        return False

def empty_space():
    #check for are there any spaces on the board
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
    #create new game
    global log_ai,mode
    mode=0
    log_ai=np.zeros((20,20))
    label.config(text='player turn')
    for i in range(20):
        for j in range(20):
            buttons[i][j].config(text="",)
window=Tk()
window.title("gomoku")
players=["x","o"]
turn=0
player='x'
mode=0
log_ai= np.zeros((20,20))
buttons= []
for i in range(20):
    x=[]
    for j in range(20):
        x.append(0)
    buttons.append(x)


def choose_algo():
    global mode
    if clicked.get() == "minimax prune":
        mode = 1
    elif clicked.get() == "ANN model":
        mode = 2


# Dropdown menu options
options = [
    "minimax prune",
    "ANN model"
]

# datatype of menu text
clicked = StringVar()

# initial menu text
clicked.set("minimax_prune")

# Create Dropdown menu
drop = OptionMenu(window, clicked, *options)
drop.pack(side=LEFT, anchor=NW)

# Create button, it will change label text
confirm_button = Button(window, text="OK", command=choose_algo).pack(side=LEFT, anchor=NW)

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