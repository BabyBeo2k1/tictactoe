from tkinter import *
import numpy as np
import minimax as mnm
#define next turn

def next_turn(row, column):
    #command khi bấm vào mỗi ô trên bàn
    global player,mode,log_ai,step
    #player turn
    step+=1
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
        print(log_ai)

        if step<2:
            x,y=mnm.opening(log_ai)
        else:
            x,y=mnm.ai_brain(log_ai,mode)
        print(x,y)
        print(log_ai)
        if buttons[x][y]['text']==""and check_winner() is False :
            log_ai[x][y]=1
            print(log_ai)
            buttons[x][y]['text'] = 'o'
            if check_winner() is False:
                label.config(text=player +' turn')
            elif check_winner() is True:
                label.config(text='ai win')
            elif check_winner() == 'Tie':
                label.config(text='Tie!')


def check_winner():
    #check điều kiện thắng
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
    #tạo game mới

    global log_ai,mode,step

    step=0
    mode=0
    log_ai=np.zeros((20,20))
    label.config(text='player turn')
    for i in range(20):
        for j in range(20):
            buttons[i][j].config(text="",)
window=Tk()
step=0

window.title("gomoku")
players=["x","o"]
turn=0
player='x'
mode=1
log_ai= np.zeros((20,20))
buttons= []
for i in range(20):
    x=[]
    for j in range(20):
        x.append(0)
    buttons.append(x)
pre_elim=[]
for i in range(20):
    x=[]
    for j in range(20):
        x.append(False)
    pre_elim.append(x)

def choose_algo():
    #chọn thuật toán khác nhau
    global mode
    new_game()
    if clicked.get() == "Minimax using manual heuristic":
        mode = 1
    elif clicked.get() == "Minimax using ANN model":
        mode = 2


# các thuật toán
options = [
    "Minimax using manual heuristic",
    "Minimax using ANN model",
]

clicked = StringVar()

# thuật toán default
clicked.set("choose the algorithm first")

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