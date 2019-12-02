import tkinter as tk
import random
from os import system, name
import time

def clear():
    if name == 'nt':
        _ = system('cls')
    else:
        _ = system('clear')

index = 0
list_index = 0
list = []
word_count = 10

f = open('words.txt', 'r')
for line in f:
    list.append(line.strip())

random.shuffle(list)

list = list[:word_count]
print("---WELCOME TO TYPING TUTOR---")
time.sleep(1)
clear()
print("Total words: ", len(list))
time.sleep(2)
clear()
print("Word "+str(list_index+1)+" out of "+str(word_count)+": "+list[list_index])

start_time = time.time()
end_time = 0
time_multiplier = 2

lives = 3

score = 0

def keypress(event):
    global index
    global list_index
    global list
    global lives
    global score
    global start_time
    global end_time
    global time_multiplier

    word = list[list_index]

    if event.char == word[index]:
        print ("yay")
        index = index + 1
        score = score + 1
    else:
        print ("boo")
        clear()
        print("Word " + str(list_index + 1) + " out of " + str(word_count) + ": " + list[list_index])
        print("wrong letter!")
        lives = lives - 1
        print("Lives left: ", lives)
        if lives == 0:
            print("Game Over!")
            print("Final Score: ", score)
            root.destroy()
        return
    #if index == len(word):
    #    clear()
    #    print("right!")
    #    index = 0
    #    list_index = list_index + 1
    #    end_time = time.time()
    #    time_taken = int(end_time - start_time)
    #    time_left = time_multiplier * len(word) - time_taken
    #    score = score + time_left
    #    print("time taken: " + str(time_taken) + " out of " + str(time_multiplier*len(word)) + " seconds.")
    #    print("Current score: ", score)
    #    time.sleep(1.5)
    #    start_time = end_time
    #    clear()

    #if list_index < len(list) and index == 0:
    #    print("Word " + str(list_index + 1) + " out of " + str(word_count) + ": " + list[list_index])

    #elif list_index == len(list):
    #    clear()
    #    print("Congratulations! you have beaten the game!")
    #    print("Final Score: ", score)
    #    root.destroy()

root = tk.Tk()
root.bind_all(keypress, '')
root.withdraw()
root.mainloop()
