from tkinter import *
from time import sleep

root = Tk()
var = StringVar()
var.set('hello')

l = Label(root, textvariable = var)
l.pack()

for i in range(20):
    sleep(1) # Need this to slow the changes down
    var.set(i)
    root.update_idletasks()
