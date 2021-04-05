# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 09:33:37 2020

@author: Mingming

how to use a stop button to stop a task in the tkinter GUI
"""

from tkinter import *

running = False  # Global flag
n=0

def scanning():
    if running:  # Only do this if the Stop button has not been clicked
        #   print ("hello")
        
        global n
        print(n+1)
        n +=1
    # After 1 second, call scanning again (create a recursive loop)
    root.after(1, scanning)
    


def start():
    """Enable scanning by setting the global flag to True."""
    global running
    running = True
    
    global n
    n=0

def stop():
    """Stop scanning by setting the global flag to False."""
    global running
    running = False

root = Tk()
root.title("Title")
root.geometry("500x500")

app = Frame(root)
app.grid()

start = Button(app, text="Start Scan", command=start)
stop = Button(app, text="Stop", command=stop)

start.grid()
stop.grid()

root.after(10, scanning)  # After 1 second, call scanning
root.mainloop()




