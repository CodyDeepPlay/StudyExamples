# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 09:33:37 2020

@author: Mingming

how to use a stop button to stop a task in the tkinter GUI
"""

from tkinter import *

running = True  # Global flag

def scanning():
    if running:  # Only do this if the Stop button has not been clicked
        print ("hello")

    # After 1 second, call scanning again (create a recursive loop)
    root.after(10, scanning)

def start():
    """Enable scanning by setting the global flag to True."""
    global running
    running = True

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
