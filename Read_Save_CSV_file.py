# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 13:35:34 2020

@author: Mingming
"""

import csv
# open a new CSV file to write data

csvfile  = open("myfile.csv",'w', newline='')   # create a file with the name you want
datawriter = csv.writer(csvfile)       # create a writer object

# csv file writer need the data to be wrote organized in tuple

# you can write string data
strings_to_write = ("mystring1", "mystring2", "mystring3", "mystring4") 
datawriter.writerow(strings_to_write) # the writerow() needs to take tuple variable in order to write one number in each cell                
# it will write a row of strings as you defined in the "strings_to_write" variable

# you can also write numberical data into the row
data_to_write = (1,  123, -158, 10.6)   
datawriter.writerow(data_to_write) 

csvfile.close() # close the .csv file after finishing save the data
