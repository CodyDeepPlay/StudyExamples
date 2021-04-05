# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 13:35:34 2020

@author: Mingming
"""

import csv
import pandas as pd
# open a new CSV file to write data


##########################################################################
########## Example of opening a file, and write data into it   ###########
##########################################################################
myfilename = "myfile.csv"

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


##########################################################################
########## Example of opening a file, and read data from it    ###########
##########################################################################

df = pd.read_csv(myfilename) # load the data in as pandas data frame
data_value  = df.values      # get all data values
keys_value  = df.keys        # get all the keys and data values
keys  = df.keys()            # get all the keys

# find if a given substring is in my current string or not
"""
if this substring is not in my string, it will return -1
"""
(myfilename.find("findMyString") != -1) 




##########################################################################
######### Example of opening a .xlxs file, and read data from it  ########
##########################################################################

# this requires you to have 'xlrd==1.2.0'
# has not tested if other newer version of 'xlrd' will work or not
df = pd.read_excel(filename, sheet_name=sheet_name)




















