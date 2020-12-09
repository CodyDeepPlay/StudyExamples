# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 12:27:44 2020

@author: Mingming
"""

from six.moves import cPickle as pickle



#%% pesudo code,
# assume you have a data file named 'example.pickle'
# retrive the data from the saved pickle file
    try:
        f_read = open('example.pickle', 'rb')    # open the file for reading
        mydata = pickle.load(f_read)['evoked_EMGs']
        f_read.close()
    except Exception as e: # capture the objects from the exception
        print('Unable to read data from', all_files, ':', e) 
        raise
        
        
        
        
#%% peudo code, save data into pickle file

pickle_file_name = 'myfile.pickle'   # assume the file name in pickle format named as 'myfile.pickle'
pickle_file = os.path.join(directory, pickle_file_name)  # directory is where you want to save you data file

try:
  f = open(pickle_file, 'wb')                   # open the file for writing
  # save the save into a dictionary format
  save = {
    'key1': data1,   # key and value pair
    'key2': data2,  
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL) # save all the data into the file named in pickle_file, 
                                                # use the highest protocol version available
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise        