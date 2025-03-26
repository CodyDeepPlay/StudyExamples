#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 22:13:28 2025

@author: mingmingzhang

ECG example
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import wfdb  # waveform database package, a library of tools for reading, writing, and processing WFDB signals and annotations.

from lib import HelpferFunctions as hf  # custom module with helperfunctions needed here
import os
import pickle

#%%
# =============================================================================
# import wfdb, os  # waveform database package, a library of tools for reading, writing, and processing WFDB signals and annotations.
# 
# data_path   = 'qt-database-1.0.0'  # this cardiac data was downloaded and saved locally
# record_name = 'sel30'   # assign one record here to show as an example
# record_path = os.path.join(data_path, record_name)
# 
# annotation = wfdb.rdann(record_path, 'pu1')
# 
# 
# =============================================================================

#%% load all the files

data_path   = 'qt-database-1.0.0'  # this cardiac data was downloaded and saved locally

all_file_names = os.listdir(data_path)

min_sig_len = None
all_pu1s =[] # get all the pu1 files

for a_file in all_file_names:
    if 'pu1' in a_file:
        a_file_name=a_file.split('.')       
        record_name = a_file_name[0] # just the file name, not including extension
        
        if record_name not in all_pu1s:  # make sure not dupilcated names, so one file will not be added twice
            all_pu1s.append(record_name) # only add the file name, not extension
            record_path = os.path.join(data_path, record_name)
            record = wfdb.rdrecord(record_path)
            signals, fields = wfdb.rdsamp(record_path)
            fs = fields['fs']
            length = fields['sig_len']
            
            if min_sig_len is not None:
                min_sig_len = min(min_sig_len, length)
            else:                
                min_sig_len = length
                
            
           
print("Minimum signal length is:" , min_sig_len)

#%%

'''
Each file is with two channels of recordings. This is the spontaneous ECG signals 
from two different channels, we can treat these two channels as feature dimension

signals is with shape[time stamp, number of channels] then, the size is [time_stamp, n_feature]

'''

All_data_list = []  # create a space to host all the data
All_annotation_sample = []
All_annotation_symbol = []
All_annotation_list = [] # host all the ECG signal annotation

for record_name in all_pu1s:
    record_path = os.path.join(data_path, record_name)
    record = wfdb.rdrecord(record_path)        # read a single file
    signals, fields = wfdb.rdsamp(record_path) # extract the signals from a single file
   
    # read the annotation for a single file
    annotation   = wfdb.rdann(record_path, 'pu1')
    annot_expand = hf.expand_annotation(annotation.sample, annotation.symbol, length)
    
    # use the smallest signal length to truncate all the data, so that 
    # we can contatenate all the signal recordings in a single large dataset
    signals2add = signals[0:min_sig_len,:]
    annotation2add = annot_expand[0:min_sig_len]
    
    All_data_list.append(signals2add)
    All_annotation_sample.append(annotation.sample)
    All_annotation_symbol.append(annotation.symbol)
    All_annotation_list.append(annotation2add)
    

    
All_data = np.asarray(All_data_list)  # size(num_obs, time_stamps, n_features)
All_annotation = np.asarray(All_annotation_list)  # size(num_obs, time_stamps)


 
'''
define a dictionary structure to host all the data

{'sel31.pu1': {signals: data,
               annotation: annot_Data,
               fs: fs,
               length = fields['sig_len']},
 'sel32.pu1': {...},
 'sel33.pu1': {...},
 }            
        
'''        

#%%

#%% save my data

# Example data to save
data = {
    "ECG": {'data': All_data, 
            'explain': 'data size is with [105, 224993, 2], 105 data files, with each recording length is 224993, and each recording has 2 channel of ECG',},
    "fs": fs,
    "All_annotation": {'data': All_annotation,
                       'explain': 'converted annotation of all data points. -1: baseline, 0: N, 1: st, 2:t, 3:iso, 4:p, 5:pq'},
    "All_annotation_sample": {'data': All_annotation_sample,
                              'explain': 'original annoation sample for all recordings files'},
    "All_annotation_symbol": {'data': All_annotation_symbol,
                              'explain': 'original annoation symbol for all recordings files'},
}

#%%


#%%
# File name for the pickle file
filename = "long_EKG_recording.pkl"
file_path = os.path.join("data", filename)


hf.save_and_split_pickle(data, file_path, max_size_mb=45)
long_EKG = hf.load_and_combine_pickle(file_path)

#%%
# Save the data to the pickle file
if os.path.exists(file_path):
        print(f"File '{file_path}' already exists. Skipping save.")
else:
    try:
        with open(file_path, "wb") as file:  # "wb" mode for writing binary data
            pickle.dump(data, file)
        print(f"Data successfully saved to {filename}")
    
    except Exception as e:
        print(f"An error occurred: {e}")#%%
    




#%%


try:
    with open(file_path, "rb") as file: # "rb" mode for reading binary data
        loaded_data = pickle.load(file)
    print("Loaded data:", loaded_data.keys())

except FileNotFoundError:
    print(f"File {filename} not found.")
except Exception as e:
    print(f"An error occured during loading: {e}")






#%%  save all long ECG file into smaller/individual ECG files

all_ECG_data = loaded_data['ECG']['data']
all_annotation_expand = loaded_data['All_annotation']['data']
All_annotation_sample = loaded_data['All_annotation_sample']['data']
All_annotation_symbol = loaded_data['All_annotation_symbol']['data']


n = len(all_ECG_data) # the total number of ECG recording files

# initialize the object to hold data in each iteration of the long ECG recording file.
all_ECG_array = None
all_annotation = None

# iterate through each long ECG long recordings
for i in range(n):
    my_signal_all_chs = all_ECG_data[i]  # all channels ECG signals for a single recording example
    
    one_anntype = All_annotation_symbol[i]
    one_annsamp = All_annotation_sample[i]
    one_expand  = all_annotation_expand[i]
    
    # the first channel ECG data
    my_signal1 = my_signal_all_chs[:, 0]
    (single_ECG_list1, single_annotation_list1) = hf.seg_single_ECGs(my_signal1, one_anntype, one_annsamp, one_expand)
    # the 2nd channel ECG data
    my_signal2 = my_signal_all_chs[:, 1]
    (single_ECG_list2, single_annotation_list2) = hf.seg_single_ECGs(my_signal2, one_anntype, one_annsamp, one_expand)
    ''' single_annotation_list1 and single_annotation_list2 should be the same, 
        as they started the same annotation for long ECG recordings. 
    '''
    
    # convert all the chs of single ECG signals into one big file,
    # size (num_obs, time_length, chs)
    single_ECG_array1 = np.asarray(single_ECG_list1)[:,:,np.newaxis]
    single_ECG_array2 = np.asarray(single_ECG_list2)[:,:,np.newaxis]
    single_ECG_array = np.concatenate((single_ECG_array1, single_ECG_array2),axis=2)
 
    # size (num_obs, time_length)
    single_annotation = np.asarray(single_annotation_list2)
    
    ## concatenate the single ECG file and annotation file into larger file with more observations
    if all_ECG_array is None: all_ECG_array = single_ECG_array
    else: all_ECG_array = np.concatenate((all_ECG_array, single_ECG_array),axis=0) # add more observations extracted from this new long recording file
        
    if all_annotation is None: all_annotation = single_annotation
    else: all_annotation = np.concatenate((all_annotation, single_annotation),axis=0) # add more observations extracted from this new long recording file
        
#%%


# Example data to save
ECG_data = {
    "ECG": {'data': all_ECG_array, 
            'explain': 'data size is with [111167, 140, 2], 111167 individual ECG recordings, with each recording length is 140, and each recording has 2 channel of ECG',},
    "fs": fs,
    "all_annotation": {'data': all_annotation,
                       'explain': 'converted annotation of all data points for each individual ECG. -1: baseline, 0: N, 1: st, 2:t, 3:iso, 4:p, 5:pq'},
}

# File name for the pickle file
filename = "individual_EKG_recording.pkl"
file_path = os.path.join("data", filename)


hf.save_and_split_pickle(ECG_data, file_path, max_size_mb=45)
long_EKG = hf.load_and_combine_pickle(file_path)

#%%


# Save the data to the pickle file
if os.path.exists(file_path):
        print(f"File '{file_path}' already exists. Skipping save.")
else:
    try:
        with open(file_path, "wb") as file:  # "wb" mode for writing binary data
            pickle.dump(data, file)
        print(f"Data successfully saved to {filename}")
    
    except Exception as e:
        print(f"An error occurred: {e}")#%%
    


#%%
try:
    with open(file_path, "rb") as file: # "rb" mode for reading binary data
        individual_data = pickle.load(file)
    print("Loaded data:", individual_data.keys())

except FileNotFoundError:
    print(f"File {filename} not found.")
except Exception as e:
    print(f"An error occured during loading: {e}")









#%% load a single file
# read the .hea file, header file, describing signal file contents and format
data_path   = 'qt-database-1.0.0'  # this cardiac data was downloaded and saved locally
record_name = 'sel31'   # assign one record here to show as an example
record_path = os.path.join(data_path, record_name)
record = wfdb.rdrecord(record_path)
print(f"Record Name: {record.record_name}")
print(f"Sampling rate: {record.fs}")


# Read the signal
signals, fields = wfdb.rdsamp(record_path)
fs = fields['fs']
length = fields['sig_len']
print(f"Signals shape is: {signals.shape}")
print(f"Signals fields are: {fields.keys()}")

#%%

# Read the annotation file (.atr)
annotation = wfdb.rdann(record_path, 'pu1')
example_sample = annotation.sample
example_symbol = annotation.symbol

# expand the annotation
annsamp_0  = annotation.sample
anntype_0 = annotation.symbol
annot_expand = hf.expand_annotation(annsamp_0, anntype_0, length)



#%%
which_channel = 1
display_length = 1000
my_signal     = signals[0:display_length, which_channel]
my_annotation = annot_expand[0:display_length]
time_vector = np.arange( 0, display_length/fs, 1/fs)  # time vector for selected signal


plt.figure()
# plot the raw signal
plt.subplot(2,1,1)
plt.plot(time_vector, my_signal,)
plt.title('raw signal')
plt.xticks([]) # hide x axis
plt.ylabel('mV')

# plot the raw signal with different annotations for each section within the signal
plt.subplot(2,1,2)
plt.plot(time_vector, my_signal,)
plt.title('raw signal with annotation')
myN = np.where(my_annotation==0)[0]
plt.scatter(time_vector[myN], my_signal[myN], color='r', marker='o' )

myst = np.where(my_annotation==1)[0]
plt.scatter(time_vector[myst], my_signal[myst], color='g', marker='o')

myt = np.where(my_annotation==2)[0]
plt.scatter(time_vector[myt], my_signal[myt], color='b', marker='o')

myiso = np.where(my_annotation==3)[0]
plt.scatter(time_vector[myiso], my_signal[myiso], color='k', marker='o')

myp = np.where(my_annotation==4)[0]
plt.scatter(time_vector[myp], my_signal[myp], color='c', marker='o')

mypq = np.where(my_annotation==5)[0]
plt.scatter(time_vector[mypq], my_signal[mypq], color='y', marker='o')
plt.ylabel('mV')
plt.xlabel('time (s)')



#%%

single_ECG_list = []       # a big list to store all the ECG signals
single_annotation_list =[] # a big list to store all the annotations for each individual ECG signals

annsamp_0 = annotation.sample
anntype_0 = annotation.symbol

# iterate through each batch to the get raw signals and annotations.
one_anntype  = anntype_0  # one long annotation type for a long recording, it has string, i.e.['(', 'p', ')', '(', 'N', ')'...]
one_annsamp  = annsamp_0  # the index of the related annotation in the long time series
one_expand   = annot_expand # recomputed annotation, using 0,1,2,3.. to represent different sections of ECG signal



#%% how to prepare a long recording file into a single EMG data file

(single_ECG_list, single_annotation_list) = hf.seg_single_ECGs(my_signal, one_anntype, one_annsamp, one_expand)


#%% plot an example to see the ECG signal with its annotation
which_ECG_example = 0
my_single_ECG = single_ECG_list[which_ECG_example]
my_annotation = single_annotation_list[which_ECG_example]
#time_vector = np.arange( 0, display_length/fs, 1/fs)  # time vector for selected signal

time_vector = np.arange( 0, len(my_single_ECG)/fs, 1/fs)  # time vector for selected signal


plt.figure()
plt.subplot(2,1,1)
plt.plot(time_vector, my_single_ECG)
plt.title('raw signal')
plt.xticks([]) # hide x axis
plt.ylabel('mV')

#%
ax=plt.subplot(2,1,2)
plt.plot(time_vector, my_single_ECG)
plt.title('raw signal with annotation')


#%
max_amp = max(my_single_ECG)  # the max amplitude of this single ECG recording
min_amp = min(my_single_ECG)  # the max amplitude of this single ECG recording


# annotate each segment of the signal, based on the annotations we have
myN = np.where(my_annotation==0)[0]
if len(myN) !=0:
    plt.scatter(time_vector[myN], my_single_ECG[myN], color='r', marker='o' )
    mid_point = len(myN)//2
    ax.text(time_vector[myN[mid_point]], max_amp*1.02, 'ORS')
    
myst = np.where(my_annotation==1)[0]
if len(myst) !=0:
    plt.scatter(time_vector[myst], my_single_ECG[myst], color='g', marker='o')
    mid_point = len(myst)//2
    ax.text(time_vector[myst[mid_point]], max_amp*1.02, 'ST')

myt = np.where(my_annotation==2)[0]
if len(myt) !=0:
    plt.scatter(time_vector[myt], my_single_ECG[myt], color='b', marker='o')
    mid_point = len(myt)//2
    ax.text(time_vector[myt[mid_point]], max_amp*1.02, 'T')

myiso = np.where(my_annotation==3)[0]
if len(myiso) !=0:
    plt.scatter(time_vector[myiso], my_single_ECG[myiso], color='k', marker='o')
    mid_point = len(myiso)//2
    ax.text(time_vector[myiso[0]], max_amp*1.02, 'baseline')

myp = np.where(my_annotation==4)[0]
if len(myp) !=0:
    plt.scatter(time_vector[myp], my_single_ECG[myp], color='c', marker='o')
    mid_point = len(myp)//2
    ax.text(time_vector[myp[mid_point]], max_amp*1.02, 'P')

mypq = np.where(my_annotation==5)[0]
if len(myp) !=0:
    plt.scatter(time_vector[mypq], my_single_ECG[mypq], color='y', marker='o')
    mid_point = len(mypq)//2
    ax.text(time_vector[mypq[mid_point]], max_amp*1.02, 'PQ')
       
plt.ylabel('mV')
plt.xlabel('time (s)')
plt.ylim([min_amp-min_amp*0.02, max_amp*1.05])



#%%
