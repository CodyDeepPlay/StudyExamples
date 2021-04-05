# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 14:51:33 2020

@author: Mingming

tutorial to understand the ECG data set from cardio module
"""

import os, sys
import cardio.batchflow as bf   # the old 'cardio.dataset' has been renamed to this 'cardio.batchflow'
import numpy as np
import matplotlib.pyplot as plt
from cardio import EcgBatch
from cardio import EcgDataset
# https://www.physionet.org/content/qtdb/1.0.0/
# where the data was saved
#SIGNALS_FOLDER = "\\qt-database-1.0.0"                  # the data was saved within the same folder

#SIGNALS_FOLDER = "C:/Users/Mingming/Desktop/Desktoppath_to_QT_database/qt-database-1.0.0"
current_dir = os.getcwd()
SIGNALS_FOLDER = current_dir + "\\qt-database-1.0.0"
SIGNALS_MASK = os.path.join(SIGNALS_FOLDER,  "*.hea")  # .hea, header file, describing signal file contents and format

# when data comes from a file system, it might be convenient to use 'FileIndex',
# it is coming from batchflow module.
index = bf.FilesIndex(path=SIGNALS_MASK,  # directory of the file location
                      no_ext=True,        # when filenames contain extensions which are not part of the id, then they maybe stripped with this option
                      sort=True)          # sort the order of your file index

# now ECG is indexed with its filename without extension, as it is defined by "no_ext" argument of FileIndex
# indices are stored in index.indices
print(index.indices)



dtst  = bf.Dataset(index, batch_class=EcgBatch)  # batch_class holds the data and contains processing functions. refer to documentation 'batch classes'
    




#%%


    
def expand_annotation(annsamp_0, anntype_0, length):
    """
    Unravel annotation
    
    anntype_0 and annsamp_0 are with the same size.
    
    anntype_0:  contain the actual annotation with strings to indicate what each section of the signal is
        ['(', 'p', ')', '(', 'N'...]
    
    annsamp_0: contain the corresponding index of the annotation in anntype_0
        ([ 92, 106, 115, 129, 140 ...])
    
    """
    begin = -1
    end = -1
    s = 'none'
    states = {'N':0, 'st':1, 't':2, 'iso':3, 'p':4, 'pq':5}
    annot_expand = -1 * np.ones(length)

    for j, samp in enumerate(annsamp_0):
        
        # FOR A NEW ANNOTATION STARTS
        if anntype_0[j] == '(': # if the annotation starts with "(", that means a new annotation starts
            begin = samp        # the index that is stored in annsamp_0
            # Determine what is the label number for this coming new annotation string.
            if (end > 0) & (s != 'none'):
                              
                # the previous annotation is 'N', 'N' stands for normal, which is a QRS peak,
                # so the next section from [end:begin] should be 'st', which is after 'N'
                if s == 'N':
                    annot_expand[end:begin] = states['st']               
                               
                # the previous annotation is 't'
                # so the next section from [end:begin] should be 'iso', which is after 't'
                
                elif s == 't': # if current string is 't', 
                    annot_expand[end:begin] = states['iso']
                
                # previous annotation string is 'p', when coming to the if-statement here, the current 
                # string is '(', and the current index is assigned to 'begin'
                # previous end index for previous annotation is already stored in 'end'
                elif s == 'p':                              # since previous annotation string is 'p'
                    annot_expand[end:begin] = states['pq']  # so between last index 'end', to current index 'begin' is 'pq', which is the next section after 'p'
       
        # FOR THE CURRENT ANNOTATION ENDS
        elif anntype_0[j] == ')':    # if the annotation string is "(", that means the current annotation ends
            end = samp     # update the index that is stored in annsamp_0
            # update the last annotation label in this section with the currrent annotation string
            if (begin > 0) & (s != 'none'):
                annot_expand[begin:end] = states[s]
        
       
        # TAKE THE CURRENT ANNOTATION STRING, which is in between the annotation of '(' and ')'.
        else:
            s = anntype_0[j]   # take the current annotation string


    return annot_expand


def get_annsamples(batch):
    return [ann["annsamp"] for ann in batch.annotation]

def get_anntypes(batch):
    return [ann["anntype"] for ann in batch.annotation]


#%%

template_ppl_inits = (
    bf.Pipeline()  # refer pipeline API for more information
                   # https://analysiscenter.github.io/batchflow/api/batchflow.pipeline.html
      .init_variable("annsamps", init_on_each_run=list) # create a variable if not exists, before each run, initiate it a list, a method in Pipeline class
      .init_variable("anntypes", init_on_each_run=list)
      
      .load(       # this is coming from ECGBatch
            fmt='wfdb',        # (optional), source format
            components=["signal",      # from ECGBatch class, store ECG signals in numpy array
                        "annotation",  # from ECGBatch class, array of dicts with different types of annotations, e.g. array of R-peaks
                        "meta"],       # from ECGBatch class, array of dicts with metadata about ECG records, e.g. signal frequency
                        # (str or array like) components to load
            ann_ext='pu1')     # (str, optional), extension of the annotation file
            
   
      .update_variable(                       # update a value of a given variable lazily during pipeline executuion
                      "annsamps",             # name of the variable
                       bf.F(get_annsamples),  # a callable which take a batch (could be a batch class method or an arbitrary function)
                                              # here 'get_annsamples' was defined as a callable before
                       mode='e')              # mode 'e' extend a variable with new value, if a variable is a list  
      
      .update_variable("anntypes", 
                       bf.F(get_anntypes),    # a callable which take a batch (could be a batch class method or an arbitrary function)
                       mode='e')              # mode 'e' extend a variable with new value, if a variable is a list    
      

      .run(  # coming from pipeline
           batch_size=20,    # number of items in the batch
           shuffle=False,    # no conduct random shuffle
           drop_last=False,  # if True, drop the last batch if it contains fewer than batch_size items
           n_epochs=1,       # number of epoches required
           lazy=True) # execute all lazy actions for each batch in the dataset
      
)

ppl_inits = (dtst >> template_ppl_inits).run() # pass dataset to pipeline and run



'''
dtst,      size of 105,105 batches of data
anntype,   size of 105, annotation type for ECG signal
            each batch contains the annotype for each batch of recording, different length
            anntype[0]: has size 10616, ['(','p',')','(','N',')'...]
            anntype[1]: has size 6939
            anntype[2]: has size 7289
            ...
annsamp,   size of 105, index for the annotation type in 'anntype'
            annsamp[0]: has size 10616, [ 92, 106, 115, 129, 140, 150...]
            annsamp[1]: has size 6939
            annsamp[2]: has size 7289
            ...
            
'''        


anntype = ppl_inits.get_variable("anntypes")  # return the variable value
annsamp = ppl_inits.get_variable("annsamps")
total_batches = len(anntype)

batch = dtst.next_batch(total_batches) # ask to load 20 batches of data
batch_with_data = batch.load(fmt='wfdb', components=["signal", "annotation", "meta"])

allsignals = batch_with_data.signal  # retrive all the bactches of data
allmetas   = batch_with_data.meta    # retrive all the meta information for the data


# get the time series length for each batch
lengths      = [my_signal.shape[1] for my_signal in allsignals]

all_expanded = []
# get the expanded annotation for all the batches of signals
for samp, types, length in zip(annsamp, anntype, lengths):
    all_expanded.append(expand_annotation(samp, types, length))


#%%

annsamp_0 = annsamp[0]
anntype_0 = anntype[0]
lengths_0 = lengths[0]
signal_0 = allsignals[0]

expanded_0 = expand_annotation(samp, types, length)


annsamp_0[0:20]
anntype_0[0:20]

#%%  plot a single batch of data with its annotation.

which_batch    = 0
display_start = 50
display_end   = 250
display_length = display_end - display_start

expanded      = all_expanded[which_batch]
my_signal     = allsignals[which_batch][1, display_start:display_end]
my_annotation = expanded[display_start:display_end]

fs = allmetas[which_batch]['fs']                      # for this batch, get its sampling frequency
time_vector = np.arange( 0, display_length/fs, 1/fs)  # time vector for selected signal

plt.figure()
plt.subplot(2,1,1)
plt.plot(time_vector, my_signal,)
plt.title('raw signal')
plt.xticks([]) # hide x axis
plt.ylabel('mV')


# {'N':0, 'st':1, 't':2, 'iso':3, 'p':4, 'pq':5}
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


#%% Segment the data based on each ECG heart beat, from p-QRS-t.
'''
a regular ECG signal starts from P, PQ, QRS, ST, T.
Here, we use this physiological properties of the ECG signal, to segment an entire ECG signals, 
from a super long recording.
'''
single_ECG_list = []       # a big list to store all the ECG signals
single_annotation_list =[] # a big list to store all the annotations for each individual ECG signals


for which_batch in range(len(allsignals)):

    # iterate through each batch to the get raw signals and annotations.
    anntype_batch  = anntype[which_batch]  # this returns a list, with each list has a string, i.e.['(', 'p', ')', '(', 'N', ')'...]
    annsamp_batch  = annsamp[which_batch]  # this returns a list, with each list has index corresponds to anntype
    expanded_batch = all_expanded[which_batch]
    signal_batch   = allsignals[which_batch]
    
    P_locs = np.where(np.asarray(anntype_batch) == 'p')[0]
    T_locs = np.where(np.asarray(anntype_batch) == 't')[0]
    QRS_locs = np.where(np.asarray(anntype_batch) == 'N')[0]
    
    
    for n in range(len(QRS_locs)):   
        
        if n == 0:
            pass
        
        else:
            # in anntype for a batch, a QRS is annotated with  '(', 'N', ')'..., 
            QRS_start = annsamp[which_batch][QRS_locs[n]-1]  # here extract the index for QRS start "("
            QRS_end   = annsamp[which_batch][QRS_locs[n]+1]  # here extract the index for QRS end   ")"
            
            display_start  = QRS_start-50 # 50 data points before the start of QRS, which likely include P wave  
            display_end    = QRS_start+90 # 90 data points after the end of QRS, which likely to include T wave
            display_length = display_end - display_start
        
            my_single_ECG = signal_batch[1, display_start:display_end]  # a single ECG signal
            my_annotation = expanded_batch[display_start:display_end]   # the annotation related to this single ECG signal
        
            single_ECG_list.append(my_single_ECG)
            single_annotation_list.append(my_annotation)


#%% plot an example to see the ECG signal with its annotation
my_single_ECG = single_ECG_list[0]
my_annotation = single_annotation_list[0]

fs = allmetas[which_batch]['fs']                      # for this batch, get its sampling frequency
time_vector = np.arange( 0, display_length/fs, 1/fs)  # time vector for selected signal

plt.figure()
plt.subplot(2,1,1)
plt.plot(time_vector, my_single_ECG)
plt.title('raw signal')
plt.xticks([]) # hide x axis
plt.ylabel('mV')

ax=plt.subplot(2,1,2)
plt.plot(time_vector, my_single_ECG)
plt.title('raw signal with annotation')

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

















