# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 10:06:58 2020

@author: Mingming

ECG data segmentation
"""


import os, sys
import cardio.batchflow as bf   # the old 'cardio.dataset' has been renamed to this 'cardio.batchflow'
import numpy as np
import matplotlib.pyplot as plt
from cardio import EcgBatch
from cardio import EcgDataset
import datetime

import tensorflow as tf   # tf.__version__ : '2.0.0'
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
#%%

# https://www.physionet.org/content/qtdb/1.0.0/
# where the data was saved
#SIGNALS_FOLDER = "\\qt-database-1.0.0"                  # the data was saved within the same folder

current_dir = os.getcwd()
SIGNALS_FOLDER = current_dir + "\\qt-database-1.0.0"
SIGNALS_MASK = os.path.join(SIGNALS_FOLDER,  "*.hea")  # .hea, header file, describing signal file contents and format

# when data comes from a file system, it might be convenient to use 'FileIndex',
# it is coming from batchflow module.
index = bf.FilesIndex(path=SIGNALS_MASK,  # directory of the file location
                      no_ext=True,        # when filenames contain extensions which are not part of the id, then they maybe stripped with this option
                      sort=True)          # sort the order of your file index

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
        # in anntype for a batch, a QRS is annotated with  '(', 'N', ')'..., 
        QRS_start = annsamp[which_batch][QRS_locs[n]-1]  # here extract the index for QRS start "("
        QRS_end   = annsamp[which_batch][QRS_locs[n]+1]  # here extract the index for QRS end   ")"
        
        display_start  = QRS_start-50 # 50 data points before the start of QRS, which likely include P wave  
        display_end    = QRS_start+90 # 90 data points after the end of QRS, which likely to include T wave
        display_length = display_end - display_start
    
        my_single_ECG = signal_batch[1, display_start:display_end]  # a single ECG signal
        my_annotation = expanded_batch[display_start:display_end]   # the annotation related to this single ECG signal
    
        # get rid of the annotations where it contains -1,
        # this is the left over non-used annotation label from previous segmentation
        if -1 in set(my_annotation):
            pass
        else:
            single_ECG_list.append(my_single_ECG)
            single_annotation_list.append(my_annotation)


# reorganize all the data into numpy array

each_recording_length = len(single_ECG_list[0]) # take the first recording as example to see how long one recording is 

All_ECG_data       = np.zeros((len(single_ECG_list), each_recording_length))     # signal and annotation should have the same size
All_ECG_annotation = np.zeros((len(single_annotation_list), each_recording_length))

for n in range(len(single_annotation_list)):    
    All_ECG_data[n,:]       = single_ECG_list[n]
    All_ECG_annotation[n,:] = single_annotation_list[n]


#%% USING LSTM and raw signal to conduct signal segmentation. 



def PrepareRawDatalstm(myAllData, myAllLabels,):        
        
    (num_rec, rec_len) = myAllData.shape  # number of observations for EMG data, and length for each recording.
    
    X = myAllData
    y = myAllLabels
    
    
    # reshape the data to meet the lstm layer input format requirement.
    lstm_data  = X.reshape(num_rec, rec_len, 1)  # [batch, timesteps, feature]
    lstm_label = y.reshape(num_rec, rec_len, 1)  # [batch, timesteps, label]
    
    X_train, X_test, y_train, y_test = train_test_split(lstm_data, lstm_label, test_size=0.2)  # 20% of the total data left as testing data
    
    # get some data within training data, to be used as validation during training.
    num_validation = 100
    start_index    = 500
    mask = range(start_index, start_index + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
            
    return X_train, y_train, X_val, y_val, X_test, y_test






def bilstm_raw(rec_len, num_classes):
    '''
    use a bidirectional lstm layer
    '''
    
    #rec_len = 140
    input_shape = (rec_len, 1)
    hidden_size = 250
    
    initializer = tf.initializers.VarianceScaling(scale=2.0)
    regularizer = tf.keras.regularizers.l2(0.02)
    layers = [
        tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(input_shape=input_shape,
                             units = hidden_size,
                             kernel_initializer=initializer,
                             kernel_regularizer=regularizer,
                             return_sequences=True,  # output in the sequence
                             dropout=0.3,  # fraction of the units to drop for the linear transformation of the inputs
                             )),        
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        
        
        tf.keras.layers.Dense(num_classes, activation='softmax',
                              kernel_initializer=initializer),
        ]

    model = tf.keras.Sequential(layers)
    return model


def lstm_raw(rec_len, num_classes):
    '''
    use a unidirectional lstm layer, not bidirectional
    '''
    #rec_len = 140
    input_shape = (rec_len, 1)
    hidden_size= 250
    
    initializer = tf.initializers.VarianceScaling(scale=2.0)
    regularizer = tf.keras.regularizers.l2(0.02)
    layers = [   
        tf.keras.layers.LSTM(input_shape=input_shape,
                             units = hidden_size,
                             kernel_initializer=initializer,
                             kernel_regularizer=regularizer,
                             return_sequences=True,  # output in the sequence
                             dropout=0.3,  # fraction of the units to drop for the linear transformation of the inputs
                             ),                
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),       
        tf.keras.layers.Dense(num_classes, activation='softmax',
                              kernel_initializer=initializer),
        ]

    model = tf.keras.Sequential(layers)
    return model  


#%%

directory= current_dir # at the beginning, we already get the current working directory

feature = "raw"
X_train, y_train, X_val, y_val, X_test, y_test = PrepareRawDatalstm(All_ECG_data, All_ECG_annotation,)

rec_len = X_train.shape[1]
num_classes = len(set(All_ECG_annotation[0]))

model = lstm_raw(rec_len, num_classes)

# build a learning decay schedule, use this during optimizer building.
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(  # use exponetial learning rate decay during training
    initial_learning_rate= 0.01,
    decay_steps=100,  # decay the learning rate every given decay_steps
    decay_rate=0.96,  # every time for decay learning rate, decay with decay_rate of exponential decay base.
    staircase=False)


model.compile(
    optimizer=optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True),  #decay=1e-6,
    loss='sparse_categorical_crossentropy',
    metrics=[tf.keras.metrics.sparse_categorical_accuracy])

tensorboard_cbk = tf.keras.callbacks.TensorBoard(
                              log_dir=directory,    # directory where to write logs
                              histogram_freq=0,     # How often to log histogram visualizations
                              embeddings_freq=0,    # How often to log embedding visualizations
                              update_freq='epoch')  # How often to write logs (default: once per epoch)

num_epoches = 4
model.fit(X_train, y_train, batch_size=32, epochs=num_epoches, shuffle=True, 
          callbacks=[tensorboard_cbk],     # use tensorboard call back to visualize the training process
          validation_data=(X_val, y_val))



# plot the training history to see the training procees, in terms of loss and acc.
epoches = range(num_epoches)
train_loss = model.history.history['loss']
train_acc  = model.history.history['sparse_categorical_accuracy']
val_loss   = model.history.history['val_loss']
val_acc    = model.history.history['val_sparse_categorical_accuracy']

plt.figure()
plt.subplot(2,1,1)
plt.plot(epoches, train_loss, '--.')
plt.plot(epoches, val_loss, '--.')
plt.legend(['Train loss', 'val loss'])

plt.subplot(2,1,2)
plt.plot(epoches, train_acc, '--.')
plt.plot(epoches, val_acc, '--.')
plt.legend(['Train acc', 'val acc'])



# the traing data is applied on the testing data
loss, acc = model.evaluate(X_test, y_test)    # evaluate function returns loss and overall accuracy
mypred    = model.predict(X_test)             # predict return the score for each class, for each time point, for each recording example
mypred_labels=np.argmax(mypred, axis=2)


# save the mode if you need to.
SAVE_MODEL = False
if SAVE_MODEL:
    dateinfo = str( datetime.datetime.now() )                   # get current time staps 
    new_folder = 'lstm_seg_'+ feature+dateinfo[0:4] +  dateinfo[5:7]+ dateinfo[8:10] + '-' + dateinfo[11:13] + dateinfo[14:16]
    
    filepath = directory + '\\' + new_folder +'\\'  # create a new folder each time for new training
    tf.saved_model.save(model, filepath)  # this will create a folder and save models there





#%%
#------ USE KERAS saved_model.load_model() method to load the model ----------#
loaded_model = tf.keras.models.load_model(filepath)
mypred  = loaded_model.predict(X_test)
mypred_labels=np.argmax(mypred, axis=2)
loss, acc  = loaded_model.evaluate(X_test, y_test, verbose=0)

#%% plot an example to see how the sigmentation looks like
n = 249
my_signal      = X_test[n]
my_annotation  = y_test[n]

mypred = loaded_model.predict(my_signal[np.newaxis,:,:])
mypred_annotation=np.argmax(mypred, axis=2)[0]

# just take the fs from the first batch, the sampling rate is the same for all data batches.
which_batch = 0
fs = allmetas[which_batch]['fs']  # for this batch, get its sampling frequency
     
display_length = my_signal.shape[0]
time_vector = np.arange( 0, display_length/fs, 1/fs)  # time vector for selected signal


plt.figure()
# {'N':0, 'st':1, 't':2, 'iso':3, 'p':4, 'pq':5}
plt.subplot(2,1,1)
plt.plot(time_vector, my_signal,)
plt.title('true annotation')
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


# {'N':0, 'st':1, 't':2, 'iso':3, 'p':4, 'pq':5}
plt.subplot(2,1,2)
plt.plot(time_vector, my_signal,)
plt.title('predicted annotation')
myN = np.where(mypred_annotation==0)[0]
plt.scatter(time_vector[myN], my_signal[myN], color='r', marker='o' )

myst = np.where(mypred_annotation==1)[0]
plt.scatter(time_vector[myst], my_signal[myst], color='g', marker='o')

myt = np.where(mypred_annotation==2)[0]
plt.scatter(time_vector[myt], my_signal[myt], color='b', marker='o')

myiso = np.where(mypred_annotation==3)[0]
plt.scatter(time_vector[myiso], my_signal[myiso], color='k', marker='o')

myp = np.where(mypred_annotation==4)[0]
plt.scatter(time_vector[myp], my_signal[myp], color='c', marker='o')

mypq = np.where(mypred_annotation==5)[0]
plt.scatter(time_vector[mypq], my_signal[mypq], color='y', marker='o')
plt.ylabel('mV')
plt.xlabel('time (s)')