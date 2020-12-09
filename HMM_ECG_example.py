# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:13:33 2019

@author: Mingming

HMM ECG example
https://github.com/analysiscenter/cardio/blob/master/cardio/models/hmm/hmmodel_training.ipynb

"""

import os
import sys

from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm

sys.path.append(os.path.join("..", "..", ".."))
import cardio.batchflow as bf   # the old 'cardio.dataset' has been renamed to this 'cardio.batchflow'
from cardio import EcgBatch

# Refer to this webpage for more information
# https://analysiscenter.github.io/batchflow/intro/named_expr.html
from cardio.batchflow import B, V, F
# B('name')-a batch class attribute or component name
# V('name')-a pipeline variable name
# C('name')-a pipeline config option
# F(...)- a callable which takes a batch (could be a batch class method or an arbitrary function)
from cardio.models.hmm import HMModel, prepare_hmm_input

import warnings
warnings.filterwarnings('ignore')



#%% Here are helper functions to generate data for learning, to get values from pipeline, 
# and to perform calculation of initial parameters of the HMM.

def get_annsamples(batch):
    return [ann["annsamp"] for ann in batch.annotation]

def get_anntypes(batch):
    return [ann["anntype"] for ann in batch.annotation]

    # expand_annotation(samp, types, length)
    # annsamp_0 = samp
    # anntype_0 = types
    
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



def prepare_means_covars(hmm_features, clustering, states=[3, 5, 11, 14, 17, 19], num_states=19, num_features=3):
    """
    This function is specific to the task and the model configuration, thus contains hardcode.
    hmm_features, size of (23624810, 3)
    clustering,   size of (23624810, 1)
    
    """
    means = np.zeros((num_states, num_features))  # size of (num_states 19, num_features 3)
    covariances = np.zeros((num_states, num_features, num_features)) # size of (19, 3, 3)
    
    # Prepearing means and variances
    last_state = 0
    unique_clusters = len(np.unique(clustering)) - 1         # Excluding value -1, which represents undefined state
    for state, cluster in zip(states, np.arange(unique_clusters)):
        value = hmm_features[clustering == cluster, :]       # Get all the values in the features for a particular label (a state)  
                                                             #      values has a size of (N, 3), N is the number of examples found within this cluster
                                                             #      3 is the number of features
        means[last_state:state, :] = np.mean(value, axis=0)  # Got a particular state, average all the values within a feature, and store them in the "mean" matrix
                                                             #      np.mean(value, axis=0) has a size of (1, 3), 3 is the number of features
                                                             #      then, from previous state to current state, all means in each state are assigned to be this mean value
        covariances[last_state:state, :, :] = value.T.dot(value) / np.sum(clustering == cluster)
                                                             # Got a particular state find the clustering, the covariance matrix is (3, 3), because there is 3 features
                                                             #      from previous state to current state, all the covariance matrix is assigned with this matrix
        last_state = state    # in the for loop, update the state                 
        
        
    return means, covariances

def prepare_transmat_startprob():
    """ This function is specific to the task and the model configuration, thus contains hardcode.
    """
    # Transition matrix - each row should add up to 1
    transition_matrix = (np.diag(19 * [14/15.0]) +    # construct a diagonal array, size (number of states 19, 19)
                                                      # for a state to stay within this state, the probability is 0.933
                         np.diagflat(18 * [1/15.0], 1) + # construct a 2-D array with the flattend input as a diagonal
                                                         # for a state to move on to next state, the probability is 0.067
                         np.diagflat([1/15.0], -18)) # baeucase of the matrix, for the last state to move into the first state, prob is 0.067
    
    # We suppose that absence of P-peaks is possible
    transition_matrix[13,14]=0.9*1/15.0
    transition_matrix[13,17]=0.1*1/15.0

    # Initial distribution - should add up to 1
    start_probabilities = np.array(19 * [1/np.float(19)]) # each state intialize with the same probability
    
    return transition_matrix, start_probabilities

#%% First, we need to specify paths to ECG signals:
#  you need to download the data file, https://www.physionet.org/content/qtdb/1.0.0/     
#SIGNALS_PATH = "C:/Users/Mingming/Desktop/Desktoppath_to_QT_database/training2017"
SIGNALS_PATH = "C:/Users/Mingming/Desktop/Desktoppath_to_QT_database/qt-database-1.0.0"
SIGNALS_MASK = os.path.join(SIGNALS_PATH,  "*.hea")  # .hea, header file, describing signal file contents and format

#%%
# when data comes from a file system, it might be convenient to use 'FileIndex',
# it is coming from batchflow module.
index = bf.FilesIndex(path=SIGNALS_MASK,  # directory of the file location
                      no_ext=True,        # when filenames contain extensions which are not part of the id, then they maybe stripped with this option
                      sort=True)          # sort the order of your file index
print(index.indices)
dtst  = bf.Dataset(index, 
                   batch_class=EcgBatch)  # batch_class holds the data and contains processing functions. refer to documentation 'batch classes'
                                          # EcgBatch defines how ECGs are stored and includes actions for ECG processing.
dtst.split(0.9)  # dataset can be split into training and testing

#dtst.index()

# Join various path components  
#print(os.path.join(path, "/home", "file.txt")) 

#%% Preprocessing
'''
We need to calculate initial parameters for the model to ease the learning process.
To do it, we will define template pipeline that loads data and accumulates the data 
we need for initial parameters in pipeline variables:
'''
template_ppl_inits = (
    bf.Pipeline()  # refer pipeline API for more information
                   # https://analysiscenter.github.io/batchflow/api/batchflow.pipeline.html
      .init_variable("annsamps", init_on_each_run=list) # create a variable if not exists, before each run, initiate it a list, a method in Pipeline class
      .init_variable("anntypes", init_on_each_run=list)
      .init_variable("hmm_features", init_on_each_run=list)
      
      .load(       # this is coming from ECGBatch
            fmt='wfdb',        # (optional), source format
            components=["signal",      # from ECGBatch class, store ECG signals in numpy array
                        "annotation",  # from ECGBatch class, array of dicts with different types of annotations, e.g. array of R-peaks
                        "meta"],       # from ECGBatch class, array of dicts with metadata about ECG records, e.g. signal frequency
                        # (str or array like) components to load
            ann_ext='pu1')     # (str, optional), extension of the annotation file
            
      .cwt(        # conduct continous wavelet transform to the signal
           src="signal",       # source to get data from
           dst="hmm_features", # the destination to put the results in, should be string, it is a component name
           scales=[4,8,16],    # wavelet scales to use
           wavelet="mexh")     # what type of wavelet to use during the transform, use 'Mexican hat' wavelet
      
      .standardize(# this is coming from ECGBatch, standardize data along specified axes by removing the mean and scaling to unit variance
              axis=-1,           # along which axis to conduct standardization
              src="hmm_features", # source, batch attribute or component name to get the data from.
              dst="hmm_features") # destination, batch attribute or component name to put the data in.
                   
      .update_variable(                       # update a value of a given variable lazily during pipeline executuion
                      "annsamps",             # name of the variable
                       bf.F(get_annsamples),  # a callable which take a batch (could be a batch class method or an arbitrary function)
                                              # here 'get_annsamples' was defined as a callable before
                       mode='e')              # mode 'e' extend a variable with new value, if a variable is a list  
      
      .update_variable("anntypes", 
                       bf.F(get_anntypes),    # a callable which take a batch (could be a batch class method or an arbitrary function)
                       mode='e')              # mode 'e' extend a variable with new value, if a variable is a list    
      
      .update_variable("hmm_features", 
                       bf.B("hmm_features"),  # batch component,  at each iteration B('features') and B('labels') will be replaced with
                                              # 'current_batch.features' and 'current_batch.labels'
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
Next, we get values from pipeline variables and perform calculation of 
the models' initial parameters: means, covariances, transition matrix and initial probabilities.
'''
lengths      = [hmm_features.shape[2] for hmm_features in ppl_inits.get_variable("hmm_features")]
hmm_features = np.concatenate([hmm_features[0,:,:].T for hmm_features in ppl_inits.get_variable("hmm_features")])
# hmm_features has size of (23624810,3)
# ppl_inits.get_variable("hmm_features") returns a list has a size of (105,1)
#   for each element in this list, it has a size of (2,3,225000)
#       not sure what 2 is ,  
#       3 probably means the scales=[4,8,16] in wavelets data, 
#       225000 may means the time points for this single ECG recording, 15 min long (fs=250 Hz), exacted data points might change for each file in the list
#   for some reason??? the program here only concatenate the first dimesion of this data [hmm_features[0,:,:], don't know why.

'''
for hmm_features in ppl_inits.get_variable("hmm_features"):
    hi = hmm_features


plt.figure(2)
plt.plot(hi[1,2,:])
'''



anntype = ppl_inits.get_variable("anntypes")  # return the variable value
    # anntype, is a list with a length of 105, 
    #  within anntype, each list has signs of patenthesis and annotation characters,
    #  the length is different in each list, i.e. 10616. 

annsamp = ppl_inits.get_variable("annsamps")
    # annsamp, is a list with a length of 105
    #  within annsamp, each list has the location of index, 
    #  it looks like this should be used together with the information in parameter 'anntype'

'''
After this, we expand the annotation with helper function so that each observation has its own label: 
    0 for QRS complex, 
    1 for ST segment, 
    2 for T-wave, 
    3 for ISO segment, 
    4 for P-wave, 
    5 for PQ segment and -1 for all other observations.
'''
expanded = np.concatenate([expand_annotation(samp, types, length) for samp, types, length in zip(annsamp, anntype, lengths)])
# expanded is the annotation of each data points in the signal, hmm_features.
# it has a size of (23624810, 1)

'''
Now using unravelled annotation calculate means and covariances for the states of the model:
And, finally, define matrix of transitions between states and probabilities of the states:
'''
means, covariances = prepare_means_covars(hmm_features, expanded, states = [3, 5, 11, 14, 17, 19], num_features = 3)
# means,       size of (num of states, num of features)
# covariances, size of (num of states, num of features, num of features)
# And, finally, define matrix of transitions between states and probabilities of the states:
transition_matrix, start_probabilities = prepare_transmat_startprob()
# transition_matrix,   size of (num of states, num of states), transition prob between one state to another
# start_probabilities, size of (num of states, 1), initial prob of each state


#%%
# TRAINING, using GaussianHMM()
'''
In config we first specify that we want model to be build from scratch rather than loaded.
 Next, pass the desired estimator and initial parameters that we calculated in previous section.
'''

config_train = {
    'build': True,
     # hidden Markov model with Gaussian emission
    'estimator': hmm.GaussianHMM(n_components=19,         # number of states 
                                 n_iter=25,               # maximum number of iterations to perform
                                 covariance_type="full",  # each state uses a full covariance matrix
                                 random_state=42,         # if assign a number, the data split will not be random, but keep the same way of spliting from trial to trial. 
                                 init_params='',          # controls which parameters are initialized prior to training. Default to all parameters
                                 verbose=False),          # when True, per-iteration convergence reports are printed to sys.stderr. You can diagnose convergence via the monitor_ attribute
    'init_params': {'means_':     means, 
                    'covars_':    covariances, 
                    'transmat_':  transition_matrix,
                    'startprob_': start_probabilities}
                }

'''
Training pipeline consists of the following actions:

Model initialization (building)
Data loading
Preprocessing (e.g. wavelet transformation)
Train step on current batch data
Let's create template pipeline:
'''
ppl_train_template = (
    bf.Pipeline()  # refer pipeline API for more information
      .init_model("dynamic",           # initialize a dynamic model
                  HMModel,             # a model class, 
                  "HMM",               # name of the model class
                  config=config_train) # model configuration parameters, a dictionary with keys and values
                  
      .load( # this is coming from ECGBatch
            fmt='wfdb',         # (optional), source format
            components=["signal",     # in ECGBatch, signal is array of 2D with ECG signals in channels first format
                        "annotation", # array of dicts with different types of annotations 
                        "meta"],      # array of dicts with metadata about signals
                        # most of the ECGBatch actions work under the assumption that both 'signal' and 'meta' components are loaded. 
                        # normal operation of the actions is not gauranteed if this condition is not fullfilled.
            ann_ext='pu1')      # (str, optional), extension of the annotation file
      
      .cwt(  # conduct continous wavelet transform to the signal
           src="signal",        # source where to get the signal from
           dst="hmm_features",  # the destination to put the results in, should be string, it is a component name 
           scales=[4,8,16],     # wavelet scales to use
           wavelet="mexh")      # what type of wavelet to use during the transform, use 'Mexican hat' wavelet
      .standardize(axis=-1, 
                   src="hmm_features",  # source where to get the signal
                   dst="hmm_features")  # destination where to store the results
      .train_model( # train a model
                  "HMM",        # a model name
                   make_data=partial(  # partial is a python function to allow us to fix a certain number of arguments of a function and generate a new function
                                     prepare_hmm_input,        # the function that partial() is going to generate, with its using parameter in the following
                                     features="hmm_features", 
                                     channel_ix=0))       # from 'Source code for cardio.models.hmm.hmm', index of channel, which data should be used in training and predicting
      .run(batch_size=20, shuffle=False, drop_last=False, n_epochs=1, lazy=True)
)

# now link the training data with the model
ppl_train = (dtst >> ppl_train_template).run()

#%%
#From resulting pipeline we can save the trained model to file.
model_save_path = 'C:/Users/Mingming/Desktop/Work projects/Veressa/Python_repo/Study examples'
model_path = os.path.join(model_save_path,  "hmmodel.dill")

ppl_train.save_model("HMM", path=model_path)

#%% PREDICTION, recall the previously saved model


config_predict = {
    'build': False,
    'load': {'path': model_path} #"hmmodel.dill"}
                 }
'''
Prediction pipeline is somewhat more complex and composed of:

Model initialization (loading)
Initialization of pipeline variable
Data loading
Preprocesing
Calculation of ECG parameters, such as HR
Update of the variable
'''

template_ppl_predict = (
    bf.Pipeline()
      .init_model("static", HMModel, "HMM", config=config_predict)   # indicate that we want to load the pre-trained model
      .load(fmt="wfdb", components=["signal", "annotation", "meta"], ann_ext="pu1")
      .cwt(src="signal", dst="hmm_features", scales=[4,8,16], wavelet="mexh")
      .standardize(axis=-1, src="hmm_features", dst="hmm_features")
      .predict_model("HMM",      # predict using a model, model name
                     # partial() makes a callable object
                     make_data=partial(prepare_hmm_input, features="hmm_features", channel_ix=0),
                     save_to=bf.B("hmm_annotation"), mode='w') 
      .calc_ecg_parameters(src="hmm_annotation")
      .run(batch_size=20, shuffle=False, drop_last=False, n_epochs=1, lazy=True)
)

# Let's link pipieline to the dataset:
ppl_predict = (dtst >> template_ppl_predict)

# Now we can generate new batches with next_batch() method of the pipeline. Let's take a look at the first one:
batch = ppl_predict.next_batch()
#%%
# Next, select fifth and ninth signal from that batch:


example_1 = 5
example_2 = 9
# plot an ECG signal, optionally highlight QRS complexes along with P and T waves.
# Each channel is displayed on a separate subplot.
# source code for show_ecg()
# https://analysiscenter.github.io/cardio/_modules/cardio/core/ecg_batch.html#EcgBatch.show_ecg
batch.show_ecg(batch.indices[example_1],   # index, index of a signal to plot
               5,                 # the start point of the displayed part of the signal, in seconds
               10,                # the end point of the displayed part of the signal, in seconds 
               "hmm_annotation")  # specifies attribtute that stores annotation obtained from cardio.models.HMModel
print("Heart rate: %d bpm" %batch.meta[example_1]["hr"])


batch.show_ecg(batch.indices[example_2], 0, 5, "hmm_annotation")
print("Heart rate: %d bpm" %batch.meta[example_2]["hr"])



#%%

# extract the raw recording of the signal
mysig= batch.signal[example_1]  # get the first example within this batch,
                        # a single exmaple has two channels of recording, they are the same length

myfeature = batch.hmm_features[example_1]  # get the features of the first example

end_point  = 10000 # end point of the signal choose to display
plt.figure()
plt.subplot(3,1,1)
plt.plot(mysig[0, 0:end_point])  # look at the first channel of recording, 0 to 2500 points
plt.xlim([0, end_point])

# look at the feature from the hmm_feature
plt.subplot(3,1,2)
plt.imshow(myfeature[0], aspect='auto', extent = [0 , end_point,  1, 3], cmap='RdBu')
plt.ylim([1,3])


import pywt
fs=250

#%
length = len(mysig)
scales = np.arange(1, 20)
# continuous wavelet transform, return: 
#   features, are the wavelet transform coefficients, 
#   freq,     is the corresponding frequency for each scale.
dt = 1/fs

# try to use the same scales as what the hmm_features uses
features, frq = pywt.cwt(mysig, scales = [4,8,16], wavelet='mexh', sampling_period=dt)
frequencies = pywt.scale2frequency('mexh', [1]) / dt

plt.subplot(3,1,3)
plt.imshow(features[:, 0:end_point], aspect='auto', extent =  [0 , end_point,  1, 3], cmap='RdBu')
plt.show()
plt.xlabel('ms')
plt.ylabel('Frequency (To be adjusted)')



