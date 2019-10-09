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
import cardio.batchflow as bf
from cardio import EcgBatch
from cardio.batchflow import B, V, F
from cardio.models.hmm import HMModel, prepare_hmm_input

import warnings
warnings.filterwarnings('ignore')

#%% Here are helper functions to generate data for learning, to get values from pipeline, 
# and to perform calculation of initial parameters of the HMM.

def get_annsamples(batch):
    return [ann["annsamp"] for ann in batch.annotation]

def get_anntypes(batch):
    return [ann["anntype"] for ann in batch.annotation]

def expand_annotation(annsamp, anntype, length):
    """Unravel annotation
    """
    begin = -1
    end = -1
    s = 'none'
    states = {'N':0, 'st':1, 't':2, 'iso':3, 'p':4, 'pq':5}
    annot_expand = -1 * np.ones(length)

    for j, samp in enumerate(annsamp):
        if anntype[j] == '(':
            begin = samp
            if (end > 0) & (s != 'none'):
                if s == 'N':
                    annot_expand[end:begin] = states['st']
                elif s == 't':
                    annot_expand[end:begin] = states['iso']
                elif s == 'p':
                    annot_expand[end:begin] = states['pq']
        elif anntype[j] == ')':
            end = samp
            if (begin > 0) & (s != 'none'):
                annot_expand[begin:end] = states[s]
        else:
            s = anntype[j]

    return annot_expand

def prepare_means_covars(hmm_features, clustering, states=[3, 5, 11, 14, 17, 19], num_states=19, num_features=3):
    """This function is specific to the task and the model configuration, thus contains hardcode.
    """
    means = np.zeros((num_states, num_features))
    covariances = np.zeros((num_states, num_features, num_features))
    
    # Prepearing means and variances
    last_state = 0
    unique_clusters = len(np.unique(clustering)) - 1 # Excuding value -1, which represents undefined state
    for state, cluster in zip(states, np.arange(unique_clusters)):
        value = hmm_features[clustering == cluster, :]
        means[last_state:state, :] = np.mean(value, axis=0)
        covariances[last_state:state, :, :] = value.T.dot(value) / np.sum(clustering == cluster)
        last_state = state
        
    return means, covariances

def prepare_transmat_startprob():
    """ This function is specific to the task and the model configuration, thus contains hardcode.
    """
    # Transition matrix - each row should add up tp 1
    transition_matrix = np.diag(19 * [14/15.0]) + np.diagflat(18 * [1/15.0], 1) + np.diagflat([1/15.0], -18)
    
    # We suppose that absence of P-peaks is possible
    transition_matrix[13,14]=0.9*1/15.0
    transition_matrix[13,17]=0.1*1/15.0

    # Initial distribution - should add up to 1
    start_probabilities = np.array(19 * [1/np.float(19)])
    
    return transition_matrix, start_probabilities

#%% First, we need to specify paths to ECG signals:
#  you need to download the data file, https://www.physionet.org/content/qtdb/1.0.0/     
SIGNALS_PATH = "C:/Users/Mingming/Desktoppath_to_QT_database/qt-database-1.0.0"
SIGNALS_MASK = os.path.join(SIGNALS_PATH,  "*.hea")

#%%

index = bf.FilesIndex(path=SIGNALS_MASK, no_ext=True, sort=True)
dtst  = bf.Dataset(index, batch_class=EcgBatch)
dtst.split(0.9)


#path = "User/Documents"
  
# Join various path components  
#print(os.path.join(path, "/home", "file.txt")) 

#%%
import cardio.dataset as ds
index = ds.FilesIndex(path="ecg/tests/data/*.hea", no_ext=True, sort=True)








