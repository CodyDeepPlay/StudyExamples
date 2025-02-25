#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 21:34:59 2025

@author: mingmingzhang


host helper functions

"""

import numpy as np


#%%

'''
Functions that will help to convert "anntype" and "annsamp" into annotation data
that can corresponding to the ECG raw signal recording.
'''    
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




#%%


def seg_single_ECGs(my_signal, one_anntype, one_annsamp, one_expand): 
    '''
    Input:
    one_anntype: 
        annotation from the orignial data, it has string, i.e.['(', 'p', ')', '(', 'N', ')'...]
    one_annsamp: 
        the index of the related annotation in the long time series
    one_expand: 
        recomputed annotation, using 0,1,2,3.. to represent different sections of ECG signal,
        it has a annotation for each data point in the original time series
        
    
    Output:
    single_annotation_list:
        each list element is a single ECG signal
    single_annotation_list:
        each list element is a series of annotation for a single ECG signal
        
    '''
    single_ECG_list = []       # a big list to store all the ECG signals
    single_annotation_list =[] # a big list to store all the annotations for each individual ECG signals

 
    P_locs   = np.where(np.asarray(one_anntype) == 'p')[0]
    T_locs   = np.where(np.asarray(one_anntype) == 't')[0]
    QRS_locs = np.where(np.asarray(one_anntype) == 'N')[0]
    
    
    for n in range(len(QRS_locs)):   
    
        # in anntype for a batch, a QRS is annotated with  '(', 'N', ')'..., 
        QRS_start = one_annsamp[QRS_locs[n]-1]  # here extract the index for QRS start "("
        QRS_end   = one_annsamp[QRS_locs[n]+1]  # here extract the index for QRS end   ")"
        
        display_start  = QRS_start-50 # 50 data points before the start of QRS, which likely include P wave  
        display_end    = QRS_start+90 # 90 data points after the end of QRS, which likely to include T wave
        display_length = display_end - display_start
    
        my_single_ECG = my_signal[ display_start:display_end]  # a single ECG signal
        my_annotation = one_expand[display_start:display_end]   # the annotation related to this single ECG signal
    
        # get rid of the annotations where it contains -1,
        # this is the left over non-used annotation label from previous segmentation
        # "-1" is likely only appear at the beginning of each super long recording
        if -1 in set(my_annotation):
            pass
        else:
            single_ECG_list.append(my_single_ECG)
            single_annotation_list.append(my_annotation)


    return (single_ECG_list, single_annotation_list)




