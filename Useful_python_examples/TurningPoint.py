# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:47:59 2021

@author: Mingming

Find my turning points


This function is helping to identify the turning point of a curve. It can also be modified to 
detect peaks




"""


def DetectTurning(mysignal, turning_type = 'both', first=False, vis=False):
    
    '''
    input,
        mysignal: 
            input time series.
        turning_type: 
            what type of turning point we want to detect.
            'positive', turning point that forms a curve pointing positive direction
            'negative', turning point that forms a curve pointing negative direction
            'both',     need all turning point to be detected
        first:
            Ture,  Stop detection when it see the first turning point
            False, continue to detect all the request turning points
            
        vis: 
            if True, will plot a figure
    '''

    # starting from 2, 
    # because the 1st and 0th data point shows the sign (moving trend) of a section of curve
    # the 1st and 2nd data point show the sign (moving trend) of the next section of the curve.
    for n in range( 2, len(mysignal) ):
               
        first_diff  = mysignal[n-1] - mysignal[n-2] # the number from first section
        second_diff = mysignal[n] - mysignal[n-1]   # the number from second section
        product     = first_diff*second_diff        # the product of these two sections
    
        # when product is negative, detect a turning point
        if product<0: 
    
            # positive peak
            if turning_type == 'positive' and second_diff<0:   
                turning_list.append(n-1)
                # user request to only detect the first turning point
                if first==True: break
            # negative peak
            elif turning_type == 'negative' and second_diff>0:     
                turning_list.append(n-1)
                # user request to only detect the first turning point
                if first==True: break
            # need both types
            elif turning_type == 'both':
                turning_list.append(n-1)
                # user request to only detect the first turning point
                if first==True: break
    
            
     
    if vis == True:    
        plt.figure()
        plt.plot(mysignal) 
        plt.plot(turning_list, mysignal[turning_list], 'o') 
    

    return turning_list





