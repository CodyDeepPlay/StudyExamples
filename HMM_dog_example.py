# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 17:59:15 2019

@author: Mingming

This is a study example, from the online blog located here
http://www.blackarbs.com/blog/introduction-hidden-markov-models-python-networkx-sklearn/2/9/2017
to understand HMM model from mathmatical equations and examples, refer to this link
http://www.cim.mcgill.ca/~latorres/Viterbi/va_alg.htm

"""


import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
#%matplotlib inline

# create state space and initial state probabilities

states = ['sleeping', 'eating', 'pooping']
p_observ = [0.35, 0.35, 0.3]
state_space = pd.Series(p_observ, index=states, name='states')
print(state_space)
print(state_space.sum())


# create transition matrix
# equals transition probability matrix of changing states given a state
# matrix is size (M x M) where M is number of states

q_df = pd.DataFrame(columns=states, index=states)
q_df.loc[states[0]] = [0.4, 0.2, 0.4]
q_df.loc[states[1]] = [0.45, 0.45, 0.1]
q_df.loc[states[2]] = [0.45, 0.25, .3]

print(q_df)

q = q_df.values
print('\n', q, q.shape, '\n')
print(q_df.sum(axis=1))


# create state space and initial state probabilities

hidden_states = ['healthy', 'sick']
p_hidden_states = [0.5, 0.5]     
state_space = pd.Series(p_hidden_states, index=hidden_states, name='states')
print(state_space)
print('\n', state_space.sum())


# create hidden transition matrix
# a or alpha 
#   = transition probability matrix of changing states given a state
# matrix is size (M x M) where M is number of states

a_df = pd.DataFrame(columns=hidden_states, index=hidden_states)
a_df.loc[hidden_states[0]] = [0.7, 0.3]
a_df.loc[hidden_states[1]] = [0.4, 0.6]

print(a_df)

p_hiddden_trans = a_df.values   # probability change from one hidden state to another hidden state
print('\n', p_hiddden_trans, p_hiddden_trans.shape, '\n')
print(a_df.sum(axis=1))


# create matrix of observation (emission) probabilities
# b or beta = observation probabilities given a state
# matrix is size (M x O) where M is number of states 
# and O is number of different possible observations

observable_states = states

b_df = pd.DataFrame(columns=observable_states, index=hidden_states)
b_df.loc[hidden_states[0]] = [0.2, 0.6, 0.2]
b_df.loc[hidden_states[1]] = [0.4, 0.1, 0.5]

print(b_df)

p_emission = b_df.values   # emission probability, given a hidden state, what is the prbability for an certain observation
print('\n', p_emission, p_emission.shape, '\n')
print(b_df.sum(axis=1))


# observations are encoded numerically

obs_map = {'sleeping':0, 'eating':1, 'pooping':2}
obs = np.array([1,1,2,1,0,1,2,1,0,2,2,0,1,0,1])

inv_obs_map = dict((v,k) for k, v in obs_map.items())
obs_seq = [inv_obs_map[v] for v in list(obs)]

print( pd.DataFrame(np.column_stack([obs, obs_seq]), 
                columns=['Obs_code', 'Obs_seq']) )


#%%
# define Viterbi algorithm for shortest path
# code adapted from Stephen Marsland's, Machine Learning An Algorthmic Perspective, Vol. 2
# https://github.com/alexsosn/MarslandMLAlgo/blob/master/Ch16/HMM.py

def viterbi(p_hidden_states, p_hiddden_trans, p_emission, obs):
    
    '''
    p_hidden_states: 
        probability of all the hidden states, prior
    p_hiddden_trans: 
        hidden transition matrix, transition probability matrix of 
        change states given a state
    
    p_emission: 
        create matrix of observation (emission) probabilities
        conditional probabilty: P( an observation | a state )
        each row is a single state, each column is an observation
        this matrix tells the probability of an observation given a state
        so it is also called the emission probabilities.
        likelihood
    
    obs: observations
    '''
    
    nStates = np.shape(p_emission)[0]        # number of states
    T = np.shape(obs)[0]            # number of total observation we received, a sequence data
    
    # init blank path
    path = np.zeros(T)              # the path should be the same length as the observation
    # delta --> highest probability of any path that reaches state i
    delta = np.zeros((nStates, T))  # size is [# of states, # of observations]
    # phi --> argmax by time step for each state
    phi = np.zeros((nStates, T))    # size is [# of states, # of observations]
     
    # init delta and phi 
    phi[:, 0] = 0
    delta[:, 0] = p_hidden_states * p_emission[:, obs[0]] # state probability * the emission probability of the first observation
    
    '''
    delta: proportional to P(states|observations), posterior
    
    Bayes' rule:
        P(states|observations) = P(observations|states)*P(states) / P(observations)
    
    P(states|observations) is called a 'posterior'
    P(states) is called a 'priori'
    P(observations|states) is called a likelihood
    because P(observations) is independent of the states,
    
    so, maximum a posterior (MAP) is the same as maximizing the following:
        P(observations|states)*P(states)  
    '''
    
    
    print('\nStart Walk Forward\n')    
    # the forward algorithm extension
    for t in range(1, T):  # iterate through all the time points, starting from the 2nd one
        
        # within each time point, iterate through all the states
        for s in range(nStates):
            
            
            # for a given time point t, 
            # 'delta[:, t-1]' is the posterior of previous time point
            # at time point t-1, what is the probability of an observation in each hidden state
            #    proportional to P(states|a_observation)
            
            # 'p_hiddden_trans' is the hidden transition matrix between different states.
            #   'p_hiddden_trans[:,s]' is given the hidden state s, what is the probability of transfering to all hidden state at next time point 
            #    P(states|a_state) 
            
            # 'delta[:, t-1] * p_hiddden_trans[:, s]' gives you the 
            #  P(states|a_observation)*P(states|a_state)
            #  from one time to another time, state might change, so this operation gives you
            #    the new P(states) at current time t .   
            
            # 'np.max(delta[:, t-1] * p_hiddden_trans[:, s])', gives you the maximum probability of which state at this time point t
                     
            # 'p_emission[s, obs[t]]' is for an observation probability of a state, given a hidden state at time t, 
            #    p(a_observation|states), likelihood
            
            delta[s, t] = np.max(delta[:, t-1] * p_hiddden_trans[:, s]) * p_emission[s, obs[t]] # the maximum probability reaaching a state at time t for this observation
                          
            phi[s, t] = np.argmax(delta[:, t-1] * p_hiddden_trans[:, s]) # record the index of maximum within the new P(states)
            print('s={s} and t={t}: phi[{s}, {t}] = {phi}'.format(s=s, t=t, phi=phi[s, t]))
    
    
    # find optimal path
    print('-'*50)
    print('Start Backtrace\n')
    path[T-1] = np.argmax(delta[:, T-1])  # the last time point, reach to which hidden state has the highest probability
    #p('init path\n    t={} path[{}-1]={}\n'.format(T-1, T, path[T-1]))
    for t in range(T-2, -1, -1):
        path[t] = phi[int(path[t+1]), [t+1]]
        #p(' '*4 + 't={t}, path[{t}+1]={path}, [{t}+1]={i}'.format(t=t, path=path[t+1], i=[t+1]))
        print('path[{}] = {}'.format(t, path[t]))
        
    return path, delta, phi

path, delta, phi = viterbi(p_hidden_states, 
                           p_hiddden_trans, 
                           p_emission, 
                           obs)
print('\nsingle best state path: \n', path)
print('delta:\n', delta)
print('phi:\n', phi)


#%%
state_map = {0:'healthy', 1:'sick'}
state_path = [state_map[v] for v in path]

(pd.DataFrame()
 .assign(Observation=obs_seq)
 .assign(Best_Path=state_path))