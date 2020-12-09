# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 11:13:29 2020

@author: Mingming

example of how to conduct power analysis.

This file contain an example of how to conduct power analysis to estimate an 
sample size for a study.

https://towardsdatascience.com/introduction-to-power-analysis-in-python-e7b748dfa26

"""

import numpy as np
import pandas as pd

from statsmodels.stats.power import TTestIndPower
from scipy.stats import ttest_ind

import seaborn as sns
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *


#%%

# parameters for the analysis 

'''
how to measure effect_size using different methods.
'''

effect_size = 0.8

'''
alpha
significance level, 
the probability of rejecting the null hypothesis(H0) when it is in fact true.
'''
alpha = 0.05 
power = 0.8

power_analysis = TTestIndPower()
sample_size = power_analysis.solve_power(effect_size = effect_size, 
                                         power = power, 
                                         alpha = alpha)

print('Required sample size: {0:.2f}'.format(sample_size))