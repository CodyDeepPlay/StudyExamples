# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 17:28:11 2019

@author: Mingming

HMM model regime detection example.
http://www.blackarbs.com/blog/introduction-hidden-markov-models-python-networkx-sklearn/2/9/2017
"""

import pandas as pd
import pandas_datareader.data as web
import sklearn.mixture as mix

import numpy as np
import scipy.stats as scs

import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator

import seaborn as sns
import missingno as msno
from tqdm import tqdm
p=print

#%%
'''
Using pandas we can grab data from Yahoo Finance and FRED. 
'''
# get fed data

f1 = 'TEDRATE' # ted spread
f2 = 'T10Y2Y' # constant maturity ten yer - 2 year
f3 = 'T10Y3M' # constant maturity 10yr - 3m

start = pd.to_datetime('2002-01-01')
end = pd.datetime.today()

mkt = 'SPY'
MKT = (web.DataReader([mkt], 'yahoo', start, end)['Adj Close']
       .rename(columns={mkt:mkt})
       .assign(sret=lambda x: np.log(x[mkt]/x[mkt].shift(1)))
       .dropna())

data = (web.DataReader([f1, f2, f3], 'fred', start, end)
        .join(MKT, how='inner')
        .dropna()
       )

p(data.head())

# gives us a quick visual inspection of the data
msno.matrix(data)

#%%
'''
Next we will use the sklearn's GaussianMixture to fit a model that estimates these regimes. 
We will explore mixture models  in more depth in part 2 of this series. 
The important takeaway is that mixture models implement a closely related unsupervised form of density estimation. 
It makes use of the expectation-maximization algorithm to estimate the means and covariances of the hidden states (regimes). 
For now, it is ok to think of it as a magic button for guessing the transition and emission probabilities, and most likely path. 

We have to specify the number of components for the mixture model to fit to the time series. 
In this example the components can be thought of as regimes. We will arbitrarily classify the regimes as High, 
Neutral and Low Volatility and set the number of components to three.
'''
# code adapted from http://hmmlearn.readthedocs.io
# for sklearn 18.1

col = 'sret'
select = data.iloc[:].dropna() # remove missing values

ft_cols = [f1, f2, f3, 'sret']
X = select[ft_cols].values

# from sklearn, using mixture Gaussian models to guess the means and covariances of the hidden states
model = mix.GaussianMixture(n_components=3,         # the number of mixture components (the hidden states or latent variables)
                            covariance_type="full", # each component has its own general covariance matrix
                            n_init=100,             # the number of initializations to perform, The best results are kept
                            random_state=7).fit(X)  # if int, is the seed used by the random number generator???

# Predict the optimal sequence of internal hidden state
hidden_states = model.predict(X)

print("Means and vars of each hidden state")
for i in range(model.n_components):                 # for the variables in a particular hidden state
    print("{0}th hidden state".format(i)) 
    print("mean = ", model.means_[i])               # what is the means for this hidden state, one mean for each feature  
    print("var = ", np.diag(model.covariances_[i])) # extract the diagonal values in the covariance matrix,
                                                    # for a single component, there is one variance for each feature
    print()
  
sns.set(font_scale=1.25)    # set aesthetic (plotting) parameters in one step
style_kwds = {'xtick.major.size': 3,               'ytick.major.size': 3,
              'font.family':u'courier prime code', 'legend.frameon': True}
sns.set_style('white', style_kwds)

fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True, figsize=(12,9))
colors = cm.rainbow(np.linspace(0, 1, model.n_components))  # return 3 color maps from rainbow color map, the 3 color maps values are 
                                                            # are even distribute from 0 to 1.

for i, (ax, color) in enumerate(zip(axs, colors)):
    # Use fancy indexing to plot data in each state.
    mask = hidden_states == i
    ax.plot_date(select.index.values[mask], # x axis of the plot, just time information
                 select[col].values[mask],  # y-axis of the plot, selected values 
                 ".-", c=color)
    ax.set_title("{0}th hidden state".format(i), fontsize=16, fontweight='demi')

    # Format the ticks.
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_minor_locator(MonthLocator())
    sns.despine(offset=10)

plt.tight_layout()
# fig.savefig('Hidden Markov (Mixture) Model_Regime Subplots.png')  # save the figure

#%%

sns.set(font_scale=1.5)
states = (pd.DataFrame(hidden_states, columns=['states'], index=select.index)  # construct a data frame with the column name 'states'
          .join(select, how='inner')              # join the newly created column into a existed DataFrame, here is 'select'
          .assign(mkt_cret=select.sret.cumsum())  # cumulative sum of the column 'scret' and assign it to 'mkt_cret' as a new column into this DataFrame
          .reset_index(drop=False)                # reset the index of DataFrame, drop=False: reset the DataFrame index to the default integer index
          .rename(columns={'index':'Date'}))      
p(states.head())                                  # DataFrame.head(), return the first n rows; by default, return the first 5 rows.
 
sns.set_style('white', style_kwds)
order = [0, 1, 2]
# multiple-plot grid for plotting conditional relationships.
fg = sns.FacetGrid(data=states, 
                   hue='states', hue_order=order, # legend for the coloring of the plot
                   palette=colors, aspect=1.31, size=12)
fg.map(plt.scatter, 'Date', mkt, alpha=0.8).add_legend()
sns.despine(offset=10)
fg.fig.suptitle('Historical SPY Regimes', fontsize=24, fontweight='demi')
# fg.savefig('Hidden Markov (Mixture) Model_SPY Regimes.png')  # save the figure
















