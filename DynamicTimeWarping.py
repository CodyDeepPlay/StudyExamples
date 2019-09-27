# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 21:48:02 2018

@author: Mingming

Dynamic Time Wraping study example
https://nipunbatra.github.io/blog/2014/dtw.html

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#%% create example signals and plot the two example signals

x = np.array([1, 1, 2, 3, 2, 0])
y = np.array([0, 1, 1, 2, 3, 2, 1])

plt.figure(1)
plt.plot(x,'r', label='x')
plt.plot(y, 'g', label='y')
plt.legend()

# initialize the space to compute the distance between all pair of points in the two signals
distances = np.zeros((len(y), len(x)))
# using Euclidean distance between the pair of points
for i in range(len(y)):
    for j in range(len(x)):
        distances[i,j] = (x[j]-y[i])**2  


# calculate the distance across all the points and display the distance matrix
# customize a function to visualize the matrix
def distance_cost_plot(distances):
    plt.imshow(distances, interpolation='nearest', cmap='Reds') 
    plt.gca().invert_yaxis()
    plt.xlabel("X, from first to last point")
    plt.ylabel("Y, from first to last point")
    plt.grid()
    plt.colorbar()
    
plt.figure(2)
distance_cost_plot(distances) # plot the matrix
plt.title('distance matrix')
# From this plot here, it seems like the diagonal entries have low distances, 
# which means that the distance between similar index points in x and y is low.
#%%

"""
Warping path
In order to create a mapping between the two signals, 
we need to create a path in the above plot. The path should start at (0,0) and want to reach (M,N) where (M, N) are the lengths of the two signals. 
Our aim is to find the path of minimum distance. To do this, we thus build a matrix similar to the distances matrix. This matrix would contain 
the minimum distances to reach a specific point when starting from (0,0). We impose some restrictions on the paths which we would explore:

The path must start at (0,0) and end at (M,N)
We cannot go back in time, so the path only flows forwards,
which means that from a point (i, j), we can only right (i+1, j) or upwards (i, j+1) or diagonal (i+1, j+1).
These restrictions would prevent the combinatorial explosion and convert the problem to a 
Dynamic Programming problem which can be solved in O(MN) time.
"""

accumulated_cost = np.zeros((len(y), len(x)))

# If we were to move along the first row, i.e. from (0,0) in the right direction only, one step at a time
for i in range(1, len(x)):
    accumulated_cost[0,i] = distances[0,i] + accumulated_cost[0, i-1]  

# If we were to move along the first column, i.e. from (0,0) in the upwards direction only, one step at a time
for i in range(1, len(y)):
    accumulated_cost[i,0] = distances[i, 0] + accumulated_cost[i-1, 0] 


# Accumulated Cost (D(i,j))=min{D(i−1,j−1),D(i−1,j),D(i,j−1)}+distance(i,j)

# calculate all the accumulated cost
for i in range(1, len(y)):
    for j in range(1, len(x)):
        accumulated_cost[i, j] = min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]) + distances[i, j]

plt.figure(3)
distance_cost_plot(accumulated_cost)
plt.title('Accumulated cost matrix')


"""
So, now we have obtained the matrix containing cost of all paths starting from (0,0). We now need to find the path 
minimizing the distance which we do by backtracking.

Backtracking and finding the optimal warp path
Backtracking procedure is fairly simple and involves trying to move back from the last point (M, N) and 
finding which place we would reached there from (by minimizing the cost) and do this in a repetitive fashion.
"""

def path_cost(x, y, accumulated_cost, distances):
    path = [[len(x)-1, len(y)-1]]
    cost = 0
    i = len(y)-1
    j = len(x)-1
    while i>0 and j>0:
        if i==0:
            j = j - 1
        elif j==0:
            i = i - 1
        else:
            if accumulated_cost[i-1, j] == min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]):
                i = i - 1
            elif accumulated_cost[i, j-1] == min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]):
                j = j-1
            else:
                i = i - 1
                j= j- 1
        path.append([j, i])
    path.append([0,0])
    for [y, x] in path:
        cost = cost +distances[x, y]
    return path, cost    

path, cost = path_cost(x, y, accumulated_cost, distances)

path_x = [point[0] for point in path]
path_y = [point[1] for point in path]
plt.figure(3)
distance_cost_plot(accumulated_cost)
plt.plot(path_x, path_y)
plt.title('The lowest accumulated cost path')


# place these two signals on the same plot
plt.figure(4)
plt.plot(x, 'bo-' ,label='x')
plt.plot(y, 'g^-', label = 'y')
plt.legend();
paths = path_cost(x, y, accumulated_cost, distances)[0]
for [map_x, map_y] in paths:
    print( map_x, x[map_x], ":", map_y, y[map_y])    
    plt.plot([map_x, map_y], [x[map_x], y[map_y]], 'r')






