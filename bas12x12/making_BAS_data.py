# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 13:08:58 2022

@author: phillips_pc
"""

#%% prelims 
import numpy as np 

#set seed for reproducibility, set other params
np.random.seed(1)
pix = 12**2
n = 4000
features = np.empty((n, pix), dtype = int)
labels = np.empty(n, dtype = int)


#%% functions to make ONE bars or stripes image 
def gen_bars(pix): #this makes bars with pix number of pixels 
    m = round(np.sqrt(pix))
    image = np.zeros((m,m))
    for i in range(m): 
        if np.random.randint(0,2) == 1:
            image[:,i] = np.ones(m)
    return np.ndarray.flatten(image)
        
def gen_stripes(pix): #this makes stripes with pix numbers of pixels 
    m = round(np.sqrt(pix))
    image = np.zeros((m,m))
    for i in range(m): 
        if np.random.randint(0,2) == 1:
            image[i,:] = np.ones(m)
    return np.ndarray.flatten(image)

#%% make data 
#labels first
for i in range(n): 
    labels[i] = np.random.randint(0,2)

for i in range(n):
    if labels[i] == 0:
        features[i, :] = np.copy(gen_bars(pix))
    else:
        features[i, :] = np.copy(gen_stripes(pix))
        
#%% split as train and test, save it
split = round(0.75*n)

train_feats = features[0:split, :]
train_labels = labels[0:split]

test_feats = features[split:, :]
test_labels = labels[split:]


np.save('BAS_train_feats.npy', train_feats)
np.save('BAS_train_labels.npy', train_labels)

np.save('BAS_test_feats.npy', test_feats)
np.save('BAS_test_labels.npy', test_labels)
        



    



