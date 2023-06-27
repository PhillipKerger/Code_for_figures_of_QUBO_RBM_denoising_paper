# -*- coding: utf-8 -*-

import numpy as np
from sklearn.preprocessing import minmax_scale
from PIL import Image
import matplotlib.pyplot as plt


#%% get data 
train_mnist_feats = np.load('MNIST_features.npy')
train_mnist_labels = np.load('MNIST_labels.npy')

test_mnist_feats = np.load('MNIST_features_test.npy')
test_mnist_labels = np.load('MNIST_labels_test.npy')

pixroot = int(np.round(np.sqrt(len(train_mnist_feats[0,:]))))

#normalize images
train_mnist_feats= minmax_scale(train_mnist_feats, feature_range=(0, 255))
test_mnist_feats = minmax_scale(test_mnist_feats, feature_range=(0, 255))

#%% Only grab zeros and ones
'''
trainmask0 = (train_mnist_labels == 0)
trainmask1 = (train_mnist_labels == 1)
trainmask = trainmask0 + trainmask1
train_mnist_feats = train_mnist_feats[trainmask]
train_mnist_labels = train_mnist_labels[trainmask]

testmask0 = (test_mnist_labels == 0)
testmask1 = (test_mnist_labels == 1)
testmask = testmask0 + testmask1
test_mnist_feats = test_mnist_feats[testmask]
test_mnist_labels = test_mnist_labels[testmask]
'''

ntrain = len(train_mnist_labels)
ntest = len(test_mnist_labels)

#%%
train_mnist12x12_feats = np.empty((ntrain, 12**2))
train_mnist12x12_labels = train_mnist_labels

test_mnist12x12_feats= np.empty((ntest, 12**2))
test_mnist12x12_labels = test_mnist_labels

#%% downsize the images to 12x12

#train data
for i in range(ntrain):
    og_img = np.reshape(train_mnist_feats[i,:], (28, 28))
    img = Image.fromarray(np.uint8(og_img))
    img = img.resize((12,12))
    img = np.array(img)
    train_mnist12x12_feats[i,:] = np.reshape(img, (1,12*12))

#test data
for i in range(ntest):
    og_img = np.reshape(test_mnist_feats[i,:], (28, 28))
    img = Image.fromarray(np.uint8(og_img))
    img = img.resize((12,12))
    img = np.array(img)
    test_mnist12x12_feats[i,:] = np.reshape(img, (1,12*12))
    
np.save('mnist12x12/mnist12x12_trainfeats.npy', train_mnist12x12_feats)
np.save('mnist12x12/mnist12x12_trainlabels.npy', train_mnist12x12_labels)

np.save('mnist12x12/mnist12x12_testfeats.npy', test_mnist12x12_feats)
np.save('mnist12x12/mnist12x12_testlabels.npy', test_mnist12x12_labels)

#%% Plot an image
#img = np.round(img/255)
plt.figure(figsize = (6,6))
plt.subplot(4,2,1)
plt.imshow(img.reshape((12,12)), cmap=plt.cm.gray_r,
               interpolation='nearest')

    
    
    
    
    
    
    
    
    
    