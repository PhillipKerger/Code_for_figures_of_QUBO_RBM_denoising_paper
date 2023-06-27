

"""
A Python script using the sklearn BernoulliRBM. 
Gets MNIST digits, then fits RBM to model their distribution and generates some.
Original code is here: 
https://scikit-learn.org/stable/auto_examples/neural_networks/plot_rbm_logistic_classification.html#sphx-glr-download-auto-examples-neural-networks-plot-rbm-logistic-classification-py
Original Authors Authors: Yann N. Dauphin, Vlad Niculae, Gabriel Synnaeve
Modified by Phillip Kerger for GRIPS Sendai use. 
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import minmax_scale
from sklearn.base import clone
from sklearn.datasets import fetch_openml
np.set_printoptions(precision=4, suppress = True)



# #############################################################################
# Setting up
pix = 144
pixroot = round(np.sqrt(pix))


#%% Load Data
mnist12_train_feats = np.load('mnist12x12_trainfeats.npy')
mnist12_train_labels = np.load('mnist12x12_trainlabels.npy') #I don't need the labels here 



# #############################################################################
#%% Training

#can go up to 160 hidden units with pegasus. Hopefully can do less though
n_components = 64
learning_rate = 0.01
batch_size = 50
n_iter = 200

n_components = 64
learning_rate = 0.01
batch_size = 50
n_iter = 150

rbm = BernoulliRBM(n_components=n_components, learning_rate=learning_rate,
                   batch_size=batch_size, n_iter=n_iter,
                   random_state=1, verbose=1)
X = mnist12_train_feats
X = minmax_scale(X, feature_range=(0, 1))  # 0-1 scaling
rbm.fit(X)

np.save('mnist_trained_rbm/mnist12_rbm_components_.npy', rbm.components_)
np.save('mnist_trained_rbm/mnist12_rbm_intercept_hidden_.npy', rbm.intercept_hidden_)
np.save('mnist_trained_rbm/mnist12_rbm_intercept_visible_.npy', rbm.intercept_visible_)

#%% Sampling, Plotting
'''
#Do some (Gibbs) sampling on the fitted model 
num_samples = 10 #how many samples?
gibbs_steps = 1000 #how many Gibbs steps for each sample?
for j in range(num_samples):
    v = np.random.randint(0, 2, pix)
    for i in range (gibbs_steps):
        v = rbm.gibbs(v)
    #filename = str(j)+'th_generated_digit.pdf'
    foldername = 'YOUR DATA FOLDER NAME/'
    filename = foldername + str(j)+'th_generated_img.jpg'
    
    #plt.figure()
    plt.imsave(filename, v.reshape((pixroot,pixroot)), cmap=plt.cm.gray_r)
    
print('We used', n_components, 'hidden nodes.')
'''

'''
#visualize the components of the data that the BM learned
plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(rbm.components_):
    plt.subplot(np.ceil(int(rbm.n_components/10)), 10, i +1)
    plt.imshow(comp.reshape((pixroot,pixroot)), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('Components extracted by RBM', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

plt.savefig('components.pdf')
'''
    
'''
num_samples = 1
gibbs_steps = 600
for j in range(num_samples):
    v = np.random.randint(0, 2, 28*28)
    for i in range (gibbs_steps):
        v = rbm.gibbs(v)
        if i%100 == 0: 
            plt.figure()
            plt.imshow(v.reshape((28,28)), cmap=plt.cm.gray_r,
                       interpolation='nearest')
#'''

'''
pointnum = 6 #which number data point of handwritten image to plot
plt.figure()
plt.imshow(X[pointnum,:].reshape((28,28)), cmap=plt.cm.gray_r,
               interpolation='nearest')
print('The plot should be a ', Y[pointnum])
#'''
