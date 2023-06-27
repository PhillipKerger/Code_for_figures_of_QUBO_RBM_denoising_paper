
"""
This script tests the QUBO denoising method across the test set and different
noise levels against other methods. 
It produces a plot of noise levels versus denoising quality at the end.   

@author: phillip kerger
"""



import numpy as np
np.set_printoptions(precision=4, suppress = True)
import kergerqubo as kq
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from sklearn.neural_network._rbm import BernoulliRBM
from skimage import filters
import seaborn as sns
import pandas as pd



#%% Our main denoising function to use, with lots of optional arguments for flexibility. 
def restore_all_w_qubo (noisydata, sigma, components_, intercept_hidden_, intercept_visible_, anneal_params, 
                approx_sigma =0, approx_absolute = False, bias_factor = 1, fix_rho = None):
    
    
    denoised_images = np.zeros(noisydata.shape)
    for i in range(len(noisydata)):
        noisyimg = noisydata[i,:]
        sigma_to_use = sigma
        
        #if approxing sigma, instead pick random sigma in interval of size approx_sigma*sigma centered at sigma
        if approx_sigma != 0:
            if approx_absolute: 
                sigma_to_use = approx_sigma*(0.5-np.random.rand()) + sigma 
            else: 
                sigma_to_use = approx_sigma*(0.5-np.random.rand())*sigma + sigma 
        
        if bias_factor != 1: 
            sigma_to_use = bias_factor*sigma_to_use
        
        #put sigma in a desired range in case the approx is too big or small
        sigma_to_use = max(sigma_to_use, 0.01*bias_factor)
        sigma_to_use = min(sigma_to_use, 0.5*bias_factor)
                
        #set rho based on the guess of sigma         
        rho = np.log((1-sigma_to_use)/sigma_to_use) #this is our rho 

        #option to just fix rho instead (e.g. rho = 2 works well, from sigma 0.1)        
        if fix_rho != None:
            rho = fix_rho
            
        #use the denoising function from kergerqubo    
        sampled_images = kq.rho_penalty_image_restore(
                noisyimg,
                rho, 
                components_, intercept_hidden_, intercept_visible_, 
                anneal_params)
        
        denoised_images[i,:] = sampled_images[0,:]
    return denoised_images


        
#%% main starts here

#noise range to use
noiseinc = 0.01 #noise increments
noisemin = noiseinc   #noise min value to start at
noisemax = 0.4  #max noise to stop at
sigma_range = np.arange(noisemin, noisemax+noiseinc/2, noiseinc)

#set seed for reproducibility
np.random.seed(1)

#add possibility to test on fewer images
test_on_fewer_images = True
testnum = 200

#will test qubo method with exact, true sigma and with approximate, biased (robust) sigma: 
approx_absolute = False #make the randomness of the guess not scale with sigma if True
approx_sigma = 0.5 #will guess random sigma in interval of size approx_sigma*sigma centered at sigma
#this is to simulate the situation when we only have a rough guess for sigma
bias_factor = 0.75 #bias the guess of sigma for robustness 

#%%  Import data, trained RBM, and build Model
#import the testing data 
test_feats = np.load('BAS_test_feats.npy')
test_labels = np.load('BAS_test_labels.npy')

#initialize BM
rbm = BernoulliRBM(random_state=0, verbose=True) 

pix = 12**2 #number of pixels in the image 
pixroot = round(np.sqrt(pix))

#the testing data
X = test_feats
X = minmax_scale(X, feature_range=(0, 1))  # 0-1 scaling

X = np.round(X)

#get pre-trained weights for BM
folder = 'trained_rbm/'
rbm.components_  = np.load(folder +'bas_rbm_components_.npy')
rbm.intercept_hidden_ = np.load(folder + 'bas_rbm_intercept_hidden_.npy')
rbm.intercept_visible_ = np.load(folder +'bas_rbm_intercept_visible_.npy')  

flip = True

#flip signs if needed (sklearn works with negative)
if flip:    
    [rbm.components_,  rbm.intercept_hidden_, rbm.intercept_visible_] = \
        [-1*rbm.components_, -1*rbm.intercept_hidden_, -1*rbm.intercept_visible_]


#%% Here, create noisy test set and denoise it. 
#record the average overlap each time. 


#declare annealing parameters
#these may not be carefully tuned for best performance, but they do the trick. 
anneal_params = {'start': 0.01}
anneal_params['end'] = 100 #n_steps = 200; 
anneal_params['num_sweeps'] = 5000
anneal_params['num_reads'] = 2
anneal_params['num_results'] = anneal_params['num_reads']
#num_samples = 1
anneal_params['seed'] = 1 #OR for random seed: np.random.randint(0,2**31-1)

#if you want to just test on the same set across different sigmas 
#if test_on_fewer_images == True:
#    X = X[0:testnum,:]
    
    
true_imgs = np.copy(X)

    
overlap_qubo = np.zeros((len(sigma_range), testnum))
overlap_qubo_approxsigma = np.zeros((len(sigma_range), testnum))
overlap_flow = np.zeros((len(sigma_range), testnum))
overlap_median = np.zeros((len(sigma_range), testnum))
overlap_gaussian = np.zeros((len(sigma_range), testnum))
overlap_gibbs = np.zeros((len(sigma_range), testnum))


savedatafolder = 'data_from_full_eval_BAS/' #to save all data from the next loop
#delete that above line to make it faster but not save all the images (true, noisy, denoised)
imgsavefolder = savedatafolder + 'images/'

for i,sigma in enumerate(sigma_range):
     
     #grab a random subset of test images
     X = np.copy(true_imgs)
     np.random.shuffle(X)
     X = X[:testnum] 
     #save the true images
     np.save(imgsavefolder + 'true_images_for_sigma_' +str(round(sigma, 2))+'.npy', X)
     
     #make and save noisy versions
     noisydata = kq.add_salt_and_pepper(X, sigma)
     np.save(imgsavefolder + 'noisy_images_for_sigma_' +str(round(sigma, 2))+'.npy', noisydata)
     
     #Print sigma value in console to see where we are
     print('Sigma:', sigma, '\n')
     
     
     #qubo denoise
     denoised_qubo = restore_all_w_qubo(noisydata, sigma, 
            rbm.components_, rbm.intercept_hidden_, rbm.intercept_visible_, 
            anneal_params)
     for j in range(len(noisydata)):
         overlap_qubo[i, j] = 1-np.mean(np.abs(X[j,:]-denoised_qubo[j,:]))
     np.save(imgsavefolder + 'denoised_qubo_for_sigma_' +str(round(sigma, 2))+'.npy', 
             denoised_qubo)
     
     #qubo denoise with approximate sigma and bias for robustness
     denoised_qubo_approxsigma = restore_all_w_qubo(noisydata, sigma, 
            rbm.components_, rbm.intercept_hidden_, rbm.intercept_visible_, 
            anneal_params, approx_sigma = approx_sigma, approx_absolute = approx_absolute, bias_factor = bias_factor)
     for j in range(len(noisydata)):
         overlap_qubo_approxsigma[i, j] = 1-np.mean(np.abs(X[j,:]-denoised_qubo_approxsigma[j,:]))
     np.save(imgsavefolder + 'denoised_qubo_approxsigma'+str(approx_sigma)+'bias'+str(bias_factor)+'_for_sigma_' +str(round(sigma, 2))+'.npy', 
             denoised_qubo_approxsigma)
     
 
     #median denoise
     denoised_median = np.empty(denoised_qubo.shape)
     for j in range(len(noisydata)):
         img_noisy = np.reshape(noisydata[j,:], (pixroot, pixroot))
         fixedimg = filters.median(img_noisy)
         denoised_median[j,:] = np.reshape(fixedimg, denoised_median[j,:].shape)
         overlap_median[i, j] = 1-np.mean(np.abs(X[j,:]-denoised_median[j,:]))
     np.save(imgsavefolder + 'denoised_median_for_sigma_' +str(round(sigma, 2))+'.npy', 
             denoised_median)
         
     
     
     #gaussian denoise
     denoised_gaussian = np.empty(denoised_qubo.shape)
     for j in range(len(noisydata)):
         img_noisy = np.reshape(noisydata[j,:], (pixroot, pixroot))
         fixedimg = np.round(filters.gaussian(img_noisy))
         denoised_gaussian[j,:] = (np.reshape(fixedimg, denoised_gaussian[j,:].shape))
         overlap_gaussian[i, j] = 1-np.mean(np.abs(X[j,:]-denoised_gaussian[j,:]))
     np.save(imgsavefolder + 'denoised_gaussian_for_sigma_' +str(round(sigma, 2))+'.npy', 
             denoised_gaussian)
     
     
     
    #flow denoise
     denoised_flow = np.empty(denoised_qubo.shape)
     for j in range(len(noisydata)):
         img_noisy = np.reshape(noisydata[j,:], (pixroot, pixroot))
         fixedimg = kq.flow_recover(img_noisy) #can give error if img is pure white (rare)
         denoised_flow[j,:] = (np.reshape(fixedimg, denoised_flow[j,:].shape))
         overlap_flow[i, j] = 1-np.mean(np.abs(X[j,:]-denoised_flow[j,:]))
     np.save(imgsavefolder + 'denoised_flow_for_sigma_' +str(round(sigma, 2))+'.npy', 
             denoised_flow)
     
     #gibbs denoise
     #un-flip signs for using sklearn's rbm.gibbs
     if flip:    
         [rbm.components_,  rbm.intercept_hidden_, rbm.intercept_visible_] = \
             [-1*rbm.components_, -1*rbm.intercept_hidden_, -1*rbm.intercept_visible_]
     denoised_gibbs = np.empty(denoised_qubo.shape)
     gibbs_steps = 20 #Number of gibbs steps 
     alpha = 0.8 #decay factor for averaging
     for j in range(len(noisydata)):
        b = rbm.gibbs(noisydata[j,:])
        x_gibbs = np.zeros(pixroot**2) + np.copy(b)
        for k in range(gibbs_steps):
            b = rbm.gibbs(b)
            x_gibbs += (alpha**(k+1))*b.astype(float) # Averaging the images
        #create the final image based on threshold 
        denoised_gibbs[j,:] = np.where(x_gibbs > 0.5*np.max(x_gibbs), 1, 0)
        overlap_gibbs[i, j] = 1-np.mean(np.abs(X[j,:]-denoised_gibbs[j,:]))
     np.save(imgsavefolder + 'denoised_gibbs_for_sigma_' +str(round(sigma, 2))+'.npy', 
            denoised_gibbs)
     #re-flip signs for QUBO
     if flip:    
         [rbm.components_,  rbm.intercept_hidden_, rbm.intercept_visible_] = \
             [-1*rbm.components_, -1*rbm.intercept_hidden_, -1*rbm.intercept_visible_]

     
#%% plotting with error bars    

sigmas = sigma_range*np.ones((testnum, len(sigma_range)))
sigmas = np.reshape(sigmas.T, (testnum*len(sigma_range),1))

overlap_qubo = np.reshape(overlap_qubo, (sigmas.shape))
overlap_qubo_approxsigma = np.reshape(overlap_qubo_approxsigma, (sigmas.shape))
overlap_gaussian = np.reshape(overlap_gaussian, (sigmas.shape))
overlap_median = np.reshape(overlap_median, (sigmas.shape))
overlap_flow = np.reshape(overlap_flow, (sigmas.shape))
overlap_gibbs = np.reshape(overlap_gibbs, (sigmas.shape))

data = np.concatenate((sigmas, overlap_qubo, overlap_qubo_approxsigma,overlap_gaussian, overlap_median, 
                       overlap_flow, overlap_gibbs), axis = 1)
column_names = ['sigma', 'qubo', 'qubo approxsigma', 'gaussian', 'median',  'minmaxFlow', 'gibbs']
overlap_df = pd.DataFrame(data, columns = column_names)

#seaborn by default uses 95% confidence intervals from 
#bootstrapping to make the error bars. This is appropriate here. 
fig1 = plt.figure()
ax = fig1.add_subplot(1, 1, 1)
sns.lineplot(data=overlap_df, x = 'sigma', y = 'qubo')
sns.lineplot(data=overlap_df, x = 'sigma', y = 'qubo approxsigma')
sns.lineplot(data=overlap_df, x = 'sigma', y = 'gaussian')
sns.lineplot(data=overlap_df, x = 'sigma', y = 'median')
sns.lineplot(data=overlap_df, x = 'sigma', y = 'minmaxFlow')
sns.lineplot(data=overlap_df, x = 'sigma', y = 'gibbs')
ax.legend(['QUBO (known sigma)','_nolegend_',
           'QUBO '+str(round(approx_sigma, 2))+'-approx $\sigma$, bias '+str(bias_factor),'_nolegend_',
           'Gaussian Filter','_nolegend_','Median Filter','_nolegend_',
            'Graph Cut','_nolegend_', 'Gibbs Denoising'])
plt.ylabel('Overlap')
plt.xlabel('noise level $\sigma$')
plt.title('Comparison of Methods on BAS')
plt.savefig(savedatafolder+'figures/'+
            'fig_with_'+'noiseinc'+str(noiseinc)+'testnum'+str(testnum)+'.pdf')

#%% take a look?
'''
testimg = X[1,:]

noisytest = kq.add_salt_and_pepper(testimg, 0.1)

img_noisy = np.reshape(noisytest, (pixroot, pixroot))
fixedimg = filters.median(img_noisy)

plt.figure(figsize = (6,6))
plt.subplot(4,2,1)
plt.imshow(testimg.reshape((pixroot,pixroot)), cmap=plt.cm.gray_r,
               interpolation='nearest')
plt.subplot(4,2,3)
plt.imshow(img_noisy.reshape((pixroot,pixroot)), cmap=plt.cm.gray_r,
               interpolation='nearest')
plt.subplot(4,2,5)
plt.imshow(fixedimg.reshape((pixroot,pixroot)), cmap=plt.cm.gray_r,
               interpolation='nearest')
plt.subplot(4,2,7)
plt.imshow(np.round(fixedimg.reshape((pixroot,pixroot))), cmap=plt.cm.gray_r,
               interpolation='nearest')

'''