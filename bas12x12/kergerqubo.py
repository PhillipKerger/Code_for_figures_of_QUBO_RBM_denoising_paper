# -*- coding: utf-8 -*-
"""
A module of functions to work across PyQUBO, Sklearn's BernoulliRBM, and d-wave's neal module. 

@author: Phillip Kerger
"""

import sys
sys.path.append('/import/mqhome/grips01/VectorAnnealing/libexec/VectorAnnealing/python/')
import numpy as np
import neal
import kergerqubo as kq
import igraph


#working as intended!!
def rho_penalty_image_restore(noisyimg, rho, components_, intercept_hidden_,
                              intercept_visible_, anneal_params):
    '''
    Parameters
    ----------
    noisyimg : ARRAY
        Flattened noisy image to be restored.
    rho : Float
        Penalty adjustment for changing a pixel from the noisy image.
    components_, intercept_hidden_, intercept_visible_: array 
        The trained RBM to be used to restore the image.
        If rbm is the sklearn trained RBM, these are rbm.components_, rbm.intercept_hidden_,...
    anneal_params : dictionary
        a dictionary containing annealing parameters.
        must have entries for 'start', 'end', 'num_sweeps', 'num_reads',  and 'seed'
        note that init_spin is made from the noisy image. 

    Returns
    -------
    sampled_images : Array
        Each row is an reconstructed image.

    '''

    penalty_adj = rho*(-2*(noisyimg)+1)
    intercept_visible_mod = intercept_visible_ + penalty_adj
     
    AP = anneal_params
    #Parameters
    start = AP['start']; end = AP['end']; # n_steps = 200; 
    num_sweeps = AP['num_sweeps']
    num_reads = AP['num_reads']
    num_results = num_reads
    #num_samples = 1
    seed = AP['seed']
    beta_range = (start, end)
    
    init_spin = kq.init_spin_from_image(noisyimg)
    
    sampled_images = kq.neal_sample_visibles(
            components_, 
            intercept_hidden_,
            intercept_visible_mod,
            beta_range = beta_range,
            num_reads = num_reads,
            num_sweeps = num_sweeps,
            num_results = num_results, 
            init_spin = init_spin,
            seed = seed)
    
    return sampled_images




def neal_sample_visibles(
        components_, 
        intercept_hidden_,
        intercept_visible_,
        beta_range = None,
        num_reads = 1,
        num_sweeps = 2000,
        num_results = 1, 
        init_spin = None, 
        seed = np.random.randint(0,2**31-1),
        ve = None, vector_mode = 'Speed'):
    '''
    This samples the visible nodes from the given rbm with SimulatedAnnealingSampler.
    The sampling parameters are determined by the arguments.
    In particular, we can either sample or optimize depending on choice of beta_range. 


    Parameters
    ----------
    components_ : m x n ARRAY
            weights of hidden to visible connections in the rbm as m x n array. 
    intercept_hidden_ : m x 1 ARRAY
        offset for the qubo model of the hidden nodes, or partial diagonal of the Q matrix.
    intercept_visible_ : n x 1 ARRAY
        offset for the qubo model of the visible nodes, or remainder of the diagonal. 
    beta_range : 2-TUPLE, optional
        (start, end) for beginning and end of beta parameter for annealing. beta = 1/T
        is the inverse of the annealing temperature. 
    num_reads : INT, optional
        Number of samples to take. The default is 1.
    num_sweeps : INT, optional
        Number of steps to take for each sample. The default is 2000.
    num_results : INT, optional
        Number of results to return. Must be >= num_rads. If less than num_reads, will return 
        the num_results lowest energy samples produced. The default is 1.
    init_spin : Dictionary (I think? 1/30), optional
        initial values to start annealing. The default is None (goes to PyQUBO's default random)
    seed : INT, optional
        seed to start the RNG of the annealing process with. The default is np.random.randint(0,2**31-1).
    ve : TBD, optional
        DESCRIPTION. The default is None.
    vector_mode : TYPE, optional
        DESCRIPTION. The default is 'Speed'.

    Returns
    -------
    sampled_images : num_results x len(intercept_visible_) Array. 
        Each row of this array is a flattened sampled image.

    '''
    
    #make the qubo
    qubo = kq.rbm_to_qubodict(components_, intercept_hidden_, intercept_visible_)
    va_model = qubo
    
    
    #solve the qubo using neal sampler (HERE can use QA instead?)
    sa = neal.SimulatedAnnealingSampler()
    results = sa.sample_qubo(va_model, beta_range = beta_range, num_reads=num_reads, 
          num_sweeps=num_sweeps, num_results=num_results, init_spin= init_spin, 
          seed=seed, ve=ve, vector_mode= vector_mode)
    
    # res = results[0]  #1/30 previosuly res was this 
    samples = results.samples()  #each entry is a dictionary form sample
    
    numh = len(intercept_hidden_)
    numv = len(intercept_visible_)
    
    samples_h = np.empty((num_results, numh)) #1/30: here num_results was previously 'num_samples'. 
    samples_v = np.empty((num_results, numv))
    
    # h_and_v  = kq.neal_samples_out_to_h_v(res.spin, numh, numv) #1/30 previously line
    for imgnum in range(num_results):
        h_and_v = kq.neal_samples_out_to_h_v(samples[imgnum], numh, numv)
    
        samples_h[imgnum, :] = h_and_v[0]
        samples_v[imgnum, :] =  h_and_v[1]
    
        sampled_images = samples_v
    
    return sampled_images



def add_salt_and_pepper(A, p):
    '''
    Adds salt and pepper noise to an image. 
    Parameters
    ----------
    A : ARRAY of BINARY
        Binary array (flattened or not) that represents the image we want to add noise too. 
    
    p: FLOAT in [0,1]
        probability of each pixel being flipped. 

    Returns
    -------
    A with salt and pepper noise added. 
    '''
    
    swaps = np.random.uniform(low = 0, high = 1, size = A.shape)
    mask = (swaps < p)
    
    B = (A + mask)%2
    return B



def init_spin_from_image(x):
    '''
    gives init_spins in dictionary form for PyQUBO / neal / VA type syntax from array x. 
    '''
    init_spin_list =  {}
    
    if len(np.shape(x)) != 1: 
        x = np.ndarray.flatten(x)

    xdim = len(x)
    n = int(np.sqrt(len(x)))
    
    for i in range(xdim):
            tempstr = 'v['+str(i)+']'
            init_spin_list[tempstr] = x[i]
    return init_spin_list

def mean_pixel_difference(restored_images, original_image):
    '''
    This function takes in an array of restored or noisy images and an original image. 
    It returns the average proportion of matching pixels across the restored images. 

    Parameters
    ----------
    restored_images : Array, m x n
        array of images to be compared to true image.Each ROW is a flattened image. 
    original_image : Vector, length n or 1 x n 
        flattened original image that restored or noisy images were derived from. 

    Returns
    -------
    float
        proportion of matching pixels. 

    '''
    right_wrong = (restored_images == original_image)
    return np.mean(right_wrong)
        

def rbm_to_qubodict(components_, intercept_hidden_, intercept_visible_):
    '''
    function to take the components and intercepts of rbm and turn it into 
    output as qubo dictionary for annealer
    
    Authors: Phillip Kerger

    Parameters
    ----------
    components_ : array-like of shape (n_components, n_features)
        Weight matrix, where n_features in the number of
        visible units and n_components is the number of hidden units.
        This is all the weight conections of hidden to visible units. 
        
    intercept_hidden_ : array-like of shape (n_components,)
        Biases of the hidden units.

    intercept_visible_ : array-like of shape (n_features,)
        Biases of the visible units.
    
    ^^ Note: these can be passed as for example rbm.components_

    Returns
    -------
    qubodict : dictionary form of Q to give to SX Aurora.
    '''
    hnum = len(intercept_hidden_)  #number of hidden nodes
    vnum = len(intercept_visible_) #number of visible nodes
    
    qubodict = {}
    #put connections of hiddens to visible into qubo dictionary
    for i in range(hnum):
        for j in range(vnum): 
            #only take nonzero entries for sparse representation
            mystr1 = 'h['+str(i)+']'
            mystr2 = 'v['+str(j)+']'
            qubodict[(mystr1, mystr2)] = components_[i,j] 
    
    #put intercepts into qubo dictionary
    for i in range(hnum):
        mystr1 = 'h['+str(i)+']'
        mystr2 = 'h['+str(i)+']'
        qubodict[(mystr1, mystr2)] = intercept_hidden_[i]
    for i in range(vnum):
        mystr1 = 'v['+str(i)+']'
        mystr2 = 'v['+str(i)+']'
        qubodict[(mystr1, mystr2)] = intercept_visible_[i]
    return qubodict 

def neal_samples_out_to_h_v(spins, numh, numv):
    '''
    function to turn output of SX Aurora into np vector.
     
    Authors: Phillip Kerger

    Parameters
    ----------
    spins : dictionary outputted by SX-Aurora.
    numh: number of hidden units
    numv: number of visible units

    Returns
    -------
    list where 0 entry is array of hiddens, 1 entry is array of visibles 

    '''
    hiddens= np.empty(numh)
    visibles = np.empty(numv)
    for i in range(numh):
        h_i =  'h[' + str(i) +']'
        hiddens[i] = int(spins[h_i])
    for i in range(numv):
        v_i =  'v[' + str(i) +']'
        visibles[i] = int(spins[v_i])
    return [hiddens,visibles]
    
#%% MINMAX FLOW DENOISING STUFF

#create graph for the minmax denoising 
def create_graph_for_flow(img, K=1, lam=3):
     max_num = len(img)*len(img[0])
     s,t = max_num, max_num + 1
     edge_list = []
     weights = []
     for r_idx, row in enumerate(img):
         for idx, pixel in enumerate(row):
                  px_id = (r_idx*len(row)) + idx
                  #add edge to cell to the left
                  if px_id!= 0:
                      edge_list.append((px_id -1, px_id))
                      weights.append( K )
                   #add edge to cell to the right
                  if px_id != len(row) -1:
                       edge_list.append((px_id +1, px_id))
                       weights.append( K )
                   #add edge to cell to the above
                  if r_idx!= 0:
                       edge_list.append((px_id - len(row), px_id))
                       weights.append( K )
                    #add edge to cell to the below
                  if r_idx != len(img) -1:
                       edge_list.append((px_id + len(row), px_id))
                       weights.append( K )
                    #add an edge to either s (source) or t (sink)
                  if pixel == 1:
                      edge_list.append((s,px_id))
                      weights.append( lam )
                  else:
                      edge_list.append((px_id, t))
                      weights.append( lam )
     return edge_list, weights, s, t
 
    
def flow_recover(noisy, K=1, lam =3.5):
       edge_list, weights, s, t = create_graph_for_flow(noisy, K,lam)
       g = igraph.Graph(edge_list)
       output = g.maxflow(s, t, weights)
       recovered = np.array(output.membership[:-2]).reshape(noisy.shape)
       recovered = np.mod(recovered+1, 2) #flip because of implementation 0-1
       return recovered
    
    