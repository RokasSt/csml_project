""" 
Author: Rokas Stanislovas
MSc Project: Likelihood Approximations
for Energy-Based Models
MSc Computational Statistics and 
Machine Learning 
"""

import numpy as np
import theano.tensor as T
import theano
import sys
import matplotlib
matplotlib.use('agg',warn=False, force=True)
from matplotlib import pyplot as plt

def energy_function(W, b, x):
    
    """ to compute energy function for fully visible Boltzmann machine
    
    W - D x D matrix of weight variables in theano
    
    b - D-dimensional column vector of bias terms in theano
    
    x - D x N matrix of N input vectors
    
    return:
    
    - xTWx - bT x 
    
    as a symbolic variable to add to the theano
    computational graph.
    
    """
  
    return - T.dot(T.transpose(x), T.dot(W, x)) - T.dot(T.transpose(b), x)
   
def compute_class_ratios(dataset_labels):
    
    """function to compute class ratios from one-hot-encoded dataset
    
    dataset_labels - N x C matrix with N rows, each row corresponding
    to one-hot encoding with C classes.
    
    """
    
    num_instances, num_classes  = dataset_labels.shape
    
    counts = np.zeros(num_classes)
    
    for index in range(num_classes):
        
        counts[index] = 0
        
    for inst_ind in xrange(num_instances):
        
        counts[np.argmax(dataset_labels[inst_ind,:])] += 1
    
    return counts / float(num_instances)
    
def make_raster_plots(images, 
                      num_samples, 
                      num_chains, 
                      reshape_to, 
                      save_to_path,
                      init_with_images = True):
    
    """ function to generate a raster of image plots 
    
    images - (num_samples x num_chains) x num_pixels  matrix with
    num_samples x num_chains rows and num_pixels columns
    
    num_samples - number of required rows in the raster plot
    
    num_chains - number of required columns in the raster plot
    
    reshape_to - shape of the image [x_dim, y_dim ] (e.g. [28, 28])
    
    save_to_path - full path to save images.
    """
    
    num_rows =  num_samples

    num_cols =  num_chains
    
    if init_with_images:
    
       assert images.shape[0] == num_rows*num_cols + num_cols
       
       num_rows = num_rows +1
       
    else:
        
       assert images.shape[0] == num_rows*num_cols

    _, ax = plt.subplots(num_rows, num_cols, sharex=False ,\
    figsize=  (3 * num_cols, 3 * num_rows) )
    
    ax = ax.ravel()
    
    plot_index = 0
    
    chain_index = 0
    
    for image_index in range(images.shape[0]):
        
        if init_with_images:
        
           if image_index <= num_chains-1:
        
              ax[plot_index].set_title("Test image %d"%(chain_index), size = 13)
           
              chain_index +=1
           
           if (image_index >= num_chains) and (image_index < 2*num_chains):
            
              ax[plot_index].set_title("Samples", size = 13)
              
        else:
            
           if image_index <= num_chains-1:
        
              ax[plot_index].set_title("Samples", size = 13)
           
        ax[plot_index].imshow(np.reshape(images[image_index,:], reshape_to))
               
        ax[plot_index].set_xticks([])
        ax[plot_index].set_yticks([])
        ax[plot_index].set_yticklabels([])
        ax[plot_index].set_xticklabels([])
            
        plot_index += 1
        
    plt.tight_layout()
       
    plt.savefig(save_to_path)
        
    plt.clf() 
    
def get_noise_matrix(gamma, D, N):
    
    """ function to select which pixels are missing for 
    the reconstruction task """
    
    select_pixels = np.random.rand(N,D) > 0.5
    
    return select_pixels
    
def hamming_distance(array1, array2):
    
    """ function to compute the hamming distance between binary arrays"""
    
    return np.sum(np.abs(array1- array2))
    
def save_images_per_class(images, labels, root_dir):
    
    """ function to save images for each individual class separately
    
    images - N x D matrix of training images
    
    labels - N x K matrix of one-hot encodings of image labels"""
    
    num_images  = images.shape[0]
    
    D = images.shape[1]
    
    assert num_images == labels.shape[0]
    
    num_classes = labels.shape[1]
    
    cl_arr = {0:np.zeros([1,D]), 
              1:np.zeros([1,D]), 
              2:np.zeros([1,D]), 
              3:np.zeros([1,D]), 
              4:np.zeros([1,D]), 
              5:np.zeros([1,D]), 
              6:np.zeros([1,D]), 
              7:np.zeros([1,D]), 
              8:np.zeros([1,D]), 
              9:np.zeros([1,D])}
    
    for ind in range(num_images):
        
        if cl_arr[int(np.argmax(labels[ind,:]))].shape[0] == 1:
            
           cl_arr[int(np.argmax(labels[ind,:]))] = images[ind,:]
        
        else:
           
           cl_arr[int(np.argmax(labels[ind,:]))] = \
           np.vstack([cl_arr[int(np.argmax(labels[ind,:]))], images[ind,:]])
        
    for cli in cl_arr.keys():
        
        n = len(cl_arr[cli])
        
        class_images = np.reshape(cl_arr[cli],[n,D])
        
        save_to_path  = os.path.join(root_dir,"CLASS%d.dat"%cli)
        
        np.savetxt(save_to_path, class_images)
        
def select_subset(list_of_paths, n, D):

    """ function to select a training subset
    
    list_filenames - list of filenames """
    
    num_classes = len(list_filenames)
    
    selected_images = np.zeros([num_classes*num_per_class, D])
    
    for cli in range(num_classes):
        
        cl_img = np.loadtxt(list_of_paths[cli])
        
        inds = np.random.choice(range(cl_img.shape[0]), n, replace=False)
                                
        assert len(inds) == n
        
        selected_images[cli*n:(cli+1)*n,:] = cl_img[inds,:]
        
    return selected_images
    
if __name__ == "__main__":
    
    ### this block is for testing functions
    
    W = T.matrix('W')
    
    x = T.col('x')
    
    b = T.col('b')
    
    quadratic_form = energy_function(W, b, x)
    
    compute_quad_form = theano.function([W,x,b],quadratic_form)
    
    print(compute_quad_form([[1,2],[3,4]], [[1],[1]], [[1],[1]])[0][0] == 12)
    
    grad_W, grad_b = T.grad(quadratic_form[0][0], [W,b])
    
    comp_grad_W = theano.function([W,b,x], grad_W)
    
    comp_grad_b = theano.function([W,b,x], grad_b)
    
    print(comp_grad_W([[0,2],[2,0]], [[1],[1]], [[1],[4]]))
    sys.exit()
    print(comp_grad_b([[0,2],[2,0]], [[1],[1]], [[1],[4]]))
    
    compute_trace = T.diag(W)
    
    x_matrix = T.matrix('x_matrix')
    
    quadratic_form_mat = energy_function(W, b, x_matrix) 
    
    compute_quad_matrix = theano.function([W,x_matrix,b],T.sum(T.diag(quadratic_form_mat)))
    
    # test when x is a matrix 
    
    print(compute_quad_matrix([[1,2],[3,4]], [[1,1,2],[1,1,2]], [[1],[1]]))
    
    diag_quad_terms, updates = \
    theano.scan(lambda i: energy_function(W,b,x_matrix[:,i]), sequences = [T.arange(3)])
    
    compute_diag_terms =  theano.function([W,x_matrix,b], diag_quad_terms)
    
    print(compute_diag_terms([[1,2],[3,4]], [[1,1,2],[1,1,2]], [[1],[1]]))
    
    
    
    
    

    
    
    
    
    
