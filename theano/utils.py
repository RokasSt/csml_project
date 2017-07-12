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
import os
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
    
def get_missing_pixels(gamma, D, N):
    
    """ function to select which pixels are missing for 
    the reconstruction task """
    
    select_pixels = np.random.rand(N,D) > 0.5
    
    return select_pixels
    
def get_noisy_pixels(pflip, D, N):
    
    """ function to select which pixels are corrupted with noise"""
    
    select_pixels = np.random.rand(N,D) < pflip
    
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
    
    num_classes = len(list_of_paths)
    
    selected_images = np.zeros([num_classes*n, D])
    
    for cli in range(num_classes):
        
        cl_img = np.loadtxt(list_of_paths[cli])
        
        inds = np.random.choice(range(cl_img.shape[0]), n, replace=False)
                                
        assert len(inds) == n
        
        selected_images[cli*n:(cli+1)*n,:] = cl_img[inds,:]
        
    return selected_images
    
def plot_reconstructions(correct_images,
                         corrupted_images,
                         reconstructed_images,
                         save_to_path):
                             
    """ function to plot test images, their corrupted versions and 
    their reconstructions """
    
    num_reconstruct = correct_images.shape[0]
    
    reconstruction_errors = np.zeros([1,num_reconstruct])
    
    num_rows = num_reconstruct

    num_cols = 3
                                          
    _, ax = plt.subplots(num_rows, num_cols, sharex=False ,
    figsize=  (3 * num_cols, 3 * num_rows) )
    
    ax = ax.ravel()
    
    plot_index = 0
    
    for xi in range(num_reconstruct):
    
        ax[plot_index].imshow(np.reshape(correct_images[xi,:], [28,28]))
               
        ax[plot_index].set_xticks([])
        ax[plot_index].set_yticks([])
        ax[plot_index].set_yticklabels([])
        ax[plot_index].set_xticklabels([])
            
        plot_index += 1
    
        ax[plot_index].imshow(np.reshape(corrupted_images[xi,:], [28,28]))
               
        ax[plot_index].set_xticks([])
        ax[plot_index].set_yticks([])
        ax[plot_index].set_yticklabels([])
        ax[plot_index].set_xticklabels([])
            
        plot_index += 1
    
        ax[plot_index].imshow(np.reshape(reconstructed_images[xi,:], [28,28]))
               
        ax[plot_index].set_xticks([])
        ax[plot_index].set_yticks([])
        ax[plot_index].set_yticklabels([])
        ax[plot_index].set_xticklabels([])
            
        plot_index += 1
    
        dist_val = hamming_distance(correct_images[xi,:], 
                                    reconstructed_images[xi,:])
                                      
        print("Image --- %d ----"%xi+\
        " hamming distance between the true image and "+\
        "its reconstruction: %f"%dist_val)
        
        reconstruction_errors[0,xi] = dist_val
    
    plt.tight_layout()
    plt.savefig(save_to_path)
        
    plt.clf()
    
    return reconstruction_errors

def compare_reconstructions(correct_images,
                            corrupted_images,
                            reconstructed_images,
                            save_to_path):
                             
    """ function to plot test images, their corrupted versions and 
    their reconstructions under different training algorithms"""
    
    num_reconstruct = correct_images.shape[0]
    
    num_rows = len(reconstructed_images.keys()) + 2
    
    num_cols = num_reconstruct
    
    _, ax = plt.subplots(num_rows, num_cols, sharex=False ,
    figsize=  (3 * num_cols, 3 * num_rows) )
    
    ax = ax.ravel()
    
    plot_index = 0
    
    for xi in range(num_reconstruct):
        
        if xi ==0:
               
           ax[plot_index].set_title("Correct", size = 13) 
    
        ax[plot_index].imshow(np.reshape(correct_images[xi,:], [28,28]))
               
        ax[plot_index].set_xticks([])
        ax[plot_index].set_yticks([])
        ax[plot_index].set_yticklabels([])
        ax[plot_index].set_xticklabels([])
            
        plot_index += 1
        
    for xi in range(num_reconstruct):
        
        if xi ==0:
               
           ax[plot_index].set_title("Corrupted", size = 13) 
    
        ax[plot_index].imshow(np.reshape(corrupted_images[xi,:], [28,28]))
               
        ax[plot_index].set_xticks([])
        ax[plot_index].set_yticks([])
        ax[plot_index].set_yticklabels([])
        ax[plot_index].set_xticklabels([])
            
        plot_index += 1
        
    for algorithm in reconstructed_images.keys():
        
        for xi in range(num_reconstruct):
            
            if xi ==0:
               
               ax[plot_index].set_title("%s"%algorithm, size = 13) 
               
            img = reconstructed_images[algorithm][xi,:] 
    
            ax[plot_index].imshow(np.reshape(img, [28,28]))
               
            ax[plot_index].set_xticks([])
            ax[plot_index].set_yticks([])
            ax[plot_index].set_yticklabels([])
            ax[plot_index].set_xticklabels([])
            
            plot_index += 1
    
    plt.tight_layout()
    plt.savefig(save_to_path)
        
    plt.clf()
##########################################################################    
def plot_sequences(mean_dict, 
                   xlabel_dict,
                   ylabel_dict,
                   save_to_path, 
                   param_name ="", 
                   std_dict = None):
    
    """plot temporal sequences of w norms"""
    
    num_rows = len(w_norms_dict.keys())
    
    num_cols = 1
    
    _, ax = plt.subplots(num_rows, num_cols, sharex=False )
    #figsize=  ( 1*num_cols, 1*num_rows) )
    
    ax = ax.ravel()
    
    plot_index = 0
    
    use_legend = False
    
    for exp_tag in w_norms_dict.keys():
        
        ax[plot_index].set_title(exp_tag, size = 13) 
        
        if not isinstance(w_norms_dict[exp_tag], dict):
        
           max_val = np.max(w_norms_dict[exp_tag]) +10
    
           iters = range(len(w_norms_dict[exp_tag]))
           
           if isinstance(w_norms_std, dict):
        
              ax[plot_index].errorbar(iters,
                                   w_norms_dict[exp_tag],
                                   yerr= w_norms_std[exp_tag])
                                   
           else:
            
              ax[plot_index].plot(iters, w_norms_dict[exp_tag])
              
        else:
            
           use_legend = True
           
           for x_val in w_norms_dict[exp_tag].keys():
               
               max_val = np.max(w_norms_dict[exp_tag][x_val]) +10
    
               iters = range(len(w_norms_dict[exp_tag][x_val]))
               
               if isinstance(w_norms_std, dict):
        
                  ax[plot_index].errorbar(iters,
                                          w_norms_dict[exp_tag][x_val],
                                          yerr= w_norms_std[exp_tag][x_val],
                                          label = "%s %s"%(param_name,str(x_val)))
                                   
               else:
            
                  ax[plot_index].plot(iters, 
                                      w_norms_dict[exp_tag][x_val],
                                      label ="%s %s"%(param_name,str(x_val)))
        
        ax[plot_index].set_xlabel(xlabel_dict[exp_tag])
    
        ax[plot_index].set_ylabel(ylabel_dict[exp_tag])
        
        ax[plot_index].yaxis.set_ticks(np.arange(0, max_val, max_val//5))
        
        plot_index +=1
        
    if use_legend:
        
       plt.legend(loc='lower center', 
                  bbox_to_anchor=(0.5, -1.2),
                  ncol = 3)
                  #borderaxespad=0.)
    
    plt.tight_layout()
    
    plt.savefig(save_to_path, bbox_inches='tight')
       
    plt.clf()
    
def process_err_dict(means_dict, 
                     std_dict, 
                     update_dict, 
                     bar_plots = True):
    
    """ function to process error dictionary into lists for bar plots"""
    
    if bar_plots == True:
    
       output_dict                      = {}
       output_dict['MISSING']           = {}
       output_dict['NOISY']             = {}
    
       output_dict['LABELS']            = []
    
       output_dict['MISSING']['MEANS']  = []
    
       output_dict['MISSING']['STD']    = []
    
       output_dict['NOISY']['MEANS']    = []
    
       output_dict['NOISY']['STD']      = []

       for alg in means_dict.keys():
    
           output_dict['LABELS'].append(alg)
        
           output_dict['MISSING']['MEANS'].append(means_dict[alg]['MISSING'])
    
           output_dict['MISSING']['STD'].append(std_dict[alg]['MISSING'])
    
           output_dict['NOISY']['MEANS'].append(means_dict[alg]['NOISY'])
    
           output_dict['NOISY']['STD'].append(std_dict[alg]['NOISY'])
        
       return output_dict
       
    else:
       ## initialize update_dict if it is empty 
       if update_dict.keys() == []:
              
          update_dict['NOISY']   = {}
          
          update_dict['MISSING'] = {}
          
          for alg in means_dict.keys():
          
              update_dict['NOISY'][alg]   = {}
          
              update_dict['MISSING'][alg] = {}
              
              update_dict['NOISY'][alg]['MEANS']  = []
          
              update_dict['NOISY'][alg]['STD']  = []
              
              update_dict['MISSING'][alg]['MEANS']  = []
          
              update_dict['MISSING'][alg]['STD']  = []
              
       for alg in means_dict.keys():   
               
           mean_val = means_dict[alg]['MISSING']
              
           update_dict['MISSING'][alg]['MEANS'].append(mean_val)
          
           std_val = std_dict[alg]['MISSING']
          
           update_dict['MISSING'][alg]['STD'].append(std_val)
          
           mean_val = means_dict[alg]['NOISY']
              
           update_dict['NOISY'][alg]['MEANS'].append(mean_val)
          
           std_val = std_dict[alg]['NOISY']
          
           update_dict['NOISY'][alg]['STD'].append(std_val)
           
       return update_dict  
#######################################################################
def generate_bar_plots(array_dict, 
                       num_exps, 
                       save_to_path,
                       plot_std = True):
    
    """ function to generate bar plots """    
    fig, ax = plt.subplots(1, 2, sharex=False)

    ax = ax.ravel()

    width = 0.7
    
    x_axis   = np.arange(num_exps)
    
    ordered_labels = array_dict['LABELS']
    
    plot_index =0
    
    for key in array_dict.keys():
    
        if key !=  'LABELS':
            
           str_spec = key.lower()
           
           if plot_std:
    
              ax[plot_index].bar(x_axis, 
                                 array_dict[key]['MEANS'], 
                                 width = width, 
                                 color = 'b', 
                                 yerr  = array_dict[key]['STD'])
                                 
           else:
               
              ax[plot_index].bar(x_axis, 
                                 array_dict[key]['MEANS'], 
                                 width = width, 
                                 color = 'b')

           ax[plot_index].set_ylabel('Mean Reconstruction Errors')
           ax[plot_index].set_xticks(x_axis + width / 2)
           ax[plot_index].set_xticklabels(ordered_labels)
           ax[plot_index].set_title('Reconstruction of %s pixels'%str_spec)
           
           plot_index +=1
           
    plt.tight_layout()
    plt.savefig(save_to_path)
    plt.clf()
########################################################################
def plot_regressions(y_dict, 
                     x_values,
                     x_label,
                     y_label, 
                     save_to_path,
                     plot_std = False):
                         
    """ function to generate regression plots """
    
    num_exps = len(y_dict.keys())
    
    num_cols = num_exps
    
    _, ax = plt.subplots(1, num_cols, sharex=False )
    #figsize=  ( 1*num_cols, 1*num_rows) )
    
    ax = ax.ravel()
    
    plot_index = 0
    
    sorting_inds = list(np.argsort(x_values))
    
    x_values     = np.sort(x_values)
    
    for exp_type in y_dict.keys():
        
        max_val = 0
        
        for alg in y_dict[exp_type].keys():
            
            y_values = np.array(y_dict[exp_type][alg]['MEANS'])
            
            y_values = y_values[sorting_inds]
            
            max_val_check = np.max(y_values)
            
            if max_val_check > max_val:
                
               max_val = max_val_check
            
            if plot_std:
               std_y_values = np.array(y_dict[exp_type][alg]['STD'])
               std_y_values = std_y_values[sorting_inds]
               ax[plot_index].errorbar(x_values,
                                       y_values,
                                       yerr  = std_y_values,
                                       label = "%s"%alg)
            else:
            
               ax[plot_index].plot(x_values, 
                                   y_values,
                                   label ="%s"%alg)
        
        ax[plot_index].set_xlabel(x_label)
    
        ax[plot_index].set_ylabel(y_label)
        
        str_spec = exp_type.lower()
        
        ax[plot_index].set_title('Reconstruction of %s pixels'%str_spec)
        
        ax[plot_index].yaxis.set_ticks(np.arange(0, max_val+50, (max_val+50)//5))
        
        plot_index +=1
        
    plt.legend(loc='lower center', bbox_to_anchor=(0.05, -0.2), ncol = 5)
    
    plt.tight_layout()
    
    plt.savefig(save_to_path, bbox_inches='tight')
       
    plt.clf()
    
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
    
    
    
    
    

    
    
    
    
    
