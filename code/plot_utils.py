""" 
Author: Rokas Stanislovas
MSc Project: Complementary Sum Sampling 
for Learning in Boltzmann Machines
MSc Computational Statistics and 
Machine Learning 

Plotting functions.
"""

import matplotlib
matplotlib.use('agg',warn=False, force=True)
from matplotlib import pyplot as plt
from matplotlib import rc,rcParams
from matplotlib.pyplot import cm 
import subprocess
import os
import sys
import json
import argparse
import numpy as np
import utils

rc('text', usetex=True)
rc('axes', linewidth=2)
rc('font', weight='bold')
rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

def make_raster_plots(images, 
                      num_samples, 
                      num_chains, 
                      reshape_to, 
                      save_to_path,
                      init_with_images = True):
    
    """ function to generate a raster of image plots 
    
    images - N x num_pixels matrix of images; 
    
             if function argument init_with_images is set to True (default)
             N must be equal to num_samples*num_chains + num_chains;
             the first num_chains rows are interpreted as original test
             images; then each consecutive block of num_chains images 
             correspond to samples (e.g. gibbs-based reconstructions)
             of individual test images;
             if function argument init_with_images is set to False
             N must be equal to num_samples*num_chains and function
             assumes that original test images are not included in images
    
    num_samples - number of samples per chain/ test image or number of 
                  rows in the raster plot containing sample images.
    
    num_chains - number of test images which corresponds to the number of
                 columns in the raster plot
    
    reshape_to - shape of the image [x_dim, y_dim ] (e.g. [28, 28])
    
    save_to_path - full path to save images.
    
    init_with_images - (default True) . See comment on images. """
    
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
########################################################################
def plot_reconstructions(correct_images,
                         corrupted_images,
                         reconstructed_images,
                         save_to_path):
                             
    """ function to plot test images, their corrupted versions and 
    their reconstructions.
    
    correct_images - N x D matrix of correct versions of images
                     (N - number of images, D- number of pixels)
                     
    corrupted_images - N x D matrix of corrupted versions of the images
                       in correct_images
                       
    reconstructed_images - N x D matrix of recosntructed images
    
    save_to_path         - full path to save plots.
    
    return: 
                        1 x N array of reconstruction errors.   """
    
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
    
        dist_val = utils.hamming_distance(correct_images[xi,:], 
                                          reconstructed_images[xi,:])
                                      
        print("Image --- %d ----"%xi+\
        " hamming distance between the true image and "+\
        "its reconstruction: %f"%dist_val)
        
        reconstruction_errors[0,xi] = dist_val
    
    plt.tight_layout()
    plt.savefig(save_to_path)
        
    plt.clf()
    
    return reconstruction_errors
########################################################################
def compare_reconstructions(correct_images,
                            corrupted_images,
                            reconstructed_images,
                            save_to_path):
                             
    """ function to plot test images, their corrupted versions and 
    their reconstructions under different training algorithms.
    
    correct_images - N x D matrix of correct versions of images
                     (N - number of images, D- number of pixels)
                     
    corrupted_images - N x D matrix of corrupted versions of the images
                       in correct_images
                       
    reconstructed_images - dictionary with K fields. A given entry
                           stores N x D matrix of reconstructed images.
                           Key of a given entry specifies the name
                           of the algorithm used during model training.
    
    save_to_path         - full path to save plots. """
    
    num_reconstruct = correct_images.shape[0]
    
    num_rows = len(reconstructed_images.keys()) + 2
    
    num_cols = num_reconstruct
    
    _, ax = plt.subplots(num_rows, num_cols, sharex=False,
    figsize=  (3 * num_cols, 3 * num_rows) )
    
    ax = ax.ravel()
    
    plot_index = 0
    
    for xi in range(num_reconstruct):
        
        if xi ==0:
           t = "Correct"
           ax[plot_index].set_title(r'\textbf{%s}'%t, fontsize = 15) 
    
        ax[plot_index].imshow(np.reshape(correct_images[xi,:], [28,28]))
               
        ax[plot_index].set_xticks([])
        ax[plot_index].set_yticks([])
        ax[plot_index].set_yticklabels([])
        ax[plot_index].set_xticklabels([])
            
        plot_index += 1
        
    for xi in range(num_reconstruct):
        
        if xi ==0:
           t = "Corrupted"
           ax[plot_index].set_title(r'\textbf{%s}'%t, fontsize = 15) 
    
        ax[plot_index].imshow(np.reshape(corrupted_images[xi,:], [28,28]))
               
        ax[plot_index].set_xticks([])
        ax[plot_index].set_yticks([])
        ax[plot_index].set_yticklabels([])
        ax[plot_index].set_xticklabels([])
            
        plot_index += 1
        
    for algorithm in reconstructed_images.keys():
        
        for xi in range(num_reconstruct):
            
            if xi ==0:
               t = "%s"%algorithm
               ax[plot_index].set_title(r'\textbf{%s}'%t, fontsize = 15) 
               
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
def plot_sequences(means_dict, 
                   xlabel_dict,
                   ylabel_dict,
                   save_to_path, 
                   param_name ="", 
                   std_dict = None):
    
    """function to plot temporal sequences stored in numpy arrays.
    
    means_dict - dictionary containing one dimensional value arrays;
                 a given key is interpreted as the name of the
                 algorithm; the corresponding dictionary value must be
                 either one-dimensional numpy array (e.g. learning curve)
                 or dictionary.
                  
                 if it is a dictionary, the individual entries of the 
                 inner dictionary must store arrays obtained
                 under different values of a given control parameter.
                 Individual keys should correspond to the unique values
                 of this parameter. Name of this parameter can be
                 optionally specified by the function argument param_name.
                 
    xlabel_dict - dictionary of labels for x axis with keys corresponding
                  to the names of training algorithms.
                  
    ylabel_dict - dictionary of labels for y axis with keys corresponding
                  to the names of training algorithms.
                  
    save_to_path - full path to save plots to.
    
    param_name   - optional string specifier (default "") to label
                   multiple curves in the algorithm-specific plot. 
                   Used when means_dict contains inner dictionaries that
                   store multiple curves (see above).
                   
    std_dict     - (default None) optional dictionary of standard
                   deviations per individual entries in the arrays
                   stored in means_dict; used to plot error bars.
                   Must have the same structure as means_dict.        """
    
    num_rows = len(means_dict.keys())
    
    num_cols = 1
    
    _, ax = plt.subplots(num_rows, num_cols, sharex=False,
    figsize=  ( 9*num_cols, 3*num_rows) )
    
    if num_rows >1:
    
       ax = ax.ravel()
    
    plot_index = 0
    
    use_legend = False
    
    for exp_tag in means_dict.keys():
        
        if num_rows ==1:
            
           ax_obj = ax
           
        elif num_rows >1:
            
           ax_obj = ax[plot_index]
        
        if not isinstance(means_dict[exp_tag], dict):
        
           max_val = np.max(means_dict[exp_tag])
           
           min_val = np.min(means_dict[exp_tag])
           
           max_val = max_val + 0.2*max_val
           
           iters = range(len(means_dict[exp_tag]))
           
           if isinstance(std_dict, dict):
        
              ax_obj.errorbar(iters,
                              means_dict[exp_tag],
                              yerr= std_dict[exp_tag],
                              linewidth = 2)
                                   
           else:
            
              ax_obj.plot(iters, 
                          means_dict[exp_tag], 
                          linewidth = 2)
              
        else:
            
           use_legend = True
           
           for x_val in means_dict[exp_tag].keys():
               
               max_val = np.max(means_dict[exp_tag][x_val])
               
               min_val = np.min(means_dict[exp_tag][x_val])
               
               max_val = max_val + 0.2*max_val
    
               iters = range(len(means_dict[exp_tag][x_val]))
               
               if isinstance(std_dict, dict):
        
                  ax_obj.errorbar(iters,
                                  means_dict[exp_tag][x_val],
                                  yerr= std_dict[exp_tag][x_val],
                                  label =r"\textbf{%s %s}"
                                   %(param_name, str(x_val)),
                                  linewidth = 2)
                                   
               else:
                  
                  ax_obj.plot(iters, 
                              means_dict[exp_tag][x_val],
                              label =r"\textbf{%s %s}"
                              %(param_name, str(x_val)),
                              linewidth = 2)
                              
        if "_" in exp_tag:
           title_str = exp_tag.replace("_"," ") 
        else:
          title_str = exp_tag
                                      
        ax_obj.set_title(r'\textbf{%s}'%title_str, size = 15)
        
        ax_obj.set_xlabel(r'\textbf{%s}'%xlabel_dict[exp_tag], 
                          fontsize= 15)
    
        ax_obj.set_ylabel(r'\textbf{%s}'%ylabel_dict[exp_tag], 
                          fontsize= 15)
        
        ax_obj.locator_params(nbins=8, axis='y')
        
        ax_obj.yaxis.set_tick_params(labelsize = 14)
        
        ax_obj.xaxis.set_tick_params(labelsize = 14)
        
        plot_index +=1
        
    if use_legend:
        
       plt.legend(loc='lower center', 
                  bbox_to_anchor=(0.5, -0.8),
                  ncol = 3)
                  #borderaxespad=0.)
    
    plt.tight_layout()
    
    plt.savefig(save_to_path, bbox_inches='tight')
       
    plt.clf()
#########################################################################
def plot_end_values(means_dict, 
                    xlabel_dict,
                    ylabel_dict,
                    save_to_path, 
                    param_name, 
                    std_dict = None):
    
    """plot last values of learning curves against the array of parameter 
    values.
    
    means_dict - dictionary containing one dimensional value arrays;
                 a given key is interpreted as the name of the
                 algorithm; the corresponding dictionary value must be
                 either a one-dimensional numpy array, storing the
                 learning curve obtained by a given algorithm, or dictionary;
                  
                 if it is a dictionary, the individual entries of the 
                 inner dictionary must store numpy arrays such that 
                 each key corresponds to the unique value of the control 
                 parameter and the associated dictionary value
                 stores the corresponding learning curve.
                 
    ylabel_dict - dictionary of labels for y axis with keys corresponding
                  to the names of training algorithms.
                  
    save_to_path - full path to save plots to.
    
    param_name   - string used to label x axis (control parameter name).
                   
    std_dict     - (default None) optional dictionary of standard
                   deviations per individual entries in the arrays
                   stored in means_dict; used to plot error bars.
                   Must have the same structure as means_dict. """
    
    num_rows = len(ylabel_dict.keys())
    
    num_cols = 1
    
    _, ax = plt.subplots(num_rows, num_cols, sharex=False)
    #figsize=  ( 1*num_cols, 1*num_rows) )
    
    use_indexing = False
    if num_rows > 1 or num_cols > 1: 
       use_indexing = True
       ax = ax.ravel()
    
    plot_index = 0
    
    width = 0.7
    
    for exp_tag in ylabel_dict.keys():
        
        y_axis     = []
        
        x_axis     = []
        
        y_axis_std = []
        
        if "_" in exp_tag:
           title_str = exp_tag.replace("_"," ") 
        else:
           title_str = exp_tag
        
        if not isinstance(means_dict[exp_tag], dict):
            
           x_axis = np.arange(1)
        
           end_val = means_dict[exp_tag][-1]
           
           y_axis.append(end_val)
           
           if isinstance(std_dict, dict):
               
              y_axis_std.append(std_dict[exp_tag][-1])
              
        else:
           
           for x_val in means_dict[exp_tag].keys():
               
               end_val = means_dict[exp_tag][x_val][-1]
               
               y_axis.append(end_val)
               x_axis.append(x_val)
               
               if isinstance(std_dict, dict):
                  
                  y_axis_std.append(std_dict[exp_tag][x_val][-1])
        
        sorting_inds = list(np.argsort(x_axis))
    
        x_axis       = np.sort(x_axis)
        
        y_axis       = np.array(y_axis)
        
        y_axis       = y_axis[sorting_inds]
        
        if use_indexing:
           if y_axis_std != []:
           
              ax[plot_index].errorbar(x_axis, 
                                      y_axis, 
                                      yerr  = y_axis_std,
                                      linewidth = 2,
                                      marker = "o")
              
           else:
            
              ax[plot_index].plot(x_axis, 
                                  y_axis,
                                  linewidth = 2,
                                  marker = "o")
           
           ax[plot_index].set_title(r'\textbf{%s}'%title_str, fontsize = 15)  
        
           ax[plot_index].set_xlabel(r'\textbf{%s}'%param_name, 
                                     fontsize =14)
    
           ax[plot_index].set_ylabel(r'\textbf{%s}'%ylabel_dict[exp_tag],
                                     fontsize = 14)
        
           ax[plot_index].locator_params(nbins=8, axis='y')
           
           ax[plot_index].set_xlim([x_axis[0] - 0.2*x_axis[0], 
                                    x_axis[-1]+ 0.05*x_axis[-1]])
                                    
           x_list = list(np.arange(x_axis[0], x_axis[-1], (x_axis[-1])/10))  
        
           x_list.append(x_axis[-1])
                            
           ax[plot_index].xaxis.set_ticks(x_list)
        
           plot_index +=1
           
        else:
        
           if y_axis_std != []:
           
              ax.errorbar(x_axis, 
                          y_axis, 
                          yerr  = y_axis_std,
                          linewidth = 2,
                          marker = "o")
              
           else:
            
              ax.plot(x_axis, 
                      y_axis, 
                      linewidth =2,
                      marker = "o")
                               
           ax.set_title(r'\textbf{%s}'%title_str, size = 15)  
        
           ax.set_xlabel(r'\textbf{%s}'%param_name, fontsize =14)
    
           ax.set_ylabel(r'\textbf{%s}'%ylabel_dict[exp_tag], fontsize =14)
        
           ax.locator_params(nbins=8, axis='y')
           
           ax.set_xlim([x_axis[0] - 0.2*x_axis[0], x_axis[-1]+ 0.05*x_axis[-1]])
           
           x_list = list(np.arange(x_axis[0], x_axis[-1], (x_axis[-1])/10))  
        
           x_list.append(x_axis[-1])
                            
           ax.xaxis.set_ticks(x_list)
           
    plt.tight_layout()
    
    plt.savefig(save_to_path, bbox_inches='tight')
       
    plt.clf()
##########################################################################   
def process_dict_to_list(target_dict):
    
    """ function to convert dictionary format to list format for
    bar plotting.
    
    target_dict - dictionary in the following format:
    
                  target_dict[field name 0] = field value 0
                  
                  target_dict[field name 1] = field value 1
                  
                  .........................................
                  
                  target_dict[field_name n] = field value n.
                  
    return:       output_dict in the following format:
    
                  output_dict['Y'] = [field value 0, field value 1, 
                  field value 2, field value 3, ..., field value n]
                  
                  output_dict['LABELS'] = [field name 0, field name 1,
                  field name 2, ..., field name n]. """
    
    output_dict = {}
    
    output_dict['LABELS'] = []
    
    output_dict['Y']      = []
    
    for field_name in target_dict.keys():
        
        output_dict['LABELS'].append(field_name)
        
        output_dict['Y'].append(target_dict[field_name])
        
    return output_dict
 
def process_err_dict(means_dict,
                     target_fields,
                     std_dict =[],
                     update_dict = None, 
                     bar_plots = True):
    
    """ function to process error dictionary into lists for plotting
    
    means_dict    - dictionary storing reconstruction errors.
                    Two formats are allowed.
                    
                    Format 1
                    mean_dict["algorithm_name"]["MISSING" or "NOISY" ] =
                    value of mean error
                 
                    Format 2
                    means_dict["algorithm_name"]["MISSING" or "NOISY"]
                    ["regressor_name its_value"] = value of mean error .
                 
    target_fields - list of names for target algorithms.
    
    std_dict      - (default []) if specified, must store standard
                    deviations of reconstruction errors and must take
                    the same format as means_dict.
                    
    update_dict   - (default None) if bar_plots is True, update_dict
                    must be a dictionary; if it is empty ({}) it is
                    initialized using the following format:
                    
                    update_dict['NOISY'][algorithm name]['MEANS']   = []
                    update_dict['NOISY'][algorithm name]['STD']     = []
                    update_dict['MISSING'][algorithm name]['MEANS'] = []
                    update_dict['MISSING'][algorithm name]['STD']   = []
                    
                    if update_dict is not empty, it must be specified
                    in the format defined above, in order to iteratively
                    append values to the stored lists.
                    
    bar_plots     - if set to True, means_dict (and, optionally, std_dict)
                    is processed into output_dict (see code), but
                    update_dict is not used. 
                    If set to False, update_dict is updated with values
                    from means_dict and std_dict.
                    
    return:         output_dict if bar_plots == True;
    
                    or 
                    
                    update_dict, regressor_values if bar_plots False;
                    
                    regressor_values is a list of regressor values extracted
                    from the keys at the lowest level of means_dict;
                    (see comments on means_dict).                     """
    
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
           
           if alg not in target_fields:
              continue
           
           if (not isinstance(means_dict[alg]['MISSING'], dict)) and\
           (not isinstance(means_dict[alg]['NOISY'], dict)):
    
              output_dict['LABELS'].append(alg)
        
              output_dict['MISSING']['MEANS'].append(means_dict[alg]['MISSING'])
              
              output_dict['NOISY']['MEANS'].append(means_dict[alg]['NOISY'])
              
              if std_dict != []:
                  
                 output_dict['MISSING']['STD'].append(std_dict[alg]['MISSING'])
    
                 output_dict['NOISY']['STD'].append(std_dict[alg]['NOISY'])
              
           else:
               
              #### assumes that both "MISSING" and "NOISY" fields
              #### share the same set subfields
              for field in means_dict[alg]['MISSING'].keys():
                  
                  if "_" in field:
                     split_field = field.split("_")
                     get_val = split_field[1].split(" ")
                     
                     field_short=split_field[0][0].upper()+\
                     split_field[1][0].upper()
                     field_short += " %s"%get_val[1]
                  
                  output_dict['LABELS'].append("%s %s"%(alg,field_short))
                  
                  output_dict['MISSING']['MEANS'].append(\
                  means_dict[alg]['MISSING'][field])
                  
                  output_dict['NOISY']['MEANS'].append(\
                  means_dict[alg]['NOISY'][field])
                  
                  if std_dict != []:
                      
                     output_dict['MISSING']['STD'].append(\
                     std_dict[alg]['MISSING'][field])
                  
                     output_dict['NOISY']['STD'].append(\
                     std_dict[alg]['NOISY'][field])
                  
       return output_dict
       
    else:
       ## initialize update_dict if it is empty 
       regressor_values = []
       if update_dict.keys() == []:
              
          update_dict['NOISY']   = {}
          
          update_dict['MISSING'] = {}
          
          for alg in means_dict.keys():
              
              if alg not in target_fields:
                 continue
          
              update_dict['NOISY'][alg]   = {}
          
              update_dict['MISSING'][alg] = {}
              
              update_dict['NOISY'][alg]['MEANS']  = []
          
              update_dict['NOISY'][alg]['STD']  = []
              
              update_dict['MISSING'][alg]['MEANS']  = []
          
              update_dict['MISSING'][alg]['STD']  = []
              
       for alg in means_dict.keys():  
           
           if alg not in target_fields:
              continue
           
           if not isinstance(means_dict[alg]['MISSING'], dict):
               
              mean_val = means_dict[alg]['MISSING']
              
              update_dict['MISSING'][alg]['MEANS'].append(mean_val)
          
              std_val = std_dict[alg]['MISSING']
          
              update_dict['MISSING'][alg]['STD'].append(std_val)
              
           else:
               
              for field in means_dict[alg]['MISSING'].keys():
                  
                  val = float(field.split(" ")[1])
                  
                  mean_val = means_dict[alg]['MISSING'][field]
                  
                  update_dict['MISSING'][alg]['MEANS'].append(mean_val)
          
                  std_val = std_dict[alg]['MISSING'][field]
          
                  update_dict['MISSING'][alg]['STD'].append(std_val)
                  
                  regressor_values.append(val)
              
           if not isinstance(means_dict[alg]['NOISY'], dict):
          
              mean_val = means_dict[alg]['NOISY']
              
              update_dict['NOISY'][alg]['MEANS'].append(mean_val)
          
              std_val = std_dict[alg]['NOISY']
          
              update_dict['NOISY'][alg]['STD'].append(std_val)
              
           else:
               
              tmp_values = [] 
               
              for field in means_dict[alg]['NOISY'].keys():
                  
                  val = float(field.split(" ")[1])
                  
                  mean_val = means_dict[alg]['NOISY'][field]
                  
                  update_dict['NOISY'][alg]['MEANS'].append(mean_val)
          
                  std_val = std_dict[alg]['NOISY'][field]
          
                  update_dict['NOISY'][alg]['STD'].append(std_val)
                  
                  tmp_values.append(val)
              
              assert tmp_values == regressor_values
           
       return update_dict, regressor_values
#######################################################################
def generate_bar_plot(y_list,
                      labels,
                      save_to_path,
                      ylabel,
                      title = None,
                      std_list = []):
    
    """ function to generate bar plots for a given array.
    
    y_list       - list of y values.
    
    labels       - list of labels on x-axis to mark individual bars.
    
    save_to_path - full path to save bar plot.
    
    ylabel       - label for y axis.
    
    title        - (default None) optional title for bar plot.
    
    std_list     - (default []) optional list of values specifying size
                   of error bars.                                    """ 
                      
    fig, ax = plt.subplots(1, 1, sharex=False)
    
    width = 0.7
    
    x_axis   = np.arange(len(y_list))
    
    for label_x in range(len(labels)):
        
        if "_" in labels[label_x]:
           labels[label_x] = labels[label_x].replace("_"," ")
        
        labels[label_x] = r'\textbf{%s}'%labels[label_x]
    
    if std_list != []:
              
       ax.bar(x_axis, 
              y_list, 
              width = width, 
              color = 'b', 
              yerr  = std_list,
              error_kw=dict(ecolor='red', elinewidth=2))
                                 
    else:
               
       ax.bar(x_axis, 
              y_list, 
              width = width, 
              color = 'b')

    ax.set_ylabel(r'\textbf{%s}'%ylabel, fontsize = 14)
                                     
    ax.set_xticks(x_axis + width / 2)
           
    ax.set_xticklabels(labels,
                       rotation= "vertical",
                       fontsize = 14)
    
    if title != None:
       ax.set_title(r'\textbf{%s}'%title, fontsize = 15)
           
    ax.yaxis.set_tick_params(labelsize = 14)
        
    ax.xaxis.set_tick_params(labelsize = 14)
    
    plt.tight_layout()
    plt.savefig(save_to_path)
    plt.clf()
#######################################################################
def display_recon_errors(array_dict, 
                         num_exps, 
                         save_to_path,
                         ylabel = 'Mean Reconstruction Errors',
                         plot_std = True):
    
    """ function to generate bar plots of reconstruction errors.
    
    array_dict - dictionary containing values for mean reconstruction 
                 errors in the following form:
                 array_dict["LABELS"] = list of labels for x axis for
                 each subplot
                 array_dict["task_name"]["MEANS"] = list of mean values
                 array_dict["task_name"]["STD"] (optional entry) 
                 = list of std  values
                 
                 at the moment function only works with two different
                 tasks, generating two different subplots.
                 
    num_exps   - number of experiments (bars) per task; should be the
                 same for both tasks.
                 
    save_to_path - full path to save generated plot.
    
    ylabel       - label for y axis.
    
    plot_std     - (default True) boolean indicator for whether to plot
                   error bars; if True, array_dict["task_name"] must 
                   contain "STD" field for each task_name.           """
                       
    fig, ax = plt.subplots(1, 2, sharex=False)

    ax = ax.ravel()

    width = 0.7
    
    x_axis   = np.arange(num_exps)
    
    labels = array_dict['LABELS']
    
    for label_x in range(len(labels)):
        
        if "_" in labels[label_x]:
           labels[label_x] = labels[label_x].replace("_"," ")
        
        labels[label_x] = r'\textbf{%s}'%labels[label_x]
    
    plot_index =0
    
    for key in array_dict.keys():
    
        if key !=  'LABELS':
            
           str_spec = key.lower()
           
           if plot_std:
              
              ax[plot_index].bar(x_axis, 
                                 array_dict[key]['MEANS'], 
                                 width = width, 
                                 color = 'b', 
                                 yerr  = array_dict[key]['STD'],
                                 error_kw=dict(ecolor='red', elinewidth=2))
                                 
           else:
               
              ax[plot_index].bar(x_axis, 
                                 array_dict[key]['MEANS'], 
                                 width = width, 
                                 color = 'b')

           ax[plot_index].set_ylabel(r'\textbf{%s}'%ylabel, 
                                     fontsize = 14)
                                     
           ax[plot_index].set_xticks(x_axis + width / 2)
           
           ax[plot_index].set_xticklabels(labels,
                                          rotation= "vertical",
                                          fontsize = 14)
                                           
           ax[plot_index].set_title(r'\textbf{%s}'
           %("Reconstruction of %s pixels"%str_spec), fontsize = 15)
           
           ax[plot_index].yaxis.set_tick_params(labelsize = 14)
        
           ax[plot_index].xaxis.set_tick_params(labelsize = 14)
           
           plot_index +=1
           
    plt.tight_layout()
    plt.savefig(save_to_path)
    plt.clf()
########################################################################
def generate_scatter_plot(x_dict, 
                          y_dict,
                          x_label,
                          y_label, 
                          title_dict,
                          save_to_path):
                              
    """ function to generate scatter plot for given arrays.
    
    x_dict - dictionary containing list of x values; each entry of
             x_dict stores a dictionary over x-value lists which are
             used to generate an individual subplot; number of fields
             in x_dict is a number of subplots in the final figure;
             entries of inner dictionary per x_dict entry correspond
             to different populations of x-values per x_dict entry-specific 
             scatter subplot.
             
    y_dict - identical to x_dict but for y axis (must have the same
             dictionary keys as x_dict).
             
    x_label- label for x axis on every subplot.
    
    y_label- label for y axis on every subplot.
    
    title_dict - dictionary containing tiltes for individual subplots;
                 keys must identical to the keys of x_dict.
                 
    save_to_path - full path to save figure.                         """
    
    num_cols = len(x_dict.keys())
    
    _, ax = plt.subplots(1, num_cols, sharex=False)
    
    if num_cols > 1:
       ax = ax.ravel()
    
    plot_index = 0
    
    for exp_type in x_dict.keys():
        
        if num_cols > 1:
           ax_obj = ax[plot_index]
           
        else:
           ax_obj = ax
           
        num_required =  len(x_dict[exp_type].keys())
           
        color_selector=iter(cm.rainbow(np.linspace(0,1,num_required)))
        
        for alg in x_dict[exp_type].keys():
            
            print(alg)
            
            x_values = np.array(x_dict[exp_type][alg])
            
            if exp_type in y_dict.keys():
                
               y_values = np.array(y_dict[exp_type][alg])
               
            else:
                
               y_values = np.array(y_dict[alg]) 
            
            assert len(x_values) == len(y_values)
            
            if "_" in alg:
               
               alg_label = alg.replace("_"," "); 
               
            else:
               alg_label = alg
            
            ax_obj.scatter(x_values,
                           y_values,
                           label = r"\textbf{%s}"%alg_label,
                           marker = "o",
                           c = next(color_selector),
                           s = 25)
        
        ax_obj.set_xlabel(r'\textbf{%s}'%x_label, fontsize=15)
    
        ax_obj.set_ylabel(r'\textbf{%s}'%y_label, fontsize =15)
        
        ax_obj.set_title(r'\textbf{%s}'%title_dict[exp_type], fontsize=15)
        
        ax_obj.locator_params(nbins=8, axis='y')
        
        ax_obj.locator_params(nbins=8, axis='x')
        
        ax_obj.yaxis.set_tick_params(labelsize = 14)
        
        ax_obj.xaxis.set_tick_params(labelsize = 14)
        
        plot_index +=1
        
    plt.legend(loc='lower center', bbox_to_anchor=(0.05, -0.2), ncol = 5)
    
    plt.tight_layout()
    
    plt.savefig(save_to_path, bbox_inches='tight')
       
    plt.clf()
########################################################################
def plot_regressions(y_dict, 
                     x_values,
                     x_label,
                     y_label, 
                     save_to_path,
                     plot_std = False):
                         
    """ function to generate regression plots. 
    
    y_dict - dictionary containing lists of y values; each entry of
             y_dict stores a dictionary over y-value lists which are
             used to generate an individual subplot; number of fields
             in y_dict is a number of subplots in the final figure;
             entries of inner dictionary per y_dict entry correspond
             to different populations of y values per y_dict entry-specific 
             subplot; both means and standard deviations can be specified:
             y_dict["task_name"]["algorithm_name"]["MEANS"] = list of y values
             (optional) y_dict["task_name"]["algorithm_name"]["STD"] =
             list of values for error bars.
    
    x_values     -  a single list of x values associated with each y list. 
    
    x_label      -  label for x-axis on each subplot.
    
    y_label      -  label for y-axis on each subplot.
    
    save_to_path -  full path to save figure.
    
    plot_std     -  (default False) boolean indicator whether to add error
                    bars using "STD" values in y_dict.                """
    
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
        
        min_val = 0
        
        for alg in y_dict[exp_type].keys():
            
            y_values = np.array(y_dict[exp_type][alg]['MEANS'])
            
            y_values = y_values[sorting_inds]
            
            max_val_check = np.max(y_values)
            
            min_val_check = np.min(y_values)
            
            if max_val_check > max_val:
                
               max_val = max_val_check
               
            if min_val_check < min_val:
                
               min_val = min_val_check
               
            label = alg
            if "_" in label:
               label = label.replace("_"," ") 
            
            if plot_std:
               std_y_values = np.array(y_dict[exp_type][alg]['STD'])
               std_y_values = std_y_values[sorting_inds]
               ax[plot_index].errorbar(x_values,
                                       y_values,
                                       yerr  = std_y_values,
                                       label = r"\textbf{%s}"%label,
                                       linewidth = 2)
            else:
            
               ax[plot_index].plot(x_values, 
                                   y_values,
                                   label =r"\textbf{%s}"%label,
                                   linewidth = 2)
        
        ax[plot_index].set_xlabel(r'\textbf{%s}'%x_label, fontsize=15)
    
        ax[plot_index].set_ylabel(r'\textbf{%s}'%y_label, fontsize =15)
        
        str_spec = exp_type.lower()
        
        str_spec = str_spec[0].upper() + str_spec[1:]
        
        ax[plot_index].set_title(r'\textbf{%s pixels}'%str_spec, 
                                 fontsize=15)
        
        if min_val >= 0:
        
           ax[plot_index].yaxis.set_ticks(np.arange(0, max_val+5, (max_val+5)//5))
           
        else:
            
           ax[plot_index].locator_params(nbins=8, axis='y')
        
        ax[plot_index].yaxis.set_tick_params(labelsize = 14)
        
        ax[plot_index].xaxis.set_tick_params(labelsize = 14)
        
        ax[plot_index].set_xlim([x_values[0] -0.2*x_values[0], 
                                 x_values[-1]+0.05*x_values[-1]])
        
        x_list = list(np.arange(x_values[0], x_values[-1], (x_values[-1])/10))  
        
        x_list.append(x_values[-1])
                            
        ax[plot_index].xaxis.set_ticks(x_list)
        
        plot_index +=1
        
    plt.legend(loc='lower center', bbox_to_anchor=(0.05, -0.2), ncol = 5)
    
    plt.tight_layout()
    
    plt.savefig(save_to_path, bbox_inches='tight')
       
    plt.clf()
##########################################################################
def add_data(target_path,
             look_at_dict,
             target_field,
             param_value,
             avg_axis = None):
                   
    """ function to extract temporal sequences of measurements from the 
    numpy array for plotting purposes.
    
    target_path  - full path to stored numpy array.
    
    look_at_dict - dictionary which is updated with the
                   value array stored in the target_path.
                   
    target_field and 
    param_value  - target_field and param_value are keys to access the
                   array in look_at_dict which is extended (using numpy
                   vstack function) with the array loaded from target_path;
                   specifically,
                   look_at_dict[target_field][param_value] = current array
                   or, if param_value == None,
                   look_at_dict[target_field] = current array;
                   if target_field and param_value are not found in 
                   look_at_dict, these are initialized using the array
                   loaded form target_path.
                   
    avg_axis     - (default None) axis along which values in the loaded
                   numpy array are averaged if it is two-dimensional.  """

    if os.path.exists(target_path):
                      
       X = np.loadtxt(target_path)
       
       if len(X.shape) == 1:
       
          X = np.reshape(X,[1,-1])
          
       elif len(X.shape) == 2:
           
          assert avg_axis != None
           
          X = np.mean(X, axis = avg_axis)
          
          X = np.reshape(X, [1,-1])
       
       if target_field in look_at_dict.keys():
           
          if param_value != None and \
          (param_value in look_at_dict[target_field].keys()):
             
             look_at_dict[target_field][param_value] =\
             np.vstack((look_at_dict[target_field][param_value], X))
             
          elif param_value != None and \
          (param_value not in look_at_dict[target_field].keys()):
            
             look_at_dict[target_field][param_value] = X
          
          else:
             
             look_at_dict[target_field] =\
              np.vstack((look_at_dict[target_field], X))
          
       else:
          
          if param_value != None:
              
             look_at_dict[target_field] ={}
             
             look_at_dict[target_field][param_value] = X
        
          else:
           
             look_at_dict[target_field] = X
          
##############################################################################  
def plot_temporal_data(list_target_dirs,
                       target_dict,
                       xlabel_dict,
                       ylabel_dict,
                       file_name,
                       param_dict_name,
                       regressor= None,
                       algorithm_spec = None,
                       average_over_axis = None,
                       end_values_dict = None,
                       error_bars = False):
    
    """ function to plot temporal data.
    
    list_target_dirs  - list of full paths to individual training experiments.
    
    target_dict       - dictionary whose keys are names of the algorithms
                        and stored values are names of target files from
                        which temporal data must be extracted.
    
    xlabel_dict       - dictionary whose keys are the names of the algorithms
                        and the corresponding values are x-axis labels for
                        algorithm-specific plots.
    
    ylabel_dict       - dictionary whose keys are the names of the algorithms
                        and the corresponding values are y-axis labels for
                        algorithm-specific plots.
    
    file_name         - string identifier for naming figures generated
                        by this function.
    
    param_dict_name   - name of json file which stores dictionary of
                        global and algorithms-specific parameters.
    
    regressor         - (default None) name of regressor variable 
                        (e.g. "num_samples").
    
    algorithm_specific- (default None) optional boolean indicator of whether
                        a given regressor is specific to one of the algorithms.
    
    average_over_axis - (default None) axis along which values 
                        are averaged if numpy arrays loaded 
                        from target files are two-dimensional. 
    
    end_value_dict    - (default None) optional dictionary whose keys are 
                        the names of the algorithms
                        and the corresponding values are y-axis labels for
                        the plots of the last values of temporal data
                        arrays.
    
    error_bars        - (default False) boolean indicator of whether to
                        to plot data with error bars. """
    
    all_records = {}
    
    num_runs = 0
    
    for target_dir in list_target_dirs:
        
        if isinstance(regressor, str) and isinstance(algorithm_spec, bool):
           regressor = regressor.lower() 
           print("Regressor is specified: %s"%regressor)
           path_to_params = os.path.join(target_dir, param_dict_name)
           
           regressor_val = utils.get_regressor_value(path_to_params, 
                                                     regressor,
                                                     algorithm_spec)
           
           assert regressor_val != None
           
           if not isinstance(regressor_val, list):
               
              all_reg_values = []
               
              all_reg_values.append(regressor_val)
              
           else:
               
              all_reg_values = regressor_val
              
           if "_" in regressor:
               split_reg_name = regressor.split("_")
               param_name = split_reg_name[0][0].upper()+\
               split_reg_name[1][0].upper()
           else:
              param_name = regressor
               
        else:
           print("Regressor is not specified")
           all_reg_values = [None]
        
        for sub_folder in os.listdir(target_dir):
    
            selected_path = os.path.join(target_dir, sub_folder)
    
            if os.path.isdir(selected_path): # and "run" in sub_folder:
    
               print("Processing %s"%sub_folder)
              
               num_runs +=1
           
               list_all = os.listdir(selected_path)
           
               for sub_item in list_all:
               
                   check_path = os.path.join(selected_path, sub_item)
               
                   if os.path.isdir(check_path):
                       
                      for field in target_dict.keys():
                          
                          if field in sub_item:
                             
                             found_regressor = False 
                             
                             if regressor in sub_item:
                                
                                get_val = sub_item.split(regressor)[1]
                                
                                for reg_val in all_reg_values:
                                    
                                    if str(reg_val) == get_val or\
                                    len(all_reg_values) ==1:
                                       
                                       check_file =\
                                       os.path.join(check_path, 
                                                 target_dict[field])
                                       
                                       add_data(target_path = check_file, 
                                                look_at_dict= all_records,
                                                target_field = field,
                                                param_value = reg_val,
                                                avg_axis = average_over_axis)
                                    
                                       found_regressor = True
                                       break
                                       
                             if not found_regressor:
                                for reg_val in all_reg_values:
                                    
                                    check_file =\
                                    os.path.join(check_path, 
                                                 target_dict[field])
                                    
                                    add_data(target_path = check_file, 
                                             look_at_dict= all_records,
                                             target_field = field,
                                             param_value = reg_val,
                                             avg_axis = average_over_axis)
                                                 
                             break
                            
        ###################################################################
        if len(list_target_dirs) == 1 and (not isinstance(regressor, str)):
           
           if error_bars:
            
              all_records_std = {}
             
           else:
               
              all_records_std = None
           
           for field in all_records.keys():
               
               if isinstance(all_records_std, dict):
    
                  all_records_std[field] = np.std(all_records[field], axis= 0)
    
               all_records[field] = np.mean(all_records[field], axis =0)
               
           save_plot_path = os.path.join(target_dir, "%s.jpeg"%file_name)
           
           plot_sequences(means_dict   = all_records, 
                          xlabel_dict  = xlabel_dict,
                          ylabel_dict  = ylabel_dict,
                          save_to_path = save_plot_path,
                          std_dict     = all_records_std)
        ################################################################
        if len(list_target_dirs) == 1 and isinstance(regressor, str):  
            
           if error_bars:
              all_records_std = {}
           else:
              all_records_std = None    
              
           for algorithm in all_records.keys():
               
               if error_bars:
                  all_records_std[algorithm] = {}
              
               for x in all_records[algorithm].keys():
                   
                   if error_bars:
                      all_records_std[algorithm][x] = \
                      np.std(all_records[algorithm][x], axis= 0)
    
                   all_records[algorithm][x] = \
                   np.mean(all_records[algorithm][x], axis =0)
               
           root_dir = os.path.split(list_target_dirs[0])[0]
           
           save_plot_path = os.path.join(target_dir, 
                                     "%s_%s.jpeg"%(file_name,regressor))
           
           plot_sequences(means_dict   = all_records, 
                          xlabel_dict  = xlabel_dict,
                          ylabel_dict  = ylabel_dict,
                          save_to_path = save_plot_path,
                          param_name   = param_name,
                          std_dict     = all_records_std)
                          
           if end_values_dict != None:
               
              save_plot_path = os.path.join(target_dir, 
                                            "END_%s_%s.jpeg"%
                                            (file_name,regressor))
              
              plot_end_values(means_dict   = all_records, 
                              xlabel_dict  = xlabel_dict,
                              ylabel_dict  = end_values_dict,
                              save_to_path = save_plot_path, 
                              param_name   = param_name, 
                              std_dict     = all_records_std)
    ####################################################################
    if len(list_target_dirs) > 1 and isinstance(regressor, str):
        
       if error_bars:
          all_records_std = {}
       else:
          all_records_std = None
           
       for algorithm in all_records.keys():
           
           if error_bars:
               
              all_records_std[algorithm] = {}
              
           for x in all_records[algorithm].keys():
               
               if error_bars:
    
                  all_records_std[algorithm][x] = \
                  np.std(all_records[algorithm][x], axis= 0)
    
               all_records[algorithm][x] = \
               np.mean(all_records[algorithm][x], axis =0)
               
       root_dir = os.path.split(list_target_dirs[0])[0]
       
       save_plot_path = os.path.join(root_dir, 
                                     "%s_%s.jpeg"%(file_name,regressor))
       
       plot_sequences(means_dict   = all_records, 
                      xlabel_dict  = xlabel_dict,
                      ylabel_dict  = ylabel_dict,
                      save_to_path = save_plot_path,
                      param_name   = param_name,
                      std_dict     = all_records_std)
                      
       if end_values_dict != None:
               
          save_plot_path = os.path.join(root_dir, 
                                        "END_%s_%s.jpeg"%(file_name,regressor))
              
          plot_end_values(means_dict   = all_records, 
                          xlabel_dict  = xlabel_dict,
                          ylabel_dict  = end_values_dict,
                          save_to_path = save_plot_path, 
                          param_name   = param_name, 
                          std_dict     = all_records_std)
########################################################################
def plot_recon_errors(list_target_dirs,
                      list_algorithms,
                      param_dict_name,
                      regressor= None,
                      algorithm_spec = None,
                      error_bars = False):
    
    """ function to generate plots of reconstruction errors.
    
    list_target_dirs  - list of full paths to individual training experiments.
    
    list_algorithms   - list of target algorithms (e.g. ["CSS", "PCD1", "CD1"]).
                       
    param_dict_name   - name of json file that stores the parameter dictionary.
                       
    regressor         - (default None) name of regressor variable 
                        (e.g. "num_samples").
    
    algorithm_specific- (default None) optional boolean indicator of whether
                        a given regressor is specific to one of the 
                        algorithms. 
    
    error_bars        - (default False) boolean indicator of whether to
                        to plot data with error bars.                   """
    
    num_runs = 0
    
    regressor_values = []
    
    dict_to_update   = {}
    
    for target_dir in list_target_dirs:
        print("Processing %s"%target_dir)
        if isinstance(regressor, str) and isinstance(algorithm_spec, bool):
           regressor = regressor.lower()
           print("Regressor is specified: %s"%regressor)
           path_to_params = os.path.join(target_dir, param_dict_name)
           
           regressor_val = utils.get_regressor_value(path_to_params, 
                                                     regressor,
                                                     algorithm_spec)
           
           assert regressor_val != None
           
           if not isinstance(regressor_val, list):
               
              regressor_values.append(regressor_val)
           
        else:
           print("Regressor is not specified")
           regressor_val = None
                   
        for sub_folder in os.listdir(target_dir):
    
            selected_path = os.path.join(target_dir, sub_folder)
            
            if "MEAN_RECON_ERRORS" in selected_path:
                
               with open(selected_path, 'r') as json_file:
    
                    means_dict = json.load(json_file)
                
            if "STD_RECON_ERRORS"  in selected_path:
                
               with open(selected_path, 'r') as json_file:
    
                    std_dict = json.load(json_file)
                    
        assert means_dict != None
        assert std_dict   != None
        
        ### if no regressor is provided bar plots are regenerated 
        ### for each individual experiment
        if regressor == None:
            
           dict_of_lists = process_err_dict(means_dict,
                                            list_algorithms,
                                            std_dict,
                                            bar_plots = True)
            
           save_to_path = os.path.join(target_dir ,"BAR_ERRORS.jpeg")
           
           num_bars = len(dict_of_lists['LABELS'])
           
           display_recon_errors(array_dict   = dict_of_lists,
                                num_exps     = num_bars,            
                                save_to_path = save_to_path,
                                plot_std     = error_bars)
                                    
        else: 
            
           dict_to_update, found_reg_list =\
           process_err_dict(means_dict,
                            list_algorithms,
                            std_dict,
                            dict_to_update,
                            bar_plots = False)
                            
           if len(list_target_dirs) ==1  and (algorithm_spec == True):     
               
              regressor_values = found_reg_list
              
              dict_to_update = \
              utils.tile_the_lists(dict_lists = dict_to_update,
                                   num_reg_values = len(regressor_values))
                              
    if regressor != None:
       
       if len(list_target_dirs) > 1:
          root_dir    = os.path.split(list_target_dirs[0])[0]
          save_plot_to = os.path.join(root_dir,
                                      "RECON_ERRORS_%s.jpeg"%regressor)
                                      
       elif len(list_target_dirs) ==1:
          save_plot_to = os.path.join(list_target_dirs[0],
                                      "RECON_ERRORS_%s.jpeg"%regressor)
                                      
       if "_" in regressor:
          split_reg_name = regressor.split("_")
          x_label = split_reg_name[0][0].upper()+\
          split_reg_name[1][0].upper()
           
       else:
           
          x_label = regressor
          
       plot_regressions(y_dict = dict_to_update,
                        x_values = regressor_values,
                        save_to_path = save_plot_to,
                        x_label  = x_label,
                        y_label  = "Reconstruction Errors",
                        plot_std = error_bars)                                                       
########################################################################

      
      
      
      
      
        
        
        
        

