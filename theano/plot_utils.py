""" 
Author: Rokas Stanislovas
MSc Project: Likelihood Approximations
for Energy-Based Models
MSc Computational Statistics and 
Machine Learning 
"""

import matplotlib
matplotlib.use('agg',warn=False, force=True)
from matplotlib import pyplot as plt
import subprocess
import os
import sys
import json
import argparse
import numpy as np
import utils

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
########################################################################
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
    their reconstructions under different training algorithms"""
    
    num_reconstruct = correct_images.shape[0]
    
    num_rows = len(reconstructed_images.keys()) + 2
    
    num_cols = num_reconstruct
    
    _, ax = plt.subplots(num_rows, num_cols, sharex=False,
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
def plot_sequences(means_dict, 
                   xlabel_dict,
                   ylabel_dict,
                   save_to_path, 
                   param_name ="", 
                   std_dict = None):
    
    """function to plot temporal sequences"""
    
    num_rows = len(means_dict.keys())
    
    num_cols = 1
    
    _, ax = plt.subplots(num_rows, num_cols, sharex=False,
    figsize=  ( 9*num_cols, 3*num_rows) )
    
    ax = ax.ravel()
    
    plot_index = 0
    
    use_legend = False
    
    for exp_tag in means_dict.keys():
        
        ax[plot_index].set_title(exp_tag, size = 13) 
        
        if not isinstance(means_dict[exp_tag], dict):
        
           max_val = np.max(means_dict[exp_tag])
           
           min_val = np.min(means_dict[exp_tag])
           
           max_val = max_val + 0.2*max_val
           
           iters = range(len(means_dict[exp_tag]))
           
           if isinstance(std_dict, dict):
        
              ax[plot_index].errorbar(iters,
                                      means_dict[exp_tag],
                                      yerr= std_dict[exp_tag])
                                   
           else:
            
              ax[plot_index].plot(iters, means_dict[exp_tag])
              
        else:
            
           use_legend = True
           
           for x_val in means_dict[exp_tag].keys():
               
               max_val = np.max(means_dict[exp_tag][x_val])
               
               min_val = np.min(means_dict[exp_tag][x_val])
               
               max_val = max_val + 0.2*max_val
    
               iters = range(len(means_dict[exp_tag][x_val]))
               
               if isinstance(std_dict, dict):
        
                  ax[plot_index].errorbar(iters,
                                          means_dict[exp_tag][x_val],
                                          yerr= std_dict[exp_tag][x_val],
                                          label = "%s %s"%(param_name,str(x_val)))
                                   
               else:
            
                  ax[plot_index].plot(iters, 
                                      means_dict[exp_tag][x_val],
                                      label ="%s %s"%(param_name,str(x_val)))
        
        ax[plot_index].set_xlabel(xlabel_dict[exp_tag])
    
        ax[plot_index].set_ylabel(ylabel_dict[exp_tag])
        
        ax[plot_index].locator_params(nbins=8, axis='y')
        
        ## alternative to using locat_params:
        #if min_val > 0:
           #ax[plot_index].yaxis.set_ticks(np.arange(0, max_val, max_val/5))
           #pass
        #else:
           #r = max_val-min_val 
           #ax[plot_index].yaxis.set_ticks(np.arange(min_val, max_val, r/5))
        
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
                    param_name ="", 
                    std_dict = None):
    
    """plot last values from the learning curves"""
    
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
           
              ax[plot_index].errorbar(x_axis, y_axis, yerr  = y_axis_std)
              
           else:
            
              ax[plot_index].plot(x_axis, y_axis)
                               
                               
           ax[plot_index].set_title(exp_tag, size = 13)  
        
           ax[plot_index].set_xlabel(param_name)
    
           ax[plot_index].set_ylabel(ylabel_dict[exp_tag])
        
           ax[plot_index].locator_params(nbins=8, axis='y')
        
           plot_index +=1
           
        else:
        
           if y_axis_std != []:
           
              ax.errorbar(x_axis, y_axis, yerr  = y_axis_std)
              
           else:
            
              ax.plot(x_axis, y_axis)
                               
           ax.set_title(exp_tag, size = 13)  
        
           ax.set_xlabel(param_name)
    
           ax.set_ylabel(ylabel_dict[exp_tag])
        
           ax.locator_params(nbins=8, axis='y')
           
    plt.tight_layout()
    
    plt.savefig(save_to_path, bbox_inches='tight')
       
    plt.clf()
##########################################################################    
def process_err_dict(means_dict, 
                     std_dict, 
                     update_dict = None, 
                     bar_plots = True):
    
    """ function to process error dictionary into lists for plotting"""
    
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
           
           if (not isinstance(means_dict[alg]['MISSING'], dict)) and\
           (not isinstance(means_dict[alg]['NOISY'], dict)):
    
              output_dict['LABELS'].append(alg)
        
              output_dict['MISSING']['MEANS'].append(means_dict[alg]['MISSING'])
    
              output_dict['MISSING']['STD'].append(std_dict[alg]['MISSING'])
    
              output_dict['NOISY']['MEANS'].append(means_dict[alg]['NOISY'])
    
              output_dict['NOISY']['STD'].append(std_dict[alg]['NOISY'])
              
           else:
               
              #### assumes that both "MISSING" and "NOISY" fields
              #### share the same set subfields
              for field in means_dict[alg]['MISSING'].keys():
                  
                  output_dict['LABELS'].append("%s %s"%(alg,field))
                  
                  output_dict['MISSING']['MEANS'].append(\
                  means_dict[alg]['MISSING'][field])
                  
                  output_dict['MISSING']['STD'].append(\
                  std_dict[alg]['MISSING'][field])
                  
                  output_dict['NOISY']['MEANS'].append(\
                  means_dict[alg]['NOISY'][field])
                  
                  output_dict['NOISY']['STD'].append(\
                  std_dict[alg]['NOISY'][field])
                  
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
           ax[plot_index].set_xticklabels(ordered_labels,
                                          rotation= "vertical")
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
        
        #if min_val >= 0:
        
         #  ax[plot_index].yaxis.set_ticks(np.arange(0, max_val+5, (max_val+5)//5))
           
        #else:
            
        ax[plot_index].locator_params(nbins=8, axis='y')
        
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
    target file for plotting purposes"""

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
          
    return look_at_dict
##############################################################################  
def get_regressor_value(target_path, 
                        param_name,
                        algorithm_specific):
    
    """ function to obtain a regressor value from a given json file
    assumes that there is a single parameter dictionary per each individual
    algorithm (one-to-one mapping), for example
    
    exp1 - CSS
    
    exp2 - CD
    
    exp3 - PCD
    
    but then
    
    exp4 - PCD is not allowed.
    
    """
    
    param_value  = None

    with open(target_path, 'r') as json_file:
    
         param_dict = json.load(json_file)
          
    for exp_tag in param_dict.keys():
         
        exp_params = param_dict[exp_tag]
        
        if algorithm_specific:
            
           if param_name in exp_params['algorithm_dict'].keys():
           
              param_value = exp_params['algorithm_dict'][param_name] 
              
              break
           
        else:
            
           if param_name in exp_params.keys():
              
              param_value = exp_params[param_name]
             
              break
          
    return param_value
##########################################################################
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
    
    """ function to plot temporal data"""
    
    all_records = {}
    
    num_runs = 0
    
    for target_dir in list_target_dirs:
        
        if isinstance(regressor, str) and isinstance(algorithm_spec, bool):
           regressor = regressor.lower() 
           print("Regressor is specified: %s"%regressor)
           path_to_params = os.path.join(target_dir, param_dict_name)
           
           regressor_val = get_regressor_value(path_to_params, 
                                               regressor,
                                               algorithm_spec)
           
           assert regressor_val != None
           
           if not isinstance(regressor_val, list):
               
              all_reg_values = []
               
              all_reg_values.append(regressor_val)
              
           else:
               
              all_reg_values = regressor_val
           
        else:
           print("Regressor is not specified")
           all_reg_values = [None]
        print(all_reg_values)
        
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
                          print(field)
                          if field in sub_item:
                             print("%s is in %s"%(field, sub_item))
                             found_regressor = False
                             for reg_val in all_reg_values:
                                 
                                 if str(reg_val) in sub_item or\
                                 len(all_reg_values) ==1:
                              
                                    check_file =\
                                    os.path.join(check_path, 
                                                 target_dict[field])
                                    print("Processing %s"%check_file)
                                    all_records = \
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
                                    os.path.join(check_path, target_dict[field])
                                    print("Processing %s"%check_file)
                                    all_records = \
                                    add_data(target_path = check_file, 
                                             look_at_dict= all_records,
                                             target_field = field,
                                             param_value = reg_val,
                                             avg_axis = average_over_axis)
                                    
                             ### break inner for loop once field found                       
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
           print(save_plot_path)
           sys.exit()
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
           print("Printing target directory: %s"%target_dir)
           save_plot_path = os.path.join(target_dir, 
                                     "%s_%s.jpeg"%(file_name,regressor))
       
           plot_sequences(means_dict   = all_records, 
                          xlabel_dict  = xlabel_dict,
                          ylabel_dict  = ylabel_dict,
                          save_to_path = save_plot_path,
                          param_name   = regressor,
                          std_dict     = all_records_std)
                          
           if end_values_dict != None:
               
              save_plot_path = os.path.join(target_dir, 
                                        "END_%s_%s.jpeg"%(file_name,regressor))
              
              plot_end_values(means_dict   = all_records, 
                          xlabel_dict  = xlabel_dict,
                          ylabel_dict  = end_values_dict,
                          save_to_path = save_plot_path, 
                          param_name   = regressor, 
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
                      param_name   = regressor,
                      std_dict     = all_records_std)
                      
       if end_values_dict != None:
               
          save_plot_path = os.path.join(root_dir, 
                                        "END_%s_%s.jpeg"%(file_name,regressor))
              
          plot_end_values(means_dict   = all_records, 
                          xlabel_dict  = xlabel_dict,
                          ylabel_dict  = end_values_dict,
                          save_to_path = save_plot_path, 
                          param_name   = regressor, 
                          std_dict     = all_records_std)
########################################################################
def plot_recon_errors(list_target_dirs,
                      list_experiments,
                      param_dict_name,
                      regressor= None,
                      algorithm_spec = None,
                      error_bars = False):
    
    """ function to generate plots of w norms averaged over multiple runs"""
    
    num_runs = 0
    
    regressor_values = []
    
    dict_to_update   = {}
    
    for target_dir in list_target_dirs:
        print("Processing %s"%target_dir)
        if isinstance(regressor, str) and isinstance(algorithm_spec, bool):
           regressor = regressor.lower() 
           print("Regressor is specified: %s"%regressor)
           path_to_params = os.path.join(target_dir, param_dict_name)
           
           regressor_val = get_regressor_value(path_to_params, 
                                               regressor,
                                               algorithm_spec)
           
           assert regressor_val != None
           
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
        
        num_algorithms = len(means_dict.keys())
        
        assert num_algorithms == len(std_dict.keys())
        
        ### if no regressor is provided bar plots are regenerated 
        ### for each individual experiment
        if regressor == None:
            
           dict_of_lists = process_err_dict(means_dict,
                                            std_dict,
                                            bar_plots = True)
            
           save_to_path = os.path.join(target_dir ,"BAR_ERRORS.jpeg")
           
           generate_bar_plots(array_dict   = dict_of_lists,
                              num_exps     = num_algorithms,            
                              save_to_path = save_to_path,
                              plot_std     = error_bars)
                                    
        else:
            
           dict_to_update = process_err_dict(means_dict,
                                             std_dict,
                                             dict_to_update,
                                             bar_plots = False)
                                                   
    if regressor != None:
        
       root_dir    = os.path.split(list_target_dirs[0])[0]
       
       save_plot_to = os.path.join(root_dir,
                                  "RECON_ERRORS_%s.jpeg"%regressor)
        
       plot_regressions(y_dict = dict_to_update,
                        x_values = regressor_values,
                        save_to_path = save_plot_to,
                        x_label  = regressor,
                        y_label  = "Reconstruction Errors",
                        plot_std = error_bars)                                                       
########################################################################

      
      
      
      
      
        
        
        
        

