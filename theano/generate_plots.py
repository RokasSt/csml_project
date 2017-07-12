""" 
Author: Rokas Stanislovas
MSc Project: Likelihood Approximations
for Energy-Based Models
MSc Computational Statistics and 
Machine Learning
"""

import subprocess
import os
import sys
import json
import argparse
import numpy as np
import utils

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--target_dir', type=str,required= True)

arg_parser.add_argument('--avg_w_norms', type=str,required= True)

arg_parser.add_argument('--avg_recon_errors', type=str,required= True)

arg_parser.add_argument('--avg_p_tilda', type=str,required= True)

arg_parser.add_argument('--regressor_name', type=str,required= False)

arg_parser.add_argument('--algorithm_specific', type=str,required= False)

arg_parser.add_argument('--include_paths_with', type=str,required= False)

FLAGS, _    = arg_parser.parse_known_args()

target_dir  = FLAGS.target_dir

avg_w_norms        = bool(int(FLAGS.avg_w_norms))

avg_recon_errors   = bool(int(FLAGS.avg_recon_errors))

avg_p_tilda        = bool(int(FLAGS.avg_p_tilda))

regressor_name     = FLAGS.regressor_name

include_paths_with = FLAGS.include_paths_with

algorithm_specific = None

all_target_dirs    = []

if regressor_name != None:
    
   assert FLAGS.algorithm_specific != None

   algorithm_specific = bool(int(FLAGS.algorithm_specific))
   
   for sub_folder in os.listdir(target_dir):
       
       exp_path = os.path.join(target_dir, sub_folder)
    
       if os.path.isdir(exp_path) and regressor_name in sub_folder:
          
          all_target_dirs.append(exp_path)
          
elif include_paths_with != None:
   
   for sub_folder in os.listdir(target_dir):
       
       exp_path = os.path.join(target_dir, sub_folder)
    
       if os.path.isdir(exp_path) and include_paths_with in sub_folder:
          
          all_target_dirs.append(exp_path)  
else:
   
   all_target_dirs.append(target_dir)

###########################################################################
def add_data(target_path,
             look_at_dict,
             algorithm,
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
       
       if field in look_at_dict.keys():
           
          if param_value != None and \
          (param_value in look_at_dict[field].keys()):
             
             look_at_dict[field][param_value] =\
             np.vstack((look_at_dict[field][param_value], X))
             
          elif param_value != None and \
          (param_value not in look_at_dict[field].keys()):
            
             look_at_dict[field][param_value] = X
          
          else:
             
             look_at_dict[field] = np.vstack((look_at_dict[field], X))
          
       else:
          
          if param_value != None:
              
             look_at_dict[field] ={}
             
             look_at_dict[field][param_value] = X
        
          else:
           
             look_at_dict[field] = X
          
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
                       average_over_axis = None
                       error_bars = False):
    
    """ function to generate plots of w norms averaged over multiple runs"""
    
    records_all = {}
    
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
           
        else:
           print("Regressor is not specified")
           regressor_val = None
                   
        for sub_folder in os.listdir(target_dir):
    
            selected_path = os.path.join(target_dir, sub_folder)
    
            if os.path.isdir(selected_path): # and "run" in sub_folder:
    
               print("Processing %s"%sub_folder)
              
               num_runs +=1
           
               list_all = os.listdir(selected_path)
           
               for sub_item in list_all:
               
                   check_path = os.path.join(selected_path, sub_item)
               
                   if os.path.isdir(check_path):
                       
                      for field in target_dir.keys():
                          
                          if field in sub_item:
                              
                             check_file = os.path.join(check_path, 
                                                target_dict[field])
                   
                             records_all = add_data(target_path = check_file, 
                                                    look_at_dict= records_all,
                                                    target      = field,
                                                    param_value = regressor_val,
                                                    avg_axis = average_over_axis)
                              ### break inner for loop once field found                       
                              break
        
        if len(list_target_dirs) == 1 and regressor_val == None:
            
           if error_bars:
            
              all_records_std = {}
             
           else:
               
              all_records_std = None
           
           for field in all_records.keys():
               
               if isinstance(all_records_std, dict):
    
                  all_records_std[field] = np.std(all_records[field], axis= 0)
    
               all_records[field] = np.mean(all_records[field], axis =0)
               
           save_plot_path = os.path.join(target_dir, "%s.jpeg"%file_name)
           
           utils.plot_sequences(mean_dict    = all_records, 
                                xlabel_dict  = xlabel_dict,
                                ylabel_dict  = ylabel_dict,
                                save_to_path = save_plot_path,
                                std_dict     = all_records_std)
           
           
    if len(list_target_dirs) > 1 and regressor_val != None:
        
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
       # plot_w_norms -> plot_sequences
       utils.plot_sequences(mean_dict    = all_records, 
                            xlabel_dict  = xlabel_dict,
                            ylabel_dict  = ylabel_dict,
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
            
           dict_of_lists = utils.process_err_dict(means_dict,
                                                  std_dict,
                                                  bar_plots = True)
            
           save_to_path = os.path.join(target_dir ,"BAR_ERRORS.jpeg")
           
           utils.generate_bar_plots(array_dict   = dict_of_lists,
                                    num_exps     = num_algorithms,            
                                    save_to_path = save_to_path,
                                    plot_std     = error_bars)
                                    
        else:
            
           dict_to_update = utils.process_err_dict(means_dict,
                                                   std_dict,
                                                   dict_to_update,
                                                   bar_plots = False)
                                                   
    if regressor != None:
        
       root_dir    = os.path.split(list_target_dirs[0])[0]
       
       save_plot_to = os.path.join(root_dir,
                                  "RECON_ERRORS_%s.jpeg"%regressor)
        
       utils.plot_regressions(y_dict = dict_to_update,
                              x_values = regressor_values,
                              save_to_path = save_plot_to,
                              x_label  = regressor,
                              y_label  = "Reconstruction Errors",
                              plot_std = error_bars)                                                       
########################################################################
if __name__ == "__main__":
   if avg_w_norms:
      print("Will generate plots of w norms averaged over multiple runs")
      plot_temporal_data(list_target_dirs = all_target_dirs,
                         target_dict = {"CSS":"W_NORMS.dat",
                                        "PCD":"W_NORMS.dat",
                                        "CD": "W_NORMS.dat",
                                        },
                         xlabel_dict = {"CSS":'Iteration number',
                                        "PCD":'Iteration number',
                                        "CD":'Iteration number',
                                        },
                         ylabel_dict = {"CSS":'L2-norm on W',
                                        "PCD":'L2-norm on W',
                                        "CD" :'L2-norm on W',
                                        },
                         file_name        = "MEAN_W_NORMS",
                         param_dict_name  = "PARAMETERS.json",
                         regressor        = regressor_name,
                         algorithm_spec   = algorithm_specific,
                         average_over_axis= None,
                         error_bars       = False)
                       
   if avg_recon_errors:
      print("Will generate plots of reconstruction errors")
      
      plot_recon_errors(list_target_dirs = all_target_dirs,
                        list_experiments = ["CSS", "PCD", "CD"],
                        param_dict_name  = "PARAMETERS.json",
                        regressor        = regressor_name,
                        algorithm_spec   = algorithm_specific,
                        error_bars       = True)
                        
   if avg_p_tilda:
      print("Will generate plots of p tilda values over training time"+\
      " averaged over training points")
      
      plot_temporal_data(list_target_dirs = all_target_dirs,
                         target_dict = {"CSS":"TRAIN_P_TILDA.dat",
                                        "PCD":"TRAIN_PSEUDO_LOSSES.dat",
                                         "CD" :"TRAIN_PSEUDO_LOSSES.dat",
                                        },
                         xlabel_dict = {"CSS":'Iteration number',
                                        "PCD":'Iteration number',
                                        "CD":'Iteration number',
                                        },
                         ylabel_dict = {"CSS":'Average P tilda',
                                        "PCD":'Average pseudo likelihood',
                                        "CD" :'Average pseudo likelihood',
                                        },
                         file_name        = "LEARNING_CURVES",
                         param_dict_name  = "PARAMETERS.json",
                         regressor        = regressor_name,
                         algorithm_spec   = algorithm_specific,
                         average_over_axis= 1,
                         error_bars       = False)
      
      
      
      
      
        
        
        
        

