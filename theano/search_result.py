""" 
Author: Rokas Stanislovas
MSc Project: Likelihood Approximations
for Energy-Based Models
MSc Computational Statistics and 
Machine Learning
Script to regenerate samples.
"""

import subprocess
import os
import argparse
import sys
import numpy as np
import json
import plot_utils

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--target_dir', type=str,required= True)

FLAGS, _    = arg_parser.parse_known_args()

target_dir  = FLAGS.target_dir

target_file_names = {'NOISY'  :'RECONST_NOISY_ERRORS.dat',
                     'MISSING':'RECONST_MISSING_ERRORS.dat'
                     }
                     
best_runs        = {}

check_min_errors = {}

worst_runs       = {}

check_max_errors = {}

for sub_f1 in os.listdir(target_dir):
    
    sub_dir = os.path.join(target_dir, sub_f1)
    
    if os.path.isdir(sub_dir): 
        
       print("Processing %s"%sub_dir)
       
       for sub_f2 in os.listdir(sub_dir):
           
           sub_dir2 = os.path.join(sub_dir, sub_f2)
           
           if os.path.isdir(sub_dir2):
              
              for file_name in os.listdir(sub_dir2):
                  
                  for exp in target_file_names.keys():
                      
                      if file_name == target_file_names[exp]:
                          
                         if sub_f2 not in best_runs.keys():
                            
                            best_runs[sub_f2] =    {}
                           
                            check_min_errors[sub_f2] = {}
                            
                         if sub_f2 not in worst_runs.keys():
                            
                            worst_runs[sub_f2] =    {}
                           
                            check_max_errors[sub_f2] = {}
                          
                         path_to = os.path.join(sub_dir2, file_name)
                         
                         errors = np.loadtxt(path_to)
                         
                         mean_error = np.mean(errors)
                         
                         if exp not in best_runs[sub_f2].keys():
                            
                            best_runs[sub_f2][exp] = sub_f1 #run directory
                            
                            check_min_errors[sub_f2][exp] = mean_error
                            
                         else: 
                             
                            if mean_error < check_min_errors[sub_f2][exp]:
                                
                               best_runs[sub_f2][exp]    = sub_f1
                               
                               check_min_errors[sub_f2][exp] = mean_error
                               
                         if exp not in worst_runs[sub_f2].keys():
                            
                            worst_runs[sub_f2][exp] = sub_f1 #run directory
                            
                            check_max_errors[sub_f2][exp] = mean_error
                            
                         else: 
                             
                            if mean_error > check_max_errors[sub_f2][exp]:
                                
                               worst_runs[sub_f2][exp]    = sub_f1
                               
                               check_max_errors[sub_f2][exp] = mean_error
                                 
print("Runs with the lowest reconstruction errors:")
print(best_runs)
print("Lowest mean errors:")
print(check_min_errors)
print("Runs with the highest reconstruction errors:")
print(worst_runs)
print("Lowest mean errors:")
print(check_max_errors)
########################################################################
best_runs_path = os.path.join(target_dir,"BEST_RUNS.json")

best_runs_err_path  = os.path.join(target_dir,"BEST_RUNS_ERRORS.json")

with open(best_runs_path, 'w') as json_file:
    
     json.dump(best_runs, json_file) 
     
with open(best_runs_err_path, 'w') as json_file:
    
     json.dump(check_min_errors, json_file)
#######################################################################     
worst_runs_path = os.path.join(target_dir,"WORST_RUNS.json")

worst_runs_err_path  = os.path.join(target_dir,"WORST_RUNS_ERRORS.json")

with open(worst_runs_path, 'w') as json_file:
    
     json.dump(worst_runs, json_file) 
     
with open(worst_runs_err_path, 'w') as json_file:
    
     json.dump(check_max_errors, json_file)
     
num_algorithms = len(check_min_errors.keys())
#######################################################################
dict_of_lists = plot_utils.process_err_dict(means_dict = check_min_errors,
                                            bar_plots = True)

save_bar_plots_to =  os.path.join(target_dir,"MIN_BAR_ERRORS.jpeg")

plot_utils.generate_bar_plots(array_dict   = dict_of_lists,
                              num_exps     = num_algorithms,            
                              save_to_path = save_bar_plots_to,
                              ylabel = "Minimum Reconstruction Error",
                              plot_std     = False)
#######################################################################                            
dict_of_lists = plot_utils.process_err_dict(means_dict = check_max_errors,
                                            bar_plots = True)

save_bar_plots_to =  os.path.join(target_dir,"MAX_BAR_ERRORS.jpeg")

plot_utils.generate_bar_plots(array_dict   = dict_of_lists,
                              num_exps     = num_algorithms,            
                              save_to_path = save_bar_plots_to,
                              ylabel = "Maximum Reconstruction Error",
                              plot_std     = False)
                         
                         
                          
                         
                  
               
              
           
           
       
       
    
       
