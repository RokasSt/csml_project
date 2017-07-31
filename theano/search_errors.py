""" 
Author: Rokas Stanislovas
MSc Project: Complementary Sum Sampling 
for Learning in Boltzmann Machines
MSc Computational Statistics and 
Machine Learning
Script to search for reconstruction errors
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

arg_parser.add_argument('--regressor', type=str,required= False)

arg_parser.add_argument('--compare_with', type=str,required= False)

FLAGS, _    = arg_parser.parse_known_args()

target_dir  = FLAGS.target_dir

target_file_names = {'NOISY'  :'RECONST_NOISY_ERRORS.dat',
                     'MISSING':'RECONST_MISSING_ERRORS.dat'
                     }
                     
best_runs         = {}

check_min_errors  = {}

worst_runs        = {}

check_max_errors  = {}

check_means_file  = os.path.join(target_dir, "MEAN_RECON_ERRORS.json")

check_std_file    = os.path.join(target_dir, "STD_RECON_ERRORS.json")

get_stats = False

target_fields = []

if not os.path.exists(check_means_file) or (not os.path.exists(check_std_file)):
    
   get_stats   = True
   error_dict = {}
   
for sub_f1 in os.listdir(target_dir):
    
    sub_dir = os.path.join(target_dir, sub_f1)
    
    if os.path.isdir(sub_dir): 
        
       print("Processing %s"%sub_dir)
       
       for sub_f2 in os.listdir(sub_dir):
           
           sub_dir2 = os.path.join(sub_dir, sub_f2)
           
           if os.path.isdir(sub_dir2):
               
              alg = sub_f2
              
              if isinstance(FLAGS.regressor, str):
                 if FLAGS.regressor in alg:
                    alg = alg.replace(FLAGS.regressor, "")
              if "_" in alg:
                 alg = alg.replace("_", " ")
            
              target_fields.append(alg)
              
              for file_name in os.listdir(sub_dir2):
                  
                  for exp in target_file_names.keys():
                      
                      if file_name == target_file_names[exp]:
                          
                         if get_stats:
                            if alg not in error_dict.keys():
                               error_dict[alg] = {}
                         
                         if alg not in best_runs.keys():
                            
                            best_runs[alg] =    {}
                           
                            check_min_errors[alg] = {}
                            
                         if alg not in worst_runs.keys():
                            
                            worst_runs[alg] =    {}
                           
                            check_max_errors[alg] = {}
                          
                         path_to = os.path.join(sub_dir2, file_name)
                         
                         errors = np.loadtxt(path_to)
                         
                         mean_error = np.mean(errors)
                         
                         if get_stats:
                            if exp not in error_dict[alg].keys():
                               error_dict[alg][exp] = []
                               error_dict[alg][exp].append(mean_error) 
                                
                            else:
                               error_dict[alg][exp].append(mean_error)  
                         
                         if exp not in best_runs[alg].keys():
                            
                            best_runs[alg][exp] = sub_f1 #run directory
                            
                            check_min_errors[alg][exp] = mean_error
                            
                         else: 
                             
                            if mean_error < check_min_errors[alg][exp]:
                                
                               best_runs[alg][exp]    = sub_f1
                               
                               check_min_errors[alg][exp] = mean_error
                               
                         if exp not in worst_runs[alg].keys():
                            
                            worst_runs[alg][exp] = sub_f1 #run directory
                            
                            check_max_errors[alg][exp] = mean_error
                            
                         else: 
                             
                            if mean_error > check_max_errors[alg][exp]:
                                
                               worst_runs[alg][exp]    = sub_f1
                               
                               check_max_errors[alg][exp] = mean_error
                                 
print("Runs with the lowest reconstruction errors:")
print(best_runs)
print("Lowest mean errors:")
print(check_min_errors)
print("Runs with the highest reconstruction errors:")
print(worst_runs)
print("Highest mean errors:")
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
target_fields  = list(set(target_fields))
#######################################################################
dict_of_lists = plot_utils.process_err_dict(means_dict = check_min_errors,
                                            target_fields = target_fields,
                                            bar_plots = True)

save_bar_plots_to =  os.path.join(target_dir,"MIN_BAR_ERRORS.jpeg")

plot_utils.display_recon_errors(array_dict   = dict_of_lists,
                                num_exps     = num_algorithms,            
                                save_to_path = save_bar_plots_to,
                                ylabel = "Minimum Reconstruction Error",
                                plot_std     = False)
#######################################################################                            
dict_of_lists = plot_utils.process_err_dict(means_dict = check_max_errors,
                                            target_fields = target_fields,
                                            bar_plots = True)

save_bar_plots_to =  os.path.join(target_dir,"MAX_BAR_ERRORS.jpeg")

plot_utils.display_recon_errors(array_dict   = dict_of_lists,
                                num_exps     = num_algorithms,            
                                save_to_path = save_bar_plots_to,
                                ylabel = "Maximum Reconstruction Error",
                                plot_std     = False)
                              
if get_stats:
    
   std_dict = {}
   
   for f1 in error_dict.keys():
       
       std_dict[f1] = {}
       
       for f2 in error_dict[f1].keys():
           
           std_dict[f1][f2]   = np.std(error_dict[f1][f2])
           
           error_dict[f1][f2] = np.mean(error_dict[f1][f2])
    
   with open(check_means_file, 'w') as json_file:
    
        json.dump(error_dict, json_file) 
                            
   with open(check_std_file, 'w') as json_file:
    
        json.dump(std_dict, json_file)
        
   dict_of_lists = plot_utils.process_err_dict(means_dict = error_dict,
                                               target_fields = target_fields,
                                               std_dict = std_dict,
                                               bar_plots = True)
                                               
   save_bar_plots_to =  os.path.join(target_dir, "BAR_ERRORS.jpeg")
   
   num_bars = len(target_fields)
        
   plot_utils.display_recon_errors(array_dict   = dict_of_lists,
                                   num_exps     = num_bars,            
                                   save_to_path = save_bar_plots_to,
                                   plot_std     = True)
   print("Means:")    
   print(error_dict)                             
   print("Standard deviation:")
   print(std_dict)
   
if FLAGS.compare_with != None:
    
   compare_path = FLAGS.compare_with
    
   path_to_means = os.path.join(compare_path, "MEAN_RECON_ERRORS.json")

   path_to_std   = os.path.join(compare_path, "STD_RECON_ERRORS.json") 
   
   if os.path.exists(path_to_means) and os.path.exists(path_to_std):
       
      print("Results for %s"%compare_path)
      
      with open(path_to_means, 'r') as json_file:
           means_to_compare = json.load(json_file)
           
      with open(path_to_std, 'r') as json_file:
           std_to_compare = json.load(json_file)
           
      print("Means:")
      print(means_to_compare) 
      print("Standard deviaton")
      print(std_to_compare)
      
   else:
     print("Error: Cannot compare results to %s"%compare_path)
     print("%s and/or %s do not exist"%(path_to_means, 
                                        path_to_std))
     sys.exit()
    
   print("Results for %s"%target_dir)
   
   if get_stats:
      print("Means")
      print(error_dict)
      print("Standard deviation:")
      print(std_dict)
      
   else:
       
      with open(check_means_file, 'r') as json_file:
           err_means = json.load(json_file)
           
      with open(check_std_file, 'r') as json_file:
           err_std = json.load(json_file)
                                   
      print("Means:")
      print(err_means)   
      print("Standard deviation:")
      print(err_std)
    
                                   

                         
                         
                          
                         
                  
               
              
           
           
       
       
    
       
