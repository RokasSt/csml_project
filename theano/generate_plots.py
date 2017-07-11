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
import sys
import argparse
import numpy as np
import utils

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--target_dir', type=str,required= True)

arg_parser.add_argument('--avg_w_norms', type=str,required= True)

FLAGS, _    = arg_parser.parse_known_args()

target_dir  = FLAGS.target_dir

avg_w_norms = bool(int(FLAGS.avg_w_norms))

w_norms_all = {}

if avg_w_norms:
   print("Will generate plots of w norms averaged over multiple runs")
   
num_runs = 0

def add_w_norm(path_to_check,
               norms_dict,
               algorithm):
                   
    """ function to add sum temporal sequences of W norms
    for plotting their averages """

    check_w_file = os.path.join(path_to_check, "W_NORMS.dat")
                   
    if os.path.exists(check_w_file):
                      
       w_norms = np.loadtxt(check_w_file)
       
       w_norms = np.reshape(w_norms,[1,-1])
       
       if algorithm in norms_dict.keys():
           
          norms_dict[algorithm] = np.vstack((norms_dict[algorithm], w_norms))
          
       else:
           
          norms_dict[algorithm] = w_norms
          
    return norms_dict
                   
for sub_folder in os.listdir(target_dir):
    
    selected_path = os.path.join(target_dir, sub_folder)
    
    if os.path.isdir(selected_path):
    
       print("Processing %s"%sub_folder)
    
       if  "run" in sub_folder:
           
           num_runs +=1
           
           list_all = os.listdir(selected_path)
           
           for sub_item in list_all:
               
               check_path = os.path.join(selected_path, sub_item)
               
               if os.path.isdir(check_path) and "CSS" == sub_item:
                   
                  if avg_w_norms:
                      
                     w_norms_all = add_w_norm(path_to_check  = check_path, 
                                              norms_dict     = w_norms_all,
                                              algorithm      = "CSS") 
                    
               if os.path.isdir(check_path) and "CD" == sub_item:
                   
                  if avg_w_norms:
                      
                     w_norms_all = add_w_norm(path_to_check  = check_path, 
                                              norms_dict     = w_norms_all,
                                              algorithm      = "CD")  
                   
               if os.path.isdir(check_path) and "PCD" == sub_item:
                  
                  if avg_w_norms:
                      
                     w_norms_all = add_w_norm(path_to_check  = check_path, 
                                              norms_dict     = w_norms_all,
                                              algorithm      = "PCD") 
                                            
w_norms_std = {}
for algorithm in w_norms_all.keys():
    
    w_norms_std[algorithm] = np.std(w_norms_all[algorithm], axis= 0)
    
    w_norms_all[algorithm] = np.mean(w_norms_all[algorithm], axis =0)
                                              
utils.plot_w_norms(w_norms_all, 
                   os.path.join(target_dir, "MEAN_W_NORMS.jpeg"))
                   #, w_norms_std = w_norms_std)
                  
      
        
        
        
        

