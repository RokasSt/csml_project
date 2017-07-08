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

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--target_dir', type=str,required= True)

FLAGS, _    = arg_parser.parse_known_args()

target_dir  = FLAGS.target_dir

for sub_folder in os.listdir(target_dir):
    
    print("Processing %s"%sub_folder)
    
    sub_dir = os.path.join(target_dir, sub_folder)
    
    all_files = os.listdir(sub_dir)
    
    if  "LEARNT_INSTANCES.dat" in all_files:
        
        num_chains = 10
        
        if  "BS1N" in sub_dir:
        
            num_chains = 1
             
        only_subset = 1
         
    else:
        
        num_chains = 10
        
        only_subset = 0
        
    command_string = "python test_bm.py "
        
    command_string+= " --num_samples 5"
        
    command_string+= " --num_chains %d"%num_chains
        
    command_string+= " --trained_subset %d"%only_subset 
    
    command_string+= " --sampler GIBBS"
    
    command_string+= " --init_with_dataset 1"
    
    all_commands = []
    
    inter_model_tag = "TRAINED_PARAMS.model"
    
    final_model_tag = "TRAINED_PARAMS_END.model"
             
    if inter_model_tag in  all_files:
       
       target_path     = os.path.join(sub_dir, inter_model_tag)
        
       command_string1 = command_string
       
       command_string1+= " --path_to_params %s"%target_path
       
       all_commands.append(command_string1)
    
    if final_model_tag in all_files:
        
       target_path     = os.path.join(sub_dir, final_model_tag)
        
       command_string2 = command_string
       
       command_string2+= " --path_to_params %s"%target_path
       
       all_commands.append(command_string2)
       
    for num_steps in [1, 1000]:
        
        for command in all_commands:
            
            final_command = command+" --num_steps %d"%num_steps
            
            print("Executing %s"%final_command)
            
            subprocess.call(final_command, shell=True)
        
        
