""" 
Author: Rokas Stanislovas
MSc Project: Complementary Sum Sampling 
for Learning in Boltzmann Machines
MSc Computational Statistics and 
Machine Learning

Script to generate various plots.
"""

import subprocess
import os
import sys
import json
import argparse
import numpy as np
import plot_utils

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--target_dir', type=str,required= True)

arg_parser.add_argument('--avg_w_norms', type=str,required= False)

arg_parser.add_argument('--avg_recon_errors', type=str,required= False)

arg_parser.add_argument('--avg_p_tilda', type=str,required= False)

arg_parser.add_argument('--regressor_name', type=str,required= False)

arg_parser.add_argument('--algorithm_specific', type=str,required= False)

arg_parser.add_argument('--include_paths_with', type=str,required= False)

arg_parser.add_argument('--css_gibbs', type=str,required= False)

FLAGS, _    = arg_parser.parse_known_args()

target_dir  = FLAGS.target_dir

if FLAGS.avg_w_norms != None:

   avg_w_norms = bool(int(FLAGS.avg_w_norms))
   
else:
    
   avg_w_norms = False
   
if FLAGS.avg_recon_errors != None:

   avg_recon_errors = bool(int(FLAGS.avg_recon_errors))
   
else:
    
   avg_recon_errors = False
   
if FLAGS.avg_p_tilda != None:

   avg_p_tilda = bool(int(FLAGS.avg_p_tilda))
   
else:
    
   avg_p_tilda = False
   
if FLAGS.css_gibbs != None:
    
   css_gibbs = bool(int(FLAGS.css_gibbs))
   
else:
    
   css_gibbs = False

regressor_name     = FLAGS.regressor_name

include_paths_with = FLAGS.include_paths_with

algorithm_specific = None

all_target_dirs    = []

if regressor_name != None:
    
   assert FLAGS.algorithm_specific != None

   algorithm_specific = bool(int(FLAGS.algorithm_specific))
   
   if not algorithm_specific:
   
      for sub_folder in os.listdir(target_dir):
       
          exp_path = os.path.join(target_dir, sub_folder)
    
          if os.path.isdir(exp_path) and regressor_name in sub_folder:
          
             all_target_dirs.append(exp_path)
             
   else:
       
     all_target_dirs.append(target_dir)
          
elif include_paths_with != None:
   
   for sub_folder in os.listdir(target_dir):
       
       exp_path = os.path.join(target_dir, sub_folder)
    
       if os.path.isdir(exp_path) and include_paths_with in sub_folder:
          
          all_target_dirs.append(exp_path)  
else:
   
   all_target_dirs.append(target_dir)
                                                     
########################################################################
if __name__ == "__main__":
   if avg_w_norms:
      print("Will generate plots of w norms averaged over multiple runs")
      target_dict = {"CSS":"W_NORMS.dat",
                     "PCD1":"W_NORMS.dat",
                     "CD1": "W_NORMS.dat"}
                     
      xlabel_dict = {"CSS":'Iteration number',
                     "PCD1":'Iteration number',
                     "CD1":'Iteration number'}
                     
      ylabel_dict = {"CSS":'L2-norm on W',
                     "PCD1":'L2-norm on W',
                     "CD1" :'L2-norm on W'}               
      
      if css_gibbs:
         target_dict["CSS_GIBBS"] = "W_NORMS.dat" 
         xlabel_dict["CSS_GIBBS"] = 'Iteration number'
         ylabel_dict["CSS_GIBBS"] = "L2-norm on W"
         del target_dict["CSS"] 
         del xlabel_dict["CSS"]
         del ylabel_dict["CSS"]
         
      plot_utils.plot_temporal_data(list_target_dirs = all_target_dirs,
                                    target_dict = target_dict,
                                    xlabel_dict = xlabel_dict,
                                    ylabel_dict = ylabel_dict,
                                    file_name        = "MEAN_W_NORMS",
                                    param_dict_name  = "PARAMETERS.json",
                                    regressor        = regressor_name,
                                    algorithm_spec   = algorithm_specific,
                                    average_over_axis= None,
                                    error_bars       = False)
                       
   if avg_recon_errors:
      print("Will generate plots of reconstruction errors")
      list_algorithms = ["CSS", "PCD1", "CD1"]
      if css_gibbs:
         list_algorithms.append("CSS_GIBBS")
         del list_algorithms[0]
      
      plot_utils.plot_recon_errors(list_target_dirs = all_target_dirs,
                                   list_algorithms  = list_algorithms,
                                   param_dict_name  = "PARAMETERS.json",
                                   regressor        = regressor_name,
                                   algorithm_spec   = algorithm_specific,
                                   error_bars       = True)
                        
   if avg_p_tilda:
      print("Will generate plots of p tilda values during training time"+\
      " averaged over training points")
      target_dict = {"CSS":"TRAIN_P_TILDA.dat",
                     "PCD1":"TRAIN_PSEUDO_LOSSES.dat",
                     "CD1" :"TRAIN_PSEUDO_LOSSES.dat"}
                     
      xlabel_dict = {"CSS":'Iteration number',
                     "PCD1":'Iteration number',
                     "CD1" :'Iteration number'}
                     
      ylabel_dict = {"CSS":'P tilda',
                     "PCD1":'Pseudo Likelihood',
                     "CD1" :'Pseudo Likelihood'}
                     
      end_values_dict = {"CSS":'P tilda'}
      
      if css_gibbs:
         target_dict["CSS_GIBBS"]     = "TRAIN_P_TILDA.dat"
         xlabel_dict["CSS_GIBBS"]     = "Iteration number"
         ylabel_dict["CSS_GIBBS"]     = "P tilda"
         end_values_dict["CSS_GIBBS"] = "P tilda"
         del target_dict["CSS"]
         del xlabel_dict["CSS"]
         del ylabel_dict["CSS"]
         del end_values_dict["CSS"]
      
      plot_utils.plot_temporal_data(list_target_dirs = all_target_dirs,
                                    target_dict = target_dict,
                                    xlabel_dict = xlabel_dict,
                                    ylabel_dict = ylabel_dict,
                                    file_name        = "LEARNING_CURVES",
                                    param_dict_name  = "PARAMETERS.json",
                                    regressor        = regressor_name,
                                    algorithm_spec   = algorithm_specific,
                                    average_over_axis= 1,
                                    end_values_dict  = end_values_dict,
                                    error_bars       = False)
      
      
      
      
      
        
        
        
        

