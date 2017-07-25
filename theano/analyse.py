""" 
Author: Rokas Stanislovas
MSc Project: Likelihood Approximations
for Energy-Based Models
MSc Computational Statistics and 
Machine Learning
"""
import numpy as np
import argparse
import os
import json
from   model_classes_v2 import BoltzmannMachine
import plot_utils

#############################
num_hidden = 0

D  = 784

full_dataset = False

#############################

if full_dataset:
   print("Importing data ...")
   all_train_images, all_train_labels = \
   utils.get_data_arrays(file_name = "mnist-original")

###############################

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--target_dir', type=str,required= True)

FLAGS, _       = arg_parser.parse_known_args()

target_dir     = FLAGS.target_dir

bm = BoltzmannMachine(num_vars        = D, 
                      num_hidden      = num_hidden,
                      training        = False)

if full_dataset:
   energy_gaps = bm.test_energy_gap(target_dir = target_dir,
                                    given_inputs = all_train_images)
                      
else:
   energy_gaps = bm.test_energy_gap(target_dir = target_dir)
   
std_dict  = {}

mean_dict = {}
   
for exp_tag in energy_gaps.keys():
    
    energy_gaps[exp_tag] = energy_gaps[exp_tag]/np.max(energy_gaps[exp_tag])
    
    std_dict[exp_tag]    = np.std(energy_gaps[exp_tag])
    
    mean_dict[exp_tag]   = np.mean(energy_gaps[exp_tag])

print("Means:")
print(mean_dict)
print("Standard deviations:")
print(std_dict)

means_and_labels = plot_utils.process_dict_to_list(mean_dict)

list_of_means  = means_and_labels['Y']

list_of_labels = means_and_labels['LABELS']

std_and_labels = plot_utils.process_dict_to_list(std_dict)

list_of_stds   = std_and_labels['Y']

save_to_path = os.path.join(target_dir, "BAR_ENERGY_GAPS.jpeg")

plot_utils.generate_bar_plot(y_list = list_of_means,
                             ordered_labels = list_of_labels,
                             save_to_path = save_to_path,
                             ylabel = "Normalized Energy Gap",
                             title = "Average Energy Gaps (40 runs)",
                             std_list = list_of_stds)
    






              
    
               
   










