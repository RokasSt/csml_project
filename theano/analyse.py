""" 
Author: Rokas Stanislovas
MSc Project: Complementary Sum Sampling 
for Learning in Boltzmann Machines
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
   images, labels = utils.get_data_arrays(file_name = "mnist-original")

###############################

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--target_dir', type=str,required= True)

FLAGS, _       = arg_parser.parse_known_args()

target_dir     = FLAGS.target_dir

bm = BoltzmannMachine(num_vars        = D, 
                      num_hidden      = num_hidden,
                      training        = False)

if full_dataset:
    
   energy_gaps, recon_errors, test_p_tilda, w_norms =\
   bm.analyse_results(target_dir  = target_dir,
                      given_inputs= images)
                      
else:
    
   energy_gaps, recon_errors, test_p_tilda, w_norms =\
   bm.analyse_results(target_dir = target_dir)
   
std_dict  = {}

mean_dict = {}

titles    = {}

for algorithm in energy_gaps.keys():
    
    std_dict[algorithm]  = np.std(energy_gaps[algorithm])
    
    mean_dict[algorithm] = np.mean(energy_gaps[algorithm])
    
for recon_exp  in recon_errors.keys():
    
    build_title = recon_exp[0].upper() + recon_exp[1:].lower()
    
    build_title += " pixels"
    
    titles[recon_exp] = build_title

means_and_labels = plot_utils.process_dict_to_list(mean_dict)

list_of_means  = means_and_labels['Y']

list_of_labels = means_and_labels['LABELS']

std_and_labels = plot_utils.process_dict_to_list(std_dict)

list_of_stds   = std_and_labels['Y']

save_to_path = os.path.join(target_dir, "BAR_ENERGY_GAPS.jpeg")

bar_title = "Difference between Data Term and Complementary Term"

plot_utils.generate_bar_plot(y_list = list_of_means,
                             labels = list_of_labels,
                             save_to_path = save_to_path,
                             ylabel = "Log (Difference)",
                             title = bar_title,
                             std_list = list_of_stds)
                             
save_to_path = os.path.join(target_dir, "ENERGY_GAPS_VS_ERRORS.jpeg")
                             
plot_utils.generate_scatter_plot(x_dict  = recon_errors, 
                                 y_dict  = energy_gaps,
                                 x_label = "Average Reconstruction Error",
                                 y_label = "Log(Data Term - IS Term)", 
                                 title_dict = titles,
                                 save_to_path = save_to_path)
                                 
save_to_path = os.path.join(target_dir, "ENERGY_GAPS_VS_P_TILDA.jpeg")
                                 
plot_utils.generate_scatter_plot(x_dict  = recon_errors, 
                                 y_dict  = test_p_tilda,
                                 x_label = "Average Reconstruction Error",
                                 y_label = "Average p tilda", 
                                 title_dict   = titles,
                                 save_to_path = save_to_path)
                                 
save_to_path = os.path.join(target_dir, "ENERGY_GAPS_VS_W_NORMS.jpeg")
                                 
plot_utils.generate_scatter_plot(x_dict  = recon_errors, 
                                 y_dict  = w_norms,
                                 x_label = "Average Reconstruction Error",
                                 y_label = "L2-norm on W", 
                                 title_dict   = titles,
                                 save_to_path = save_to_path)
                                 
                             

    






              
    
               
   










