""" 
Author: Rokas Stanislovas
MSc Project: Complementary Sum Sampling 
for Learning in Boltzmann Machines
MSc Computational Statistics and 
Machine Learning
"""

import numpy as np
import argparse
import shutil
import os
import sys
import json
import matplotlib
matplotlib.use('agg',warn=False, force=True)
from matplotlib import pyplot as plt
import theano
import theano.tensor as T
import datetime
import utils
import plot_utils
import argparse
import timeit
import os
from   model_classes_v2 import BoltzmannMachine

np_rand_gen    = np.random.RandomState(1234)

N_train        = 60000

num_to_plot    = 20

num_to_reconst = 20

pflip          = 0.2

pmiss          = 0.9

num_iters      = 10

assert num_to_reconst >= num_to_plot

print("Importing data ...")
all_images, all_labels = utils.get_data_arrays()

test_images    = all_images[N_train:,:]

test_labels    = all_labels[N_train:,:]

train_images   = all_images[0:N_train,:]

train_labels   = all_labels[0:N_train,:]

D              = all_images.shape[1]

assert D == 784

arg_parser     = argparse.ArgumentParser()

arg_parser.add_argument('--target_dir', type=str,required= True)

arg_parser.add_argument('--use_train_set', type=str,required= True)

arg_parser.add_argument('--algorithm', type=str,required= False)

FLAGS, _       = arg_parser.parse_known_args()

target_dir     = FLAGS.target_dir

use_train_set  = bool(int(FLAGS.use_train_set))

split_path     = os.path.split(target_dir)

found_images   = False

images_file    = "TRAIN_IMAGES.dat"

params_file    = "PARAMETERS.json"

if use_train_set:
   print("Will run reconstruction tasks on the training set")
   for item in os.listdir(target_dir):
       if item == images_file:
          found_images = True
          break
       
   if  found_images:
       print("Found %s"%images_file)
       test_inputs = np.loadtxt(os.path.join(target_dir, images_file))
       
   else:
       test_inputs = train_images
else:
   print("Will run reconstruction tasks on the test set") 
   test_inputs    = test_images

num_inputs        = test_inputs.shape[0]

split_root_path   = target_dir.split("/run")

root_root_path    = split_root_path[0]

check_params_file = os.path.join(root_root_path, params_file)

if num_inputs > num_to_reconst:
   
   inds_to_reconst = np.random.choice(num_inputs, 
                                      num_to_reconst, 
                                      replace = False)
                                   
else:
    
   inds_to_reconst  = np.array(range(num_to_reconst))
   
if num_to_reconst > num_to_plot:
   
   inds_to_plot = np.random.choice(range(len(inds_to_reconst)), 
                                   num_to_plot, 
                                   replace = False)
                                   
else:
    
   inds_to_plot  = np.array(range(len(inds_to_reconst)))
   
test_inputs = test_inputs[inds_to_reconst,:]
                                   
if "RH" in target_dir:

   ind0 = target_dir.find("RH")
   ind1 = target_dir.find("LR")
   num_hidden = int(target_dir[ind0+2:ind1])
   
elif "RH" not in target_dir:
    
   num_hidden = 0
   
   if os.path.exists(check_params_file):
      print("%s exists"%check_params_file)
      with open(check_params_file, 'r') as json_file:
           param_dict = json.load(json_file) 
           
      num_hidden = param_dict['GLOBAL']['num_hidden']

bm = BoltzmannMachine(num_vars    = D, 
                      num_hidden  = num_hidden,
                      training    = False)
                      
if isinstance(FLAGS.algorithm, str):
    
   path_to_params = os.path.join(target_dir, FLAGS.algorithm)
   
else:
    
   path_to_params = target_dir
                      
path_to_params = os.path.join(path_to_params,"TRAINED_PARAMS_END.model")
      
bm.load_model_params(full_path = path_to_params)

mean_values = {}

std_values  = {}

###### reconstruction of missing pixels

curr_time = datetime.datetime.now().strftime("%I%M%p_%B%d_%Y" )

filename = "RECONST_MISSING_%s"%curr_time

save_to_path = os.path.join(split_path[0],filename+".jpeg")

which_pixels = utils.get_missing_pixels(gamma = pmiss, 
                                        D     = D, 
                                        N     = num_to_reconst)

images_to_reconst = np.copy(test_inputs)
blocked_images    = np.copy(test_inputs)
images_to_reconst[which_pixels] =  1
blocked_images[which_pixels]    =  0.5 #-1

images_to_reconst = bm.reconstruct_missing(num_iters    = num_iters, 
                                           recon_images = images_to_reconst, 
                                           which_pixels = which_pixels)
                                                                            
plot_utils.plot_reconstructions(test_inputs[inds_to_plot,:],
                                blocked_images[inds_to_plot,:],
                                images_to_reconst[inds_to_plot,:],
                                save_to_path)
                                         
recon_errors = np.zeros(num_to_reconst)
           
for xi in range(num_to_reconst):
    
    recon_errors[xi] =\
    utils.hamming_distance(test_inputs[xi,:], 
                           images_to_reconst[xi,:])

np.savetxt(os.path.join(target_dir,"%s_ERRORS.dat"%filename), recon_errors)

mean_val = np.mean(recon_errors)

std_val  = np.std(recon_errors)

mean_values['MISSING'] = mean_val

std_values['MISSING']  = std_val

############### reconstruction of noisy data
filename = "RECONST_NOISY_%s"%curr_time

save_to_path = os.path.join(target_dir,filename+".jpeg")

which_pixels = utils.get_noisy_pixels(pflip = pflip, 
                                      D     = D, 
                                      N     = num_to_reconst)
                                         
noisy_images = np.copy(test_inputs)

noisy_images[which_pixels] = 1- noisy_images[which_pixels]

images_to_reconst= np.copy(noisy_images)

images_to_reconst = bm.reconstruct_noisy(num_iters      = num_iters, 
                                         correct_images = test_inputs,
                                         recon_images   = images_to_reconst, 
                                         noisy_images   = noisy_images,
                                         pflip          = pflip)
                                                
plot_utils.plot_reconstructions(test_inputs[inds_to_plot,:],
                                noisy_images[inds_to_plot,:],
                                images_to_reconst[inds_to_plot,:],
                                save_to_path)
                        
recon_errors = np.zeros(num_to_reconst)
           
for xi in range(num_to_reconst):
    
    recon_errors[xi] =\
    utils.hamming_distance(test_inputs[xi,:], 
                           images_to_reconst[xi,:])
                                         
np.savetxt(os.path.join(target_dir,"%s_ERRORS.dat"%filename), recon_errors)

mean_val = np.mean(recon_errors)

std_val  = np.std(recon_errors)

mean_values['NOISY'] = mean_val

std_values['NOISY']  = std_val

save_means_to = os.path.join(target_dir,"MEAN_ERRORS_%s.json"%curr_time)

save_std_to   = os.path.join(target_dir,"STD_ERRORS_%s.json"%curr_time)

with open(save_means_to, 'w') as json_file:

     json.dump(mean_values, json_file)
     
with open(save_std_to, 'w') as json_file:

     json.dump(std_values, json_file)
     
print("Mean errors for reconstruction tasks:")
print(mean_values)

print("Standard deviations for reconstruction tasks:")
print(std_values)






