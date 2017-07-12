""" 
Author: Rokas Stanislovas
MSc Project: Likelihood Approximations
for Energy-Based Models
MSc Computational Statistics and 
Machine Learning
"""

import numpy as np
import argparse
import shutil
import os
import sys
import json
import tensorflow as tf
from   tensorflow.examples.tutorials.mnist import input_data
import matplotlib
matplotlib.use('agg',warn=False, force=True)
from matplotlib import pyplot as plt
import theano
import theano.tensor as T
import datetime
import utils
import argparse
import timeit
import os
from   model_classes import BoltzmannMachine


np_rand_gen = np.random.RandomState(1234)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

test_images      = np.round(mnist.test.images)

test_labels      = mnist.test.labels

train_images     = np.round(mnist.train.images)

train_labels     = mnist.train.labels

num_train_images = train_images.shape[0]

D                = train_images.shape[1]

assert D == 784

validate_images  = np.round(mnist.validation.images)

validate_labels  = mnist.validation.labels

arg_parser       = argparse.ArgumentParser()

arg_parser.add_argument('--path_to_params', type=str,required= True)

arg_parser.add_argument('--num_iters', type=str,required= True)

arg_parser.add_argument('--num_reconstruct', type=str,required= True)

arg_parser.add_argument('--trained_subset', type = str, required = True)

arg_parser.add_argument('--test_mode', type = str, required = False)

FLAGS, _          = arg_parser.parse_known_args()

path_to_params    = FLAGS.path_to_params

num_iters         = int(FLAGS.num_iters)

num_reconstruct   = int(FLAGS.num_reconstruct)

trained_subset    = int(FLAGS.trained_subset)

if FLAGS.test_mode != None:

   test_mode      = bool(int(FLAGS.test_mode))
   
else:
    
   test_mode = False
   
if "RH" in path_to_params:

   ind0 = path_to_params.find("RH")
   ind1 = path_to_params.find("LR")
   num_hidden = int(path_to_params[ind0+2:ind1])
   
else:
    
   num_hidden = 0

split_path        = os.path.split(path_to_params)

if bool(trained_subset):
    
   indices =np.loadtxt(os.path.join(split_path[0],"LEARNT_INSTANCES.dat"))
   
   indices = np.array(indices, dtype = np.int64)
   
   test_inputs = train_images[indices,:]
   
   if indices.size ==1:
       
      test_inputs = np.reshape(test_inputs,[1,D])
       
      num_reconstruct = 1
      
      num_test        = 1
   
else:
    
   test_inputs = test_images
   
   num_test = test_inputs.shape[0]

if num_test > num_reconstruct:
    
   select_inputs = np.random.choice(num_test, 
                                    num_reconstruct, 
                                    replace = False)
   
   test_inputs = test_inputs[select_inputs,:]
   
elif num_test < num_reconstruct:
    
   num_reconstruct = num_test

bm = BoltzmannMachine(num_vars        = D, 
                      num_hidden      = num_hidden,
                      training        = False)
      
bm.load_model_params(full_path = path_to_params)

###### reconstruction of missing pixels
filename = "RECONST_MISSING"

save_to_path = os.path.join(split_path[0],filename+".jpeg")

which_pixels = utils.select_missing_pixels(gamma = 0.5, 
                                           D= D, 
                                           N= num_reconstruct)

images_to_reconst = np.copy(test_inputs)
blocked_images    = np.copy(test_inputs)
images_to_reconst[which_pixels] =  1
blocked_images[which_pixels]    = -1

images_to_reconst = \
bm.reconstruct_missing_pixels(num_iters = num_iters, 
                              recon_images = images_to_reconst, 
                              which_pixels = which_pixels,
                              test_mode = test_mode)
                                                                            
recon_errors= utils.plot_reconstructions(test_inputs,
                                         blocked_images,
                                         images_to_reconst,
                                         save_to_path)
                                         
np.savetxt(os.path.join(split_path[0],"%s_ERRORS.dat"%filename), 
           recon_errors)

############### reconstruction of noisy data

pflip = 0.1

filename = "RECONST_NOISY"

save_to_path = os.path.join(split_path[0],filename+".jpeg")

which_pixels = utils.select_noisy_pixels(pflip = pflip, 
                                         D     = D, 
                                         N     = num_reconstruct)
                                         
noisy_images = np.copy(test_inputs)

noisy_images[which_pixels] = 1- noisy_images[which_pixels]

images_to_reconst= np.copy(noisy_images)

images_to_reconst = bm.reconstruct_noisy_pixels(num_iters    = num_iters, 
                                                correct_images = test_inputs,
                                                recon_images = images_to_reconst, 
                                                noisy_images = noisy_images,
                                                pflip        = pflip)
                                                
recon_errors= utils.plot_reconstructions(test_inputs,
                                         noisy_images,
                                         images_to_reconst,
                                         save_to_path)
                                         
np.savetxt(os.path.join(split_path[0],"%s_ERRORS.dat"%filename), 
           recon_errors)




